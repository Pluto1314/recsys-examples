# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""NPU: paged_kvcache_ops / hstu(FBGEMM) -> kvcache_npu_cpp / ops_rec 适配层。

- append_kvcache: GPU 14 参（尾参 num_sms）-> kvcache_npu_cpp.append_kvcache 13 参；
  nnz 由 metadata.new_history_nnz_npu（设备侧）给出，k/v 仍传 unbind 视图（kernel 按 stride 收）。
- hstu_attn_varlen_func: 保持例程（paged_hstu_infer_layer）的原位参序，内部分发：
  * 带 kv_cache  -> ops_rec.ascendc.attention.hstu_paged（传整表 [num_pages,2,page_size,N,D]，不 unbind；
    window (-1,0)->mask_type=0、(-1,-1)->mask_type=2；scaling_seqlen(除法)->silu_scale(乘法)；
    显式传 alpha=1/sqrt(head_dim)——NPU_GR.patch 漏传会丢 qk 缩放）。
  * 无 kv_cache -> ops_rec.ascendc.attention.hstu_jagged（equal overload）。
- HSTU_NPU_DRY_RUN=1 时改走本文件内的纯 torch 参考实现（scatter 版 append、
  gather 页后的 masked silu-attention），供无 NPU 环境结构/数值干检。
"""
import os
from typing import Optional

import torch
import torch.nn.functional as F

from .kvcache_compat import HSTU_NPU_DRY_RUN

if not HSTU_NPU_DRY_RUN:
    import kvcache_npu_cpp  # noqa: F401  NPU: pybind 名与 GPU 包不同（kvcache_cpp -> kvcache_npu_cpp）
    import ops_rec.ascendc.attention as _ascendc_attention


# ---------------------------------------------------------------------------
# append_kvcache
# ---------------------------------------------------------------------------
def _append_kvcache_reference(
    key,
    value,
    batch_indices,
    position,
    seqlen_offsets,
    nnz_npu,
    nnz,
    paged_k_cache,
    paged_v_cache,
    kv_indices,
    kv_indptr,
    kv_last_page_len,
    kv_layout,
):
    """CUDA AppendPagedKVCacheKernel 的 torch 等价（就地 scatter）。

    第 i 个新 token（全局序号 i < nnz_npu[0]）：
      源行 = i + seqlen_offsets[batch_indices[i]]；目的页 =
      kv_indices[kv_indptr[batch_indices[i]] + position[i] // page_size]，
      页内槽位 = position[i] % page_size（NHD 布局）。
    """
    assert kv_layout == 0, "reference only supports NHD layout"
    n = int(nnz_npu.reshape(-1)[0].item())
    if n == 0:
        return
    page_size = paged_k_cache.size(1)
    bi = batch_indices[:n].to(torch.long)
    pos = position[:n].to(torch.long)
    src_rows = torch.arange(n, device=key.device) + seqlen_offsets.to(torch.long)[bi]
    page_ids = kv_indices.to(torch.long)[kv_indptr.to(torch.long)[bi] + pos // page_size]
    slots = pos % page_size
    paged_k_cache[page_ids, slots] = key[src_rows].to(paged_k_cache.dtype)
    paged_v_cache[page_ids, slots] = value[src_rows].to(paged_v_cache.dtype)


def append_kvcache(
    key,
    value,
    batch_indices,
    position,
    seqlen_offsets,
    new_history_nnz_npu,
    new_history_nnz,
    paged_k_cache,
    paged_v_cache,
    kv_indices,
    kv_indptr,
    kv_last_page_len,
    kv_layout,
    num_sms=0,  # NPU: 仅作 GPU 兼容参数保留，kernel 核数由 PlatformAscendCManager 推导
):
    if HSTU_NPU_DRY_RUN:
        _append_kvcache_reference(
            key,
            value,
            batch_indices,
            position,
            seqlen_offsets,
            new_history_nnz_npu,
            new_history_nnz,
            paged_k_cache,
            paged_v_cache,
            kv_indices,
            kv_indptr,
            kv_last_page_len,
            kv_layout,
        )
    else:
        # NPU: 13 参绑定（去尾参 num_sms）；nnz 走设备侧 new_history_nnz_npu
        kvcache_npu_cpp.append_kvcache(
            key,
            value,
            batch_indices,
            position,
            seqlen_offsets,
            new_history_nnz_npu,
            new_history_nnz,
            paged_k_cache,
            paged_v_cache,
            kv_indices,
            kv_indptr,
            kv_last_page_len,
            kv_layout,
        )


# ---------------------------------------------------------------------------
# hstu attention
# ---------------------------------------------------------------------------
def _window_to_mask_type(window_size) -> int:
    if window_size == (-1, 0):
        return 0  # tril (causal)
    if window_size == (-1, -1):
        return 2  # no mask
    raise ValueError(f"Unsupported window_size for ops_rec hstu: {window_size}")


def _hstu_attn_reference(
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
    scaling_seqlen, num_targets, target_group_size, alpha,
    kv_cache=None, page_offsets=None, page_ids=None, last_page_lens=None,
    cu_seqlens_t=None,
):
    """纯 torch 参考：gather 页后的 masked silu-attention（golden 对齐
    examples/hstu/test/test_paged_hstu_attn_kernel.py::_hstu_paged_kv_attention）。

    out = ( silu(q @ k^T * alpha) * silu_scale * tril_mask ) @ v，与 ops_rec 公式一致；
    paged 时 K/V = 缓存页(gather, 历史) ++ 直接输入 k/v（q 段尾部的 target 行），
    delta-q：query 绝对位置 = kv_len - Lq + i。
    """
    cu_q = cu_seqlens_q.to(torch.long).tolist()
    cu_k = cu_seqlens_k.to(torch.long).tolist()
    batch = len(cu_q) - 1
    silu_scale = 1.0 / (max_seqlen_q if scaling_seqlen in (-1, None) else scaling_seqlen)
    out_dtype = q.dtype
    outs = []
    for b in range(batch):
        q_b = q[cu_q[b] : cu_q[b + 1]].to(torch.float32)  # (Lq, H, D)
        Lq, H, D = q_b.shape
        kv_len = cu_k[b + 1] - cu_k[b]
        if kv_cache is not None:
            # 直接输入只取 q 段尾部 num_target 行（target/candidate）；历史走缓存页
            num_tgt = int(num_targets[b].item()) if num_targets is not None else 0
            k_new = k[cu_q[b + 1] - num_tgt : cu_q[b + 1]].to(torch.float32)
            v_new = v[cu_q[b + 1] - num_tgt : cu_q[b + 1]].to(torch.float32)
            page_size = kv_cache.size(2)
            pg_off = page_offsets.to(torch.long)
            pages = kv_cache.to(torch.float32)[
                page_ids.to(torch.long)[pg_off[b] : pg_off[b + 1]]
            ]
            cached_len = int(
                (pages.size(0) - 1) * page_size + last_page_lens.to(torch.long)[b]
            )
            k_cached = pages[:, 0].reshape(-1, H, D)[:cached_len]
            v_cached = pages[:, 1].reshape(-1, H, D)[:cached_len]
            k_b = torch.cat([k_cached, k_new], dim=0)
            v_b = torch.cat([v_cached, v_new], dim=0)
            assert kv_len == cached_len + num_tgt, (
                f"paged kv_len mismatch: kv_len={kv_len} cached={cached_len} tgt={num_tgt}"
            )
        else:
            k_b = k[cu_k[b] : cu_k[b + 1]].to(torch.float32)
            v_b = v[cu_k[b] : cu_k[b + 1]].to(torch.float32)
        scores = torch.einsum("ihd,jhd->hij", q_b, k_b) * alpha
        attn = F.silu(scores) * silu_scale
        # tril 语义（含 target 单位块）：query i 可见 key j <= 绝对位置
        abs_pos = torch.arange(kv_len - Lq, kv_len, device=q.device)
        mask = abs_pos.view(1, -1) <= abs_pos.view(-1, 1)  # (Lq, Sk)
        attn = attn * mask.view(1, Lq, -1).to(attn.dtype)
        out_b = torch.einsum("hij,jhd->ihd", attn, v_b)
        outs.append(out_b.to(out_dtype))
    return torch.cat(outs, dim=0)


def hstu_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_q=None,
    seqused_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    scaling_seqlen=-1,
    num_contexts=None,
    num_targets=None,
    target_group_size=1,
    window_size=(-1, -1),
    alpha=1.0,
    kv_cache=None,
    page_offsets=None,
    page_ids=None,
    last_page_lens=None,
    cu_seqlens_t=None,
):
    # NPU: 例程（paged_hstu_infer_layer）的原位参序在此收敛映射到 ops_rec
    assert seqused_q is None and seqused_k is None, "seqused_* unsupported on NPU path"
    assert num_contexts is None, "num_contexts unsupported on NPU path"
    mask_type = _window_to_mask_type(window_size)
    silu_scale = 1.0 / (
        max_seqlen_q if scaling_seqlen in (-1, None) else scaling_seqlen
    )

    if HSTU_NPU_DRY_RUN:
        return _hstu_attn_reference(
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            scaling_seqlen, num_targets, target_group_size, alpha,
            kv_cache=kv_cache, page_offsets=page_offsets, page_ids=page_ids,
            last_page_lens=last_page_lens, cu_seqlens_t=cu_seqlens_t,
        )

    if kv_cache is not None:
        # paged：传整表 [num_pages, 2, page_size, N, D]（与 NPU kvcache 表布局一致）
        return _ascendc_attention.hstu_paged(
            q,
            k,
            v,
            kv_cache=kv_cache,
            mask=None,
            attn_bias=None,
            mask_type=mask_type,
            max_seq_len=max_seqlen_q,
            max_seq_len_k=max_seqlen_k,
            silu_scale=silu_scale,
            alpha=alpha,
            seq_offset=cu_seqlens_q,
            seq_offset_k=cu_seqlens_k,
            seq_offset_t=cu_seqlens_t,
            page_offsets=page_offsets,
            page_ids=page_ids,
            last_page_len=last_page_lens,
            num_target=num_targets,
            target_group_size=target_group_size,
        )
    # jagged：equal overload（不传 max_seq_len_k / seq_offset_k）
    return _ascendc_attention.hstu_jagged(
        q,
        k,
        v,
        mask=None,
        attn_bias=None,
        mask_type=mask_type,
        max_seq_len=max_seqlen_q,
        silu_scale=silu_scale,
        alpha=alpha,
        seq_offset=cu_seqlens_q,
        num_context=num_contexts,
        num_target=num_targets,
        target_group_size=target_group_size,
    )
