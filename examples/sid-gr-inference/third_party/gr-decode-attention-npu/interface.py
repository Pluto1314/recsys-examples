# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0.
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
# NPU: gr_decode_atten 的 python 等价实现（NPU spike 占位 stub）。
#
# 归属：数学分解逐行取自本仓库分支 main_gr_inference_npu 的
# inference_framework/src/gr_inference/kernels/backends/torch/decode_attention.py
# （torch reference backend），仅做如下适配：
#   1. 模块命名为 interface，使 ExistingGRDecodeAttentionBackend 的 kernel
#      locator（existing_kernel_backend.py，root 解析顺序：显式 root ->
#      GR_DECODE_ATTEN_ROOT -> third_party/gr-decode-attention -> 本目录）在
#      无 CUDA kernel 的 NPU/CPU 环境无需环境变量即可发现本实现；
#   2. backend 参数除 "torch" 外同时接受 GPU 原生 interface.py 的 "dsl"/
#      "3kernel"（框架侧 backend_name 默认 "dsl" 会透传进来），三者均路由到
#      同一份数学实现。
# API/语义与 corelib/gr_decode_atten/interface.py::beam_decode_attn 对齐：
# K1 context attention（dense / dense+seqused_k / jagged cu_seqlens_k）、
# K2 beam sparse attention（topk_indices 绝对下标 gather，GQA 组内共享下标）、
# K3 LSE 合并；全程 fp32 累加。
#
# 这是 950PR 上 gr_decode_atten NPU 算子（ops 侧开发中）就绪前的占位实现：
# 性能不作为目标，正确性以 corelib/gr_decode_atten/tests/reference.py 为
# golden（见 tests/test_npu_beam_decode_attn_golden.py）。CUDA 环境下 locator
# 优先命中 third_party/gr-decode-attention 的原生 CuTe DSL kernel，本文件
# 不参与，GPU 行为不变。
"""Beam decode attention, python-equivalent NPU stub (see module NPU note)."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

_ALLOWED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _validate_inputs(
    q, k_context, v_context, k_beam, v_beam, topk_indices, decode_nums, jagged_k_context
) -> None:
    if q.dim() != 5:
        raise ValueError(f"q must be 5-D [B, Sq, W, Hq, D], got {q.dim()}-D")
    batch, seqlen_q, beam_width, head_q, dim = q.shape
    if seqlen_q != 1:
        raise ValueError(f"decode mode requires seqlen_q == 1, got {seqlen_q}")

    if jagged_k_context:
        if k_context.dim() != 3:
            raise ValueError(
                f"k_context must be 3-D [total_k, Hkv, D] in jagged mode, got {k_context.dim()}-D"
            )
        if v_context.shape != k_context.shape:
            raise ValueError("v_context must match k_context")
        head_kv = k_context.shape[1]
        if k_context.shape[2] != dim:
            raise ValueError("k_context head dim must match q")
    else:
        if k_context.dim() != 4:
            raise ValueError(
                f"k_context must be 4-D [B, Sk, Hkv, D], got {k_context.dim()}-D"
            )
        if v_context.shape != k_context.shape:
            raise ValueError("v_context must match k_context")
        if k_context.shape[0] != batch:
            raise ValueError("k_context batch must match q")
        head_kv = k_context.shape[2]
        if k_context.shape[3] != dim:
            raise ValueError("k_context head dim must match q")

    seqlen_beam = decode_nums * beam_width
    expected_beam = (batch, seqlen_beam, head_kv, dim)
    if tuple(k_beam.shape) != expected_beam:
        raise ValueError(f"k_beam shape: expected {expected_beam}, got {tuple(k_beam.shape)}")
    if v_beam.shape != k_beam.shape:
        raise ValueError("v_beam must match k_beam")

    if topk_indices.dim() != 5:
        raise ValueError("topk_indices must be 5-D [B, Sq, Hq, max_decode_nums, W]")
    if tuple(topk_indices.shape[:3]) != (batch, seqlen_q, head_q):
        raise ValueError(f"topk_indices leading dims must be {(batch, seqlen_q, head_q)}")
    if topk_indices.shape[4] != beam_width:
        raise ValueError("topk_indices beam dim must match q")
    if decode_nums > topk_indices.shape[3]:
        raise ValueError("decode_nums exceeds topk_indices capacity")

    if head_q % head_kv != 0:
        raise ValueError(f"Hq ({head_q}) must be divisible by Hkv ({head_kv})")
    if q.dtype not in _ALLOWED_DTYPES:
        raise ValueError(f"q dtype must be fp16/bf16/fp32, got {q.dtype}")
    dtypes = {t.dtype for t in (q, k_context, v_context, k_beam, v_beam)}
    if len(dtypes) != 1:
        raise ValueError(f"Q/K/V dtypes must agree, got {dtypes}")
    if topk_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("topk_indices must be int32/int64")


def _context_attention(
    q: torch.Tensor,
    k_context: torch.Tensor,
    v_context: torch.Tensor,
    softmax_scale: float,
    seqused_k: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Q x context KV -> fp32 partials ``([B,Sq,W,Hq,D], [B,Sq,W,Hq])``."""
    batch, _seqlen_q, beam_width, head_q, dim = q.shape
    head_kv = k_context.shape[1] if cu_seqlens_k is not None else k_context.shape[2]
    ngroups = head_q // head_kv

    outs, lses = [], []
    bounds = (
        zip(cu_seqlens_k.tolist()[:-1], cu_seqlens_k.tolist()[1:])
        if cu_seqlens_k is not None
        else [(0, k_context.shape[1])] * batch
    )
    for b, (start, end) in enumerate(bounds):
        q_b = q[b, 0].float()  # [W, Hq, D]
        if cu_seqlens_k is not None:
            k_b = k_context[start:end].float()  # [Sk_b, Hkv, D]
            v_b = v_context[start:end].float()
        else:
            k_b = k_context[b, start:end].float()  # [Sk_b, Hkv, D]
            v_b = v_context[b, start:end].float()
        if k_b.shape[0] == 0:
            outs.append(torch.zeros(beam_width, head_q, dim, device=q.device))
            lses.append(torch.full((beam_width, head_q), float("-inf"), device=q.device))
            continue
        if ngroups > 1:
            k_b = k_b.repeat_interleave(ngroups, dim=1)
            v_b = v_b.repeat_interleave(ngroups, dim=1)
        scores = torch.einsum("whd,shd->whs", q_b * softmax_scale, k_b)
        if seqused_k is not None:
            valid = int(seqused_k[b].item()) - start
            if valid <= 0:
                outs.append(torch.zeros(beam_width, head_q, dim, device=q.device))
                lses.append(torch.full((beam_width, head_q), float("-inf"), device=q.device))
                continue
            scores[..., valid:] = float("-inf")
        lses.append(torch.logsumexp(scores, dim=-1))
        attn = torch.softmax(scores, dim=-1)
        outs.append(torch.einsum("whs,shd->whd", attn, v_b))
    out = torch.stack(outs, dim=0).unsqueeze(1)
    lse = torch.stack(lses, dim=0).unsqueeze(1)
    return out, lse


def _beam_sparse_attention(
    q: torch.Tensor,
    k_beam: torch.Tensor,
    v_beam: torch.Tensor,
    topk_indices: torch.Tensor,
    decode_nums: int,
    softmax_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Q x gathered beam KV -> fp32 partials, same layout as context part."""
    batch, seqlen_q, beam_width, head_q, _dim = q.shape
    head_kv = k_beam.shape[2]
    ngroups = head_q // head_kv

    if decode_nums == 0:
        out = torch.zeros(batch, seqlen_q, beam_width, head_q, q.shape[-1], device=q.device)
        lse = torch.full((batch, seqlen_q, beam_width, head_q), float("-inf"), device=q.device)
        return out, lse

    q_f = q.float()
    k_f = k_beam.float()
    v_f = v_beam.float()
    if ngroups > 1:
        k_f = k_f.repeat_interleave(ngroups, dim=2)
        v_f = v_f.repeat_interleave(ngroups, dim=2)

    # [B, Sq, Hq, Dn, W] -> [B, Sq, W, Hq, Dn]
    idx = topk_indices[:, :, :, :decode_nums, :].permute(0, 1, 4, 2, 3).contiguous()
    b_idx = torch.arange(batch, device=q.device)[:, None, None, None, None]
    h_idx = torch.arange(head_q, device=q.device)[None, None, None, :, None]
    k_gathered = k_f[b_idx, idx, h_idx]  # [B, Sq, W, Hq, Dn, D]
    v_gathered = v_f[b_idx, idx, h_idx]

    scores = torch.einsum("bqwhd,bqwhnd->bqwhn", q_f * softmax_scale, k_gathered)
    lse = torch.logsumexp(scores, dim=-1)
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("bqwhn,bqwhnd->bqwhd", attn, v_gathered)
    return out, lse


def _combine(
    out1: torch.Tensor,
    lse1: torch.Tensor,
    out2: torch.Tensor,
    lse2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Log-sum-exp merge of the two fp32 partials."""
    m = torch.maximum(lse1, lse2)
    safe_m = torch.where(m.isinf() & (m < 0), torch.zeros_like(m), m)
    exp1 = torch.exp(lse1 - safe_m)
    exp2 = torch.exp(lse2 - safe_m)
    total = exp1 + exp2
    lse = torch.where(total == 0, torch.full_like(safe_m, float("-inf")), safe_m + torch.log(total))
    inv_total = torch.where(total == 0, torch.zeros_like(total), 1.0 / total)
    out = (exp1 * inv_total).unsqueeze(-1) * out1 + (exp2 * inv_total).unsqueeze(-1) * out2
    return out, lse


# NPU: GPU 原生 interface 接受 "dsl"/"3kernel"；框架 ExistingGRDecodeAttentionBackend
# 会把 backend_name（默认 "dsl"）透传为 backend，这里三者同名路由到同一实现。
_STUB_BACKEND_ALIASES = ("torch", "dsl", "3kernel")


def beam_decode_attn(
    q: torch.Tensor,
    k_context: torch.Tensor,
    v_context: torch.Tensor,
    k_beam: torch.Tensor,
    v_beam: torch.Tensor,
    topk_indices: torch.Tensor,
    decode_nums: int,
    softmax_scale: Optional[float] = None,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    backend: str = "torch",
    seqused_k: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Beam decode attention (python-equivalent NPU stub). See the module NPU note."""
    if backend not in _STUB_BACKEND_ALIASES:
        raise ValueError(
            f"backend must be one of {_STUB_BACKEND_ALIASES} on this stub, got {backend!r}"
        )
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    if seqused_k is not None:
        if seqused_k.dim() != 1 or seqused_k.shape[0] != q.shape[0]:
            raise ValueError(f"seqused_k must be [B={q.shape[0]}], got {tuple(seqused_k.shape)}")
        if seqused_k.dtype != torch.int32:
            raise ValueError("seqused_k must be int32")
        if cu_seqlens_k is not None:
            raise ValueError("cu_seqlens_k and seqused_k are mutually exclusive")
    if cu_seqlens_k is not None:
        if cu_seqlens_k.dim() != 1 or cu_seqlens_k.shape[0] != q.shape[0] + 1:
            raise ValueError(
                f"cu_seqlens_k must be [B+1={q.shape[0] + 1}], got {tuple(cu_seqlens_k.shape)}"
            )
        if cu_seqlens_k.dtype != torch.int32:
            raise ValueError("cu_seqlens_k must be int32")
    _validate_inputs(
        q, k_context, v_context, k_beam, v_beam, topk_indices, decode_nums,
        jagged_k_context=cu_seqlens_k is not None,
    )

    out_ctx, lse_ctx = _context_attention(q, k_context, v_context, softmax_scale, seqused_k, cu_seqlens_k)
    out_beam, lse_beam = _beam_sparse_attention(q, k_beam, v_beam, topk_indices, decode_nums, softmax_scale)
    combined_out, combined_lse = _combine(out_ctx, lse_ctx, out_beam, lse_beam)

    result = combined_out.to(q.dtype)
    if out is not None:
        if out.shape != q.shape or out.dtype != q.dtype:
            raise ValueError("pre-allocated out must match q's shape and dtype")
        out.copy_(result)
        result = out
    return result, combined_lse if return_lse else None


def make_decode_attention_backends():
    return {"beam_decode_attn": beam_decode_attn}


__all__ = ["beam_decode_attn", "make_decode_attention_backends"]
