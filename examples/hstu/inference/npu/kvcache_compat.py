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
"""NPU: recsys_kvcache_manager(GPU) -> recsys_kvcache_manager_npu 适配层。

- 例程侧保持 `from <compat> import KVCacheManager / KVCacheConfig / ...` 的 GPU 包用法不变。
- 差异收敛在此处（详见穿刺方案 §C）：
  * 包名 recsys_kvcache_manager -> recsys_kvcache_manager_npu；pybind kvcache_cpp -> kvcache_npu_cpp。
  * GPUKVCacheManager -> NPUKVCacheManager；属性 .gpu_kvcache_mgr -> .npu_kvcache_mgr，
    .gpu_kvcache_tables -> .npu_kvcache_tables。
  * KVCacheMetadata 字段改名：page_ids_gpu_buffer/metadata_gpu_buffer -> *_npu_buffer，
    new_history_nnz_cuda -> new_history_nnz_npu（消费点在 paged_hstu_infer_layer 中直接改名）。
  * KVLookupResult: gpu_cached_* -> npu_cached_*（例程只读 .cached_lengths，不受影响）。
  * host_kvstorage_backend 仅支持 "native_host"（无 flexkv）。
  * NPU 包无 copy_kvcache_metadata：自 corelib/.../kvcache_metadata.py:134-157 逐行移植（仅 cudagraph 路径消费）。
- HSTU_NPU_DRY_RUN=1 时注入 CPU 参考 KVCacheManager（始终 miss、整批重算），供无 NPU 环境结构干检。
"""
import os
from dataclasses import dataclass
from typing import List, Optional

import torch

# NPU: 干检开关集中在适配层，不散落到各调用点
HSTU_NPU_DRY_RUN = os.environ.get("HSTU_NPU_DRY_RUN", "0") == "1"

if HSTU_NPU_DRY_RUN:
    # ---- CPU 参考实现（仅 Windows/无卡结构干检；镜像 NPU 包 dataclass 形状） ----

    @dataclass
    class KVCacheConfig:
        num_layers: int = 3
        num_heads: int = 4
        head_dim: int = 128
        page_size: int = 32
        offload_chunksize: int = 1024
        num_primary_cache_pages: int = 10240
        num_buffer_pages: int = 0
        host_capacity_per_layer: int = 0
        max_batch_size: int = 1
        max_seq_len: int = 8192
        dtype: torch.dtype = torch.bfloat16
        device: int = 0
        host_kvstorage_backend: str = "native_host"
        onload_timeout_ms: float = 0.0
        offload_timeout_ms: float = 1000.0
        offload_mode: str = "lazy"

    def get_kvcache_config(**kwargs) -> KVCacheConfig:
        return KVCacheConfig(**kwargs)

    @dataclass
    class KVCacheMetadata:
        # NPU: 字段名与 recsys_kvcache_manager_npu 对齐
        page_ids_npu_buffer: Optional[torch.Tensor] = None
        metadata_npu_buffer: Optional[torch.Tensor] = None
        kv_indices: Optional[torch.Tensor] = None
        kv_indptr: Optional[torch.Tensor] = None
        kv_last_page_len: Optional[torch.Tensor] = None
        total_history_lengths: Optional[torch.Tensor] = None
        total_history_offsets: Optional[torch.Tensor] = None
        new_history_offsets: Optional[torch.Tensor] = None
        batch_indices: Optional[torch.Tensor] = None
        position: Optional[torch.Tensor] = None
        new_history_nnz: int = 0
        new_history_nnz_npu: Optional[torch.Tensor] = None
        kv_seqlens: Optional[torch.Tensor] = None
        kv_seqlen_offsets: Optional[torch.Tensor] = None
        kv_cache_table: Optional[List[torch.Tensor]] = None
        kv_onload_handle: Optional[object] = None
        max_seqlen: Optional[int] = 0

    def get_kvcache_metadata_buffer(batch_size, num_new_tokens, num_pages, device=None):
        # 与 NPU 包 get_kvcache_metadata_buffer 相同的 narrow 布局（npu buffer 名）
        metadata_npu_buffer = torch.zeros(
            (5 * batch_size + 4 + num_new_tokens * 2,), dtype=torch.int32
        )
        return KVCacheMetadata(
            page_ids_npu_buffer=torch.zeros((num_pages,), dtype=torch.int32),
            metadata_npu_buffer=metadata_npu_buffer,
            kv_indices=metadata_npu_buffer.narrow(0, 0, 0),  # 由 manager allocate 重绑
            kv_indptr=metadata_npu_buffer.narrow(0, 0, batch_size + 1),
            kv_last_page_len=metadata_npu_buffer.narrow(0, batch_size + 1, batch_size),
            total_history_lengths=metadata_npu_buffer.narrow(
                0, batch_size * 2 + 1, batch_size
            ),
            total_history_offsets=metadata_npu_buffer.narrow(
                0, batch_size * 3 + 1, batch_size + 1
            ),
            new_history_nnz_npu=metadata_npu_buffer.narrow(0, batch_size * 4 + 2, 1),
            new_history_offsets=metadata_npu_buffer.narrow(
                0, batch_size * 4 + 3, batch_size + 1
            ),
            batch_indices=metadata_npu_buffer.narrow(
                0, batch_size * 5 + 4, num_new_tokens
            ),
            position=metadata_npu_buffer.narrow(
                0, batch_size * 5 + 4 + num_new_tokens, num_new_tokens
            ),
            new_history_nnz=num_new_tokens,
        )

    @dataclass
    class KVLookupResult:
        cached_lengths: Optional[torch.Tensor] = None
        npu_cached_start_indices: Optional[torch.Tensor] = None
        npu_cached_lengths: Optional[torch.Tensor] = None
        host_cached_start_indices: Optional[torch.Tensor] = None
        host_cached_lengths: Optional[torch.Tensor] = None

    @dataclass
    class KVIndexMeta:
        user_ids: Optional[torch.Tensor] = None
        sequence_lengths: Optional[torch.Tensor] = None

    from enum import Enum

    class HostKVTaskStatus(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        FAILED = "failed"
        SKIPPED = "skipped"

    class _CPUKVOnloadHandle:
        # native 后端 onboard_wait 为 batch 级 no-op（SKIPPED），逐层 wait_layer 同样 no-op
        def __init__(self, num_layers: int = 1):
            self.handle = None
            self.status = HostKVTaskStatus.SKIPPED
            self.backend = "native"

        def stream_wait_layer(self, layer_idx: int) -> None:
            return None

    class KVCacheManager:
        """CPU 参考 manager：lookup 恒 miss -> 整批历史重算并写回页表。

        仅用于 HSTU_NPU_DRY_RUN=1 干检；页表/元数据布局与 NPU 包一致，
        使 paged_ops_compat 的 CPU 参考 append/attention 能走通完整 metadata 管道。
        """

        def __init__(self, kvcache_config: KVCacheConfig):
            self._config = kvcache_config
            self.npu_kvcache_mgr = self  # NPU: 属性名 .npu_kvcache_mgr
            ps, H, D = (
                kvcache_config.page_size,
                kvcache_config.num_heads,
                kvcache_config.head_dim,
            )
            self.npu_kvcache_tables = [
                torch.zeros(
                    (kvcache_config.num_primary_cache_pages, 2, ps, H, D),
                    dtype=kvcache_config.dtype,
                )
                for _ in range(kvcache_config.num_layers)
            ]

        @classmethod
        def from_config(cls, kvcache_config: KVCacheConfig) -> "KVCacheManager":
            return cls(kvcache_config)

        def lookup_kvcache(self, user_ids, seq_hist_lengths):
            lookup = KVLookupResult(
                cached_lengths=torch.zeros_like(
                    seq_hist_lengths, dtype=torch.int32
                ).to(torch.int64)
            )
            return KVIndexMeta(user_ids=user_ids, sequence_lengths=seq_hist_lengths), lookup

        def allocate_kvcache(self, index_meta, lookup_result):
            cfg = self._config
            hist = index_meta.sequence_lengths.to(torch.int64)
            batch_size = hist.numel()
            pages = torch.div(
                hist + cfg.page_size - 1, cfg.page_size, rounding_mode="floor"
            ).to(torch.int32)
            num_pages = int(pages.sum().item())
            metadata = get_kvcache_metadata_buffer(
                batch_size, int(hist.sum().item()), max(num_pages, 1)
            )
            metadata.kv_indices = torch.arange(num_pages, dtype=torch.int32)
            metadata.kv_indptr = torch.cat(
                [torch.zeros(1, dtype=torch.int32), pages.cumsum(0, dtype=torch.int32)]
            )
            last_page = hist - (pages - 1) * cfg.page_size
            metadata.kv_last_page_len = torch.where(
                pages > 0, last_page, torch.zeros_like(last_page)
            ).to(torch.int32)
            metadata.total_history_lengths = hist.to(torch.int32)
            metadata.total_history_offsets = torch.cat(
                [torch.zeros(1, dtype=torch.int32), hist.cumsum(0, dtype=torch.int32)]
            )
            metadata.new_history_offsets = metadata.total_history_offsets
            # 恒 miss：新 history = 全部 history；position 为用户内序号（kernel 语义）
            metadata.new_history_nnz = int(hist.sum().item())
            metadata.new_history_nnz_npu = torch.tensor(
                [metadata.new_history_nnz], dtype=torch.int32
            )
            rep_b = torch.repeat_interleave(
                torch.arange(batch_size, dtype=torch.int32),
                hist.to(torch.int32),
                output_size=metadata.new_history_nnz,
            )
            metadata.batch_indices = rep_b
            metadata.position = torch.cat(
                [torch.arange(n, dtype=torch.int32) for n in hist.tolist()]
            )
            metadata.kv_cache_table = self.npu_kvcache_tables
            metadata.max_seqlen = int(hist.max().item()) if batch_size else 0
            return metadata

        def onboard_launch(self, index_meta, lookup_result, kvcache_metadata):
            kvcache_metadata.kv_onload_handle = _CPUKVOnloadHandle(
                self._config.num_layers
            )
            return kvcache_metadata.kv_onload_handle

        def onboard_wait(self, index_meta, task_handle):
            return None

        def offload_try_wait(self):
            return None

        def offload_launch(self, index_meta, kvcache_metadata):
            return None

        def evict(self, user_ids, for_npu=False, for_host=False):
            return None

        def evict_all(self, for_npu=False, for_host=False):
            return None

else:
    # ---- 真实 NPU 路径：re-export recsys_kvcache_manager_npu ----
    from recsys_kvcache_manager_npu import (  # noqa: F401
        KVCacheConfig,
        KVCacheManager,
        KVCacheOffloadMode,
        NPUKVCacheManager,
    )
    from recsys_kvcache_manager_npu.host_kvstorage_manager import (  # noqa: F401
        HostKVTaskStatus,
    )
    from recsys_kvcache_manager_npu.kvcache_config import get_kvcache_config  # noqa: F401
    from recsys_kvcache_manager_npu.kvcache_metadata import (  # noqa: F401
        KVCacheMetadata,
        get_kvcache_metadata_buffer,
    )
    from recsys_kvcache_manager_npu.kvcache_utils import (  # noqa: F401
        KVIndexMeta,
        KVLookupResult,
    )


def copy_kvcache_metadata(dst_metadata: KVCacheMetadata, src_metadata: KVCacheMetadata):
    # NPU: 包内缺失，自 corelib/.../kvcache_metadata.py:134-157 移植；字段用 npu 名。
    # 唯一消费方是 cudagraph 路径（默认关闭）。
    def copy_tensor(dst, src):
        dst[: src.shape[0], ...].copy_(src, non_blocking=True)
        dst[src.shape[0] :, ...] = 0

    def copy_offsets(dst, src):
        dst[: src.shape[0], ...].copy_(src, non_blocking=True)
        dst[src.shape[0] :, ...] = src[-1, ...]

    copy_tensor(dst_metadata.kv_indices, src_metadata.kv_indices)
    copy_offsets(dst_metadata.kv_indptr, src_metadata.kv_indptr)
    copy_tensor(dst_metadata.kv_last_page_len, src_metadata.kv_last_page_len)
    copy_tensor(dst_metadata.batch_indices, src_metadata.batch_indices)
    copy_tensor(dst_metadata.position, src_metadata.position)
    copy_tensor(dst_metadata.new_history_nnz_npu, src_metadata.new_history_nnz_npu)
    copy_offsets(dst_metadata.total_history_offsets, src_metadata.total_history_offsets)
    copy_offsets(dst_metadata.new_history_offsets, src_metadata.new_history_offsets)
    copy_tensor(dst_metadata.kv_seqlens, src_metadata.kv_seqlens)
    copy_offsets(dst_metadata.kv_seqlen_offsets, src_metadata.kv_seqlen_offsets)

    dst_metadata.new_history_nnz = src_metadata.new_history_nnz

    dst_metadata.kv_onload_handle = src_metadata.kv_onload_handle
