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

# NPU: 纯 torch 位置编码（替代 triton_position.triton_add_position_embeddings），
# 先例：gr_npu.patch pt_position。语义对齐 _add_position_embeddings_kernel：
#   out[t] = jagged[t] * (scale if scale != 1) + dense[clamp(local_pos + ind_offset_b, max=high_inds[b])]

from typing import Optional

import torch


@torch.fx.wrap
def add_position_embeddings(
    jagged: torch.Tensor,
    jagged_offsets: torch.Tensor,
    high_inds: torch.Tensor,
    max_seq_len: int,
    dense: torch.Tensor,
    scale: float = 1.0,
    ind_offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    total, dim = jagged.shape
    batch = high_inds.size(0)
    device = jagged.device
    lengths = (jagged_offsets[1:] - jagged_offsets[:-1]).to(torch.long)
    token_pos = torch.arange(total, device=device)
    seq_id = torch.repeat_interleave(
        torch.arange(batch, device=device), lengths, output_size=total
    )
    local_pos = token_pos - jagged_offsets[:-1].to(torch.long)[seq_id]
    if ind_offsets is not None:
        local_pos = local_pos + ind_offsets.to(torch.long)[seq_id]
    pos_idx = torch.clamp(local_pos, max=high_inds.to(torch.long)[seq_id])
    out = jagged if scale == 1.0 else jagged * scale
    return out + dense[pos_idx]


@torch.fx.wrap
def add_timestamp_positional_embeddings(
    seq_embeddings: torch.Tensor,
    seq_offsets: torch.Tensor,
    pos_embeddings: torch.Tensor,
    ts_embeddings: torch.Tensor,
    timestamps: Optional[torch.Tensor],
    max_seq_len: int,
    max_contextual_seq_len: int,
    seq_lengths: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    interleave_targets: bool,
    time_bucket_fn: str,
) -> torch.Tensor:
    raise NotImplementedError(
        "NPU pt_position: timestamp positional encoding is not implemented in this "
        "spike (inference entry uses use_time_encoding=False)."
    )
