# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3

# pyre-strict

# NPU: 纯 torch 版 jagged 2D concat/split（替代 CUDA 扩展 jagged_2D_tensor_concat 与
# triton triton_split_2D_jagged），内容取自 NPU_GR.patch pt_jagged_tensors；
# jagged_to_padded_dense 在 fbgemm_ascend 可用时优先走 torch.ops.fbgemm，否则 torch 回退。

from typing import List, Optional, Tuple

import torch


def _fbgemm_pad_available() -> bool:
    try:
        torch.ops.fbgemm.jagged_to_padded_dense  # noqa: B018
        return True
    except Exception:
        return False


_FBGEMM_PAD_AVAILABLE: Optional[bool] = None


def _pad_jagged_torch(
    values: torch.Tensor, offsets: torch.Tensor, max_length: int
) -> torch.Tensor:
    batch_size = offsets.size(0) - 1
    lengths = offsets[1:] - offsets[:-1]
    out = values.new_zeros((batch_size, max_length, values.size(-1)))
    pos = torch.arange(max_length, device=values.device)
    mask = pos.unsqueeze(0) < lengths.unsqueeze(1)
    b_idx = (
        torch.arange(batch_size, device=values.device).unsqueeze(1).expand_as(mask)
    )[mask]
    l_idx = pos.unsqueeze(0).expand_as(mask)[mask]
    src_rows = offsets[:-1].unsqueeze(1).expand_as(mask)[mask] + l_idx
    out.view(batch_size * max_length, -1)[b_idx * max_length + l_idx] = values[
        src_rows
    ]
    return out


def _jagged_to_padded_dense(
    values: torch.Tensor, offsets: torch.Tensor, max_length: int
) -> torch.Tensor:
    global _FBGEMM_PAD_AVAILABLE
    if _FBGEMM_PAD_AVAILABLE is None:
        _FBGEMM_PAD_AVAILABLE = _fbgemm_pad_available()
    if _FBGEMM_PAD_AVAILABLE:
        return torch.ops.fbgemm.jagged_to_padded_dense(
            values=values, offsets=[offsets], max_lengths=[max_length], padding_value=0.0
        )
    return _pad_jagged_torch(values, offsets, max_length)


@torch.fx.wrap
def fx_arange(len: int, device: torch.device) -> torch.Tensor:
    return torch.arange(len, device=device)


def _concat_2D_jagged_jagged(
    values_left: torch.Tensor,
    values_right: torch.Tensor,
    max_len_left: int,
    max_len_right: int,
    offsets_left: torch.Tensor,
    offsets_right: torch.Tensor,
) -> torch.Tensor:
    max_seq_len = max_len_left + max_len_right
    lengths_left = offsets_left[1:] - offsets_left[:-1]
    lengths_right = offsets_right[1:] - offsets_right[:-1]
    padded_left = _jagged_to_padded_dense(values_left, offsets_left, max_len_left)
    padded_right = _jagged_to_padded_dense(values_right, offsets_right, max_len_right)
    concatted_dense = torch.cat([padded_left, padded_right], dim=1)
    mask = fx_arange(max_seq_len, device=offsets_left.device).view(1, -1)
    mask = torch.logical_or(
        mask < lengths_left.view(-1, 1),
        torch.logical_and(
            mask >= max_len_left,
            mask < max_len_left + lengths_right.view(-1, 1),
        ),
    )
    return concatted_dense.flatten(0, 1)[mask.view(-1), :]


@torch.fx.wrap
def pytorch_concat_2D_jagged(
    values_left: torch.Tensor,
    values_right: torch.Tensor,
    max_len_left: Optional[int],
    max_len_right: Optional[int],
    offsets_left: Optional[torch.Tensor],
    offsets_right: Optional[torch.Tensor],
) -> torch.Tensor:
    if offsets_left is None:
        assert max_len_left is not None
        B = values_left.shape[0] // max_len_left
        offsets_left_non_optional = max_len_left * torch.arange(
            B + 1, device=values_left.device
        )
    else:
        offsets_left_non_optional = offsets_left
    if offsets_right is None:
        assert max_len_right is not None
        B = values_right.shape[0] // max_len_right
        offsets_right_non_optional = max_len_right * torch.arange(
            B + 1, device=values_left.device
        )
    else:
        offsets_right_non_optional = offsets_right
    max_len_left = (
        int(
            (offsets_left_non_optional[1:] - offsets_left_non_optional[:-1])
            .max()
            .item()
        )
        if max_len_left is None
        else max_len_left
    )
    max_len_right = (
        int(
            (offsets_right_non_optional[1:] - offsets_right_non_optional[:-1])
            .max()
            .item()
        )
        if max_len_right is None
        else max_len_right
    )
    return _concat_2D_jagged_jagged(
        values_left=values_left,
        values_right=values_right,
        max_len_left=max_len_left,
        max_len_right=max_len_right,
        offsets_left=offsets_left_non_optional,
        offsets_right=offsets_right_non_optional,
    )


def _split_2D_jagged_jagged(
    max_seq_len: int,
    values: torch.Tensor,
    offsets_left: torch.Tensor,
    offsets_right: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    offsets = offsets_left + offsets_right
    padded_values = _jagged_to_padded_dense(values, offsets, max_seq_len).flatten(0, 1)
    lengths_left = offsets_left[1:] - offsets_left[:-1]
    lengths_right = offsets_right[1:] - offsets_right[:-1]
    mask = fx_arange(max_seq_len, device=values.device).view(1, -1)
    mask_left = mask < lengths_left.view(-1, 1)
    mask_right = torch.logical_and(
        mask >= lengths_left.view(-1, 1),
        mask < (lengths_left + lengths_right).view(-1, 1),
    )
    return padded_values[mask_left.view(-1), :], padded_values[mask_right.view(-1), :]


@torch.fx.wrap
def pytorch_split_2D_jagged(
    max_seq_len: int,
    values: torch.Tensor,
    max_len_left: Optional[int],
    max_len_right: Optional[int],
    offsets_left: Optional[torch.Tensor],
    offsets_right: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if offsets_left is None:
        assert max_len_left is not None
        assert offsets_right is not None
        offsets_left_non_optional = max_len_left * torch.arange(
            offsets_right.shape[0], device=values.device
        )
    else:
        offsets_left_non_optional = offsets_left
    if offsets_right is None:
        assert max_len_right is not None
        assert offsets_left is not None
        offsets_right_non_optional = max_len_right * torch.arange(
            offsets_left.shape[0], device=values.device
        )
    else:
        offsets_right_non_optional = offsets_right
    return _split_2D_jagged_jagged(
        max_seq_len=max_seq_len,
        values=values,
        offsets_left=offsets_left_non_optional,
        offsets_right=offsets_right_non_optional,
    )


# NPU: 以下两个包装保持被替换算子的调用签名，供 hstu_processor 无缝切换
def jagged_2D_tensor_concat(
    values_list: List[torch.Tensor],
    offsets_list: List[torch.Tensor],
    max_seqlens: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """对齐 commons.ops.cuda_ops.JaggedTensorOpFunction.jagged_2D_tensor_concat 签名：
    多个 jagged 张量按用户逐行拼接，返回 (values, lengths)。"""
    assert len(values_list) == len(offsets_list) == len(max_seqlens)
    values, offsets, max_len = values_list[0], offsets_list[0], max_seqlens[0]
    for i in range(1, len(values_list)):
        max_len_acc = max_len + max_seqlens[i]
        values = _concat_2D_jagged_jagged(
            values, values_list[i], max_len, max_seqlens[i], offsets, offsets_list[i]
        )
        lengths = (offsets[1:] - offsets[:-1]) + (
            offsets_list[i][1:] - offsets_list[i][:-1]
        )
        offsets = torch.cat(
            [torch.zeros(1, dtype=lengths.dtype, device=lengths.device), lengths.cumsum(0)]
        )
        max_len = max_len_acc
    return values, offsets[1:] - offsets[:-1]


def split_2D_jagged(
    values: torch.Tensor,
    max_seq_len: int,
    offsets_a: Optional[torch.Tensor] = None,
    offsets_b: Optional[torch.Tensor] = None,
    dense_size: int = 0,
    n_prefix_to_right: int = 0,
    seq_len_a: Optional[int] = None,  # 兼容参数：pt 实现内自算 mask，不需要
    seq_len_b: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """对齐 commons.ops.triton_ops.triton_jagged.triton_split_2D_jagged 签名。"""
    assert dense_size == 0 and n_prefix_to_right == 0, "dense variants unsupported"
    return _split_2D_jagged_jagged(
        max_seq_len=max_seq_len, values=values, offsets_left=offsets_a, offsets_right=offsets_b
    )
