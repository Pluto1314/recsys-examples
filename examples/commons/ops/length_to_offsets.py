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
import torch

# NPU: asynchronous_*_cumsum 期望由 fbgemm_ascend 提供（import 成功 != 算子已注册，
# 故探测算子本身）；缺失时回退纯 torch（语义等价，仅失去 H2D 异步性）。


def _fbgemm_cumsum_available() -> bool:
    try:
        import fbgemm_gpu  # noqa: F401  for asynchronous_complete_cumsum

        torch.ops.fbgemm.asynchronous_complete_cumsum  # noqa: B018
        return True
    except Exception:
        return False


_HAS_FBGEMM_CUMSUM = _fbgemm_cumsum_available()


def length_to_complete_offsets(length_tensor: torch.Tensor):
    if _HAS_FBGEMM_CUMSUM:
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(length_tensor)
        return offsets
    # NPU fallback: cumsum + 前置 0（长度 n+1，与 fbgemm complete 语义一致）
    return torch.cat(
        [torch.zeros(1, dtype=length_tensor.dtype, device=length_tensor.device),
         torch.cumsum(length_tensor, dim=0)]
    )


def length_to_inclusive_offsets(length_tensor: torch.Tensor):
    if _HAS_FBGEMM_CUMSUM:
        offsets = torch.ops.fbgemm.asynchronous_inclusive_cumsum(length_tensor)
        return offsets
    return torch.cumsum(length_tensor, dim=0)


def length_to_exclusive_offsets(length_tensor: torch.Tensor):
    if _HAS_FBGEMM_CUMSUM:
        offsets = torch.ops.fbgemm.asynchronous_exclusive_cumsum(length_tensor)
        return offsets
    return torch.cumsum(length_tensor, dim=0) - length_tensor
