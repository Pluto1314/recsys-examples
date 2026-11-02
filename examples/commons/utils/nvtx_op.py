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

import os
from functools import partial, wraps

import torch

# NPU: 无 CUDA NVTX 时 range_push/pop 退化为 no-op（单点收敛，免改各调用方）。
# 注意 CPU-only torch 上属性可访问但调用会抛错，故以 is_available 判定。
if torch.cuda.is_available() and hasattr(torch.cuda, "nvtx"):
    _range_push = torch.cuda.nvtx.range_push
    _range_pop = torch.cuda.nvtx.range_pop
else:
    _range_push = lambda *args, **kwargs: None  # noqa: E731
    _range_pop = lambda *args, **kwargs: None  # noqa: E731

_NVTX_HOOKS_ENABLED = os.environ.get("ENABLE_NVTX_HOOKS", "1") == "1"


def _hook_placeholder_device():
    # NPU: 占位张量设备按 npu/cuda/cpu 可达性选择（torch_npu 存在且可用则 npu）
    try:
        import torch_npu  # noqa: F401

        if torch_npu.npu.is_available():
            return "npu"
    except ImportError:
        pass
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_nvtx_hooks_enabled(enabled: bool) -> None:
    """Globally enable or disable NVTX hooks in this process."""
    global _NVTX_HOOKS_ENABLED
    _NVTX_HOOKS_ENABLED = enabled


def nvtx_hooks_enabled() -> bool:
    return _NVTX_HOOKS_ENABLED


def forward_nvtx_push_hook(module, input, msg=""):
    if _NVTX_HOOKS_ENABLED:
        _range_push(msg)


def forward_nvtx_pop_hook(module, input, output):
    if _NVTX_HOOKS_ENABLED:
        _range_pop()


def backward_nvtx_push_hook(module, grad_output, msg=""):
    if _NVTX_HOOKS_ENABLED:
        _range_push(msg)


def backward_nvtx_pop_hook(module, grad_input, grad_output):
    if _NVTX_HOOKS_ENABLED:
        _range_pop()


def register_nvtx_for_module(module: torch.nn.Module, msg=""):
    module.register_forward_pre_hook(
        partial(forward_nvtx_push_hook, msg=msg + " forward")
    )
    module.register_forward_hook(forward_nvtx_pop_hook)
    module.register_full_backward_pre_hook(
        partial(backward_nvtx_push_hook, msg=msg + " backward")
    )
    module.register_full_backward_hook(backward_nvtx_pop_hook)


# default key_or_attr_name is "data" for tensor
def hook_setter(x, y, key_or_attr_name=["data"]):
    if not isinstance(key_or_attr_name, tuple) and not isinstance(
        key_or_attr_name, list
    ):
        key_or_attr_name = [key_or_attr_name]

    leaf_key_or_attr_name = key_or_attr_name[-1]
    current = x

    access_path = []
    for key in key_or_attr_name[:-1]:
        access_path.append(str(key))
        try:
            current = current[key]
        # default path for tensor
        except (TypeError, KeyError, IndexError):
            current = getattr(current, key)
        except AttributeError as e:
            raise e(f"Error accessing member {access_path} of {x}")
        except Exception as e:
            raise e(f"Error accessing member {access_path} of {x}")
    try:
        current[leaf_key_or_attr_name] = y
    # default path for tensor
    except (TypeError, KeyError, IndexError):
        setattr(current, leaf_key_or_attr_name, y)
    except AttributeError as e:
        raise e(f"Error setting attribute {key_or_attr_name} of {x} to {y}")
    except Exception as e:
        raise e(f"Error setting attribute {key_or_attr_name} of {x} to {y}")


def hook_getter(x, key_or_attr_name=["data"]):
    if not isinstance(key_or_attr_name, tuple) and not isinstance(
        key_or_attr_name, list
    ):
        key_or_attr_name = [key_or_attr_name]
    current = x
    access_path = []
    for key in key_or_attr_name:
        access_path.append(str(key))
        try:
            current = current[key]
        # default path for tensor
        except (TypeError, KeyError, IndexError):
            current = getattr(current, key)
        except AttributeError as e:
            raise e(f"Error accessing member {access_path} of {x}")
        except Exception as e:
            raise e(f"Error accessing member {access_path} of {x}")
    return current


def register_setter_and_getter_for_nvtx(forward_func, key_or_attr_name=["data"]):
    forward_func.hook_tensor_getter = partial(
        hook_getter, key_or_attr_name=key_or_attr_name
    )
    forward_func.hook_tensor_setter = partial(
        hook_setter, key_or_attr_name=key_or_attr_name
    )


class _NvtxRangePush(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, msg="None"):
        ctx.msg = msg
        _range_push(msg)
        return input

    @staticmethod
    def backward(ctx, grad_in):
        _range_pop()
        return grad_in, None


class _NvtxRangePop(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, msg="None"):
        ctx.msg = msg
        _range_pop()
        return input

    @staticmethod
    def backward(ctx, grad_in):
        msg = ctx.msg
        _range_push(msg)
        return grad_in, None


class _binary_identity_op(torch.autograd.Function):
    """
    input must be tensor
    """

    @staticmethod
    def forward(ctx, input_l, input_r):
        return input_l, input_r

    @staticmethod
    def backward(ctx, grad_l, grad_r):
        return grad_l, grad_r


## usage
NvtxRangePush = _NvtxRangePush.apply
NvtxRangePop = _NvtxRangePop.apply
binary_identity_op = _binary_identity_op.apply


def output_nvtx_hook(nvtx_tag, backward=True, hook_key_or_attr_name="data"):
    def decorator_forward_only(module):
        @wraps(module)
        def forward(*args, **kwags):
            if not _NVTX_HOOKS_ENABLED:
                return module(*args, **kwags)
            _range_push(nvtx_tag)
            output = module(*args, **kwags)
            _range_pop()
            return output

        return forward

    def decorator_nvtx(module_or_func):
        @wraps(module_or_func)
        def wrapper(*args, **kwargs):
            if not _NVTX_HOOKS_ENABLED:
                return module_or_func(*args, **kwargs)
            # NPU: 占位张量按 npu/cuda/cpu 可达性选设备（数值上恒等，不参与计算）
            _placeholder = torch.empty(
                1, device=_hook_placeholder_device()
            ).requires_grad_()
            hook_r = NvtxRangePush(_placeholder, nvtx_tag + " forward")
            output = module_or_func(*args, **kwargs)

            getter = getattr(
                wrapper,
                "hook_tensor_getter",
                partial(hook_getter, key_or_attr_name=hook_key_or_attr_name),
            )
            setter = getattr(
                wrapper,
                "hook_tensor_setter",
                partial(hook_setter, key_or_attr_name=hook_key_or_attr_name),
            )
            try:
                if isinstance(output, torch.Tensor):
                    hook_l = output
                else:
                    hook_l = getter(output)
                hook_l, _ = binary_identity_op(hook_l, hook_r)
                hook_l = NvtxRangePop(hook_l, nvtx_tag + " backward")
                if isinstance(output, torch.Tensor):
                    output = hook_l
                else:
                    setter(output, hook_l)
            except Exception:
                # silently ignore
                NvtxRangePop(hook_r, nvtx_tag + " backward")
            return output

        return wrapper

    if not backward:
        decorator = decorator_forward_only
    else:
        decorator = decorator_nvtx
    return decorator
