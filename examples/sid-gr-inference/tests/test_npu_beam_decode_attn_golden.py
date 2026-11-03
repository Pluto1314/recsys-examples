# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NPU: python stub beam_decode_attn golden tests.

Oracle 是 ``corelib/gr_decode_atten/tests/reference.py``（GPU 原生 kernel 的
语义基线，仅依赖 torch/math）。被测对象是
``third_party/gr-decode-attention-npu/interface.py::beam_decode_attn``——
通过真实的 kernel locator（ExistingGRDecodeAttentionBackend）加载，覆盖
dense / dense+seqused_k / jagged cu_seqlens_k / decode_nums=0 四种模式与
backend 别名（dsl/3kernel/torch），fp32 atol 1e-5。

pytest 可发现本文件（pyproject 的 pythonpath=[".","src"]）；也支持无 pytest
环境直接 ``python tests/test_npu_beam_decode_attn_golden.py``（Windows 无卡
干检用）。
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]  # examples/sid-gr-inference
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from gr_inference.gr_kernels.attention import (  # noqa: E402
    ExistingGRDecodeAttentionBackend,
)

STUB_ROOT = REPO_ROOT / "third_party" / "gr-decode-attention-npu"
CUDA_KERNEL_ROOT = REPO_ROOT / "third_party" / "gr-decode-attention"


def _load_stub():
    """Load the stub through the real kernel locator (no env vars needed)."""
    backend = ExistingGRDecodeAttentionBackend().ensure_available()
    return backend._beam_decode_attn, backend.kernel_root


def _load_reference():
    root = os.environ.get("GR_DECODE_ATTEN_ROOT")
    candidates = [Path(root)] if root else []
    candidates.append(Path(__file__).resolve().parents[3] / "corelib" / "gr_decode_atten")
    for base in candidates:
        path = base / "tests" / "reference.py"
        if path.is_file():
            spec = importlib.util.spec_from_file_location(
                "gr_decode_atten_reference", path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(f"corelib gr_decode_atten reference.py not found: {candidates}")


def _case(
    *,
    dtype=torch.float32,
    batch=2,
    beam_width=4,
    head_q=8,
    head_kv=2,
    dim=64,
    seqlen_context=17,
    decode_nums=3,
    max_decode_nums=5,
    seed=0,
):
    generator = torch.Generator().manual_seed(seed)
    shape_q = (batch, 1, beam_width, head_q, dim)
    q = torch.randn(*shape_q, dtype=dtype, generator=generator)
    k_context = torch.randn(batch, seqlen_context, head_kv, dim, dtype=dtype, generator=generator)
    v_context = torch.randn(batch, seqlen_context, head_kv, dim, dtype=dtype, generator=generator)
    seqlen_beam = decode_nums * beam_width
    k_beam = torch.randn(batch, seqlen_beam, head_kv, dim, dtype=dtype, generator=generator)
    v_beam = torch.randn(batch, seqlen_beam, head_kv, dim, dtype=dtype, generator=generator)
    group = head_q // head_kv
    topk_kv = torch.randint(
        0, max(seqlen_beam, 1), (batch, 1, head_kv, max_decode_nums, beam_width),
        dtype=torch.int32, generator=generator,
    )
    topk_indices = topk_kv if group == 1 else topk_kv.repeat_interleave(group, dim=2)
    return q, k_context, v_context, k_beam, v_beam, topk_indices


def _per_sample_reference(ref, q, k_ctx, v_ctx, k_beam, v_beam, topk_indices, decode_nums, valid_lengths=None):
    """Per-sample oracle for non-uniform contexts (seqused / jagged)."""
    outs, lses = [], []
    for b in range(q.shape[0]):
        length = k_ctx.shape[1] if valid_lengths is None else valid_lengths[b]
        out_ctx, lse_ctx = ref.beam_context_attention_ref(
            q[b : b + 1], k_ctx[b : b + 1, :length], v_ctx[b : b + 1, :length]
        )
        out_beam, lse_beam = ref.beam_sparse_attention_ref(
            q[b : b + 1], k_beam[b : b + 1], v_beam[b : b + 1], topk_indices[b : b + 1], decode_nums
        )
        out_b, lse_b = ref.beam_attention_combine_ref(out_ctx, lse_ctx, out_beam, lse_beam)
        outs.append(out_b[0])
        lses.append(lse_b[0])
    return torch.stack(outs), torch.stack(lses)


def test_locator_finds_stub_without_env():
    """Locator falls back to the NPU stub root when the CUDA kernel root is absent."""
    if (CUDA_KERNEL_ROOT / "interface.py").is_file():
        # CUDA kernel vendored: locator correctly prefers it; stub not selected.
        return
    _, kernel_root = _load_stub()
    assert kernel_root == STUB_ROOT, f"expected {STUB_ROOT}, got {kernel_root}"


def test_dense_fp32_matches_reference():
    beam_decode_attn, _ = _load_stub()
    ref = _load_reference()
    q, k_context, v_context, k_beam, v_beam, topk_indices = _case()
    out, lse = beam_decode_attn(
        q, k_context, v_context, k_beam, v_beam, topk_indices, 3,
        return_lse=True, backend="dsl",
    )
    out_golden, lse_golden = ref.beam_attention_ref(
        q, k_context, v_context, k_beam, v_beam, topk_indices, 3
    )
    assert out.dtype == q.dtype
    assert torch.allclose(out.float(), out_golden, atol=1e-5, rtol=1e-5)
    assert torch.allclose(lse, lse_golden, atol=1e-5, rtol=1e-5)


def test_seqused_k_masks_padded_context():
    beam_decode_attn, _ = _load_stub()
    ref = _load_reference()
    q, k_context, v_context, k_beam, v_beam, topk_indices = _case(
        batch=2, seqlen_context=10, seed=3
    )
    seqused = torch.tensor([10, 6], dtype=torch.int32)
    out, lse = beam_decode_attn(
        q, k_context, v_context, k_beam, v_beam, topk_indices, 3,
        return_lse=True, seqused_k=seqused, backend="3kernel",
    )
    out_golden, lse_golden = _per_sample_reference(
        ref, q, k_context, v_context, k_beam, v_beam, topk_indices, 3,
        valid_lengths=[10, 6],
    )
    assert torch.allclose(out.float(), out_golden, atol=1e-5, rtol=1e-5)
    assert torch.allclose(lse, lse_golden, atol=1e-5, rtol=1e-5)


def test_jagged_context_matches_sliced_dense():
    beam_decode_attn, _ = _load_stub()
    ref = _load_reference()
    q, k_context, v_context, k_beam, v_beam, topk_indices = _case(
        batch=2, seqlen_context=8, seed=4
    )
    cu_seqlens_k = torch.tensor([0, 8, 13], dtype=torch.int32)
    k_jag = torch.cat((k_context[0], k_context[1, :5])).contiguous()
    v_jag = torch.cat((v_context[0], v_context[1, :5])).contiguous()
    out, lse = beam_decode_attn(
        q, k_jag, v_jag, k_beam, v_beam, topk_indices, 3,
        return_lse=True, cu_seqlens_k=cu_seqlens_k,
    )
    out_golden, lse_golden = _per_sample_reference(
        ref, q, k_context, v_context, k_beam, v_beam, topk_indices, 3,
        valid_lengths=[8, 5],
    )
    assert torch.allclose(out.float(), out_golden, atol=1e-5, rtol=1e-5)
    assert torch.allclose(lse, lse_golden, atol=1e-5, rtol=1e-5)


def test_zero_decode_nums_equals_pure_context_attention():
    beam_decode_attn, _ = _load_stub()
    ref = _load_reference()
    q, k_context, v_context, k_beam, v_beam, topk_indices = _case(decode_nums=0, seed=5)
    k_beam = k_beam[:, :0]
    v_beam = v_beam[:, :0]
    out, lse = beam_decode_attn(
        q, k_context, v_context, k_beam, v_beam, topk_indices, 0, return_lse=True
    )
    out_golden, lse_golden = ref.beam_attention_ref(
        q, k_context, v_context, k_beam, v_beam, topk_indices, 0
    )
    assert torch.allclose(out.float(), out_golden, atol=1e-5)
    assert torch.allclose(lse, lse_golden, atol=1e-5)


def test_backend_aliases_and_rejects_unknown():
    beam_decode_attn, _ = _load_stub()
    q, k_context, v_context, k_beam, v_beam, topk_indices = _case(seed=6)
    outs = [
        beam_decode_attn(q, k_context, v_context, k_beam, v_beam, topk_indices, 3, backend=name)[0]
        for name in ("torch", "dsl", "3kernel")
    ]
    for other in outs[1:]:
        assert torch.equal(outs[0], other)
    try:
        beam_decode_attn(q, k_context, v_context, k_beam, v_beam, topk_indices, 3, backend="cuda")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown backend must raise ValueError")


if __name__ == "__main__":
    # NPU: 无 pytest 环境的独立运行入口（Windows 无卡干检）
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {exc!r}")
    sys.exit(1 if failures else 0)
