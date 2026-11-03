# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a tiny end-to-end SID-GR inference skeleton."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TOOLS_ROOT = REPO_ROOT / "tools"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from tool_utils import resolve_torch_device  # noqa: E402

from gr_inference import (  # noqa: E402
    GRDecodeAttention,
    GRDecodeEngine,
    GRGenerationState,
    PrefillAttention,
    Qwen3GRConfig,
    Qwen3GRModel,
    ScheduledBeamPolicy,
    TokenTrie,
    TorchSDPAPrefillBackend,
    TrieItemMaskProvider,
)
from gr_inference.gr_kernels.attention import (  # noqa: E402
    ExistingGRDecodeAttentionBackend,
)


def build_tiny_config(
    *,
    layers: int,
    context_len: int,
    decode_steps: int,
    beam_width: int,
    vocab_size: int,
) -> Qwen3GRConfig:
    return Qwen3GRConfig(
        model_name="tiny-qwen3-gr-demo",
        num_layers=layers,
        hidden_size=32,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=8,
        max_context_len=context_len,
        max_seq_len=context_len + decode_steps,
        max_decode_steps=decode_steps,
        max_beam_width=beam_width,
        intermediate_size=64,
        vocab_size=vocab_size,
    )


def build_identity_topk_indices(torch, *, batch, head_q, decode_nums, beam_width, device):
    # NPU/CPU 干检：与 tools/run_qwen3_real_weight_tiny_gr.py 同构的 identity 下标
    indices = torch.arange(beam_width, dtype=torch.int32, device=device)
    indices = indices.view(1, 1, 1, 1, beam_width)
    indices = indices.expand(batch, 1, head_q, decode_nums, beam_width).contiguous()
    for decode_idx in range(decode_nums):
        indices[:, :, :, decode_idx, :] = decode_idx * beam_width + torch.arange(
            beam_width,
            dtype=torch.int32,
            device=device,
        )
    return indices


def run_demo(args) -> dict:
    import torch

    torch.manual_seed(args.seed)
    config = build_tiny_config(
        layers=args.layers,
        context_len=args.context_len,
        decode_steps=args.decode_steps,
        beam_width=args.beam_width,
        vocab_size=args.vocab_size,
    )
    # NPU: auto 探测 cuda -> npu -> cpu；torch_npu 惰性导入，无卡环境跑 cpu
    # （getattr 兼容旧调用方/测试直接传 Namespace 的用法）
    device = resolve_torch_device(torch, getattr(args, "device", "auto"))
    model = Qwen3GRModel(
        config,
        prefill_attention=PrefillAttention(TorchSDPAPrefillBackend()),
    ).to(device)
    input_ids = torch.randint(
        0, config.vocab_size, (1, args.context_len), device=device
    )

    prefill = model.forward_prefill(input_ids, return_result=True)
    generation = GRGenerationState.from_prefill(
        request_id="tiny-demo",
        prefill=prefill,
        max_decode_steps=config.max_decode_steps,
        max_beam_width=config.max_beam_width,
        fixed_beam_width=args.beam_width,
    )
    decode_engine = GRDecodeEngine(
        attention=GRDecodeAttention(backend=lambda inputs: inputs.q),
        fixed_beam_width=args.beam_width,
    )
    if getattr(args, "decode_backend", "fake") == "real":
        # NPU: 走 ExistingGRDecodeAttentionBackend；kernel locator 在无 CUDA
        # kernel 的 NPU/CPU 环境自动发现 third_party/gr-decode-attention-npu
        # 的 python 等价实现（与 tools/run_qwen3_real_weight_tiny_gr.py 的
        # --decode-backend real 同构，仅随机权重 + 单步 decode）
        return run_real_decode_step(args, torch, model, config, generation, device)
    item_mask_provider = None
    if args.constraint_demo:
        sequences = []
        for root in range(1, args.beam_width + 1):
            sequence = [root]
            for depth in range(args.decode_steps):
                sequence.append((10 * root + depth) % args.vocab_size)
            sequences.append(sequence)
        item_mask_provider = TrieItemMaskProvider(
            TokenTrie.from_sequences(sequences),
            vocab_size=config.vocab_size,
        )
    beam_width_policy = None
    if args.beam_schedule:
        schedule = {}
        for item in args.beam_schedule.split(","):
            step, width = item.split(":")
            schedule[int(step)] = int(width)
        beam_width_policy = ScheduledBeamPolicy(schedule)
    result = model.generate_fixed_beam(
        generation,
        decode_engine,
        max_steps=args.decode_steps,
        item_mask_provider=item_mask_provider,
        beam_width_policy=beam_width_policy,
    )
    return {
        "input_shape": tuple(input_ids.shape),
        "context_kv_shape": generation.prefill.context_kv.key_shape,
        "beam_kv_shape": generation.beam_kv.key_shape,
        "beam_path_steps": generation.beam_path.steps_done,
        "steps": len(result.steps),
        "final_token_ids": result.final_token_ids,
        "final_scores": result.steps[-1].scores if result.steps else (),
        "constraint_demo": args.constraint_demo,
        "beam_schedule": args.beam_schedule,
    }


def run_real_decode_step(args, torch, model, config, generation, device) -> dict:
    """Single real decode step through the existing kernel backend (stub on NPU)."""
    selection = generation.initialize_beams()
    beam_token_ids = torch.tensor(
        [selection.token_ids], dtype=torch.long, device=device
    )
    topk_indices = build_identity_topk_indices(
        torch,
        batch=1,
        head_q=config.num_attention_heads,
        decode_nums=1,
        beam_width=args.beam_width,
        device=device,
    )
    decode_engine = GRDecodeEngine(
        attention=GRDecodeAttention(backend=ExistingGRDecodeAttentionBackend()),
        fixed_beam_width=args.beam_width,
    )
    logits = model.forward_decode_step(
        beam_token_ids,
        generation,
        decode_engine,
        step=0,
        topk_indices=topk_indices,
        decode_nums=1,
        return_lse=True,
        backend_name="dsl",  # NPU stub 接受 dsl/3kernel/torch，同名路由
    )
    generation.update_beams_from_logits(logits)
    return {
        "decode_backend": "real",
        "context_kv_shape": generation.prefill.context_kv.key_shape,
        "beam_kv_shape": generation.beam_kv.key_shape,
        "decode_logits_shape": tuple(logits.shape),
        "beam_path_steps": generation.beam_path.steps_done,
        "final_token_ids": generation.beam_path.entries[-1].token_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--context-len", type=int, default=16)
    parser.add_argument("--decode-steps", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--constraint-demo", action="store_true")
    parser.add_argument(
        "--beam-schedule",
        default=None,
        help="Comma separated depth:width schedule, e.g. 0:3,1:2,2:1",
    )
    # NPU: real 模式经 ExistingGRDecodeAttentionBackend 走 decode attention
    # kernel（NPU/CPU 上为 python stub）；fake 模式保持原 dummy backend
    parser.add_argument("--decode-backend", choices=["fake", "real"], default="fake")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "npu"], default="auto")
    args = parser.parse_args()

    summary = run_demo(args)
    print("Tiny GR demo")
    print("=" * 60)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
