"""NPU-device smoke/parity script for the patched HSTU inference example.

Real Ascend host (torch 2.7.1 + torch_npu + ops_rec + recsys_kvcache_manager_npu
+ fbgemm_ascend; dynamic_emb optional). REAL path only: no sys.modules mocks,
no HSTU_NPU_DRY_RUN (refuses to run with it set). Structure copied from
scripts/hstu_npu_dry_check.py. Checks (A/D hard gates, B informational):

  A  real ops_rec hstu_paged (paged_ops_compat real branch) vs its CPU torch
     reference, same inputs: fp32 atol 1e-4, then bf16 atol 2e-2 (validates
     alpha/silu_scale/seq_offset_t/metadata mapping).
  B  same bf16 kernel call with cu_seqlens_t zeroed vs correct: identical?
  D  forward_with_kvcache on a FRESH real KVCacheManager (all-miss = full
     recompute) vs forward_nokvcache, same inputs: bs=1 without contextual
     features, bs=4 with them; bf16 tol atol 1e-2 + rtol 2e-2, fp32 tol
     atol 1e-4 + rtol 1e-5.

Usage:  python hstu_npu_smoke.py <path-to-recsys-examples> [--bf16|--fp32]
Exit 0 = SMOKE PASS (gates A and D), 1 = SMOKE FAIL.
"""

import argparse
import math
import os
import sys
import time
import traceback

CTX_NAMES = [  # kuairand-1k contextual order (commons/hstu_data_preprocessor)
    "user_id", "user_active_degree", "follow_user_num_range",
    "fans_user_num_range", "friend_user_num_range", "register_days_range",
]

GIN_CONFIG = [
    "DatasetArgs.dataset_name = 'kuairand-1k'",
    "DatasetArgs.max_history_seqlen = 48", "DatasetArgs.max_num_candidates = 8",
    "NetworkArgs.num_layers = 2", "NetworkArgs.num_attention_heads = 4",
    "NetworkArgs.hidden_size = 512", "NetworkArgs.kv_channels = 128",
    "RankingArgs.prediction_head_arch = [512, 8]", "RankingArgs.num_tasks = 8",
]

# synthetic paged-attention case for A/B (mini test_paged_hstu_attn_kernel.py:
# random q/k/v AND random cache pages -> history must come from the pages,
# only the k/v tail targets from direct input). All-miss shape: the patched
# CPU reference requires kv_len == Lq per user; single target per user keeps
# the eye-vs-tril target-block question out of this mapping check.
A_SEED = 1234
A_NUM_HEADS, A_HEAD_DIM, A_PAGE = 4, 128, 32
A_HIST = [48, 70]       # history tokens held in cache pages, per user
A_NUM_TGT = [1, 1]      # candidate rows at the tail of each q/k/v segment
A_SCALING_SEQLEN = 128  # distinctive: silu_scale must map to 1/128

_CASE = None  # lazy CPU master tensors shared by checks A and B


def log(msg):
    print(f"[smoke] {msg}", flush=True)


def setup_repo(repo):
    hstu = os.path.join(repo, "examples", "hstu")
    if not os.path.isdir(hstu):
        raise SystemExit(f"not a recsys-examples checkout: {repo}")
    os.environ["HSTU_INFERENCE_ONLY"] = "1"  # skip training-only imports
    os.chdir(hstu)  # mirror the real run (entry appends "./model/" to sys.path)
    for p in reversed([hstu, os.path.join(repo, "examples"),
                       os.path.join(hstu, "inference"),
                       os.path.join(hstu, "model")]):
        if p not in sys.path:  # final order: hstu, examples, inference, model
            sys.path.insert(0, p)


def print_env(repo):
    import torch
    import torch_npu
    try:
        torch_npu.npu.set_device(0)
    except Exception as exc:  # some builds auto-init; best effort
        log(f"note: torch_npu.npu.set_device(0) failed: {exc}")
    device = torch_npu.npu.current_device()
    try:
        soc = str(torch_npu.npu.get_soc_version())
    except Exception:
        soc = "unknown"
    log(f"repo = {os.path.abspath(repo)} | torch = {torch.__version__} | "
        f"torch_npu = {torch_npu.__version__}")
    log(f"device = {device} | soc = {soc}")


# checks A/B: paged-HSTU attention kernel parity (real ops_rec vs CPU ref)
def _pad32(n):
    return ((n + 31) // 32) * 32


def get_case():
    """CPU masters, built once (semantics: A_* constants above). Per user b:
    pages hold A_HIST[b] history tokens; the q/k/v segment is A_HIST[b] +
    A_NUM_TGT[b] rows (kv_len == Lq), of which only the tail A_NUM_TGT[b] k/v
    rows are direct attention inputs - history k/v must come from the pages."""
    global _CASE
    if _CASE is not None:
        return _CASE
    import torch

    def cums(xs):
        t = torch.tensor(xs, dtype=torch.int32)
        return torch.cat([torch.zeros(1, dtype=torch.int32), t.cumsum(0)])

    torch.manual_seed(A_SEED)
    lq = [h + t for h, t in zip(A_HIST, A_NUM_TGT)]
    pages_per_user = [(p + A_PAGE - 1) // A_PAGE for p in A_HIST]
    page_ids, page_off, off = [], [0], 0
    for n in pages_per_user:
        page_ids.extend(range(off, off + n))
        off += n
        page_off.append(off)
    last_page = [p - (n - 1) * A_PAGE for p, n in zip(A_HIST, pages_per_user)]
    ntok, hd = sum(lq), (A_NUM_HEADS, A_HEAD_DIM)
    _CASE = {
        "q": torch.randn(ntok, *hd), "k": torch.randn(ntok, *hd),
        "v": torch.randn(ntok, *hd),
        "table": torch.randn(sum(pages_per_user) + 2, 2, A_PAGE, *hd),  # +2 spare
        "cu_q": cums(lq), "cu_k": cums(lq), "cu_t": cums(A_NUM_TGT),
        "n_tgt": torch.tensor(A_NUM_TGT, dtype=torch.int32),
        "page_ids": torch.tensor(page_ids, dtype=torch.int32),
        "page_off": torch.tensor(page_off, dtype=torch.int32),
        "lpl": torch.tensor(last_page, dtype=torch.int32),
        "max_q": _pad32(max(lq)), "max_k": _pad32(max(lq)),
    }
    return _CASE


def _npu_hstu_paged(case, device, dtype, cu_t=None):
    """Real-path branch of paged_ops_compat.hstu_attn_varlen_func on NPU."""
    from inference.npu.paged_ops_compat import hstu_attn_varlen_func

    def d(t):  # floats: cast to the test dtype; ints: device move only
        return t.to(device=device, dtype=dtype) if t.is_floating_point() \
            else t.to(device=device)

    return hstu_attn_varlen_func(
        d(case["q"]), d(case["k"]), d(case["v"]), d(case["cu_q"]), d(case["cu_k"]),
        None, None,                    # seqused_q/k unsupported on the NPU path
        case["max_q"], case["max_k"], A_SCALING_SEQLEN,
        None, d(case["n_tgt"]),        # num_contexts unsupported; num_targets
        target_group_size=1, window_size=(-1, 0),  # causal -> mask_type 0
        alpha=1.0 / (A_HEAD_DIM ** 0.5), kv_cache=d(case["table"]),
        page_offsets=d(case["page_off"]), page_ids=d(case["page_ids"]),
        last_page_lens=d(case["lpl"]),
        cu_seqlens_t=d(case["cu_t"] if cu_t is None else cu_t)).cpu()



def _ref_hstu_paged(case, dtype):
    """CPU torch reference from paged_ops_compat on CPU copies of the inputs."""
    from inference.npu.paged_ops_compat import _hstu_attn_reference

    def c(t):
        return t.to(dtype) if t.is_floating_point() else t

    return _hstu_attn_reference(
        c(case["q"]), c(case["k"]), c(case["v"]), case["cu_q"], case["cu_k"],
        case["max_q"], case["max_k"], A_SCALING_SEQLEN, case["n_tgt"], 1,
        1.0 / (A_HEAD_DIM ** 0.5), kv_cache=c(case["table"]),
        page_offsets=case["page_off"], page_ids=case["page_ids"],
        last_page_lens=case["lpl"], cu_seqlens_t=case["cu_t"])


def run_check_a(device):
    import torch
    case = get_case()
    ok = True
    for dtype, atol, label in ((torch.float32, 1e-4, "fp32"),
                               (torch.bfloat16, 2e-2, "bf16")):
        out = _npu_hstu_paged(case, device, dtype)
        ref = _ref_hstu_paged(case, dtype)
        diff = (out.float() - ref.float()).abs().max().item()
        ok &= diff <= atol
        log(f"[A] hstu_paged vs CPU reference [{label}]: shape "
            f"{tuple(out.shape)}, max|diff| = {diff:.3e} (atol {atol:g}) -> "
            f"{'PASS' if diff <= atol else 'FAIL'}")
    return ok


def run_check_b(device):
    """Info: does hstu_paged consume cu_seqlens_t/seq_offset_t values?"""
    import torch
    case = get_case()
    out_ok = _npu_hstu_paged(case, device, torch.bfloat16)
    zeros = torch.zeros_like(case["cu_t"])
    out_zero = _npu_hstu_paged(case, device, torch.bfloat16, cu_t=zeros)
    diff = (out_ok.float() - out_zero.float()).abs().max().item()
    verdict = ("IDENTICAL -> kernel does not read seq_offset_t content (expected)"
               if diff == 0.0 else
               f"DIFFER (max|diff| = {diff:.3e}) -> kernel DOES consume it")
    log(f"[B] seq_offset_t zeros vs correct: {verdict}")
    return diff


# check D: kvcache(all-miss) vs nokvcache parity on the real model
def _build_model(emb_configs, max_bs, num_ctx, total_max_seqlen, dtype):
    """bf16: entry builder verbatim; fp32: local replica (entry only takes
    dtype_str='bfloat16'), kvcache sizing mirrored from the entry."""
    import torch
    import torch_npu
    if dtype == torch.bfloat16:
        from inference_gr_ranking import get_inference_hstu_model
        return get_inference_hstu_model(
            emb_configs, max_bs, num_ctx, total_max_seqlen, None, True)

    from configs import (PositionEncodingConfig, RankingConfig,
                         get_inference_hstu_config)
    from inference.npu.kvcache_compat import get_kvcache_config
    from inference_ranking_gr import get_inference_ranking_gr
    from utils import NetworkArgs, RankingArgs

    na = NetworkArgs()
    hstu_config = get_inference_hstu_config(
        hidden_size=na.hidden_size, num_layers=na.num_layers,
        num_attention_heads=na.num_attention_heads, head_dim=na.kv_channels,
        max_batch_size=max_bs, max_seq_len=_pad32(total_max_seqlen),
        scaling_seqlen=total_max_seqlen, dtype=dtype, contextual_max_seqlen=num_ctx,
        position_encoding_config=PositionEncodingConfig(
            num_position_buckets=8192, num_time_buckets=2048,
            use_time_encoding=False, static_max_seq_len=_pad32(total_max_seqlen)))
    # kvcache sizing mirrors inference_gr_ranking.get_inference_hstu_model
    cap = torch_npu.npu.get_device_capability()
    page = 128 if cap is not None and cap[0] >= 10 else 32
    n_primary = math.ceil(10240 * 32 / page)
    kv_cfg = get_kvcache_config(
        num_layers=na.num_layers, num_heads=na.num_attention_heads,
        head_dim=na.kv_channels, page_size=page, offload_chunksize=1024,
        num_primary_cache_pages=n_primary, num_buffer_pages=0,
        host_capacity_per_layer=n_primary * 2 * page
        * (na.num_attention_heads * na.kv_channels) * 2,
        max_batch_size=max_bs, max_seq_len=math.ceil(total_max_seqlen / page) * page,
        dtype=dtype, device=torch_npu.npu.current_device(),
        host_kvstorage_backend="native_host", offload_timeout_ms=100.0,
        offload_mode="lazy")
    ra = RankingArgs()
    model = get_inference_ranking_gr(
        hstu_config=hstu_config, kvcache_config=kv_cfg, use_cudagraph=False,
        cudagraph_configs={"batch_size": [1], "length_per_sequence": [128]},
        task_config=RankingConfig(
            embedding_configs=emb_configs,
            prediction_head_arch=ra.prediction_head_arch,
            prediction_head_act_type=ra.prediction_head_act_type,
            prediction_head_bias=ra.prediction_head_bias, num_tasks=ra.num_tasks,
            eval_metrics=ra.eval_metrics))
    model.load_checkpoint(None)  # no-op: random init stays
    model.eval()
    return model


def _build_batch(bs, hist_len, cand_len, ctx_names, device, seed):
    """Random batch (copied from hstu_npu_dry_check.py), moved to the device."""
    import torch
    from commons.datasets.hstu_batch import HSTUBatch
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

    torch.manual_seed(seed)  # identical inputs for the kv / nokv forwards
    act_len = [h - c for h, c in zip(hist_len, cand_len)]
    lens, vals, user_ids = [], [], None
    for idx in range(len(ctx_names)):  # 1 token per user; user_id first
        ids = torch.randint(0, 999 if idx == 0 else 7, (bs,)).long()
        if idx == 0:
            user_ids = ids
        lens += [1] * bs
        vals.append(ids)
    lens += hist_len  # video_id (history items + candidates)
    vals.append(torch.randint(0, 1000, (sum(hist_len),)).long())
    lens += act_len  # action (history only)
    vals.append(torch.randint(0, 232, (sum(act_len),)).long())
    features = KeyedJaggedTensor(
        values=torch.cat(vals), lengths=torch.tensor(lens, dtype=torch.int32),
        keys=ctx_names + ["video_id", "action_weights"])

    f2m = {n: 1 for n in ctx_names} | {
        "video_id": max(hist_len), "action_weights": max(act_len) if act_len else 0}
    batch = HSTUBatch(
        features=features, batch_size=bs, feature_to_max_seqlen=f2m,
        contextual_feature_names=list(ctx_names), item_feature_name="video_id",
        action_feature_name="action_weights", max_num_candidates=max(cand_len),
        num_candidates=torch.tensor(cand_len, dtype=torch.int32))
    # entry convention: total history = 2 * #history_items + #contextual, CPU
    total_hist = torch.tensor(
        [2 * (h - c) + len(ctx_names) for h, c in zip(hist_len, cand_len)],
        dtype=torch.int64)
    if user_ids is None:  # no contextual features -> fabricate lookup ids
        user_ids = torch.arange(100, 100 + bs, dtype=torch.int64)
    return batch.to(device), user_ids, total_hist


def run_check_d(dtype, atol, rtol):
    import gin
    import torch
    import torch_npu
    gin.parse_config(GIN_CONFIG)
    from inference_gr_ranking import get_inference_dataset_and_embedding_configs
    import modules.inference_embedding as inference_embedding

    log("[D] embedding backend: " + ("real dynamic_emb"
        if inference_embedding._HAS_DYNAMIC_EMB else "nn.Embedding fallback"))
    device = torch_npu.npu.current_device()
    cases = [  # (name, disable_ctx, bs, hist_len, cand_len)
        ("bs1_no_ctx", True, 1, [20], [6]),
        ("bs4_ctx", False, 4, [20, 26, 24, 22], [6, 8, 7, 5]),
    ]
    ok_all = True
    for name, no_ctx, bs, hist, cand in cases:
        dataset_args, emb_cfgs = get_inference_dataset_and_embedding_configs(
            disable_contextual_features=no_ctx)
        num_ctx = 0 if no_ctx else len(CTX_NAMES)
        total_max = dataset_args.max_history_seqlen * 2 + num_ctx
        torch.manual_seed(7)
        t0 = time.time()
        model = _build_model(emb_cfgs, bs, num_ctx, total_max, dtype)
        assert model.dense_module._use_kvcache and not model.dense_module.use_cudagraph
        log(f"[D] {name}: model built in {time.time() - t0:.1f}s (bs={bs}, "
            f"num_ctx={num_ctx}, total_max_seqlen={total_max})")
        ctx_names = [] if no_ctx else CTX_NAMES
        batch_kv, user_ids, total_hist = _build_batch(
            bs, hist, cand, ctx_names, device, 123)
        batch_ref, _, _ = _build_batch(bs, hist, cand, ctx_names, device, 123)
        # kvcache first: fresh manager => lookup all-miss => full recompute
        # (lookup/allocate/onboard_launch + per-layer onboard sync + offload
        #  all run inside model.forward_with_kvcache, as in the example entry)
        with torch.inference_mode():
            t0 = time.time()  # first call includes NPU kernel compile latency
            out_kv = model.forward_with_kvcache(batch_kv, user_ids, total_hist)
            t_kv = time.time() - t0
            out_ref = model.forward_nokvcache(batch_ref)
        diff = (out_kv.float().cpu() - out_ref.float().cpu()).abs().max().item()
        scale = out_ref.float().abs().max().item()
        thr = atol + rtol * scale
        ok = out_kv.shape == out_ref.shape and diff <= thr
        log(f"[D] {name}: logits {tuple(out_kv.shape)} vs {tuple(out_ref.shape)} "
            f"(rows = sum(num_candidates) = {sum(cand)}), max|kv-nokv| = "
            f"{diff:.3e}, thr = {thr:.3e} (atol {atol:g} + rtol {rtol:g} * |max| "
            f"{scale:.2f}), kv fwd {t_kv:.1f}s -> {'PASS' if ok else 'FAIL'}")
        ok_all &= ok
        try:  # best-effort cleanup (offload already ran inside the forward)
            model.dense_module.kvcache.evict_all()
            torch_npu.npu.empty_cache()
        except Exception as exc:
            log(f"[D] {name}: cleanup skipped ({type(exc).__name__})")
        del model
    return ok_all


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("repo", help="path to the recsys-examples checkout")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--bf16", dest="precision", action="store_const",
                       const="bf16", default="bf16", help="bf16 (default)")
    group.add_argument("--fp32", dest="precision", action="store_const",
                       const="fp32", help="fp32")
    args = parser.parse_args()

    if os.environ.get("HSTU_NPU_DRY_RUN", "0") == "1":
        log("ERROR: HSTU_NPU_DRY_RUN=1 is set; this smoke runs the REAL NPU "
            "path only. Unset it and rerun.")
        return 1

    setup_repo(args.repo)
    print_env(args.repo)

    import torch
    import torch_npu
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    d_atol, d_rtol = (1e-2, 2e-2) if dtype == torch.bfloat16 else (1e-4, 1e-5)
    log(f"precision = {args.precision} | tolerances: D atol {d_atol:g} + rtol "
        f"{d_rtol:g}; A fp32 atol 1e-04 / bf16 atol 2e-02")

    device = torch_npu.npu.current_device()
    checks = [
        ("A", "hard gate", lambda: run_check_a(device)),
        ("B", "info", lambda: run_check_b(device)),
        ("D", "hard gate", lambda: run_check_d(dtype, d_atol, d_rtol)),
    ]
    status = {}
    for name, kind, fn in checks:
        log(f"--- check {name} ({kind}) ---")
        try:
            status[name] = fn()
        except Exception:
            traceback.print_exc()
            log(f"[{name}] EXCEPTION (traceback above) -> FAIL")
            status[name] = False

    b = status.get("B")  # float diff, or False if check B errored
    b_note = ("B: errored" if isinstance(b, bool) else
              "B: seq_offset_t not read (identical)" if b == 0.0 else
              f"B: seq_offset_t consumed (max|diff| = {b:.3e})")
    failed = [n for n, k, _ in checks if k == "hard gate" and status.get(n) is not True]
    if failed:
        log(f"SMOKE FAIL (checks: {', '.join(failed)} failed; {b_note})")
        return 1
    log(f"SMOKE PASS (gates A and D passed; {b_note})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
