# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for repository tools."""

from __future__ import annotations

import dataclasses
import json
import sys
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Iterator


def bootstrap_repo_paths(
    anchor: str | Path,
    *,
    include_tools: bool = False,
    include_benchmarks: bool = False,
) -> Path:
    """Add repo-local import roots for standalone tool execution."""

    repo_root = Path(anchor).resolve().parents[1]
    roots = [repo_root / "src"]
    if include_tools:
        roots.append(repo_root / "tools")
    if include_benchmarks:
        roots.append(repo_root / "benchmarks")
    for root in roots:
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
    return repo_root


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if dataclasses.is_dataclass(value):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, list | tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        try:
            return jsonable(value.tolist())
        except (TypeError, ValueError):
            pass
    return repr(value)


def json_dumps(value: Any) -> str:
    return json.dumps(jsonable(value), indent=2, sort_keys=True)


# NPU: 设备解析/同步统一入口，供 tools 共用。torch_npu 一律惰性导入，
# CPU/CUDA 路径不引入 torch_npu 依赖（Windows 无卡干检可用）。
def resolve_torch_device(torch, requested: str) -> str:
    """Resolve ``--device``: explicit cuda/npu passes through, auto probes cuda -> npu -> cpu."""
    if requested == "cuda":
        return "cuda"
    if requested == "npu":
        return "npu"
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        try:
            import torch_npu  # NPU: 惰性导入
        except ImportError:
            torch_npu = None
        if torch_npu is not None and torch_npu.npu.is_available():
            return "npu"
    return "cpu"


def device_synchronize_fn(torch, device: str):
    """Return the synchronize callable for ``device`` (None on cpu)."""
    if device == "cuda":
        return torch.cuda.synchronize
    if device == "npu":
        import torch_npu  # NPU: 惰性导入

        return torch_npu.npu.synchronize
    return None


def write_json(path: Path | str, value: Any, *, trailing_newline: bool = True) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json_dumps(value)
    output_path.write_text(
        encoded + ("\n" if trailing_newline else ""),
        encoding="utf-8",
    )


def summary_metric(summary: dict[str, Any], field: str) -> Any:
    median_field = f"{field}_median"
    if median_field in summary:
        return summary[median_field]
    samples = summary.get(f"{field}_samples")
    median = numeric_median(samples)
    if median is not None:
        return median
    if field == "decode_ms":
        return (summary.get("scheduler_metrics") or {}).get("decode_ms")
    return summary.get(field)


def numeric_median(values: Any) -> Any:
    if values is None:
        return None
    rows = [float(value) for value in values]
    if not rows:
        return None
    sorted_values = sorted(rows)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


def read_jsonl(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    return [row for _, row in iter_jsonl(path, limit=limit)]


def iter_jsonl(
    path: Path,
    *,
    limit: int | None = None,
) -> Iterator[tuple[int, dict[str, Any]]]:
    rows_read = 0
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, MappingABC):
            raise ValueError(f"{path}:{line_no} must contain a JSON object")
        rows_read += 1
        yield line_no, dict(row)
        if limit is not None and rows_read >= limit:
            break


def load_optional_tokenizer(args):
    if args.no_tokenizer:
        return None
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError:
        if args.require_tokenizer:
            raise
        return None
    return AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
