#!/usr/bin/env python3
"""Aggregate LoRA eval metrics across seeds into FT_SUMMARY.md."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scambranch.eval.constants import VIEW_FULL, VIEW_PREFIX_PLUS_REACTION


def _mean_sd(vals: list[float]) -> tuple[float, float]:
    clean = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not clean:
        return float("nan"), float("nan")
    mean = sum(clean) / len(clean)
    if len(clean) == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in clean) / (len(clean) - 1)
    return mean, math.sqrt(var)


def _fmt(mean: float, sd: float, digits: int = 3) -> str:
    if not math.isfinite(mean):
        return "—"
    if not math.isfinite(sd):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def _parse_model_arm_seed(model_id: str) -> tuple[str, str, str] | None:
    """Parse short ids like ``qwen2.5-0.5b+branched-seed1``."""
    m = re.match(r"^(.+)\+([^+]+)-(seed\d+)$", model_id)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


def load_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    slices = metrics.get("slices") or metrics.get("slices_all_usable") or {}
    rows: list[dict[str, Any]] = []
    for model_id, views in slices.items():
        parsed = _parse_model_arm_seed(model_id)
        if parsed is None:
            continue
        base, arm, seed = parsed
        for view, block in views.items():
            pooled = block.get("pooled") or {}
            branch = block.get("branch") or {}
            reaction = block.get("reaction_delta") or {}
            rows.append(
                {
                    "model": base,
                    "arm": arm,
                    "seed": seed,
                    "view": view,
                    "macro_f1": pooled.get("macro_f1"),
                    "legitimate_fpr": pooled.get("legitimate_fpr"),
                    "fraud_recall": pooled.get("fraud_recall"),
                    "disagreement": branch.get("branch_disagreement_rate"),
                    "reaction_delta": reaction.get("mean_delta"),
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> str:
    # Group by model × arm
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[(row["model"], row["arm"])].append(row)

    lines = [
        "# LoRA fine-tune summary (mean ± sd across seeds)",
        "",
        "## Test quality (FULL view)",
        "",
        "| model | arm | macro-F1 | legit FPR | fraud recall |",
        "|---|---|---:|---:|---:|",
    ]
    for (model, arm) in sorted(by_key):
        full = [r for r in by_key[(model, arm)] if r["view"] == VIEW_FULL]
        f1_m, f1_s = _mean_sd([r["macro_f1"] for r in full])
        fpr_m, fpr_s = _mean_sd([r["legitimate_fpr"] for r in full])
        rec_m, rec_s = _mean_sd([r["fraud_recall"] for r in full])
        lines.append(
            f"| {model} | {arm} | {_fmt(f1_m, f1_s)} | {_fmt(fpr_m, fpr_s)} | {_fmt(rec_m, rec_s)} |"
        )
    lines.append("")

    lines.extend(
        [
            "## Sibling disagreement (complete sets)",
            "",
            "| model | arm | view | disagreement |",
            "|---|---|---|---:|",
        ]
    )
    for (model, arm) in sorted(by_key):
        for view in (VIEW_FULL, VIEW_PREFIX_PLUS_REACTION):
            subset = [r for r in by_key[(model, arm)] if r["view"] == view]
            d_m, d_s = _mean_sd([r["disagreement"] for r in subset])
            lines.append(f"| {model} | {arm} | {view} | {_fmt(d_m, d_s)} |")
    lines.append("")

    lines.extend(
        [
            "## Mean reaction Δ P(fraud) (PREFIX_PLUS_REACTION − PREFIX_ONLY)",
            "",
            "| model | arm | mean Δ |",
            "|---|---|---:|",
        ]
    )
    for (model, arm) in sorted(by_key):
        subset = [r for r in by_key[(model, arm)] if r["view"] == VIEW_PREFIX_PLUS_REACTION]
        m, s = _mean_sd([r["reaction_delta"] for r in subset])
        lines.append(f"| {model} | {arm} | {_fmt(m, s)} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize FT eval metrics")
    parser.add_argument(
        "--metrics",
        type=Path,
        nargs="+",
        help="One or more metrics.json paths (default: results/eval_hpc_ft/metrics.json)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "eval_hpc_ft" / "FT_SUMMARY.md",
    )
    args = parser.parse_args()
    paths = args.metrics or [ROOT / "results" / "eval_hpc_ft" / "metrics.json"]
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            print(f"missing metrics: {path}", flush=True)
            continue
        rows.extend(collect_rows(load_metrics(path)))
    text = summarize(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
