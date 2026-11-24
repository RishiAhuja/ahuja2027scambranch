#!/usr/bin/env python3
"""Evaluate pretrained compact detectors on a ScamBranch run directory.

Does not write into the run directory. No paid APIs — local Hugging Face
checkpoints only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scambranch.eval.constants import ALL_VIEWS, DEFAULT_MODELS
from scambranch.eval.run import EvalConfig, run_evaluation


def main() -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    parser = argparse.ArgumentParser(description="Zero-shot detector evaluation")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "full_v1")
    parser.add_argument("--eval-run-id", required=True)
    parser.add_argument("--out-dir", type=Path, help="Default: results/<eval_run_id>")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--views", nargs="+", default=list(ALL_VIEWS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-test", type=int)
    parser.add_argument("--max-dev", type=int)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--complete-sets", type=Path)
    parser.add_argument(
        "--no-complete-sets",
        action="store_true",
        help="Do not restrict to run_dir/complete_sets.json even if present",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32", "float16", "bfloat16", "float32"])
    args = parser.parse_args()

    out_dir = args.out_dir or (ROOT / "results" / args.eval_run_id)
    config = EvalConfig(
        run_dir=args.run_dir,
        out_dir=out_dir,
        eval_run_id=args.eval_run_id,
        models=tuple(args.models),
        views=tuple(args.views),
        split=args.split,
        max_test=args.max_test,
        max_dev=args.max_dev,
        sample_seed=args.sample_seed,
        complete_sets_path=args.complete_sets,
        use_complete_sets_if_present=not args.no_complete_sets,
        bootstrap_reps=args.bootstrap_reps,
        bootstrap_seed=args.bootstrap_seed,
        device=args.device,
        dtype=args.dtype,
    )
    result = run_evaluation(config)
    metrics = result["metrics"]
    proj = metrics.get("runtime_projection") or {}
    print("models:", ", ".join(metrics.get("models") or []) or "(none)")
    if metrics.get("skipped_models"):
        for item in metrics["skipped_models"]:
            print("skipped:", item.get("short_id"), item.get("skip_reason"))
    lat = metrics.get("latency_test") or {}
    pc = lat.get("per_conversation_sum") or {}
    print(
        f"test per-conversation latency median={pc.get('median_s')}s "
        f"p95={pc.get('p95_s')}s n={pc.get('n')}"
    )
    if proj:
        print(
            f"projection 5 models × 4 views on {proj.get('n_test')}+{proj.get('n_dev')} "
            f"convos: {proj.get('hours_five_models_four_views')} h "
            f"(one model {proj.get('hours_one_model')} h)"
        )
    print(f"artifacts: {result['out_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
