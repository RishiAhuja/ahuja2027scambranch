#!/usr/bin/env python3
"""LoRA fine-tune ScamBranch detectors (RQ3 arms). peft imported lazily."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scambranch.eval.data import load_usable_conversations
from scambranch.ft.data import FT_ARMS, TARGET_N, compute_matched_n, select_arm
from scambranch.ft.train import TrainConfig, train_lora


def _model_short(name: str) -> str:
    from scambranch.eval.detectors import resolve_model_spec

    short, _ = resolve_model_spec(name)
    return short


def main() -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="LoRA fine-tune on ScamBranch train split")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "full_v1")
    parser.add_argument("--model", required=True, help="Registry short id or HF id")
    parser.add_argument("--arm", choices=list(FT_ARMS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, help="Default: outputs/ft/<model>/<arm>/seed<k>")
    parser.add_argument("--n", type=int, help="Override matched N (must be ≤ matched and %%4==0 for B/C)")
    parser.add_argument("--target-n", type=int, default=TARGET_N)
    parser.add_argument("--lambda-consistency", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, help="Force left-truncation cap (default: fit longest)")
    parser.add_argument("--device")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32", "float16", "bfloat16", "float32"])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Select arms and print balance; no model download / peft",
    )
    parser.add_argument(
        "--all-arms-dry-run",
        action="store_true",
        help="Print matched N and all three arms for this seed (CPU only)",
    )
    args = parser.parse_args()

    train = load_usable_conversations(
        args.run_dir,
        split="train",
        complete_sets_path=None,
        use_complete_sets_if_present=False,
    )
    caps = compute_matched_n(train, target_n=args.target_n)
    print(json.dumps({"matched_caps": caps}, indent=2), flush=True)

    if args.all_arms_dry_run:
        for arm in FT_ARMS:
            sel = select_arm(train, arm, seed=args.seed, n=args.n, target_n=args.target_n)
            summary = {
                "arm": arm,
                "seed": args.seed,
                "n": sel.n,
                "n_histories": sel.summary["n_histories"],
                "by_label": sel.summary["by_label"],
                "by_language": sel.summary["by_language"],
                "by_condition": sel.summary["by_condition"],
            }
            print(json.dumps(summary, indent=2), flush=True)
        return 0

    short = _model_short(args.model)
    out_dir = args.out_dir or (ROOT / "outputs" / "ft" / short / args.arm / f"seed{args.seed}")
    selection = select_arm(
        train,
        args.arm,
        seed=args.seed,
        n=args.n,
        target_n=args.target_n,
    )
    print(
        json.dumps(
            {
                "arm": selection.arm,
                "seed": selection.seed,
                "n": selection.n,
                "n_histories": selection.summary["n_histories"],
                "by_label": selection.summary["by_label"],
                "by_language": selection.summary["by_language"],
                "by_condition": selection.summary["by_condition"],
            },
            indent=2,
        ),
        flush=True,
    )

    dev = load_usable_conversations(
        args.run_dir,
        split="dev",
        complete_sets_path=None,
        use_complete_sets_if_present=False,
    )
    cfg = TrainConfig(
        model=args.model,
        arm=args.arm,
        seed=args.seed,
        out_dir=out_dir,
        run_dir=args.run_dir,
        n=args.n,
        target_n=args.target_n,
        lambda_consistency=args.lambda_consistency,
        epochs=args.epochs,
        lr=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_length=args.max_length,
        device=args.device,
        dtype=args.dtype,
        dry_run=args.dry_run,
    )
    result = train_lora(cfg, selection, dev_conversations=dev)
    print(json.dumps({"status": "ok", "out_dir": result.get("out_dir"), "dry_run": args.dry_run}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
