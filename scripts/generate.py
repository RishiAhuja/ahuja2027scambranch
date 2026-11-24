#!/usr/bin/env python3
"""Generate ScamBranch conversations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv


def main() -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Generate ScamBranch conversations")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--families", nargs="+")
    parser.add_argument("--pairs", nargs="+")
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sync-review", action="store_true")
    parser.add_argument("--budget", type=float)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--histories", nargs="+")
    parser.add_argument(
        "--language-review-only",
        action="store_true",
        help="Re-run missing language reviews on an existing run and re-apply dispositions.",
    )
    parser.add_argument(
        "--semrev03",
        action="store_true",
        help="Re-run semantic_review v0.3 on all conversations, fill missing language reviews, repair post-branch.",
    )
    parser.add_argument(
        "--recompute-dispositions",
        action="store_true",
        help="Offline: snapshot then recompute dispositions under the current language_gate.",
    )
    parser.add_argument(
        "--regen-prefix-broken",
        action="store_true",
        help="Regenerate plan+prefix+4 branches for prefix_broken_v03 units.",
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="Write FROZEN.json hashes for usable conversations and detector export.",
    )
    args = parser.parse_args()

    from scambranch.llm import BudgetExceeded
    from scambranch.pipeline import run_generation

    try:
        result = run_generation(
            run_id=args.run_id,
            families=args.families,
            pairs=args.pairs,
            conditions=args.conditions,
            dry_run=args.dry_run,
            sync_review=args.sync_review,
            budget=args.budget,
            workers=args.workers,
            history_ids=args.histories,
            language_review_only=args.language_review_only,
            semrev03=args.semrev03,
            recompute_dispositions=args.recompute_dispositions,
            regen_prefix_broken=args.regen_prefix_broken,
            freeze=args.freeze,
        )
    except BudgetExceeded as exc:
        print(f"BUDGET EXCEEDED: {exc}", file=sys.stderr)
        return 2
    if result.get("projection_stop"):
        print(
            f"PROJECTION STOP: ${result.get('projected_usd'):.2f} exceeds budget",
            file=sys.stderr,
        )
    print(f"wrote {result['run_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
