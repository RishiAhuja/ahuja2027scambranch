#!/usr/bin/env python3
"""Summarise a run ledger and project Luna + Haiku-batch cost."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


GEN_ROLES = {"plan", "prefix", "caller", "recipient", "repair"}
REVIEW_ROLES = {"semantic_review", "language_review"}
AUDIT_ROLES = {"audit", "audit_semantic", "audit_language"}


def load_ledger(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_summary(run_dir: Path) -> dict:
    p = run_dir / "summary.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    n_acc = 0
    n = 0
    for f in (run_dir / "conversations").glob("*.json"):
        rec = json.loads(f.read_text(encoding="utf-8"))
        n += 1
        disp = rec.get("provenance", {}).get("generation_settings", {}).get("disposition")
        if disp in {"accepted", "repaired"}:
            n_acc += 1
    return {"n": n, "accepted": n_acc}


def group_report(rows: list[dict]) -> None:
    by_role: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    by_model: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        for bucket, key in ((by_role, row.get("role") or "?"), (by_model, row.get("model") or "?")):
            bucket[key]["usd"] += float(row.get("usd") or 0)
            bucket[key]["input_tokens"] += float(row.get("input_tokens") or 0)
            bucket[key]["cached_tokens"] += float(row.get("cached_tokens") or 0)
            bucket[key]["output_tokens"] += float(row.get("output_tokens") or 0)
            bucket[key]["reasoning_tokens"] += float(row.get("reasoning_tokens") or 0)
            bucket[key]["n"] += 1
    print("## Cost by role")
    for role, s in sorted(by_role.items()):
        print(
            f"  {role:20s} n={int(s['n']):4d}  in={int(s['input_tokens']):7d}  "
            f"cached={int(s['cached_tokens']):7d}  out={int(s['output_tokens']):7d}  "
            f"reason={int(s['reasoning_tokens']):7d}  ${s['usd']:.6f}"
        )
    print("## Cost by model")
    for model, s in sorted(by_model.items()):
        print(
            f"  {model:24s} n={int(s['n']):4d}  in={int(s['input_tokens']):7d}  "
            f"out={int(s['output_tokens']):7d}  ${s['usd']:.6f}"
        )


def project(rows: list[dict], n_conv: int, n_conditions: int, audit_frac: float) -> dict:
    """Scale observed per-conversation usage; apply Haiku batch 50% to reviews."""
    gen = [r for r in rows if r.get("role") in GEN_ROLES]
    rev = [r for r in rows if r.get("role") in REVIEW_ROLES]
    n_obs_conv = max(
        1,
        len({r.get("conversation_id") for r in rows if r.get("conversation_id")}),
    )
    # Plan/prefix are per unit (shared across conditions).
    unit_roles = [r for r in gen if r.get("role") in {"plan", "prefix"}]
    branch_roles = [r for r in gen if r.get("role") in {"caller", "recipient", "repair"}]
    n_units = max(1, n_obs_conv // max(n_conditions, 1))

    def tot(rs: list[dict], key: str) -> float:
        return sum(float(r.get(key) or 0) for r in rs)

    gen_usd = tot(gen, "usd")
    rev_usd = tot(rev, "usd")
    # If reviews were sync (full Haiku price), batch is 50%.
    # If already batch, keep as-is (ledger.batch true).
    rev_was_batch = all(bool(r.get("batch")) for r in rev) if rev else False
    rev_batch_usd = rev_usd if rev_was_batch else rev_usd * 0.5

    per_unit_plan = tot(unit_roles, "usd") / n_units
    per_conv_branch = tot(branch_roles, "usd") / n_obs_conv
    per_conv_rev_batch = rev_batch_usd / n_obs_conv
    per_conv_gen = per_unit_plan / max(n_conditions, 1) + per_conv_branch

    n_units_proj = n_conv / max(n_conditions, 1)
    proj_gen = per_unit_plan * n_units_proj + per_conv_branch * n_conv
    proj_rev = per_conv_rev_batch * n_conv

    # Sol audit: 5% of conversations, both review types, sync Sol prices.
    # Estimate from Haiku review token volume * Sol unit prices if no audit rows.
    audit_rows = [r for r in rows if r.get("role") in AUDIT_ROLES]
    if audit_rows:
        per_audit = tot(audit_rows, "usd") / max(
            1, len({r.get("conversation_id") for r in audit_rows})
        )
    else:
        # Approximate: same token counts as Haiku reviews, billed at Sol sync.
        in_tok = tot(rev, "input_tokens") / max(n_obs_conv, 1)
        out_tok = tot(rev, "output_tokens") / max(n_obs_conv, 1)
        # Sol: $2 / $10 per MTok
        per_audit = (in_tok * 2.0 + out_tok * 10.0) / 1_000_000.0
    proj_audit = per_audit * audit_frac * n_conv

    return {
        "n_conv": n_conv,
        "observed_conversations": n_obs_conv,
        "per_conversation_generation_usd": per_conv_gen,
        "per_conversation_haiku_batch_review_usd": per_conv_rev_batch,
        "projected_generation_usd": proj_gen,
        "projected_haiku_batch_review_usd": proj_rev,
        "projected_total_usd": proj_gen + proj_rev,
        "projected_sol_audit_usd": proj_audit,
        "projected_total_with_audit_usd": proj_gen + proj_rev + proj_audit,
        "generation_usd_observed": gen_usd,
        "review_usd_observed": rev_usd,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--families", type=int, default=60)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--conditions", type=int, default=4)
    parser.add_argument("--audit-frac", type=float, default=0.05)
    args = parser.parse_args()

    run_dir = args.runs_dir / args.run_id
    rows = load_ledger(run_dir / "ledger.jsonl")
    if not rows:
        print(f"no ledger at {run_dir / 'ledger.jsonl'}", file=sys.stderr)
        return 1
    summary = load_summary(run_dir)
    total = sum(float(r.get("usd") or 0) for r in rows)
    n_acc = max(int(summary.get("accepted") or 0), 1)
    print(f"# Run {args.run_id}")
    print(f"ledger rows: {len(rows)}")
    print(f"total usd: ${total:.6f}")
    print(f"accepted conversations: {summary.get('accepted')} / {summary.get('n')}")
    print(f"cost per accepted conversation: ${total / n_acc:.6f}")
    group_report(rows)

    n1440 = args.families * 2 * args.pairs * args.conditions * 1
    n2880 = args.families * 2 * args.pairs * args.conditions * 2
    print("\n# Projection (Luna generation + Haiku batch review)")
    for n in (n1440, n2880):
        proj = project(rows, n, args.conditions, args.audit_frac)
        print(
            f"\n{n} conversations "
            f"({args.families} fam × 2 cases × {args.pairs} pairs × {args.conditions} cond × "
            f"{n // (args.families * 2 * args.pairs * args.conditions)} hist)"
        )
        print(f"  generation:           ${proj['projected_generation_usd']:.2f}")
        print(f"  Haiku batch review:   ${proj['projected_haiku_batch_review_usd']:.2f}")
        print(f"  total (no audit):     ${proj['projected_total_usd']:.2f}")
        print(
            f"  + {args.audit_frac:.0%} Sol audit: ${proj['projected_sol_audit_usd']:.2f} "
            f"→ ${proj['projected_total_with_audit_usd']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
