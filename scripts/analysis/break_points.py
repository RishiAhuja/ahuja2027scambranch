#!/usr/bin/env python3
"""Where do zero-shot detectors break? Reads an eval predictions.jsonl (no model calls).

Writes BREAK_POINTS.md next to the predictions:
  1. per-family error rates on TEST (fraud misses, legit false alarms)
  2. keyword-confound families vs other legitimate families
  3. surface-keyword shortcut: legit false-alarm rate with vs without a keyword
  4. language x family false alarms (where the Tamil gap comes from)
  5. reaction flips: PREFIX_ONLY verdict -> PREFIX_PLUS_REACTION verdict by condition
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

UNRELIABLE = {"smollm2-135m", "tinyllama-1.1b"}  # 100% invalid outputs
STRONG = ["qwen2.5-1.5b", "qwen3-1.7b", "qwen2-1.5b"]
CONFOUND_FAMILIES = {
    "ec_08": "OTP mentioned in a legit delivery call",
    "jr_14": "fee disclosed legitimately (gig onboarding)",
    "jr_08": "account number / IFSC requested legitimately",
}
KEYWORDS = {
    "otp": r"\botp\b",
    "pin/cvv": r"\b(pin|cvv)\b",
    "link": r"\blink\b|https?://|\.invalid",
    "upi": r"\bupi\b",
    "fee/payment": r"\b(fee|fees|charge|charges|payment|pay|deposit|amount)\b",
    "account/ifsc": r"\b(account number|a/c|ifsc|account)\b",
    "app/install": r"\b(app|install|download|screen ?share)\b",
    "kyc": r"\bkyc\b",
}
FRAUD = {"fraud_attempt", "fraud"}


def load_preds(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_caller_text(conv_dir: Path, ids: set[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for cid in ids:
        p = conv_dir / f"{cid}.json"
        if not p.exists():
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        out[cid] = " ".join(
            t["text"] for t in rec["turns"] if t.get("speaker") == "caller"
        ).lower()
    return out


def rate(num: int, den: int) -> str:
    return f"{num / den:.2f} ({num}/{den})" if den else "—"


def fmt(x: float | None) -> str:
    return "—" if x is None else f"{x:.2f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, default=ROOT / "results/eval_hpc/predictions.jsonl")
    ap.add_argument("--conv-dir", type=Path, default=ROOT / "runs/full_v1/conversations")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    out_path = args.out or args.preds.parent / "BREAK_POINTS.md"

    rows = [r for r in load_preds(args.preds) if r["split"] == "test"]
    models = sorted({r["model"] for r in rows} - UNRELIABLE)
    rows = [r for r in rows if r["model"] in models]
    full = [r for r in rows if r["view"] == "FULL"]
    lines: list[str] = [
        f"# Break points (`{args.preds.parent.name}`, TEST)",
        "",
        f"Models: {', '.join(models)}. Excluded (100% invalid outputs): "
        f"{', '.join(sorted(UNRELIABLE))}. Strong = {', '.join(STRONG)}.",
        "Verdicts use the DEV-fit threshold for each (model, view).",
        "",
    ]

    # 1. per-family errors, FULL view
    err = defaultdict(lambda: [0, 0])  # (family, case, model) -> [errors, n]
    for r in full:
        case = "F" if r["label"] in FRAUD else "L"
        wrong = (r["pred"] == "fraud") != (case == "F")
        key = (r["scenario_family_id"], case, r["model"])
        err[key][0] += int(wrong)
        err[key][1] += 1
    fams = sorted({(k[0], k[1]) for k in err})
    lines += [
        "## 1. Per-family error rate (FULL view)",
        "",
        "F = share of fraud chats missed; L = share of legit chats flagged as fraud.",
        "",
        "| family | case | all models | strong models | " + " | ".join(STRONG) + " |",
        "|---|---|---:|---:|" + "---:|" * len(STRONG),
    ]
    fam_rows = []
    for fam, case in fams:
        tot = [err[(fam, case, m)] for m in models if (fam, case, m) in err]
        strong = [err[(fam, case, m)] for m in STRONG if (fam, case, m) in err]
        all_rate = sum(e for e, _ in tot) / max(1, sum(n for _, n in tot))
        s_rate = sum(e for e, _ in strong) / max(1, sum(n for _, n in strong))
        per = []
        for m in STRONG:
            e, n = err.get((fam, case, m), (0, 0))
            per.append(fmt(e / n) if n else "—")
        fam_rows.append((s_rate, fam, case, all_rate, per))
    for s_rate, fam, case, all_rate, per in sorted(fam_rows, key=lambda x: -x[0]):
        tag = f" *({CONFOUND_FAMILIES[fam]})*" if fam in CONFOUND_FAMILIES and case == "L" else ""
        lines.append(f"| {fam}{tag} | {case} | {fmt(all_rate)} | {fmt(s_rate)} | " + " | ".join(per) + " |")
    lines.append("")

    # 2. confound families vs other legit families
    lines += [
        "## 2. Keyword-confound legit families vs other legit families (FULL, false-alarm rate)",
        "",
        "| model | confound legit (ec_08, jr_14, jr_08) | other legit | gap |",
        "|---|---:|---:|---:|",
    ]
    for m in models:
        c = [r for r in full if r["model"] == m and r["label"] not in FRAUD and r["scenario_family_id"] in CONFOUND_FAMILIES]
        o = [r for r in full if r["model"] == m and r["label"] not in FRAUD and r["scenario_family_id"] not in CONFOUND_FAMILIES]
        cf = sum(r["pred"] == "fraud" for r in c)
        of = sum(r["pred"] == "fraud" for r in o)
        gap = (cf / len(c) - of / len(o)) if c and o else None
        lines.append(f"| {m} | {rate(cf, len(c))} | {rate(of, len(o))} | {'—' if gap is None else f'{gap:+.2f}'} |")
    lines.append("")

    # 3. keyword shortcut on legit chats
    ids = {r["conversation_id"] for r in full}
    text = load_caller_text(args.conv_dir, ids)
    kw_hits = {cid: {k for k, pat in KEYWORDS.items() if re.search(pat, t)} for cid, t in text.items()}
    lines += [
        "## 3. Surface-keyword shortcut on LEGIT chats (FULL, false-alarm rate)",
        "",
        "Keyword searched in the caller's lines only. Positive gap = the word alone pushes a genuine call toward 'fraud'.",
        "",
        "| keyword | legit chats with it | " + " | ".join(f"{m} with / without" for m in STRONG) + " |",
        "|---|---:|" + "---|" * len(STRONG),
    ]
    legit_ids = {r["conversation_id"] for r in full if r["label"] not in FRAUD}
    for kw in KEYWORDS:
        with_kw = {c for c in legit_ids if kw in kw_hits.get(c, set())}
        cells = []
        for m in STRONG:
            mr = {r["conversation_id"]: r for r in full if r["model"] == m and r["conversation_id"] in legit_ids}
            w = [mr[c] for c in with_kw if c in mr]
            wo = [mr[c] for c in legit_ids - with_kw if c in mr]
            wf = sum(r["pred"] == "fraud" for r in w) / len(w) if w else None
            wof = sum(r["pred"] == "fraud" for r in wo) / len(wo) if wo else None
            cells.append(f"{fmt(wf)} / {fmt(wof)}")
        lines.append(f"| {kw} | {len(with_kw)} | " + " | ".join(cells) + " |")
    lines.append("")

    # 4. language x family false alarms for strong models
    lines += [
        "## 4. Legit false alarms by language (FULL, strong models pooled)",
        "",
        "| family | hi_en | ta_en | te_en |",
        "|---|---:|---:|---:|",
    ]
    la = defaultdict(lambda: [0, 0])
    for r in full:
        if r["model"] in STRONG and r["label"] not in FRAUD:
            key = (r["scenario_family_id"], r["language_pair"])
            la[key][0] += int(r["pred"] == "fraud")
            la[key][1] += 1
    legit_fams = sorted({k[0] for k in la})
    for fam in legit_fams:
        cells = []
        for pair in ("hi_en", "ta_en", "te_en"):
            f, n = la.get((fam, pair), (0, 0))
            cells.append(fmt(f / n) if n else "—")
        lines.append(f"| {fam} | " + " | ".join(cells) + " |")
    tot = {p: [0, 0] for p in ("hi_en", "ta_en", "te_en")}
    for (fam, pair), (f, n) in la.items():
        tot[pair][0] += f
        tot[pair][1] += n
    lines.append("| **all** | " + " | ".join(fmt(tot[p][0] / tot[p][1]) if tot[p][1] else "—" for p in ("hi_en", "ta_en", "te_en")) + " |")
    lines.append("")

    # 5. reaction flips
    po = {(r["model"], r["conversation_id"]): r for r in rows if r["view"] == "PREFIX_ONLY"}
    lines += [
        "## 5. Verdict flips when the recipient's first line is added",
        "",
        "Share of chats whose verdict changes from PREFIX_ONLY to PREFIX_PLUS_REACTION.",
        "Fraud chats: flip = fraud -> legitimate (victim reaction clears the caller).",
        "Legit chats: flip = legitimate -> fraud (victim reaction creates a false alarm).",
        "",
        "| model | label | cooperative | clarifying | resistant | suspicious |",
        "|---|---|---:|---:|---:|---:|",
    ]
    conds = ["cooperative", "clarifying", "resistant", "suspicious"]
    for m in models:
        for is_fraud, name in ((True, "fraud"), (False, "legit")):
            cells = []
            for cond in conds:
                pr = [r for r in rows if r["model"] == m and r["view"] == "PREFIX_PLUS_REACTION"
                      and r["recipient_condition"] == cond and (r["label"] in FRAUD) == is_fraud]
                base_from = "fraud" if is_fraud else "legitimate"
                elig = [r for r in pr if (m, r["conversation_id"]) in po and po[(m, r["conversation_id"])]["pred"] == base_from]
                flips = sum(r["pred"] != base_from for r in elig)
                cells.append(rate(flips, len(elig)))
            lines.append(f"| {m} | {name} | " + " | ".join(cells) + " |")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
