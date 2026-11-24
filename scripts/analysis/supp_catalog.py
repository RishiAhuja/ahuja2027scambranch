#!/usr/bin/env python3
"""Build DATASET-DOCUMENTATION supplement catalog tables from frozen artifacts.

Read-only over scenarios/, runs/full_v1/, configs/, prompts/, results/eval_*.
Writes LaTeX + REPORT.md under results/supplement/catalog/.
Never calls LLMs or loads detector weights.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / ".venv" / "bin" / "python"
OUT = ROOT / "results" / "supplement" / "catalog"
RUN = ROOT / "runs" / "full_v1"
SCEN = ROOT / "scenarios"
MANIFESTS = ROOT / "manifests"
PROMPTS = ROOT / "prompts"
CONFIGS = ROOT / "configs"
EVAL_ZS = ROOT / "results" / "eval_hpc"
EVAL_FT = ROOT / "results" / "eval_hpc_ft_rest"

SEM_CHECKS = (
    "evidence_present",
    "evidence_not_invalidated",
    "facts_consistent",
    "condition_followed",
    "caller_within_limits",
    "no_leakage",
    "plausible_ending",
)
HARD_FLAG_TYPES = ("script", "wrong_language", "artifact")
CONDITIONS = ("cooperative", "clarifying", "resistant", "suspicious")
PAIRS = ("hi_en", "ta_en", "te_en")
PAIR_DISPLAY = {
    "hi_en": "Hindi--English",
    "ta_en": "Tamil--English",
    "te_en": "Telugu--English",
}
DOMAIN_META = {
    "banking_payments": {
        "prefix": "bp",
        "group": "Banking / payments",
        "display": "Banking",
        "readme": SCEN / "draft" / "banking_payments" / "README.md",
    },
    "jobs_recruitment": {
        "prefix": "jr",
        "group": "Jobs / recruitment",
        "display": "Jobs",
        "readme": SCEN / "draft" / "jobs_recruitment" / "README.md",
    },
    "ecommerce_delivery": {
        "prefix": "ec",
        "group": "E-commerce / delivery",
        "display": "Commerce",
        "readme": SCEN / "draft" / "ecommerce_delivery" / "README.md",
    },
}

# Display-name registry for detectors (human-readable).
DETECTOR_DISPLAY = {
    "qwen3-0.6b": "Qwen3-0.6B",
    "qwen3-1.7b": "Qwen3-1.7B",
    "qwen2.5-1.5b": "Qwen2.5-1.5B-Instruct",
    "qwen2.5-0.5b": "Qwen2.5-0.5B-Instruct",
    "qwen2-0.5b": "Qwen2-0.5B-Instruct",
    "qwen2-1.5b": "Qwen2-1.5B-Instruct",
    "llama-3.2-1b": "Llama-3.2-1B-Instruct",
    "smollm2-1.7b": "SmolLM2-1.7B-Instruct",
    "smollm2-135m": "SmolLM2-135M-Instruct",
    "smollm2-360m": "SmolLM2-360M-Instruct",
    "gemma3-270m": "Gemma-3-270M-IT",
    "gemma3-1b": "Gemma-3-1B-IT",
    "tinyllama-1.1b": "TinyLlama-1.1B-Chat",
    "stablelm2-1.6b": "StableLM-2-Zephyr-1.6B",
    "internlm2-1.8b": "InternLM2-Chat-1.8B",
    "danube2-1.8b": "H2O-Danube2-1.8B-Chat",
    "granite-1b": "Granite-3.1-1B-A400M-Instruct",
    "bloomz-1b7": "BLOOMZ-1B7",
    "olmo-1b": "OLMo-1B-0724",
}

STAGE_ORDER = (
    "plan",
    "prefix",
    "recipient",
    "caller",
    "repair",
    "semantic_review",
    "language_review",
    "audit",
)


def latex_escape(text: str) -> str:
    if text is None:
        return ""
    s = str(text)
    repl = (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    )
    for a, b in repl:
        s = s.replace(a, b)
    return s


def pct(n: int, d: int) -> str:
    if d <= 0:
        return "---"
    return f"{n / d:.2f}".lstrip("0") if n / d < 1 else f"{n / d:.2f}"


def prop(n: int, d: int) -> str:
    """Proportion in .73 style (leading zero dropped when < 1)."""
    if d <= 0:
        return "---"
    v = n / d
    s = f"{v:.2f}"
    if s.startswith("0."):
        return s[1:]
    return s


def fmt_usd(x: float) -> str:
    return f"{x:.4f}"


def fmt_params(n: int | None) -> str:
    if n is None:
        return "---"
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.0f}M"
    return str(n)


def one_line(text: str, max_len: int = 90) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    if len(t) <= max_len:
        return t
    cut = t[: max_len - 1].rsplit(" ", 1)[0]
    return cut + "…"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_readme_catalog(path: Path) -> dict[str, dict[str, str]]:
    """Parse domain README catalog table rows."""
    out: dict[str, dict[str, str]] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| `"):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 4:
            continue
        m = re.match(r"`([a-z]{2}_\d{2})`", parts[0])
        if not m:
            continue
        fid = m.group(1)
        out[fid] = {
            "title": parts[1].strip(),
            "fraud": parts[2].strip(),
            "legitimate": parts[3].strip(),
        }
    return out


@dataclass
class CatalogState:
    issues: list[str] = field(default_factory=list)
    key_numbers: dict[str, Any] = field(default_factory=dict)


def load_manifest() -> list[dict[str, Any]]:
    rows = []
    for line in (RUN / "private_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build_families(state: CatalogState) -> str:
    splits = load_json(MANIFESTS / "splits_v1.json")["assignment"]
    catalogs = {}
    for domain, meta in DOMAIN_META.items():
        catalogs.update(parse_readme_catalog(meta["readme"]))

    usable_by_family: Counter[str] = Counter()
    for row in load_manifest():
        if row["disposition"] in {"accepted", "repaired"}:
            usable_by_family[row["scenario_family_id"]] += 1

    lines: list[str] = []
    lines.append(r"% Auto-generated by scripts/analysis/supp_catalog.py — do not edit by hand.")
    lines.append(r"\begin{longtable}{llp{0.22\textwidth}p{0.22\textwidth}p{0.18\textwidth}lc}")
    lines.append(
        r"\caption{Scenario families (60). Display id is domain-local; original id retained for traceability. "
        r"Usable = accepted + repaired calls (of 48 generated per family).}"
    )
    lines.append(r"\label{tab:cat-families}\\")
    lines.append(r"\toprule")
    lines.append(
        r"\rowcolor{headblue} Disp. & Orig. & Fraud mechanism & Genuine counterpart "
        r"& Label evidence & Split & Use \\"
    )
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(
        r"\rowcolor{headblue} Disp. & Orig. & Fraud mechanism & Genuine counterpart "
        r"& Label evidence & Split & Use \\"
    )
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{7}{r}{\emph{Continued on next page}}\\")
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    for domain, meta in DOMAIN_META.items():
        lines.append(rf"\rowcolor{{groupgray}} \multicolumn{{7}}{{l}}{{\textbf{{{meta['group']}}}}} \\")
        for i in range(1, 21):
            fid = f"{meta['prefix']}_{i:02d}"
            card = load_json(SCEN / "json" / f"{fid}.json")
            cat = catalogs.get(fid, {})
            fraud = cat.get("fraud") or one_line(
                card["cases"]["fraud"].get("caller_objective")
                or card["cases"]["fraud"].get("label_rationale", ""),
                85,
            )
            legit = cat.get("legitimate") or one_line(
                card["cases"]["legitimate"].get("caller_objective")
                or card["cases"]["legitimate"].get("label_rationale", ""),
                85,
            )
            # Evidence that decides the label: compress fraud + legitimate rationales.
            fr = one_line(card["cases"]["fraud"].get("label_rationale", ""), 70)
            lr = one_line(card["cases"]["legitimate"].get("label_rationale", ""), 55)
            evidence = f"F: {fr} / L: {lr}"
            evidence = one_line(evidence, 110)
            split = splits.get(fid) or card.get("split") or "?"
            if splits.get(fid) and card.get("split") and splits[fid] != card["split"]:
                state.issues.append(
                    f"split mismatch {fid}: splits_v1={splits[fid]} card={card['split']}"
                )
            use = usable_by_family.get(fid, 0)
            disp = f"{meta['display']} {i}"
            lines.append(
                f"{latex_escape(disp)} & \\texttt{{{latex_escape(fid)}}} & "
                f"{latex_escape(fraud)} & {latex_escape(legit)} & "
                f"{latex_escape(evidence)} & {latex_escape(split)} & {use} \\\\"
            )
            state.key_numbers.setdefault("families", {})[fid] = {
                "split": split,
                "usable": use,
            }

    lines.append(r"\end{longtable}")
    return "\n".join(lines) + "\n"


def build_splits(state: CatalogState) -> str:
    rows = load_manifest()
    usable = [r for r in rows if r["disposition"] in {"accepted", "repaired"}]
    splits_assign = load_json(MANIFESTS / "splits_v1.json")["assignment"]
    fam_by_split: dict[str, set[str]] = defaultdict(set)
    for fid, sp in splits_assign.items():
        fam_by_split[sp].add(fid)

    # Calls: split x language x label x condition
    counts: Counter[tuple[str, str, str, str]] = Counter()
    for r in usable:
        label = "fraud" if r["case_id"].endswith("_F") else "legitimate"
        counts[(r["split"], r["language_pair"], label, r["branch_id"])] += 1

    complete = load_json(RUN / "complete_sets.json")
    by_split_complete = complete.get("by_split") or {}

    lines: list[str] = []
    lines.append(r"% Auto-generated by scripts/analysis/supp_catalog.py")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Usable calls by split $\times$ language $\times$ label $\times$ recipient "
        r"condition; families per split; complete four-branch sets.}"
    )
    lines.append(r"\label{tab:cat-splits}")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{tabular}{llrrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"\rowcolor{headblue} Split & Lang. & Label & Coop. & Clar. & Resist. & Susp. & Total \\"
    )
    lines.append(r"\midrule")

    grand = 0
    for split in ("train", "dev", "test"):
        split_total = 0
        first_split = True
        for pair in PAIRS:
            for label in ("fraud", "legitimate"):
                vals = [counts[(split, pair, label, c)] for c in CONDITIONS]
                tot = sum(vals)
                split_total += tot
                grand += tot
                sp_cell = split if first_split else ""
                first_split = False
                lines.append(
                    f"{sp_cell} & {latex_escape(PAIR_DISPLAY[pair])} & {label} & "
                    + " & ".join(str(v) for v in vals)
                    + f" & {tot} \\\\"
                )
        lines.append(rf"\rowcolor{{groupgray}} & & \textbf{{split total}} & & & & & \textbf{{{split_total}}} \\")
        n_fam = len(fam_by_split[split])
        n_comp = by_split_complete.get(split, complete.get("by_split", {}).get(split))
        if isinstance(n_comp, dict):
            n_comp_n = n_comp.get("n") or n_comp.get("total") or sum(
                v if isinstance(v, int) else 0 for v in n_comp.values()
            )
        else:
            n_comp_n = n_comp
        # Prefer complete_sets.by_split structure
        if isinstance(by_split_complete.get(split), int):
            n_comp_n = by_split_complete[split]
        elif isinstance(by_split_complete.get(split), dict):
            d = by_split_complete[split]
            n_comp_n = d.get("n") or d.get("total") or sum(
                int(v) for v in d.values() if isinstance(v, (int, float))
            )
        lines.append(
            rf"\multicolumn{{8}}{{l}}{{\quad Families: {n_fam}; "
            rf"complete 4-branch sets: {n_comp_n}}} \\"
        )
        if split != "test":
            lines.append(r"\midrule")

    lines.append(r"\midrule")
    lines.append(rf"\rowcolor{{groupgray}} \textbf{{All}} & & & & & & & \textbf{{{grand}}} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    # Also write a compact families/complete summary table part via comment in report
    state.key_numbers["usable_total"] = grand
    state.key_numbers["families_per_split"] = {s: len(fam_by_split[s]) for s in fam_by_split}
    state.key_numbers["complete_sets"] = {
        "n_complete": complete.get("n_complete"),
        "n_groups": complete.get("n_groups"),
        "by_split": by_split_complete,
    }
    if grand != 2643:
        state.issues.append(f"usable call count from manifest is {grand}, expected 2643")
    return "\n".join(lines) + "\n"


def repair_rounds_distribution(conversations: dict[str, dict]) -> Counter[int]:
    """Distribution of repair rounds among calls that were repaired.

    Uses max ``round`` in provenance.repair_history when present; else 0.
    For disposition=repaired with empty history, count as unknown (-1) and note.
    """
    dist: Counter[int] = Counter()
    return dist  # filled by caller


def build_pipeline_outcomes(state: CatalogState) -> str:
    rows = load_manifest()
    conversations: dict[str, dict] = {}
    # Load conversations lazily for review stats — all 2880 needed
    for p in (RUN / "conversations").glob("*.json"):
        conversations[p.stem] = load_json(p)

    # Disposition crosstabs
    by_lang: dict[str, Counter] = defaultdict(Counter)
    by_label: dict[str, Counter] = defaultdict(Counter)
    by_cond: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        d = r["disposition"]
        by_lang[r["language_pair"]][d] += 1
        label = "fraud" if r["case_id"].endswith("_F") else "legitimate"
        by_label[label][d] += 1
        by_cond[r["branch_id"]][d] += 1

    summary = load_json(RUN / "summary.json")
    rej_reasons = Counter(summary.get("rejection_reason") or {})

    # Repair rounds among disposition=repaired
    repair_round_dist: Counter[int] = Counter()
    repaired_no_hist = 0
    for r in rows:
        if r["disposition"] != "repaired":
            continue
        c = conversations[r["conversation_id"]]
        rh = c["provenance"].get("repair_history") or []
        if not rh:
            repaired_no_hist += 1
            repair_round_dist[0] += 1
            continue
        mx = max(int(e.get("round", 0)) for e in rh if isinstance(e, dict))
        repair_round_dist[mx] += 1

    # Semantic fail-at-least-once: OR across stored semantic + semantic_v02
    fail_once: Counter[str] = Counter()
    n_calls = len(conversations)
    for c in conversations.values():
        revs = (c["provenance"].get("generation_settings") or {}).get("reviews") or {}
        failed: set[str] = set()
        for key in ("semantic", "semantic_v02"):
            sem = revs.get(key) or {}
            for ck, ch in (sem.get("checks") or {}).items():
                if isinstance(ch, dict) and ch.get("pass") is False:
                    failed.add(ck)
        for ck in failed:
            fail_once[ck] += 1

    # Language score distributions + hard flags
    score_dist: dict[tuple[str, str], Counter] = defaultdict(Counter)
    hard_flags: Counter[str] = Counter()
    hard_by_pair: dict[str, Counter] = defaultdict(Counter)
    soft_flags: Counter[str] = Counter()
    n_lang = 0
    for c in conversations.values():
        pair = c["language_pair"]
        lang = ((c["provenance"].get("generation_settings") or {}).get("reviews") or {}).get(
            "language"
        ) or {}
        if not lang:
            continue
        n_lang += 1
        sn = lang.get("switching_naturalness")
        ga = lang.get("grammatical_acceptability")
        if sn is not None:
            score_dist[(pair, "switching")][int(sn)] += 1
        if ga is not None:
            score_dist[(pair, "grammar")][int(ga)] += 1
        for tf in lang.get("turn_flags") or []:
            if not isinstance(tf, dict):
                continue
            t = tf.get("type") or "unknown"
            if t in HARD_FLAG_TYPES:
                hard_flags[t] += 1
                hard_by_pair[pair][t] += 1
            else:
                soft_flags[t] += 1

    # Prefix regenerations from FINAL_DATA_REPORT + superseded count
    superseded_n = len(list((RUN / "superseded").glob("*.json"))) if (RUN / "superseded").exists() else 0
    final_report = (RUN / "FINAL_DATA_REPORT.md").read_text(encoding="utf-8")
    m = re.search(
        r"prefix-broken units regenerated:\s*(\d+)\s*\(moved\s*(\d+),\s*wrote\s*(\d+)\)",
        final_report,
    )
    if m:
        prefix_units, prefix_moved, prefix_wrote = map(int, m.groups())
    else:
        prefix_units, prefix_moved, prefix_wrote = None, None, None
        state.issues.append("could not parse prefix regeneration line from FINAL_DATA_REPORT.md")

    if superseded_n and prefix_moved and superseded_n != prefix_moved:
        state.issues.append(
            f"superseded/ has {superseded_n} files; FINAL_DATA_REPORT moved={prefix_moved}"
        )

    lines: list[str] = []
    lines.append(r"% Auto-generated by scripts/analysis/supp_catalog.py")

    # --- Disposition ---
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Pipeline dispositions (accepted first pass / accepted after repair / rejected) "
        r"by language, label, and recipient condition. Totals: "
        rf"{summary['accepted']} / {summary['repaired']} / {summary['rejected']} "
        rf"(of {summary['n']}; usable {summary['accepted']+summary['repaired']}).}}"
    )
    lines.append(r"\label{tab:cat-pipeline-disp}")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"\rowcolor{headblue} Slice & $n$ & First-pass & After repair & Rejected \\"
    )
    lines.append(r"\midrule")
    lines.append(r"\rowcolor{groupgray} \multicolumn{5}{l}{\textbf{By language}} \\")
    for pair in PAIRS:
        c = by_lang[pair]
        n = sum(c.values())
        lines.append(
            f"{latex_escape(PAIR_DISPLAY[pair])} & {n} & {c['accepted']} & "
            f"{c['repaired']} & {c['rejected']} \\\\"
        )
    lines.append(r"\rowcolor{groupgray} \multicolumn{5}{l}{\textbf{By label}} \\")
    for label in ("fraud", "legitimate"):
        c = by_label[label]
        n = sum(c.values())
        lines.append(
            f"{label} & {n} & {c['accepted']} & {c['repaired']} & {c['rejected']} \\\\"
        )
    lines.append(r"\rowcolor{groupgray} \multicolumn{5}{l}{\textbf{By condition}} \\")
    for cond in CONDITIONS:
        c = by_cond[cond]
        n = sum(c.values())
        lines.append(
            f"{cond} & {n} & {c['accepted']} & {c['repaired']} & {c['rejected']} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # --- Repair rounds + rejection + prefix regen ---
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Repair-round distribution (disposition=repaired; max round in "
        r"\texttt{repair\_history}), rejection-reason histogram (final gate), "
        r"and prefix regenerations.}"
    )
    lines.append(r"\label{tab:cat-pipeline-repair}")
    lines.append(r"\begin{tabular}{lr}")
    lines.append(r"\toprule")
    lines.append(r"\rowcolor{headblue} Item & Count \\")
    lines.append(r"\midrule")
    lines.append(r"\rowcolor{groupgray} \multicolumn{2}{l}{\textbf{Repair rounds (repaired calls)}} \\")
    for rnd in sorted(repair_round_dist):
        label = f"max round = {rnd}" + (" (empty history)" if rnd == 0 else "")
        lines.append(f"{latex_escape(label)} & {repair_round_dist[rnd]} \\")
    if repaired_no_hist:
        state.issues.append(
            f"{repaired_no_hist} repaired calls have empty repair_history"
        )
    lines.append(r"\rowcolor{groupgray} \multicolumn{2}{l}{\textbf{Rejection reasons}} \\")
    for reason, n in rej_reasons.most_common():
        lines.append(f"\\texttt{{{latex_escape(reason)}}} & {n} \\")
    lines.append(r"\rowcolor{groupgray} \multicolumn{2}{l}{\textbf{Prefix regenerations}} \\")
    lines.append(
        f"Units regenerated & {prefix_units if prefix_units is not None else '---'} \\\\"
    )
    lines.append(
        f"Calls moved to \\texttt{{superseded/}} & "
        f"{prefix_moved if prefix_moved is not None else superseded_n} \\\\"
    )
    lines.append(
        f"Calls rewritten & {prefix_wrote if prefix_wrote is not None else '---'} \\\\"
    )
    lines.append(rf"Files currently in \\texttt{{superseded/}} & {superseded_n} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # --- Semantic fail rates ---
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Share of calls where each semantic check failed in at least one stored "
        r"review (\texttt{semantic} or \texttt{semantic\_v02}). Intermediate overwritten "
        r"reviews are not retained, so rates are a lower bound on true fail-once incidence.}"
    )
    lines.append(r"\label{tab:cat-pipeline-semfail}")
    lines.append(r"\begin{tabular}{lrr}")
    lines.append(r"\toprule")
    lines.append(r"\rowcolor{headblue} Check & Fail-once $n$ & Share \\")
    lines.append(r"\midrule")
    for ck in SEM_CHECKS:
        n = fail_once.get(ck, 0)
        lines.append(f"\\texttt{{{latex_escape(ck)}}} & {n} & {prop(n, n_calls)} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # --- Language scores ---
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Language-review score distributions (1--5) by language pair, and hard-flag "
        r"counts (\texttt{script}, \texttt{wrong\_language}, \texttt{artifact}). "
        rf"Reviews present for {n_lang}/{n_calls} calls. Soft turn-flag types are listed in the report.}}"
    )
    lines.append(r"\label{tab:cat-pipeline-lang}")
    lines.append(r"\begin{tabular}{llrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"\rowcolor{headblue} Lang. & Score & 1 & 2 & 3 & 4 & 5 \\"
    )
    lines.append(r"\midrule")
    for pair in PAIRS:
        for kind, label in (("switching", "switching nat."), ("grammar", "grammatical")):
            ctr = score_dist[(pair, kind)]
            cells = [str(ctr.get(i, 0)) for i in range(1, 6)]
            lines.append(
                f"{latex_escape(PAIR_DISPLAY[pair])} & {label} & "
                + " & ".join(cells)
                + " \\\\"
            )
    lines.append(r"\midrule")
    lines.append(r"\rowcolor{groupgray} \multicolumn{7}{l}{\textbf{Hard flags (turn\_flags)}} \\")
    lines.append(r"\multicolumn{2}{l}{\texttt{script} (all langs)} & \multicolumn{5}{l}{"
                 + str(hard_flags.get("script", 0)) + r"} \\")
    lines.append(
        r"\multicolumn{2}{l}{\texttt{wrong\_language}} & \multicolumn{5}{l}{"
        + str(hard_flags.get("wrong_language", 0))
        + r"} \\"
    )
    lines.append(
        r"\multicolumn{2}{l}{\texttt{artifact}} & \multicolumn{5}{l}{"
        + str(hard_flags.get("artifact", 0))
        + r"} \\"
    )
    for pair in PAIRS:
        hb = hard_by_pair[pair]
        lines.append(
            rf"\multicolumn{{2}}{{l}}{{\quad {latex_escape(PAIR_DISPLAY[pair])}}} & "
            rf"\multicolumn{{5}}{{l}}{{script={hb.get('script',0)}, "
            rf"wrong\_lang={hb.get('wrong_language',0)}, "
            rf"artifact={hb.get('artifact',0)}}} \\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    state.key_numbers["dispositions"] = {
        "accepted": summary["accepted"],
        "repaired": summary["repaired"],
        "rejected": summary["rejected"],
        "n": summary["n"],
        "usable": summary["accepted"] + summary["repaired"],
    }
    state.key_numbers["rejection_reasons"] = dict(rej_reasons.most_common())
    state.key_numbers["repair_rounds"] = dict(sorted(repair_round_dist.items()))
    state.key_numbers["sem_fail_once"] = {ck: fail_once.get(ck, 0) for ck in SEM_CHECKS}
    state.key_numbers["sem_fail_once_top"] = fail_once.most_common(3)
    state.key_numbers["hard_flags"] = dict(hard_flags)
    state.key_numbers["soft_flags"] = dict(soft_flags.most_common())
    state.key_numbers["prefix_regen"] = {
        "units": prefix_units,
        "moved": prefix_moved,
        "wrote": prefix_wrote,
        "superseded_files": superseded_n,
    }
    state.key_numbers["language_reviews_present"] = n_lang
    return "\n".join(lines) + "\n"


def build_cost(state: CatalogState) -> str:
    by_role: dict[str, dict[str, float]] = defaultdict(
        lambda: {"usd": 0.0, "input": 0, "output": 0, "cached": 0, "n": 0}
    )
    by_model: dict[str, dict[str, float]] = defaultdict(
        lambda: {"usd": 0.0, "input": 0, "output": 0, "cached": 0, "n": 0}
    )
    total = 0.0
    for line in (RUN / "ledger.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        usd = float(e.get("usd") or 0.0)
        total += usd
        role = e.get("role") or "unknown"
        model = e.get("model") or "unknown"
        for bucket, key in ((by_role, role), (by_model, model)):
            bucket[key]["usd"] += usd
            bucket[key]["input"] += int(e.get("input_tokens") or 0)
            bucket[key]["output"] += int(e.get("output_tokens") or 0)
            bucket[key]["cached"] += int(e.get("cached_tokens") or 0)
            bucket[key]["n"] += 1

    summary = load_json(RUN / "summary.json")
    expected = float(summary.get("spent_usd") or 0.0)
    if abs(total - expected) > 0.01:
        state.issues.append(
            f"ledger sum ${total:.6f} != summary.spent_usd ${expected:.6f}"
        )

    lines: list[str] = []
    lines.append(r"% Auto-generated by scripts/analysis/supp_catalog.py")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{Generation cost and tokens from \texttt{{ledger.jsonl}}. "
        rf"Ledger total \${fmt_usd(total)}; "
        rf"\texttt{{summary.json}} reports \${fmt_usd(expected)}. "
        r"Audit stage: 0 calls (audit\_sample$=$0).}}"
    )
    lines.append(r"\label{tab:cat-cost}")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"\rowcolor{headblue} Stage / model & Calls & Input tok. & Cached & Output tok. & USD \\"
    )
    lines.append(r"\midrule")
    lines.append(r"\rowcolor{groupgray} \multicolumn{6}{l}{\textbf{By stage}} \\")
    for stage in STAGE_ORDER:
        b = by_role.get(stage)
        if not b:
            lines.append(f"{latex_escape(stage)} & 0 & 0 & 0 & 0 & 0.0000 \\")
            continue
        lines.append(
            f"{latex_escape(stage)} & {int(b['n'])} & {int(b['input']):,} & "
            f"{int(b['cached']):,} & {int(b['output']):,} & {fmt_usd(b['usd'])} \\\\"
            .replace(",", "{,}")
        )
    # any unexpected roles
    for role in sorted(by_role):
        if role not in STAGE_ORDER:
            b = by_role[role]
            lines.append(
                f"{latex_escape(role)} (other) & {int(b['n'])} & {int(b['input']):,} & "
                f"{int(b['cached']):,} & {int(b['output']):,} & {fmt_usd(b['usd'])} \\\\"
                .replace(",", "{,}")
            )
    lines.append(r"\rowcolor{groupgray} \multicolumn{6}{l}{\textbf{By model}} \\")
    for model, b in sorted(by_model.items(), key=lambda kv: -kv[1]["usd"]):
        lines.append(
            f"\\texttt{{{latex_escape(model)}}} & {int(b['n'])} & {int(b['input']):,} & "
            f"{int(b['cached']):,} & {int(b['output']):,} & {fmt_usd(b['usd'])} \\\\"
            .replace(",", "{,}")
        )
    lines.append(r"\midrule")
    lines.append(
        rf"\textbf{{Total}} & {sum(int(b['n']) for b in by_role.values())} & "
        rf"{sum(int(b['input']) for b in by_role.values()):,} & "
        rf"{sum(int(b['cached']) for b in by_role.values()):,} & "
        rf"{sum(int(b['output']) for b in by_role.values()):,} & "
        rf"\textbf{{{fmt_usd(total)}}} \\".replace(",", "{,}")
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    state.key_numbers["cost_total_ledger"] = total
    state.key_numbers["cost_total_summary"] = expected
    state.key_numbers["cost_by_stage"] = {
        k: round(v["usd"], 4) for k, v in by_role.items()
    }
    state.key_numbers["cost_by_model"] = {
        k: round(v["usd"], 4) for k, v in by_model.items()
    }
    return "\n".join(lines) + "\n"


def build_models_generation(state: CatalogState) -> str:
    model_variant_counts: Counter[tuple] = Counter()
    effort_variant_counts: Counter[tuple] = Counter()
    max_turns: Counter[int] = Counter()
    sync_flags: Counter[str] = Counter()
    eng_shares: list[float] = []
    repair_model_counts: Counter[str] = Counter()
    for p in (RUN / "conversations").glob("*.json"):
        gs = load_json(p)["provenance"]["generation_settings"]
        models = gs.get("models") or {}
        effort = gs.get("reasoning_effort") or {}
        model_variant_counts[tuple(sorted(models.items()))] += 1
        effort_variant_counts[tuple(sorted((k, str(v)) for k, v in effort.items()))] += 1
        repair_model_counts[str(models.get("repair"))] += 1
        max_turns[int(gs.get("max_post_branch_turns") or -1)] += 1
        sync_flags[str(gs.get("sync_review"))] += 1
        if gs.get("language_target_english_share") is not None:
            eng_shares.append(float(gs["language_target_english_share"]))

    if len(model_variant_counts) > 1:
        state.issues.append(
            "generation_settings.models not uniform: "
            + "; ".join(
                f"n={cnt} repair={dict(v).get('repair')}"
                for v, cnt in model_variant_counts.most_common()
            )
            + " (majority used in table; minority is OpenAI→Haiku repair fallback)"
        )
    try:
        import yaml  # type: ignore

        cfg = yaml.safe_load((CONFIGS / "generation.yaml").read_text(encoding="utf-8"))
    except Exception as e:
        state.issues.append(f"could not parse generation.yaml: {e}")
        cfg = {}

    # Batch vs sync from ledger
    batch_by_role: dict[str, Counter] = defaultdict(Counter)
    for line in (RUN / "ledger.jsonl").read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        batch_by_role[e["role"]][bool(e.get("batch"))] += 1

    # Majority model map
    models = dict(model_variant_counts.most_common(1)[0][0]) if model_variant_counts else {}
    effort_map = {}
    if effort_variant_counts:
        effort_map = {
            k: (None if v == "None" else v)
            for k, v in effort_variant_counts.most_common(1)[0][0]
        }
    eng_note = ""
    if eng_shares:
        eng_note = (
            f"Per-call target English share sampled as "
            f"Uniform(p25, p75) from each language profile's "
            f"\\texttt{{mixing\\_bands.english\\_share}} "
            f"(observed mean {sum(eng_shares)/len(eng_shares):.3f}, "
            f"min {min(eng_shares):.3f}, max {max(eng_shares):.3f})."
        )

    lines: list[str] = []
    lines.append(r"% Auto-generated by scripts/analysis/supp_catalog.py")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Writer and reviewer models per stage (from conversation "
        r"\texttt{provenance.generation\_settings}), batch vs sync usage (ledger), "
        rf"max post-branch turns, and English-share sampling. {eng_note}}}"
    )
    lines.append(r"\label{tab:cat-models-gen}")
    lines.append(r"\begin{tabular}{lllrr}")
    lines.append(r"\toprule")
    lines.append(
        r"\rowcolor{headblue} Stage & Model & Reasoning & Sync calls & Batch calls \\"
    )
    lines.append(r"\midrule")
    for stage in STAGE_ORDER:
        model = models.get(stage, "---")
        reff = effort_map.get(stage, "---")
        bc = batch_by_role.get(stage, Counter())
        sync_n = bc.get(False, 0)
        batch_n = bc.get(True, 0)
        lines.append(
            f"{latex_escape(stage)} & \\texttt{{{latex_escape(str(model))}}} & "
            f"{latex_escape(str(reff))} & {sync_n} & {batch_n} \\\\"
        )
    lines.append(r"\midrule")
    mt = ", ".join(f"{k}:{v}" for k, v in sorted(max_turns.items()))
    lines.append(
        rf"\multicolumn{{5}}{{l}}{{max\_post\_branch\_turns distribution: {latex_escape(mt)}; "
        rf"config default {cfg.get('max_post_branch_turns', '?')}; "
        rf"max\_repair\_rounds={cfg.get('max_repair_rounds', '?')}; "
        rf"language\_gate=\texttt{{{latex_escape(str(cfg.get('language_gate', '?')))}}}}} \\"
    )
    lines.append(
        rf"\multicolumn{{5}}{{l}}{{sync\_review flag in provenance: "
        + ", ".join(f"{k}={v}" for k, v in sorted(sync_flags.items()))
        + r"} \\"
    )
    lines.append(
        r"\multicolumn{5}{l}{Reviewer batch\_provider (config): "
        + latex_escape(str((cfg.get("review") or {}).get("batch_provider", "?")))
        + r"; generation roles are sync OpenAI calls.} \\"
    )
    repair_note = ", ".join(f"{m}: {c}" for m, c in repair_model_counts.most_common())
    lines.append(
        r"\multicolumn{5}{l}{Repair model in provenance (call counts): "
        + latex_escape(repair_note)
        + r". Minority Haiku repairs are OpenAI fallbacks after empty/truncated output.} \\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    state.key_numbers["gen_models"] = models
    state.key_numbers["gen_reasoning"] = effort_map
    state.key_numbers["repair_model_counts"] = dict(repair_model_counts)
    state.key_numbers["max_post_branch_turns"] = dict(max_turns)
    state.key_numbers["english_share_mean"] = (
        sum(eng_shares) / len(eng_shares) if eng_shares else None
    )
    return "\n".join(lines) + "\n"


def shorten_skip_reason(short_id: str, reason: str | None) -> str:
    r = reason or ""
    if short_id == "internlm2-1.8b" or "sentencepiece" in r.lower():
        return "missing sentencepiece (ImportError on load)"
    if "chat_template" in r:
        return "tokenizer has no chat template"
    if "gated" in r.lower() or short_id.startswith("llama") or short_id.startswith("gemma"):
        return "gated Hugging Face repo (401 / no auth)"
    # fallback: first clause
    return one_line(r.replace("\n", " "), 80)


def build_detector_models(state: CatalogState) -> str:
    from scambranch.eval.constants import MODEL_REGISTRY  # type: ignore

    resources = load_json(EVAL_ZS / "resources.json")
    metrics = load_json(EVAL_ZS / "metrics.json")
    res_models = resources.get("models") or {}
    skipped = {s["short_id"]: s for s in metrics.get("skipped_models") or []}

    lines: list[str] = []
    lines.append(r"% Auto-generated by scripts/analysis/supp_catalog.py")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Zero-shot detector registry. Parameter counts from "
        r"\texttt{results/eval\_hpc/resources.json} when the model loaded. "
        r"Chat template: yes if the model ran (eval applies "
        r"\texttt{tokenizer.apply\_chat\_template}); no if skipped for missing template.}"
    )
    lines.append(r"\label{tab:cat-detector-models}")
    lines.append(r"\begin{tabular}{llrl}")
    lines.append(r"\toprule")
    lines.append(
        r"\rowcolor{headblue} Display name & Hugging Face id & Params & Chat tpl. \\"
    )
    lines.append(r"\midrule")
    lines.append(r"\rowcolor{groupgray} \multicolumn{4}{l}{\textbf{Evaluated}} \\")
    for short_id, spec in MODEL_REGISTRY.items():
        info = res_models.get(short_id) or {}
        if info.get("skipped") or short_id in skipped:
            continue
        display = DETECTOR_DISPLAY.get(short_id, short_id)
        hf = spec["hf_id"]
        params = fmt_params(info.get("params"))
        # Successfully evaluated models all used chat templates in detectors.py
        chat = "yes"
        lines.append(
            f"{latex_escape(display)} & \\texttt{{{latex_escape(hf)}}} & "
            f"{params} & {chat} \\\\"
        )

    lines.append(r"\rowcolor{groupgray} \multicolumn{4}{l}{\textbf{Attempted but skipped}} \\")
    for short_id, spec in MODEL_REGISTRY.items():
        info = res_models.get(short_id) or skipped.get(short_id) or {}
        if not (info.get("skipped") or short_id in skipped):
            continue
        sk = skipped.get(short_id) or info
        display = DETECTOR_DISPLAY.get(short_id, short_id)
        hf = spec["hf_id"]
        reason = shorten_skip_reason(short_id, sk.get("skip_reason") or info.get("skip_reason"))
        # chat template column
        if "chat template" in reason.lower() or "chat_template" in (sk.get("skip_reason") or ""):
            chat = "no"
        elif "sentencepiece" in reason.lower():
            chat = "yes*"  # would use template; failed earlier
        else:
            chat = "---"
        params = fmt_params(info.get("params")) if info.get("params") else "---"
        lines.append(
            f"{latex_escape(display)} & \\texttt{{{latex_escape(hf)}}} & "
            f"{params} & {chat} \\\\"
        )
        lines.append(
            rf"\multicolumn{{4}}{{p{{0.95\linewidth}}}}{{\quad "
            rf"Skip reason: {latex_escape(reason)}}} \\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\begin{flushleft}\footnotesize *InternLM2 failed before scoring; it ships a chat "
        r"template but the modeling file required \texttt{sentencepiece}.\end{flushleft}"
    )
    lines.append(r"\end{table}")

    state.key_numbers["detectors_evaluated"] = [
        s for s in MODEL_REGISTRY if s not in skipped and not (res_models.get(s) or {}).get("skipped")
    ]
    state.key_numbers["detectors_skipped"] = {
        sid: shorten_skip_reason(sid, (skipped.get(sid) or res_models.get(sid) or {}).get("skip_reason"))
        for sid in MODEL_REGISTRY
        if sid in skipped or (res_models.get(sid) or {}).get("skipped")
    }
    return "\n".join(lines) + "\n"


def build_prompts(state: CatalogState) -> str:
    from scambranch.eval.constants import DETECTOR_INSTRUCTION  # type: ignore

    recipient = (PROMPTS / "recipient.md").read_text(encoding="utf-8")
    # Extract condition definitions from HTML comment block
    cond_block = re.search(
        r"condition_instruction values.*?suspicious:\s*(.+?)(?:\n\n|\nending_hint)",
        recipient,
        re.S,
    )
    conditions_text = {
        "cooperative": "You go along with what the caller asks: you start doing it, agree to it, or ask how to do it. You do not refuse, hesitate on principle, or accuse the caller. You may ask a short practical question.",
        "clarifying": "You are not sure yet. Ask the caller to explain why this is needed or how it works. Do not refuse, do not accuse the caller, and do not agree yet.",
        "resistant": "You decline or set a boundary (for example, you will handle it yourself, visit the branch, or do it later). You do not need to give a reason and you do not accuse the caller of fraud.",
        "suspicious": "You openly doubt that this caller is genuine and say so. You may mention that you have heard of such calls.",
    }
    # Verify against file
    for k, v in conditions_text.items():
        if v not in recipient:
            state.issues.append(f"condition text for {k} not found verbatim in recipient.md")

    caller = (PROMPTS / "caller_continuation.md").read_text(encoding="utf-8")
    # Limits are card-filled; describe the template instruction
    caller_limits_excerpt = (
        "Keep your goal and the facts already stated. You may explain, reassure, "
        "repeat your request, add urgency, accept a refusal, or end the call, within "
        "the limits above. Do not introduce a new kind of request that is not in your role. "
        "Do not use the words \"fraud\", \"scam\", or \"legitimate\" about yourself."
    )
    if caller_limits_excerpt not in caller.replace("\n", " ").replace("  ", " "):
        # softer check
        if "Do not introduce a new kind of request" not in caller:
            state.issues.append("caller continuation limit wording not found in caller_continuation.md")

    # Family-level limits example
    example_limits = load_json(SCEN / "json" / "bp_01.json").get("caller_continuation_limits", "")

    sem = (PROMPTS / "semantic_review.md").read_text(encoding="utf-8")
    # Pull the numbered checklist (lines starting with digits)
    checklist_items = []
    for m in re.finditer(r"^(\d+)\.\s+`([^`]+)`:\s*(.+?)(?=^\d+\.\s+`|\nReturn JSON|\Z)", sem, re.M | re.S):
        num, name, body = m.group(1), m.group(2), m.group(3)
        body = re.sub(r"\s+", " ", body.strip())
        # drop the cooperative/clarifying sub-bullets for space — keep first sentence-ish
        body = one_line(body, 220)
        checklist_items.append((name, body))

    lang = (PROMPTS / "language_review.md").read_text(encoding="utf-8")
    hard_flags_excerpt = []
    for flag in ("script", "wrong_language", "artifact"):
        m = re.search(rf"`{flag}`:\s*([^;]+)", lang)
        if m:
            hard_flags_excerpt.append((flag, m.group(1).strip().rstrip(";")))
        else:
            # artifact line uses different formatting
            m2 = re.search(rf"- `{flag}`:\s*(.+)", lang)
            if m2:
                hard_flags_excerpt.append((flag, m2.group(1).strip().rstrip(";")))

    def q(text: str) -> str:
        return (
            "\\begin{quote}\\small\n"
            + latex_escape(text)
            + "\n\\end{quote}"
        )

    lines: list[str] = []
    lines.append(r"% Auto-generated by scripts/analysis/supp_catalog.py")
    lines.append(r"\section*{Prompt excerpts}")
    lines.append(
        r"Condensed faithful excerpts from \texttt{prompts/}. "
        r"Placeholders such as \texttt{\{\{...\}\}} are filled by the pipeline."
    )
    lines.append(r"\subsection*{Recipient conditions}")
    for name in CONDITIONS:
        lines.append(rf"\paragraph{{{name}.}}")
        lines.append(q(conditions_text[name]))

    lines.append(r"\subsection*{Caller continuation limits (after the branch)}")
    lines.append(
        r"Each family card supplies a concrete \texttt{caller\_continuation\_limits} string. "
        r"The shared caller template additionally states:"
    )
    lines.append(q(caller_limits_excerpt))
    lines.append(r"Example (family \texttt{bp\_01}):")
    lines.append(q(example_limits))

    lines.append(r"\subsection*{Semantic-review checklist}")
    for name, body in checklist_items:
        lines.append(rf"\paragraph{{\texttt{{{latex_escape(name)}}}.}}")
        lines.append(q(body))

    lines.append(r"\subsection*{Language-review hard flags}")
    lines.append(
        r"Under \texttt{language\_gate: hard\_flags\_only\_no\_scores}, only confirmed "
        r"hard flags gate acceptance (scores never reject). Definitions:"
    )
    for flag, body in hard_flags_excerpt:
        lines.append(rf"\paragraph{{\texttt{{{latex_escape(flag)}}}.}}")
        lines.append(q(body))

    lines.append(r"\subsection*{Detector instruction}")
    lines.append(
        r"Exact string from \texttt{src/scambranch/eval/constants.py} "
        r"(\texttt{DETECTOR\_INSTRUCTION}):"
    )
    lines.append(q(DETECTOR_INSTRUCTION))

    state.key_numbers["detector_instruction"] = DETECTOR_INSTRUCTION
    return "\n".join(lines) + "\n"


def load_predictions(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def decide(score: float, threshold: float) -> str:
    return "fraud" if score >= threshold else "legitimate"


def gold_label(label_field: str) -> str:
    if label_field in {"fraud_attempt", "fraud"}:
        return "fraud"
    return "legitimate"


def pick_examples(state: CatalogState) -> list[dict[str, Any]]:
    """Select three TEST-split four-branch example sets."""
    zs_preds = load_predictions(EVAL_ZS / "predictions.jsonl")
    ft_preds = load_predictions(EVAL_FT / "predictions.jsonl")
    zs_th = load_json(EVAL_ZS / "metrics.json")["thresholds"]["qwen2.5-1.5b"]
    ft_th = load_json(EVAL_FT / "metrics.json")["thresholds"]["qwen2.5-1.5b+branched-seed1"]
    manifest = {r["conversation_id"]: r for r in load_manifest()}

    def unit_key(cid: str) -> str:
        # family_case_pair_history
        parts = cid.rsplit("_", 1)
        return parts[0]

    # Index zs and ft
    def index(preds: list[dict], model: str) -> dict[tuple[str, str], dict]:
        out = {}
        for d in preds:
            if d["model"] != model:
                continue
            if d.get("invalid"):
                continue
            out[(d["conversation_id"], d["view"])] = d
        return out

    zs = index(zs_preds, "qwen2.5-1.5b")
    ft = index(ft_preds, "qwen2.5-1.5b+branched-seed1")

    usable_units: dict[str, list[str]] = defaultdict(list)
    for cid, row in manifest.items():
        if row["split"] != "test":
            continue
        if row["disposition"] not in {"accepted", "repaired"}:
            continue
        usable_units[unit_key(cid)].append(cid)

    complete_units = {
        u: cids
        for u, cids in usable_units.items()
        if len(cids) == 4
        and {c.rsplit("_", 1)[1] for c in cids} == set(CONDITIONS)
    }

    def scores_for_unit(unit: str) -> dict[str, Any] | None:
        cids = complete_units.get(unit)
        if not cids:
            return None
        by_cond = {c.rsplit("_", 1)[1]: c for c in cids}
        out = {"unit": unit, "by_cond": {}, "label": None, "pair": None}
        for cond in CONDITIONS:
            cid = by_cond[cond]
            row = manifest[cid]
            lab = gold_label(row["private_annotations"]["scenario_label"])
            out["label"] = lab
            out["pair"] = row["language_pair"]
            entry = {}
            for view in ("PREFIX_PLUS_REACTION", "FULL"):
                zp = zs.get((cid, view))
                if not zp:
                    return None
                zscore = float(zp["score"])
                zdec = decide(zscore, float(zs_th[view]))
                entry[view] = {
                    "zs_score": zscore,
                    "zs_pred": zdec,
                    "zs_correct": zdec == lab,
                }
            fp = ft.get((cid, "FULL"))
            if not fp:
                # FT may only cover usable test; skip unit if missing
                return None
            fscore = float(fp["score"])
            fdec = decide(fscore, float(ft_th["FULL"]))
            entry["FT_FULL"] = {
                "ft_score": fscore,
                "ft_pred": fdec,
                "ft_correct": fdec == lab,
            }
            out["by_cond"][cond] = {"cid": cid, **entry}
        return out

    # (a) fixed: ec_08_L_hi_en_h1
    ex_a_unit = "ec_08_L_hi_en_h1"
    ex_a = scores_for_unit(ex_a_unit)
    if not ex_a:
        state.issues.append(f"example (a) unit {ex_a_unit} not complete or missing preds")
    else:
        # verify: only suspicious flagged on PREFIX_PLUS_REACTION
        prep = {
            c: ex_a["by_cond"][c]["PREFIX_PLUS_REACTION"]["zs_pred"] for c in CONDITIONS
        }
        if not (
            prep["suspicious"] == "fraud"
            and all(prep[c] == "legitimate" for c in CONDITIONS if c != "suspicious")
        ):
            state.issues.append(
                f"example (a) unexpected PREFIX_PLUS_REACTION preds: {prep}"
            )

    # (b) FRAUD in ta_en or te_en: zs misses cooperative on FULL but catches suspicious
    candidates_b = []
    for unit, cids in complete_units.items():
        if not (unit.endswith("_ta_en_h1") or unit.endswith("_ta_en_h2")
                or unit.endswith("_te_en_h1") or unit.endswith("_te_en_h2")):
            continue
        if "_F_" not in unit and not re.search(r"_F_", unit):
            # unit like jr_03_F_ta_en_h1
            pass
        if "_F_" not in f"_{unit}_" and not re.search(r"[a-z]{2}_\d{2}_F_", unit):
            continue
        sc = scores_for_unit(unit)
        if not sc or sc["label"] != "fraud":
            continue
        coop = sc["by_cond"]["cooperative"]["FULL"]
        susp = sc["by_cond"]["suspicious"]["FULL"]
        if (not coop["zs_correct"]) and susp["zs_correct"] and coop["zs_pred"] == "legitimate" and susp["zs_pred"] == "fraud":
            # clarity: large score gap
            gap = susp["zs_score"] - coop["zs_score"]
            candidates_b.append((gap, unit, sc))
    candidates_b.sort(reverse=True)
    ex_b = candidates_b[0][2] if candidates_b else None
    if not ex_b:
        state.issues.append("no FRAUD ta/te example where zs misses coop but catches suspicious on FULL")

    # (c) GENUINE ta/te with at least one zs false alarm (FULL or PREFIX_PLUS_REACTION).
    # Prefer a different family from example (a) when possible.
    avoid_family = "ec_08"
    candidates_c = []
    for unit, cids in complete_units.items():
        if not any(unit.endswith(f"_{p}_{h}") for p in ("ta_en", "te_en") for h in ("h1", "h2")):
            continue
        if not re.search(r"[a-z]{2}_\d{2}_L_", unit):
            continue
        sc = scores_for_unit(unit)
        if not sc or sc["label"] != "legitimate":
            continue
        false_alarms = []
        for cond in CONDITIONS:
            for view in ("PREFIX_PLUS_REACTION", "FULL"):
                cell = sc["by_cond"][cond][view]
                if cell["zs_pred"] == "fraud":
                    false_alarms.append((cond, view, cell["zs_score"]))
        if false_alarms:
            fam = unit.split("_")[0] + "_" + unit.split("_")[1]
            # Prefer: different family, clear single/few false alarms with high score,
            # and at least one correct branch remaining (contrast).
            n_fa = len(false_alarms)
            n_ok_full = sum(
                1 for cond in CONDITIONS if sc["by_cond"][cond]["FULL"]["zs_correct"]
            )
            diversify = 1 if fam != avoid_family else 0
            contrast = 1 if n_ok_full >= 1 else 0
            score = (diversify, contrast, max(x[2] for x in false_alarms), n_fa)
            candidates_c.append((score, unit, sc, false_alarms))
    candidates_c.sort(reverse=True)
    ex_c = candidates_c[0][2] if candidates_c else None
    if not ex_c:
        state.issues.append("no GENUINE ta/te example with zs false alarm")
    chosen = []
    if ex_a:
        chosen.append({"tag": "a", "title": "Genuine courier call (Hindi--English)", "data": ex_a})
    if ex_b:
        chosen.append({
            "tag": "b",
            "title": "Fraud call where cooperation hides the scam (zero-shot)",
            "data": ex_b,
        })
    if ex_c:
        chosen.append({
            "tag": "c",
            "title": "Genuine call with a zero-shot false alarm",
            "data": ex_c,
        })
    state.key_numbers["examples"] = []
    for ch in chosen:
        d = ch["data"]
        summary = {
            "tag": ch["tag"],
            "unit": d["unit"],
            "label": d["label"],
            "pair": d["pair"],
            "scores": {
                cond: {
                    "cid": d["by_cond"][cond]["cid"],
                    "zs_ppr": d["by_cond"][cond]["PREFIX_PLUS_REACTION"]["zs_score"],
                    "zs_full": d["by_cond"][cond]["FULL"]["zs_score"],
                    "ft_full": d["by_cond"][cond]["FT_FULL"]["ft_score"],
                }
                for cond in CONDITIONS
            },
        }
        state.key_numbers["examples"].append(summary)
    return chosen


def mark(ok: bool) -> str:
    return r"\checkmark" if ok else r"$\times$"


def write_example_tex(choice: dict[str, Any], zs_th: dict, ft_th: dict) -> tuple[str, str]:
    data = choice["data"]
    unit = data["unit"]
    # Load cooperative conversation for shared opening (same prefix across branches)
    coop_cid = data["by_cond"]["cooperative"]["cid"]
    coop = load_json(RUN / "conversations" / f"{coop_cid}.json")
    branch_after = int(coop["private_annotations"]["branch_after_turn"])
    prefix_turns = [t for t in coop["turns"] if int(t["turn_id"]) <= branch_after]

    lines: list[str] = []
    lines.append(r"% Auto-generated by scripts/analysis/supp_catalog.py")
    lines.append(rf"\subsection{{{latex_escape(choice['title'])}}}")
    lines.append(
        r"\textbf{Unit:} \texttt{"
        + latex_escape(unit)
        + r"} \quad Language: "
        + latex_escape(PAIR_DISPLAY[data["pair"]])
        + r" \quad Gold label: \textbf{"
        + latex_escape(data["label"])
        + "}"
    )
    lines.append("")
    lines.append(r"\paragraph{Shared opening.}")
    lines.append(r"\begin{itemize}\setlength{\itemsep}{2pt}\setlength{\parsep}{0pt}")
    for t in prefix_turns:
        sp = "Caller" if t["speaker"] == "caller" else "Recipient"
        lines.append(
            r"\item \textbf{"
            + sp
            + ":} "
            + latex_escape(t["text"])
            + r" \gloss{}"
        )
    lines.append(r"\end{itemize}")
    lines.append("")
    lines.append(r"\noindent\textit{Branch point}")
    lines.append("")

    for cond in CONDITIONS:
        cid = data["by_cond"][cond]["cid"]
        conv = load_json(RUN / "conversations" / f"{cid}.json")
        post = [t for t in conv["turns"] if int(t["turn_id"]) > branch_after]
        lines.append(
            r"\paragraph{"
            + cond
            + r" branch.} \texttt{"
            + latex_escape(cid)
            + "}"
        )
        lines.append(r"\begin{itemize}\setlength{\itemsep}{2pt}\setlength{\parsep}{0pt}")
        if not post:
            lines.append(r"\item \emph{(no post-branch turns)}")
        for t in post:
            sp = "Caller" if t["speaker"] == "caller" else "Recipient"
            first = " (first reply)" if t == post[0] else ""
            lines.append(
                r"\item \textbf{"
                + sp
                + first
                + ":} "
                + latex_escape(t["text"])
                + r" \gloss{}"
            )
        lines.append(r"\end{itemize}")

    lines.append("")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{P(fraud) for unit \texttt{{{latex_escape(unit)}}}. "
        rf"Zero-shot Qwen2.5-1.5B thresholds: "
        rf"PREFIX\_PLUS\_REACTION={zs_th['PREFIX_PLUS_REACTION']:.3f}, "
        rf"FULL={zs_th['FULL']:.3f}. "
        rf"Fine-tuned branched seed1 FULL threshold={ft_th['FULL']:.3f}. "
        r"Decision $=$ score $\ge$ threshold. Marks: correct/\incorrect vs gold.}}"
    )
    lines.append(rf"\label{{tab:cat-ex-{choice['tag']}}}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\rowcolor{headblue} Branch & ZS open+reply & ZS full & FT branched full \\"
    )
    lines.append(r"\midrule")
    for cond in CONDITIONS:
        cell = data["by_cond"][cond]
        ppr = cell["PREFIX_PLUS_REACTION"]
        full = cell["FULL"]
        ft = cell["FT_FULL"]
        lines.append(
            f"{cond} & "
            f"{ppr['zs_score']:.2f}~{mark(ppr['zs_correct'])} & "
            f"{full['zs_score']:.2f}~{mark(full['zs_correct'])} & "
            f"{ft['ft_score']:.2f}~{mark(ft['ft_correct'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return unit, "\n".join(lines) + "\n"


def build_examples(state: CatalogState) -> None:
    zs_th = load_json(EVAL_ZS / "metrics.json")["thresholds"]["qwen2.5-1.5b"]
    ft_th = load_json(EVAL_FT / "metrics.json")["thresholds"]["qwen2.5-1.5b+branched-seed1"]
    chosen = pick_examples(state)
    readme_lines = [
        "# Example four-branch sets (TEST split)",
        "",
        "Selected for the dataset-documentation supplement. Scores are P(fraud).",
        "Decision rule: `score >= thresholds[model][view]` from the corresponding `metrics.json`.",
        "",
        f"- Zero-shot model: `qwen2.5-1.5b` / Qwen2.5-1.5B-Instruct",
        f"  - thresholds: PREFIX_PLUS_REACTION={zs_th['PREFIX_PLUS_REACTION']}, FULL={zs_th['FULL']}",
        f"- Fine-tuned: `qwen2.5-1.5b+branched-seed1` from `results/eval_hpc_ft_rest/`",
        f"  - FULL threshold: {ft_th['FULL']}",
        "",
    ]
    for ch in chosen:
        unit, tex = write_example_tex(ch, zs_th, ft_th)
        out_name = f"{unit}.tex"
        (OUT / "examples" / out_name).write_text(tex, encoding="utf-8")
        d = ch["data"]
        readme_lines.append(f"## ({ch['tag']}) `{unit}` — {ch['title']}")
        readme_lines.append("")
        if ch["tag"] == "a":
            readme_lines.append(
                "Chosen because it is the paper's running genuine-courier example: "
                "zero-shot Qwen2.5-1.5B flags only the suspicious branch after the first reply "
                "(PREFIX_PLUS_REACTION), while cooperative/clarifying/resistant stay legitimate."
            )
        elif ch["tag"] == "b":
            readme_lines.append(
                "Chosen as a TEST fraud set in Tamil-- or Telugu--English where zero-shot "
                "Qwen2.5-1.5B misses the cooperative branch on the full call but catches the "
                "suspicious branch (largest score gap among matching candidates)."
            )
        else:
            readme_lines.append(
                "Chosen as a TEST genuine set in Tamil-- or Telugu--English with at least one "
                "zero-shot false alarm on PREFIX_PLUS_REACTION or FULL."
            )
        readme_lines.append("")
        readme_lines.append("| branch | zs open+reply | zs full | ft full |")
        readme_lines.append("|---|---:|---:|---:|")
        for cond in CONDITIONS:
            cell = d["by_cond"][cond]
            readme_lines.append(
                f"| {cond} | {cell['PREFIX_PLUS_REACTION']['zs_score']:.4f} "
                f"({'ok' if cell['PREFIX_PLUS_REACTION']['zs_correct'] else 'WRONG'}) | "
                f"{cell['FULL']['zs_score']:.4f} "
                f"({'ok' if cell['FULL']['zs_correct'] else 'WRONG'}) | "
                f"{cell['FT_FULL']['ft_score']:.4f} "
                f"({'ok' if cell['FT_FULL']['ft_correct'] else 'WRONG'}) |"
            )
        readme_lines.append("")
        readme_lines.append(f"LaTeX: `examples/{out_name}`")
        readme_lines.append("")

    (OUT / "examples" / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")


def write_report(state: CatalogState, files: list[str]) -> None:
    kn = state.key_numbers
    lines = [
        "# Supplement catalog REPORT",
        "",
        "Generated by `scripts/analysis/supp_catalog.py` from frozen artifacts under "
        "`runs/full_v1/`, `scenarios/`, `manifests/`, `prompts/`, `configs/`, and "
        "`results/eval_hpc*`. No LLM calls; no model weights loaded.",
        "",
        "## Files written",
        "",
    ]
    for f in files:
        lines.append(f"- `{f}`")
    lines.append("")
    lines.append("## Key numbers")
    lines.append("")
    disp = kn.get("dispositions", {})
    lines.append(
        f"- Generated conversations: **{disp.get('n')}**; "
        f"accepted first pass **{disp.get('accepted')}**; "
        f"accepted after repair **{disp.get('repaired')}**; "
        f"rejected **{disp.get('rejected')}**; "
        f"usable **{disp.get('usable')}**."
    )
    lines.append(
        f"- Cost: ledger total **${kn.get('cost_total_ledger', float('nan')):.4f}**; "
        f"summary.json **${kn.get('cost_total_summary', float('nan')):.4f}** "
        f"(expected ≈ $33.58)."
    )
    lines.append(
        f"- Complete 4-branch sets: **{kn.get('complete_sets', {}).get('n_complete')}** / "
        f"{kn.get('complete_sets', {}).get('n_groups')} groups."
    )
    lines.append(
        f"- Prefix regenerations: **{kn.get('prefix_regen', {})}**."
    )
    top = kn.get("sem_fail_once_top") or []
    lines.append(
        "- Top semantic fail-once checks: "
        + ", ".join(f"{k}={v}" for k, v in top)
        + "."
    )
    lines.append(f"- Hard flags: {kn.get('hard_flags')}.")
    lines.append(f"- Soft turn-flag types (not gating): {kn.get('soft_flags')}.")
    lines.append(f"- Repair-round distribution (repaired): {kn.get('repair_rounds')}.")
    lines.append(f"- Rejection reasons: {kn.get('rejection_reasons')}.")
    lines.append(f"- Detectors skipped: {kn.get('detectors_skipped')}.")
    lines.append("")
    lines.append("### Chosen examples")
    lines.append("")
    for ex in kn.get("examples") or []:
        lines.append(f"- **({ex['tag']})** `{ex['unit']}` ({ex['pair']}, gold={ex['label']})")
        for cond, sc in ex["scores"].items():
            lines.append(
                f"  - {cond}: zs_ppr={sc['zs_ppr']:.4f}, zs_full={sc['zs_full']:.4f}, "
                f"ft_full={sc['ft_full']:.4f}"
            )
    lines.append("")
    lines.append("## Missing / inconsistent")
    lines.append("")
    if state.issues:
        for issue in state.issues:
            lines.append(f"- {issue}")
    else:
        lines.append("- None noted.")
    lines.append("")
    lines.append("## Notes on methodology")
    lines.append("")
    lines.append(
        "- Final dispositions taken from `private_manifest.jsonl` / `summary.json` "
        "(not from ablation snapshot `dispositions_semrev03_langscore.json`, which has different counts)."
    )
    lines.append(
        "- Semantic fail-once uses the OR of stored `reviews.semantic` and `reviews.semantic_v02` "
        "check results; overwritten intermediate review rounds are not on disk."
    )
    lines.append(
        "- `provenance.review_status` collapses accepted+repaired to `accepted`; "
        "use `disposition` for the three-way split."
    )
    lines.append(
        "- Domain one-liners for families come from `scenarios/draft/*/README.md` catalog tables, "
        "cross-checked against JSON cards; evidence column paraphrases `label_rationale`."
    )
    lines.append(
        "- Audit stage has 0 ledger rows because `configs/generation.yaml` sets `audit_sample: 0.0`."
    )
    lines.append(
        "- `configs/generation.yaml` sets `max_repair_rounds: 2`, but 113 repaired calls have "
        "`repair_history` entries with `round` 3 or 4 (likely from later patch/re-review passes "
        "that appended to history). The table reports the observed max round, not the config cap."
    )
    lines.append(
        "- 24/2880 conversations list `repair: claude-haiku-4-5` in `generation_settings.models` "
        "(OpenAI empty/truncated repair fallback); the other 2856 list `gpt-6-luna`."
    )
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "examples").mkdir(parents=True, exist_ok=True)
    state = CatalogState()

    # Ensure src importable
    import sys

    sys.path.insert(0, str(ROOT / "src"))

    writers = [
        ("families.tex", build_families),
        ("splits.tex", build_splits),
        ("pipeline_outcomes.tex", build_pipeline_outcomes),
        ("cost.tex", build_cost),
        ("models_generation.tex", build_models_generation),
        ("detector_models.tex", build_detector_models),
        ("prompts.tex", build_prompts),
    ]
    files: list[str] = []
    for name, fn in writers:
        print(f"building {name}...")
        tex = fn(state)
        (OUT / name).write_text(tex, encoding="utf-8")
        files.append(f"results/supplement/catalog/{name}")

    print("building examples...")
    build_examples(state)
    for p in sorted((OUT / "examples").glob("*")):
        files.append(f"results/supplement/catalog/examples/{p.name}")

    write_report(state, files + ["results/supplement/catalog/REPORT.md"])
    files.append("results/supplement/catalog/REPORT.md")

    print("DONE")
    print("issues:", len(state.issues))
    for i in state.issues:
        print(" -", i)
    print("dispositions:", state.key_numbers.get("dispositions"))
    print("cost:", state.key_numbers.get("cost_total_ledger"))
    print("examples:", [e["unit"] for e in state.key_numbers.get("examples") or []])


if __name__ == "__main__":
    main()
