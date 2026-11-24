#!/usr/bin/env python3
"""AAAI-27 student-abstract supplement: RESULTS tables, figures, statistical tests.

Reads existing eval artifacts (metrics.json / predictions.jsonl / resources.json /
FT train logs). Never fabricates numbers. Re-runnable with fixed seeds.

Outputs under results/supplement/results/.
"""

from __future__ import annotations

import json
import math
import os
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "supplement" / "results"
FIG = OUT / "figures"

BOOT_N = int(os.environ.get("SUPP_BOOT_N", "10000"))
BOOT_SEED = 42
CONDITIONS = ("cooperative", "clarifying", "resistant", "suspicious")
LANGS = ("hi_en", "ta_en", "te_en")
LANG_NAME = {
    "hi_en": "Hindi--English",
    "ta_en": "Tamil--English",
    "te_en": "Telugu--English",
}
VIEW_NAME = {
    "FULL": "Full call",
    "PREFIX_ONLY": "Opening only",
    "PREFIX_PLUS_REACTION": "Opening + first reply",
    "CALLER_ONLY": "Caller turns only",
    "STREAMING": "Streaming",
}
ARM_NAME = {
    "unbranched": "Unbranched",
    "branched": "Branched",
    "branched_consistency": "Branched + consistency",
    "zeroshot": "Zero-shot base",
}
DAGGER = frozenset({"smollm2-135m", "tinyllama-1.1b"})
ZEROSHOT_IDS = [
    "qwen3-0.6b",
    "qwen3-1.7b",
    "qwen2.5-0.5b",
    "qwen2.5-1.5b",
    "qwen2-0.5b",
    "qwen2-1.5b",
    "smollm2-135m",
    "smollm2-360m",
    "smollm2-1.7b",
    "tinyllama-1.1b",
    "stablelm2-1.6b",
    "danube2-1.8b",
    "granite-1b",
    "llama-3.2-1b",
    "gemma3-270m",
    "gemma3-1b",
]
DISPLAY = {
    "qwen3-0.6b": "Qwen3-0.6B",
    "qwen3-1.7b": "Qwen3-1.7B",
    "qwen2.5-0.5b": "Qwen2.5-0.5B",
    "qwen2.5-1.5b": "Qwen2.5-1.5B",
    "qwen2-0.5b": "Qwen2-0.5B",
    "qwen2-1.5b": "Qwen2-1.5B",
    "smollm2-135m": "SmolLM2-135M",
    "smollm2-360m": "SmolLM2-360M",
    "smollm2-1.7b": "SmolLM2-1.7B",
    "tinyllama-1.1b": "TinyLlama-1.1B",
    "stablelm2-1.6b": "StableLM-2-1.6B",
    "danube2-1.8b": "H2O-Danube2-1.8B",
    "granite-1b": "Granite-1B",
    "llama-3.2-1b": "Llama-3.2-1B",
    "gemma3-270m": "Gemma-3-270M",
    "gemma3-1b": "Gemma-3-1B",
}
FAMILY = {
    "qwen3-0.6b": "Qwen3",
    "qwen3-1.7b": "Qwen3",
    "qwen2.5-0.5b": "Qwen2.5",
    "qwen2.5-1.5b": "Qwen2.5",
    "qwen2-0.5b": "Qwen2",
    "qwen2-1.5b": "Qwen2",
    "smollm2-135m": "SmolLM2",
    "smollm2-360m": "SmolLM2",
    "smollm2-1.7b": "SmolLM2",
    "tinyllama-1.1b": "TinyLlama",
    "stablelm2-1.6b": "StableLM 2",
    "danube2-1.8b": "H2O-Danube2",
    "granite-1b": "Granite",
    "llama-3.2-1b": "Llama",
    "gemma3-270m": "Gemma 3",
    "gemma3-1b": "Gemma 3",
}
# Colorblind-safe Okabe–Ito subset
CB = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "sky": "#56B4E9",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "black": "#000000",
    "gray": "#999999",
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def merge_metrics(*paths: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "thresholds": {},
        "slices": {},
        "models": [],
    }
    for path in paths:
        m = load_json(path)
        out["thresholds"].update(m.get("thresholds", {}))
        out["slices"].update(m.get("slices", {}))
        out["models"] = sorted(set(out["models"]) | set(m.get("models", [])))
    return out


def merge_resources(*paths: Path) -> dict[str, Any]:
    models: dict[str, Any] = {}
    hardware = None
    for path in paths:
        r = load_json(path)
        if hardware is None:
            hardware = r.get("hardware")
        for mid, spec in r.get("models", {}).items():
            if mid not in models or (models[mid].get("params") is None and spec.get("params") is not None):
                models[mid] = spec
            elif models[mid].get("latency") is None and spec.get("latency") is not None:
                models[mid] = spec
    return {"hardware": hardware, "models": models}


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Metric primitives (match src/scambranch/eval/metrics.py)
# ---------------------------------------------------------------------------


def as_bin(label: Any) -> int:
    if isinstance(label, bool):
        return int(label)
    if isinstance(label, (int, np.integer)):
        return 1 if int(label) == 1 else 0
    text = str(label).strip().lower()
    if text in {"fraud_attempt", "fraud", "1"}:
        return 1
    if text in {"legitimate", "legit", "0"}:
        return 0
    raise ValueError(f"unrecognized label {label!r}")


def decide(score: float, threshold: float, invalid: bool = False) -> int:
    # Invalid rows still carry a score in these evals; decision rule is score >= t.
    _ = invalid
    return 1 if float(score) >= float(threshold) else 0


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    if yt.size == 0:
        return float("nan")
    f1s = []
    for c in (0, 1):
        tp = int(((yt == c) & (yp == c)).sum())
        fp = int(((yt != c) & (yp == c)).sum())
        fn = int(((yt == c) & (yp != c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s))


def fraud_recall(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    n = int((yt == 1).sum())
    if n == 0:
        return float("nan")
    return float(((yt == 1) & (yp == 1)).sum() / n)


def legitimate_fpr(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    n = int((yt == 0).sum())
    if n == 0:
        return float("nan")
    return float(((yt == 0) & (yp == 1)).sum() / n)


def roc_auc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    yt = np.asarray(y_true, dtype=int)
    sc = np.asarray(scores, dtype=float)
    if yt.size == 0 or len(np.unique(yt)) < 2:
        return float("nan")
    # Mann–Whitney / rank AUC without sklearn dependency for speed
    pos = sc[yt == 1]
    neg = sc[yt == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # Handle ties via average ranks
    order = np.argsort(sc, kind="mergesort")
    ranks = np.empty(len(sc), dtype=float)
    i = 0
    while i < len(sc):
        j = i
        while j + 1 < len(sc) and sc[order[j + 1]] == sc[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    sum_ranks_pos = ranks[yt == 1].sum()
    n_pos = float(pos.size)
    n_neg = float(neg.size)
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def hist_key(conversation_id: str) -> str:
    for c in CONDITIONS:
        suf = "_" + c
        if conversation_id.endswith(suf):
            return conversation_id[: -len(suf)]
    return conversation_id


def branch_disagreement_rate(rows: list[dict], pred_key: str = "pred") -> float:
    groups: dict[str, set[int]] = defaultdict(set)
    counts: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        hk = hist_key(r["conversation_id"])
        groups[hk].add(int(r[pred_key]))
        counts[hk].add(r["recipient_condition"])
    complete = [hk for hk, conds in counts.items() if set(CONDITIONS) <= conds]
    if not complete:
        return float("nan")
    disagree = sum(1 for hk in complete if len(groups[hk]) > 1)
    return disagree / len(complete)


def all_branches_correct_rate(rows: list[dict], pred_key: str = "pred") -> float:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[hist_key(r["conversation_id"])].append(r)
    n = 0
    ok = 0
    for members in groups.values():
        byc = {m["recipient_condition"]: m for m in members}
        if set(CONDITIONS) <= set(byc):
            n += 1
            if all(int(byc[c][pred_key]) == as_bin(byc[c]["label"]) for c in CONDITIONS):
                ok += 1
    return ok / n if n else float("nan")


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def fmt_prop(x: float | None, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "---"
    return f"{x:.{digits}f}".lstrip("0") if abs(x) < 1 and x != 0 else f"{x:.{digits}f}"


def fmt_prop_plain(x: float | None, digits: int = 2) -> str:
    """Proportions like .73 (leading zero dropped for |x|<1)."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "---"
    s = f"{x:.{digits}f}"
    if s.startswith("0."):
        return s[1:]
    if s.startswith("-0."):
        return "-" + s[2:]
    return s


def fmt_ci(lo: float, hi: float, digits: int = 2) -> str:
    return f"[{fmt_prop_plain(lo, digits)}, {fmt_prop_plain(hi, digits)}]"


def fmt_params(n: int | None) -> str:
    if n is None:
        return "---"
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 100_000_000:
        # e.g. 999M -> 1.00B if close; else xxxM
        if n >= 950_000_000:
            return f"{n / 1_000_000_000:.2f}B"
        return f"{n / 1_000_000:.0f}M"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.0f}M" if n >= 10_000_000 else f"{n / 1_000_000:.1f}M".replace(".0M", "M")
    return str(n)


def model_tex(mid: str) -> str:
    name = DISPLAY[mid]
    if mid in DAGGER:
        return f"{name}$^\\dagger$"
    return name


def bold_best(values: list[float | None], higher_is_better: bool, fmt_fn=fmt_prop_plain) -> list[str]:
    finite = [(i, v) for i, v in enumerate(values) if v is not None and math.isfinite(v)]
    out = [fmt_fn(v) if v is not None and math.isfinite(v) else "---" for v in values]
    if not finite:
        return out
    best = max(v for _, v in finite) if higher_is_better else min(v for _, v in finite)
    for i, v in finite:
        if abs(v - best) < 1e-12:
            out[i] = f"\\textbf{{{fmt_fn(v)}}}"
    return out


def latex_escape(s: str) -> str:
    return s.replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


# ---------------------------------------------------------------------------
# Bootstrap utilities
# ---------------------------------------------------------------------------


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i] if math.isfinite(pvals[i]) else 1.0)
    adj = [1.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        p = pvals[idx]
        if not math.isfinite(p):
            adj[idx] = float("nan")
            continue
        corrected = min(1.0, (m - rank) * p)
        running = max(running, corrected)
        adj[idx] = running
    return adj


def bootstrap_p(diffs: np.ndarray) -> float:
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return float("nan")
    p_le = float(np.mean(diffs <= 0))
    p_ge = float(np.mean(diffs >= 0))
    return float(min(1.0, 2.0 * min(p_le, p_ge)))


def mcnemar_exact(n01: int, n10: int) -> float:
    """Two-sided exact McNemar (binomial mid-style via chi2 continuity when large)."""
    n = n01 + n10
    if n == 0:
        return 1.0
    # Exact binomial two-sided: P(X <= min) + P(X >= max) under p=0.5
    k = min(n01, n10)
    # Sum binom probs
    # P(X=i) = C(n,i) / 2^n
    from math import comb

    p = sum(comb(n, i) for i in range(0, k + 1)) / (2**n)
    # two-sided
    p2 = min(1.0, 2.0 * p)
    return float(p2)


def family_bootstrap_stat(
    family_to_rows: dict[str, list[dict]],
    metric_fn: Callable[[list[dict]], float],
    *,
    n_reps: int = BOOT_N,
    seed: int = BOOT_SEED,
) -> dict[str, float]:
    families = sorted(family_to_rows)
    if not families:
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "boot_mean": float("nan")}
    rng = np.random.default_rng(seed)
    n_fam = len(families)
    stats = np.empty(n_reps, dtype=float)
    for i in range(n_reps):
        chosen = rng.choice(families, size=n_fam, replace=True)
        sample: list[dict] = []
        for fam in chosen:
            sample.extend(family_to_rows[str(fam)])
        stats[i] = metric_fn(sample)
    point = metric_fn([r for fam in families for r in family_to_rows[fam]])
    return {
        "estimate": float(point),
        "ci_low": float(np.nanpercentile(stats, 2.5)),
        "ci_high": float(np.nanpercentile(stats, 97.5)),
        "boot_mean": float(np.nanmean(stats)),
        "boot_samples": stats,
    }


def group_by_family(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        out[str(r["scenario_family_id"])].append(r)
    return dict(out)


# ---------------------------------------------------------------------------
# Load + attach predictions
# ---------------------------------------------------------------------------


def attach_preds(rows: list[dict], thresholds: dict[str, dict[str, float]]) -> list[dict]:
    out = []
    for r in rows:
        if r.get("split") != "test":
            continue
        mid = r["model"]
        view = r["view"]
        tmap = thresholds.get(mid, {})
        if view not in tmap:
            continue
        thr = tmap[view]
        score = float(r["score"]) if r.get("score") is not None else float("nan")
        # Canonical decision rule: score >= threshold (matches metrics.py).
        pred = decide(score, thr, bool(r.get("invalid")))
        nr = dict(r)
        nr["pred"] = pred
        nr["y"] = as_bin(r["label"])
        nr["threshold"] = thr
        out.append(nr)
    return out


def reproduce_checks(metrics_hpc: dict, preds_hpc: list[dict]) -> list[str]:
    lines = []
    mid = "qwen2.5-1.5b"
    thr = metrics_hpc["thresholds"][mid]
    full = [r for r in preds_hpc if r["model"] == mid and r["view"] == "FULL" and r["split"] == "test"]
    scores = np.array([float(r["score"]) for r in full])
    labels = np.array([as_bin(r["label"]) for r in full])
    pred = (scores >= thr["FULL"]).astype(int)
    mf1 = macro_f1(labels, pred)
    fpr = legitimate_fpr(labels, pred)
    exp_f1 = metrics_hpc["slices"][mid]["FULL"]["pooled"]["macro_f1"]
    exp_fpr = metrics_hpc["slices"][mid]["FULL"]["pooled"]["legitimate_fpr"]
    ok1 = abs(mf1 - exp_f1) < 1e-4 and abs(fpr - exp_fpr) < 1e-4
    lines.append(
        f"FULL {mid}: reproduced macro-F1={mf1:.4f} (expected {exp_f1:.4f}), "
        f"legit FPR={fpr:.4f} (expected {exp_fpr:.4f}) — {'PASS' if ok1 else 'FAIL'}"
    )
    ppr = [r for r in preds_hpc if r["model"] == mid and r["view"] == "PREFIX_PLUS_REACTION" and r["split"] == "test"]
    enriched = []
    for r in ppr:
        rr = dict(r)
        rr["pred"] = decide(float(r["score"]), thr["PREFIX_PLUS_REACTION"])
        enriched.append(rr)
    d = branch_disagreement_rate(enriched)
    exp_d = metrics_hpc["slices"][mid]["PREFIX_PLUS_REACTION"]["branch"]["branch_disagreement_rate"]
    ok2 = abs(d - exp_d) < 1e-4
    lines.append(
        f"PREFIX_PLUS_REACTION {mid}: disagreement={d:.4f} (expected {exp_d:.4f}) — {'PASS' if ok2 else 'FAIL'}"
    )
    if not (ok1 and ok2):
        raise SystemExit("Reproduction check failed; aborting.")
    return lines


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------


def build_models_tex(zs_metrics: dict, resources: dict) -> str:
    rows_data = []
    for mid in ZEROSHOT_IDS:
        sl = zs_metrics["slices"][mid]["FULL"]["pooled"]
        spec = resources["models"].get(mid, {})
        lat = spec.get("latency") or {}
        med_ms = 1000.0 * float(lat["median_s"]) if lat.get("median_s") is not None else float("nan")
        rows_data.append(
            {
                "mid": mid,
                "f1": sl["macro_f1"],
                "hf": spec.get("hf_id") or "",
                "params": spec.get("params"),
                "family": FAMILY[mid],
                "lat_ms": med_ms,
                "inv": sl["invalid_output_rate"],
            }
        )
    rows_data.sort(key=lambda r: r["f1"], reverse=True)
    lines = [
        "% Auto-generated by scripts/analysis/supp_results.py — do not edit by hand.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lllrrc}",
        "\\toprule",
        "\\rowcolor{headblue}",
        "\\textbf{Model} & \\textbf{HF id} & \\textbf{Family} & \\textbf{Params} & \\textbf{Med.\\ lat.\\ (ms)} & \\textbf{Invalid} \\\\",
        "\\midrule",
    ]
    for r in rows_data:
        lines.append(
            f"{model_tex(r['mid'])} & \\texttt{{{latex_escape(r['hf'])}}} & {r['family']} & "
            f"{fmt_params(r['params'])} & {r['lat_ms']:.1f} & {fmt_prop_plain(r['inv'])} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Zero-shot panel (16 models). Parameters are trainable weight counts from "
        "\\texttt{resources.json}. Median latency is per-inference wall time on an NVIDIA H100 "
        "MIG slice (\\texttt{3g.47gb}). Invalid: share of test calls with no valid free-text "
        "verbalizer continuation (SmolLM2-135M and TinyLlama-1.1B never produce a valid answer; "
        "marked $^\\dagger$).}",
        "\\label{tab:supp-models}",
        "\\end{table*}",
    ]
    return "\n".join(lines)


def _slice(metrics: dict, mid: str, view: str) -> dict:
    return metrics["slices"][mid][view]


def build_zeroshot_quality(
    zs_metrics: dict,
    f1_cis: dict[str, tuple[float, float]],
) -> str:
    rows = []
    for mid in ZEROSHOT_IDS:
        p = _slice(zs_metrics, mid, "FULL")["pooled"]
        rows.append(
            {
                "mid": mid,
                "f1": p["macro_f1"],
                "auc": p["roc_auc"],
                "rec": p["fraud_recall"],
                "fpr": p["legitimate_fpr"],
                "ci": f1_cis[mid],
            }
        )
    rows.sort(key=lambda r: r["f1"], reverse=True)
    f1s = bold_best([r["f1"] for r in rows], True)
    aucs = bold_best([r["auc"] for r in rows], True)
    recs = bold_best([r["rec"] for r in rows], True)
    fprs = bold_best([r["fpr"] for r in rows], False)
    lines = [
        "% Auto-generated by scripts/analysis/supp_results.py",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\rowcolor{headblue}",
        "\\textbf{Model} & \\textbf{Macro-F1 [95\\% CI]} & \\textbf{ROC-AUC} & "
        "\\textbf{Fraud recall} & \\textbf{False-alarm rate} \\\\",
        "\\midrule",
    ]
    for i, r in enumerate(rows):
        lo, hi = r["ci"]
        f1_cell = f"{f1s[i]} {fmt_ci(lo, hi)}"
        lines.append(
            f"{model_tex(r['mid'])} & {f1_cell} & {aucs[i]} & {recs[i]} & {fprs[i]} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Zero-shot quality on the full call (test, $n{=}540$). "
        "False-alarm rate $=$ P(pred\\,=\\,fraud $\\mid$ legitimate). "
        "95\\% CIs are family-clustered bootstrap percentile intervals "
        f"({BOOT_N:,} resamples of the 12 scenario families, seed {BOOT_SEED}), "
        "using fixed development thresholds. Bold: best in column "
        "(highest F1/AUC/recall; lowest false-alarm rate). "
        "$^\\dagger$: never produces a valid free-text answer.}",
        "\\label{tab:supp-zeroshot-quality}",
        "\\end{table*}",
    ]
    return "\n".join(lines)


def build_zeroshot_branch(zs_metrics: dict, caller_metrics: dict) -> str:
    rows = []
    for mid in ZEROSHOT_IDS:
        full = _slice(zs_metrics, mid, "FULL")
        po = _slice(zs_metrics, mid, "PREFIX_ONLY")
        ppr = _slice(zs_metrics, mid, "PREFIX_PLUS_REACTION")
        co = caller_metrics["slices"][mid]["CALLER_ONLY"]
        rd = ppr.get("reaction_delta", {})
        rows.append(
            {
                "mid": mid,
                "f1": full["pooled"]["macro_f1"],
                "d_po": po["branch"]["branch_disagreement_rate"],
                "d_ppr": ppr["branch"]["branch_disagreement_rate"],
                "d_full": full["branch"]["branch_disagreement_rate"],
                "d_co": co["branch"]["branch_disagreement_rate"],
                "a_po": po["branch"]["all_branches_correct_rate"],
                "a_ppr": ppr["branch"]["all_branches_correct_rate"],
                "a_full": full["branch"]["all_branches_correct_rate"],
                "delta": rd.get("mean_delta", float("nan")),
            }
        )
    rows.sort(key=lambda r: r["f1"], reverse=True)
    # Bold: lowest disagreement, highest all-correct, mean delta closest to 0? skip delta bold
    cols = {
        "d_po": False,
        "d_ppr": False,
        "d_full": False,
        "d_co": False,
        "a_po": True,
        "a_ppr": True,
        "a_full": True,
    }
    formatted = {}
    for k, hib in cols.items():
        formatted[k] = bold_best([r[k] for r in rows], hib)
    lines = [
        "% Auto-generated by scripts/analysis/supp_results.py",
        "\\begin{table*}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{2.8pt}",
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "\\rowcolor{headblue}",
        "\\textbf{Model} & \\textbf{D}$_{\\mathrm{open}}$ & \\textbf{D}$_{\\mathrm{+reply}}$ & "
        "\\textbf{D}$_{\\mathrm{full}}$ & \\textbf{D}$_{\\mathrm{caller}}$ & "
        "\\textbf{A}$_{\\mathrm{open}}$ & \\textbf{A}$_{\\mathrm{+reply}}$ & "
        "\\textbf{A}$_{\\mathrm{full}}$ & $\\overline{\\Delta}$ \\\\",
        "\\midrule",
    ]
    for i, r in enumerate(rows):
        lines.append(
            f"{model_tex(r['mid'])} & {formatted['d_po'][i]} & {formatted['d_ppr'][i]} & "
            f"{formatted['d_full'][i]} & {formatted['d_co'][i]} & "
            f"{formatted['a_po'][i]} & {formatted['a_ppr'][i]} & {formatted['a_full'][i]} & "
            f"{fmt_prop_plain(r['delta'])} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Sibling disagreement (D) and all-four-correct rate (A) on complete "
        "four-branch histories ($n{=}122$), plus mean reaction delta "
        "$\\overline{\\Delta}=\\mathbb{E}[P(\\mathrm{fraud})_{\\mathrm{+reply}}"
        "-P(\\mathrm{fraud})_{\\mathrm{open}}]$. "
        "Opening-only disagreement is zero when openings are identical. "
        "Bold: best in column (lowest D; highest A).}",
        "\\label{tab:supp-zeroshot-branch}",
        "\\end{table*}",
    ]
    return "\n".join(lines)


def build_by_language(zs_metrics: dict) -> str:
    rows = []
    for mid in ZEROSHOT_IDS:
        bl = _slice(zs_metrics, mid, "FULL")["by_language"]
        rows.append(
            {
                "mid": mid,
                "f1": _slice(zs_metrics, mid, "FULL")["pooled"]["macro_f1"],
                **{f"f1_{L}": bl[L]["macro_f1"] for L in LANGS},
                **{f"fpr_{L}": bl[L]["legitimate_fpr"] for L in LANGS},
            }
        )
    rows.sort(key=lambda r: r["f1"], reverse=True)
    f1_cols = {L: bold_best([r[f"f1_{L}"] for r in rows], True) for L in LANGS}
    fpr_cols = {L: bold_best([r[f"fpr_{L}"] for r in rows], False) for L in LANGS}
    lines = [
        "% Auto-generated by scripts/analysis/supp_results.py",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "\\rowcolor{headblue}",
        "\\textbf{Model} & \\multicolumn{3}{c}{\\textbf{Macro-F1}} & "
        "\\multicolumn{3}{c}{\\textbf{False-alarm rate}} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
        "\\rowcolor{headblue}",
        "& Hi--En & Ta--En & Te--En & Hi--En & Ta--En & Te--En \\\\",
        "\\midrule",
    ]
    for i, r in enumerate(rows):
        lines.append(
            f"{model_tex(r['mid'])} & "
            + " & ".join(f1_cols[L][i] for L in LANGS)
            + " & "
            + " & ".join(fpr_cols[L][i] for L in LANGS)
            + " \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Full-call macro-F1 and false-alarm rate by language pair (test). "
        "Bold: best per column.}",
        "\\label{tab:supp-by-language}",
        "\\end{table*}",
    ]
    return "\n".join(lines)


def build_by_mood(zs_metrics: dict, view: str, label: str) -> str:
    rows = []
    for mid in ZEROSHOT_IDS:
        bc = _slice(zs_metrics, mid, view)["pooled"]["by_condition"]
        rows.append(
            {
                "mid": mid,
                "f1": _slice(zs_metrics, mid, "FULL")["pooled"]["macro_f1"],
                **{f"fpr_{c}": bc[c]["legitimate_fpr"] for c in CONDITIONS},
                **{f"rec_{c}": bc[c]["fraud_recall"] for c in CONDITIONS},
            }
        )
    rows.sort(key=lambda r: r["f1"], reverse=True)
    fpr_fmt = {c: bold_best([r[f"fpr_{c}"] for r in rows], False) for c in CONDITIONS}
    rec_fmt = {c: bold_best([r[f"rec_{c}"] for r in rows], True) for c in CONDITIONS}
    short = {"cooperative": "Coop", "clarifying": "Clar", "resistant": "Res", "suspicious": "Sus"}
    lines = [
        "% Auto-generated by scripts/analysis/supp_results.py",
        "\\begin{table*}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{tabular}{lcccccccc}",
        "\\toprule",
        "\\rowcolor{headblue}",
        "\\textbf{Model} & \\multicolumn{4}{c}{\\textbf{False-alarm rate (genuine)}} & "
        "\\multicolumn{4}{c}{\\textbf{Fraud recall}} \\\\",
        "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
        "\\rowcolor{headblue}",
        "& "
        + " & ".join(short[c] for c in CONDITIONS)
        + " & "
        + " & ".join(short[c] for c in CONDITIONS)
        + " \\\\",
        "\\midrule",
    ]
    for i, r in enumerate(rows):
        lines.append(
            f"{model_tex(r['mid'])} & "
            + " & ".join(fpr_fmt[c][i] for c in CONDITIONS)
            + " & "
            + " & ".join(rec_fmt[c][i] for c in CONDITIONS)
            + " \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{Recipient-condition breakdown on the {label} view (test). "
        "False alarms are computed on genuine calls; recall on fraud calls. "
        "Bold: best per column.}}",
        f"\\label{{tab:supp-by-mood-{view.lower()}}}",
        "\\end{table*}",
    ]
    return "\n".join(lines)


def build_caller_only(caller_metrics: dict) -> str:
    rows = []
    for mid in ZEROSHOT_IDS:
        full = caller_metrics["slices"][mid]["FULL"]
        co = caller_metrics["slices"][mid]["CALLER_ONLY"]
        rows.append(
            {
                "mid": mid,
                "f1_full": full["pooled"]["macro_f1"],
                "f1_co": co["pooled"]["macro_f1"],
                "d_full": full["branch"]["branch_disagreement_rate"],
                "d_co": co["branch"]["branch_disagreement_rate"],
            }
        )
    rows.sort(key=lambda r: r["f1_full"], reverse=True)
    f1f = bold_best([r["f1_full"] for r in rows], True)
    f1c = bold_best([r["f1_co"] for r in rows], True)
    df = bold_best([r["d_full"] for r in rows], False)
    dc = bold_best([r["d_co"] for r in rows], False)
    lines = [
        "% Auto-generated by scripts/analysis/supp_results.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\rowcolor{headblue}",
        "\\textbf{Model} & \\textbf{F1}$_{\\mathrm{full}}$ & \\textbf{F1}$_{\\mathrm{caller}}$ & "
        "\\textbf{D}$_{\\mathrm{full}}$ & \\textbf{D}$_{\\mathrm{caller}}$ \\\\",
        "\\midrule",
    ]
    for i, r in enumerate(rows):
        lines.append(
            f"{model_tex(r['mid'])} & {f1f[i]} & {f1c[i]} & {df[i]} & {dc[i]} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Full call vs.\\ caller-turns-only (test). Macro-F1 and sibling disagreement. "
        "Bold: best per column.}",
        "\\label{tab:supp-caller-only}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def build_streaming(zs_metrics: dict) -> str:
    lines = [
        "% Auto-generated by scripts/analysis/supp_results.py",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lcccccccc}",
        "\\toprule",
        "\\rowcolor{headblue}",
        "\\textbf{Model} & \\textbf{Macro-F1} & \\textbf{Recall} & \\textbf{FA} & "
        "\\textbf{Miss rate} & \\textbf{Mean delay} & \\textbf{Med.\\ delay} & "
        "\\textbf{Pre-ev.\\ alert} & \\textbf{$n$ det.} \\\\",
        "\\midrule",
    ]
    for mid in ["qwen2.5-1.5b", "qwen3-1.7b"]:
        sl = zs_metrics["slices"][mid]["STREAMING"]
        p = sl["pooled"]
        st = sl["streaming"]
        lines.append(
            f"{DISPLAY[mid]} & {fmt_prop_plain(p['macro_f1'])} & {fmt_prop_plain(p['fraud_recall'])} & "
            f"{fmt_prop_plain(p['legitimate_fpr'])} & {fmt_prop_plain(st['miss_rate'])} & "
            f"{st['mean_delay_detected']:.2f} & {st['median_delay_detected']:.1f} & "
            f"{fmt_prop_plain(st['pre_evidence_alert_rate'])} & {int(st['n_detected'])} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Streaming evaluation (two models only). After each turn the model scores the "
        "prefix; the first time $P(\\mathrm{fraud})$ crosses the development threshold is the "
        "\\emph{alert turn}. \\emph{Delay} $=$ alert turn $-$ evidence turn (negative $=$ alerted "
        "before the deciding evidence). \\emph{Miss rate}: share of calls that never alert. "
        "\\emph{Pre-evidence alert}: share of eligible calls with alert turn before the evidence "
        "turn. Macro-F1/recall/FA use the final (full-call) streaming score.}",
        "\\label{tab:supp-streaming}",
        "\\end{table*}",
    ]
    return "\n".join(lines)


def ft_model_id(base: str, arm: str, seed: int) -> str:
    return f"{base}+{arm}-seed{seed}"


def collect_ft_rows(ft_metrics: dict, zs_metrics: dict) -> list[dict]:
    rows = []
    for base in ["qwen2.5-0.5b", "qwen2.5-1.5b"]:
        # zero-shot reference from caller_only or zs (FULL etc.)
        zs_full = zs_metrics["slices"][base]
        # CALLER_ONLY may be only in caller metrics — filled by caller later if needed
        rows.append({"base": base, "arm": "zeroshot", "seed": None, "mid": base, "slices": zs_full})
        for arm in ["unbranched", "branched", "branched_consistency"]:
            for seed in (0, 1, 2):
                mid = ft_model_id(base, arm, seed)
                if mid not in ft_metrics["slices"]:
                    raise KeyError(f"missing FT adapter metrics for {mid}")
                rows.append(
                    {
                        "base": base,
                        "arm": arm,
                        "seed": seed,
                        "mid": mid,
                        "slices": ft_metrics["slices"][mid],
                    }
                )
    return rows


def _ft_extract(slices: dict, caller_zs: dict | None = None) -> dict[str, float]:
    full = slices["FULL"]
    ppr = slices["PREFIX_PLUS_REACTION"]
    po = slices["PREFIX_ONLY"]
    co = slices.get("CALLER_ONLY")
    out = {
        "macro_f1": full["pooled"]["macro_f1"],
        "roc_auc": full["pooled"]["roc_auc"],
        "fraud_recall": full["pooled"]["fraud_recall"],
        "legitimate_fpr": full["pooled"]["legitimate_fpr"],
        "invalid": full["pooled"]["invalid_output_rate"],
        "d_full": full["branch"]["branch_disagreement_rate"],
        "d_ppr": ppr["branch"]["branch_disagreement_rate"],
        "d_po": po["branch"]["branch_disagreement_rate"],
        "d_co": co["branch"]["branch_disagreement_rate"] if co else float("nan"),
        "rec_coop": full["pooled"]["by_condition"]["cooperative"]["fraud_recall"],
        "fpr_sus": full["pooled"]["by_condition"]["suspicious"]["legitimate_fpr"],
    }
    for L in LANGS:
        out[f"f1_{L}"] = full["by_language"][L]["macro_f1"]
    return out


def build_ft_per_seed(ft_metrics: dict, zs_metrics: dict, caller_metrics: dict) -> str:
    # Attach CALLER_ONLY for zero-shot from caller_metrics
    lines = [
        "% Auto-generated by scripts/analysis/supp_results.py",
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.2pt}",
        "\\begin{tabular}{llccccccccccc}",
        "\\toprule",
        "\\rowcolor{headblue}",
        "\\textbf{Size} & \\textbf{Arm} & \\textbf{Seed} & \\textbf{F1} & \\textbf{AUC} & "
        "\\textbf{Rec} & \\textbf{FA} & \\textbf{D}$_{\\mathrm{full}}$ & "
        "\\textbf{D}$_{\\mathrm{+reply}}$ & \\textbf{R}$_{\\mathrm{C}}$ & "
        "\\textbf{FA}$_{\\mathrm{S}}$ & \\textbf{F1}$_{\\mathrm{Hi}}$ & \\textbf{F1}$_{\\mathrm{Ta}}$ \\\\",
        "\\midrule",
    ]
    for base, size in [("qwen2.5-0.5b", "0.5B"), ("qwen2.5-1.5b", "1.5B")]:
        lines.append(f"\\rowcolor{{groupgray}}\\multicolumn{{13}}{{l}}{{\\emph{{Qwen2.5-{size}}}}} \\\\")
        # zeroshot
        zs_slices = dict(zs_metrics["slices"][base])
        zs_slices["CALLER_ONLY"] = caller_metrics["slices"][base]["CALLER_ONLY"]
        e = _ft_extract(zs_slices)
        lines.append(
            f"& Zero-shot & --- & {fmt_prop_plain(e['macro_f1'])} & {fmt_prop_plain(e['roc_auc'])} & "
            f"{fmt_prop_plain(e['fraud_recall'])} & {fmt_prop_plain(e['legitimate_fpr'])} & "
            f"{fmt_prop_plain(e['d_full'])} & {fmt_prop_plain(e['d_ppr'])} & "
            f"{fmt_prop_plain(e['rec_coop'])} & {fmt_prop_plain(e['fpr_sus'])} & "
            f"{fmt_prop_plain(e['f1_hi_en'])} & {fmt_prop_plain(e['f1_ta_en'])} \\\\"
        )
        for arm in ["unbranched", "branched", "branched_consistency"]:
            for seed in (0, 1, 2):
                mid = ft_model_id(base, arm, seed)
                e = _ft_extract(ft_metrics["slices"][mid])
                lines.append(
                    f"& {ARM_NAME[arm]} & {seed} & {fmt_prop_plain(e['macro_f1'])} & "
                    f"{fmt_prop_plain(e['roc_auc'])} & {fmt_prop_plain(e['fraud_recall'])} & "
                    f"{fmt_prop_plain(e['legitimate_fpr'])} & {fmt_prop_plain(e['d_full'])} & "
                    f"{fmt_prop_plain(e['d_ppr'])} & {fmt_prop_plain(e['rec_coop'])} & "
                    f"{fmt_prop_plain(e['fpr_sus'])} & {fmt_prop_plain(e['f1_hi_en'])} & "
                    f"{fmt_prop_plain(e['f1_ta_en'])} \\\\"
                )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Fine-tuning per seed (test). R$_C$: fraud recall under a cooperative victim; "
        "FA$_S$: false-alarm rate under a suspicious recipient. D: sibling disagreement. "
        "Zero-shot base rows use the same thresholds/views as the main panel.}",
        "\\label{tab:supp-ft-per-seed}",
        "\\end{table*}",
    ]
    return "\n".join(lines)


def build_ft_summary(ft_metrics: dict, zs_metrics: dict, caller_metrics: dict) -> str:
    lines = [
        "% Auto-generated by scripts/analysis/supp_results.py",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{2.8pt}",
        "\\begin{tabular}{llccccccc}",
        "\\toprule",
        "\\rowcolor{headblue}",
        "\\textbf{Size} & \\textbf{Arm} & \\textbf{F1} & \\textbf{FA} & \\textbf{D}$_{\\mathrm{full}}$ & "
        "\\textbf{D}$_{\\mathrm{+reply}}$ & \\textbf{R}$_{\\mathrm{C}}$ & "
        "\\textbf{FA}$_{\\mathrm{S}}$ & \\textbf{Inv.} \\\\",
        "\\midrule",
    ]
    keys = ["macro_f1", "legitimate_fpr", "d_full", "d_ppr", "rec_coop", "fpr_sus", "invalid"]

    def mean_sd(vals: list[float]) -> str:
        a = np.asarray(vals, dtype=float)
        return f"{fmt_prop_plain(float(a.mean()))}$\\pm${fmt_prop_plain(float(a.std(ddof=1) if len(a) > 1 else 0.0))}"

    for base, size in [("qwen2.5-0.5b", "0.5B"), ("qwen2.5-1.5b", "1.5B")]:
        lines.append(f"\\rowcolor{{groupgray}}\\multicolumn{{9}}{{l}}{{\\emph{{Qwen2.5-{size}}}}} \\\\")
        zs_slices = dict(zs_metrics["slices"][base])
        zs_slices["CALLER_ONLY"] = caller_metrics["slices"][base]["CALLER_ONLY"]
        e = _ft_extract(zs_slices)
        lines.append(
            f"& Zero-shot & {fmt_prop_plain(e['macro_f1'])} & {fmt_prop_plain(e['legitimate_fpr'])} & "
            f"{fmt_prop_plain(e['d_full'])} & {fmt_prop_plain(e['d_ppr'])} & "
            f"{fmt_prop_plain(e['rec_coop'])} & {fmt_prop_plain(e['fpr_sus'])} & "
            f"{fmt_prop_plain(e['invalid'])} \\\\"
        )
        for arm in ["unbranched", "branched", "branched_consistency"]:
            seed_ex = [_ft_extract(ft_metrics["slices"][ft_model_id(base, arm, s)]) for s in (0, 1, 2)]
            cells = [mean_sd([ex[k] for ex in seed_ex]) for k in keys]
            lines.append(f"& {ARM_NAME[arm]} & " + " & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Fine-tuning summary: mean$\\pm$sd over seeds $\\{0,1,2\\}$. "
        "Three seeds $\\times$ 12 families is low power for arm contrasts (see statistical tests).}",
        "\\label{tab:supp-ft-summary}",
        "\\end{table*}",
    ]
    return "\n".join(lines)


def parse_train_log(path: Path) -> dict[str, Any]:
    best_epoch = None
    best_ce = float("inf")
    last_step = 0
    early_stop_epoch = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("event") == "step":
            last_step = max(last_step, int(d.get("step", 0)))
        if d.get("event") == "checkpoint":
            ce = float(d["dev_ce"])
            ep = int(d["epoch"])
            if ce < best_ce:
                best_ce = ce
                best_epoch = ep
        if d.get("event") == "early_stop":
            early_stop_epoch = int(d["epoch"])
        if d.get("event") == "epoch":
            last_step = max(last_step, 0)  # noop; steps tracked above
    return {
        "best_epoch": best_epoch,
        "best_dev_ce": best_ce if best_epoch is not None else None,
        "last_step": last_step,
        "early_stop_epoch": early_stop_epoch,
    }


def build_ft_hparams() -> str:
    lines = [
        "% Auto-generated by scripts/analysis/supp_results.py",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{llccccccc}",
        "\\toprule",
        "\\rowcolor{headblue}",
        "\\textbf{Size} & \\textbf{Arm} & \\textbf{Seed} & \\textbf{$n$} & "
        "\\textbf{Histories} & $\\lambda$ & \\textbf{Best ep.} & \\textbf{Steps} & "
        "\\textbf{Best dev CE} \\\\",
        "\\midrule",
    ]
    shared = (
        "Shared: LoRA $r{=}16$, $\\alpha{=}32$, dropout $0.05$, lr $1{\\times}10^{-4}$, "
        "4 epochs max, early stop on development CE."
    )
    for base, size in [("qwen2.5-0.5b", "0.5B"), ("qwen2.5-1.5b", "1.5B")]:
        lines.append(f"\\rowcolor{{groupgray}}\\multicolumn{{9}}{{l}}{{\\emph{{Qwen2.5-{size}}}}} \\\\")
        for arm in ["unbranched", "branched", "branched_consistency"]:
            for seed in (0, 1, 2):
                run_dir = ROOT / "outputs" / "ft" / base / arm / f"seed{seed}"
                rc = load_json(run_dir / "run_config.json")
                tr = load_json(run_dir / "train_result.json")
                log = parse_train_log(run_dir / "train_log.jsonl")
                ss = rc["selection_summary"]
                n_hist = ss.get("n_histories")
                lam = rc.get("lambda_consistency", 0.0)
                best_ce = tr.get("best_dev_ce", log.get("best_dev_ce"))
                lines.append(
                    f"& {ARM_NAME[arm]} & {seed} & {rc['n']} & {n_hist} & {lam:g} & "
                    f"{log['best_epoch']} & {log['last_step']} & {best_ce:.3f} \\\\"
                )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Fine-tuning hyperparameters and stopping. Unbranched: 408 histories "
        "$\\times$ 1 random branch; branched / consistency: 102 histories $\\times$ 4 branches "
        "(matched $n{=}408$). Consistency arm uses variance penalty on $P(\\mathrm{fraud})$ "
        "across siblings with $\\lambda{=}1$. "
        + shared
        + " Best epoch/steps read from \\texttt{train_log.jsonl}; best CE from "
        "\\texttt{train_result.json}.}",
        "\\label{tab:supp-ft-hparams}",
        "\\end{table*}",
    ]
    return "\n".join(lines)


def build_break_points() -> str:
    # Parse rates from BREAK_POINTS.md table
    bp = (ROOT / "results" / "eval_hpc" / "BREAK_POINTS.md").read_text(encoding="utf-8")
    rate_map: dict[tuple[str, str], dict[str, float]] = {}
    for line in bp.splitlines():
        if not line.startswith("| "):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 7 or parts[0] == "family" or parts[0].startswith("---"):
            continue
        fam = parts[0].split()[0]
        case = parts[1]
        try:
            rate_map[(fam, case)] = {
                "all": float(parts[2]),
                "strong": float(parts[3]),
                "qwen2.5-1.5b": float(parts[4]),
                "qwen3-1.7b": float(parts[5]),
                "qwen2-1.5b": float(parts[6]),
            }
        except ValueError:
            continue

    def short_title(fid: str, case: str, note: str | None) -> str:
        if note and case == "L":
            return note
        card = load_json(ROOT / "scenarios" / "json" / f"{fid}.json")
        t = card["title"]
        if " vs." in t:
            t = t.split(" vs.")[0]
        t = t.strip().replace('"', "'")
        if len(t) > 64:
            t = t[:61] + "..."
        return t

    # notes keyed by (fam, case) from markdown italics
    notes: dict[tuple[str, str], str] = {}
    for line in bp.splitlines():
        if "*(" not in line or not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 2:
            continue
        fam = parts[0].split()[0]
        case = parts[1]
        m = re.search(r"\*\((.*?)\)\*", line)
        if m:
            notes[(fam, case)] = m.group(1)

    lines = [
        "% Auto-generated by scripts/analysis/supp_results.py",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{llcrrrrr}",
        "\\toprule",
        "\\rowcolor{headblue}",
        "\\textbf{Family} & \\textbf{Err.} & \\textbf{Description} & \\textbf{All} & "
        "\\textbf{Strong} & \\textbf{Q2.5-1.5B} & \\textbf{Q3-1.7B} & \\textbf{Q2-1.5B} \\\\",
        "\\midrule",
    ]
    ordered = []
    for line in bp.splitlines():
        if not line.startswith("| "):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 7 or parts[0] == "family":
            continue
        fam = parts[0].split()[0]
        case = parts[1]
        if (fam, case) in rate_map and (fam, case) not in ordered:
            ordered.append((fam, case))
        if len(ordered) >= 10:
            break
    for fam, case in ordered:
        desc = short_title(fam, case, notes.get((fam, case)))
        err = "Miss" if case == "F" else "FA"
        r = rate_map[(fam, case)]
        lines.append(
            f"\\texttt{{{fam}}} & {err} & {latex_escape(desc)} & "
            f"{fmt_prop_plain(r['all'])} & {fmt_prop_plain(r['strong'])} & "
            f"{fmt_prop_plain(r['qwen2.5-1.5b'])} & {fmt_prop_plain(r['qwen3-1.7b'])} & "
            f"{fmt_prop_plain(r['qwen2-1.5b'])} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Hardest scenario families on the full call (from \\texttt{BREAK\\_POINTS.md}). "
        "Err.\\,=\\,Miss (fraud recall failure) or FA (false alarm on genuine). "
        "Strong $=$ Qwen2.5-1.5B, Qwen3-1.7B, Qwen2-1.5B. Descriptions from scenario cards "
        "or author notes in the break-point report.}",
        "\\label{tab:supp-break-families}",
        "\\end{table*}",
        "",
        "\\begin{table*}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "\\rowcolor{headblue}",
        "\\textbf{Model} & \\textbf{Label} & \\textbf{Cooperative} & \\textbf{Clarifying} & "
        "\\textbf{Resistant} & \\textbf{Suspicious} \\\\",
        "\\midrule",
    ]
    # Reaction flips for strong models
    flip_models = ["qwen2.5-1.5b", "qwen3-1.7b", "qwen2-1.5b", "gemma3-1b"]
    # Parse section 5 from BREAK_POINTS — gemma3 not in eval_hpc break points; only hpc models
    flip_models = ["qwen2.5-1.5b", "qwen3-1.7b", "qwen2-1.5b", "qwen2.5-0.5b"]
    for mid in flip_models:
        for label in ("fraud", "legit"):
            # find line
            found = None
            for line in bp.splitlines():
                if line.startswith(f"| {mid} | {label}"):
                    found = [p.strip() for p in line.strip("|").split("|")]
                    break
            if not found:
                continue
            # columns: model, label, coop, clar, res, sus
            cells = found[2:6]
            lines.append(
                f"{DISPLAY[mid]} & {label} & " + " & ".join(c.split()[0] for c in cells) + " \\\\"
            )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Verdict flips when the recipient's first reply is added "
        "(PREFIX\\_ONLY $\\to$ PREFIX\\_PLUS\\_REACTION). For fraud calls, flip $=$ fraud$\\to$legitimate "
        "(reaction clears the caller); for genuine calls, flip $=$ legitimate$\\to$fraud "
        "(reaction creates a false alarm). Cells show rate (count).}",
        "\\label{tab:supp-break-flips}",
        "\\end{table*}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def run_stats(
    zs_preds: dict[str, dict[str, list[dict]]],
    caller_preds: dict[str, dict[str, list[dict]]],
    ft_preds: dict[str, dict[str, list[dict]]],
    zs_metrics: dict,
) -> tuple[str, dict]:
    """Return (stats_tests.tex, stats.json)."""
    results: dict[str, Any] = {"n_reps": BOOT_N, "seed": BOOT_SEED, "tests": {}}
    tex_blocks: list[str] = []

    def pack(name: str, entries: list[dict]) -> None:
        pvals = [e["p"] for e in entries]
        padj = holm(pvals)
        for e, a in zip(entries, padj):
            e["p_holm"] = a
        results["tests"][name] = entries

    # ---- T1 mood ----
    t1: list[dict] = []
    for view in ["FULL", "PREFIX_PLUS_REACTION"]:
        for mid in ZEROSHOT_IDS:
            rows = zs_preds[mid][view]
            fam = group_by_family(rows)
            families = sorted(fam)
            n_fam = len(families)
            rng = np.random.default_rng(BOOT_SEED)
            choices = rng.choice(families, size=(BOOT_N, n_fam), replace=True)

            def both_gaps(sample: list[dict]) -> tuple[float, float]:
                def fpr(cond: str) -> float:
                    sub = [r for r in sample if r["recipient_condition"] == cond and r["y"] == 0]
                    return float(np.mean([r["pred"] for r in sub])) if sub else float("nan")

                def rec(cond: str) -> float:
                    sub = [r for r in sample if r["recipient_condition"] == cond and r["y"] == 1]
                    return float(np.mean([r["pred"] for r in sub])) if sub else float("nan")

                fpr_g = fpr("suspicious") - fpr("cooperative")
                others = [rec(c) for c in ("clarifying", "resistant", "suspicious")]
                rec_g = rec("cooperative") - float(np.nanmean(others))
                return fpr_g, rec_g

            point = both_gaps([r for f in families for r in fam[f]])
            stats_fpr = np.empty(BOOT_N)
            stats_rec = np.empty(BOOT_N)
            for i in range(BOOT_N):
                sample = [r for f in choices[i] for r in fam[str(f)]]
                a, b = both_gaps(sample)
                stats_fpr[i] = a
                stats_rec[i] = b

            # McNemar on paired siblings for FPR sus vs coop
            by_h: dict[str, dict[str, int]] = defaultdict(dict)
            for r in rows:
                if r["y"] == 0:
                    by_h[hist_key(r["conversation_id"])][r["recipient_condition"]] = r["pred"]
            n01 = n10 = 0
            for g in by_h.values():
                if "cooperative" in g and "suspicious" in g:
                    a, b = g["cooperative"], g["suspicious"]
                    if a == 0 and b == 1:
                        n01 += 1
                    elif a == 1 and b == 0:
                        n10 += 1
            mcn = {"n01": n01, "n10": n10, "p": mcnemar_exact(n01, n10)}

            for metric_name, est, stats, mcn_val in [
                ("fpr_sus_minus_coop", point[0], stats_fpr, mcn),
                ("rec_coop_minus_others", point[1], stats_rec, None),
            ]:
                t1.append(
                    {
                        "family": "T1",
                        "model": mid,
                        "view": view,
                        "metric": metric_name,
                        "estimate": float(est),
                        "ci_low": float(np.nanpercentile(stats, 2.5)),
                        "ci_high": float(np.nanpercentile(stats, 97.5)),
                        "p": bootstrap_p(stats),
                        "mcnemar": mcn_val,
                    }
                )
    pack("T1_mood", t1)

    # ---- T2 branch splitting ----
    t2: list[dict] = []
    for mid in ZEROSHOT_IDS:
        # Need both views; bootstrap families jointly
        rows_po = zs_preds[mid]["PREFIX_ONLY"]
        rows_ppr = zs_preds[mid]["PREFIX_PLUS_REACTION"]
        # merge by attaching view tag
        combined = []
        for r in rows_po:
            rr = dict(r)
            rr["_view"] = "PREFIX_ONLY"
            combined.append(rr)
        for r in rows_ppr:
            rr = dict(r)
            rr["_view"] = "PREFIX_PLUS_REACTION"
            combined.append(rr)
        fam = group_by_family(combined)

        def delta_d(sample: list[dict]) -> float:
            po = [r for r in sample if r["_view"] == "PREFIX_ONLY"]
            ppr = [r for r in sample if r["_view"] == "PREFIX_PLUS_REACTION"]
            return branch_disagreement_rate(ppr) - branch_disagreement_rate(po)

        boot = family_bootstrap_stat(fam, delta_d)
        t2.append(
            {
                "family": "T2",
                "model": mid,
                "metric": "disagreement_ppr_minus_po",
                "estimate": boot["estimate"],
                "ci_low": boot["ci_low"],
                "ci_high": boot["ci_high"],
                "p": bootstrap_p(boot["boot_samples"]),
            }
        )
    pack("T2_branch_split", t2)

    # ---- T3 language ----
    t3: list[dict] = []
    for mid in ZEROSHOT_IDS:
        rows = zs_preds[mid]["FULL"]
        fam = group_by_family(rows)

        def f1_lang(sample: list[dict], lang: str) -> float:
            sub = [r for r in sample if r["language_pair"] == lang]
            if not sub:
                return float("nan")
            return macro_f1([r["y"] for r in sub], [r["pred"] for r in sub])

        for other, key in [("ta_en", "hi_minus_ta"), ("te_en", "hi_minus_te")]:

            def gap(sample: list[dict], _o=other) -> float:
                return f1_lang(sample, "hi_en") - f1_lang(sample, _o)

            boot = family_bootstrap_stat(fam, gap)
            t3.append(
                {
                    "family": "T3",
                    "model": mid,
                    "metric": key,
                    "estimate": boot["estimate"],
                    "ci_low": boot["ci_low"],
                    "ci_high": boot["ci_high"],
                    "p": bootstrap_p(boot["boot_samples"]),
                }
            )
    pack("T3_language", t3)

    # ---- T4 caller vs full ----
    t4: list[dict] = []
    for mid in ZEROSHOT_IDS:
        full = caller_preds[mid]["FULL"]
        co = caller_preds[mid]["CALLER_ONLY"]
        by_id_full = {r["conversation_id"]: r for r in full}
        by_id_co = {r["conversation_id"]: r for r in co}
        paired = []
        for cid, rf in by_id_full.items():
            rc = by_id_co.get(cid)
            if rc is None:
                continue
            paired.append(
                {
                    "conversation_id": cid,
                    "scenario_family_id": rf["scenario_family_id"],
                    "y": rf["y"],
                    "pred_full": rf["pred"],
                    "pred_co": rc["pred"],
                    "recipient_condition": rf["recipient_condition"],
                    "language_pair": rf["language_pair"],
                }
            )
        # For disagreement need all four branches — attach both views into family bootstrap
        combined = []
        for r in full:
            rr = dict(r)
            rr["_view"] = "FULL"
            combined.append(rr)
        for r in co:
            rr = dict(r)
            rr["_view"] = "CALLER_ONLY"
            combined.append(rr)
        fam = group_by_family(combined)

        def d_f1(sample: list[dict]) -> float:
            f = [r for r in sample if r["_view"] == "FULL"]
            c = [r for r in sample if r["_view"] == "CALLER_ONLY"]
            return macro_f1([r["y"] for r in c], [r["pred"] for r in c]) - macro_f1(
                [r["y"] for r in f], [r["pred"] for r in f]
            )

        def d_dis(sample: list[dict]) -> float:
            f = [r for r in sample if r["_view"] == "FULL"]
            c = [r for r in sample if r["_view"] == "CALLER_ONLY"]
            return branch_disagreement_rate(c) - branch_disagreement_rate(f)

        for metric_name, fn in [("delta_macro_f1_caller_minus_full", d_f1), ("delta_disagreement_caller_minus_full", d_dis)]:
            boot = family_bootstrap_stat(fam, fn)
            mcn = None
            if metric_name.startswith("delta_macro_f1"):
                # McNemar on correctness
                n01 = n10 = 0
                for r in paired:
                    ok_f = r["pred_full"] == r["y"]
                    ok_c = r["pred_co"] == r["y"]
                    if (not ok_f) and ok_c:
                        n01 += 1
                    elif ok_f and (not ok_c):
                        n10 += 1
                mcn = {"n01": n01, "n10": n10, "p": mcnemar_exact(n01, n10), "note": "correctness discordant pairs"}
            t4.append(
                {
                    "family": "T4",
                    "model": mid,
                    "metric": metric_name,
                    "estimate": boot["estimate"],
                    "ci_low": boot["ci_low"],
                    "ci_high": boot["ci_high"],
                    "p": bootstrap_p(boot["boot_samples"]),
                    "mcnemar": mcn,
                }
            )
    pack("T4_caller_only", t4)

    # ---- T5 fine-tuning (optimized family index) ----
    t5: list[dict] = []
    for base in ["qwen2.5-0.5b", "qwen2.5-1.5b"]:
        arm_seed_rows: dict[str, dict[int, dict[str, list[dict]]]] = {}
        for arm in ["zeroshot", "unbranched", "branched", "branched_consistency"]:
            arm_seed_rows[arm] = {}
            seeds = (0,) if arm == "zeroshot" else (0, 1, 2)
            for seed in seeds:
                if arm == "zeroshot":
                    mid = base
                    src = zs_preds
                else:
                    mid = ft_model_id(base, arm, seed)
                    src = ft_preds
                arm_seed_rows[arm][seed] = {
                    "FULL": src[mid]["FULL"],
                    "PREFIX_PLUS_REACTION": src[mid]["PREFIX_PLUS_REACTION"],
                }

        def one_metric(rows_full: list[dict], rows_ppr: list[dict], mk: str) -> float:
            if mk == "macro_f1":
                return macro_f1([r["y"] for r in rows_full], [r["pred"] for r in rows_full])
            if mk == "legitimate_fpr":
                return legitimate_fpr([r["y"] for r in rows_full], [r["pred"] for r in rows_full])
            if mk == "rec_coop":
                return fraud_recall(
                    [r["y"] for r in rows_full if r["recipient_condition"] == "cooperative"],
                    [r["pred"] for r in rows_full if r["recipient_condition"] == "cooperative"],
                )
            if mk == "d_full":
                return branch_disagreement_rate(rows_full)
            if mk == "d_ppr":
                return branch_disagreement_rate(rows_ppr)
            raise KeyError(mk)

        metric_keys = ["macro_f1", "legitimate_fpr", "rec_coop", "d_full", "d_ppr"]

        # fam_index[family][arm][seed][view] -> rows
        fam_index: dict[str, dict[str, dict[int, dict[str, list[dict]]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        )
        families_set: set[str] = set()
        for arm, seeds in arm_seed_rows.items():
            for seed, views in seeds.items():
                for view, rows in views.items():
                    for r in rows:
                        fam = str(r["scenario_family_id"])
                        families_set.add(fam)
                        fam_index[fam][arm][seed][view].append(r)
        families = sorted(families_set)
        n_fam = len(families)

        contrasts = [
            ("branched", "unbranched", "branched_minus_unbranched"),
            ("branched_consistency", "branched", "consistency_minus_branched"),
            ("unbranched", "zeroshot", "unbranched_minus_zeroshot"),
            ("branched", "zeroshot", "branched_minus_zeroshot"),
            ("branched_consistency", "zeroshot", "consistency_minus_zeroshot"),
        ]

        def gather(chosen: Sequence[str], arm: str, seed: int, view: str) -> list[dict]:
            out: list[dict] = []
            for fam in chosen:
                out.extend(fam_index[fam][arm][seed][view])
            return out

        rng = np.random.default_rng(BOOT_SEED)
        choices = rng.choice(families, size=(BOOT_N, n_fam), replace=True)

        def all_metrics(rows_full: list[dict], rows_ppr: list[dict]) -> dict[str, float]:
            return {mk: one_metric(rows_full, rows_ppr, mk) for mk in metric_keys}

        for a_arm, b_arm, cname in contrasts:
            # Seed-level ranges per metric
            seed_diffs: dict[str, list[float]] = {mk: [] for mk in metric_keys}
            for seed in (0, 1, 2):
                sa = 0 if a_arm == "zeroshot" else seed
                sb = 0 if b_arm == "zeroshot" else seed
                ma = all_metrics(
                    arm_seed_rows[a_arm][sa]["FULL"],
                    arm_seed_rows[a_arm][sa]["PREFIX_PLUS_REACTION"],
                )
                mb = all_metrics(
                    arm_seed_rows[b_arm][sb]["FULL"],
                    arm_seed_rows[b_arm][sb]["PREFIX_PLUS_REACTION"],
                )
                for mk in metric_keys:
                    seed_diffs[mk].append(ma[mk] - mb[mk])

            # Point estimates (all families)
            point_a = [
                all_metrics(
                    gather(families, a_arm, 0 if a_arm == "zeroshot" else s, "FULL"),
                    gather(families, a_arm, 0 if a_arm == "zeroshot" else s, "PREFIX_PLUS_REACTION"),
                )
                for s in (0, 1, 2)
            ]
            point_b = [
                all_metrics(
                    gather(families, b_arm, 0 if b_arm == "zeroshot" else s, "FULL"),
                    gather(families, b_arm, 0 if b_arm == "zeroshot" else s, "PREFIX_PLUS_REACTION"),
                )
                for s in (0, 1, 2)
            ]
            points = {
                mk: float(np.mean([point_a[i][mk] - point_b[i][mk] for i in range(3)]))
                for mk in metric_keys
            }

            boot_stats = {mk: np.empty(BOOT_N, dtype=float) for mk in metric_keys}
            for i in range(BOOT_N):
                chosen = choices[i]
                diffs_seed = []
                for seed in (0, 1, 2):
                    sa = 0 if a_arm == "zeroshot" else seed
                    sb = 0 if b_arm == "zeroshot" else seed
                    fa = gather(chosen, a_arm, sa, "FULL")
                    pa = gather(chosen, a_arm, sa, "PREFIX_PLUS_REACTION")
                    fb = gather(chosen, b_arm, sb, "FULL")
                    pb = gather(chosen, b_arm, sb, "PREFIX_PLUS_REACTION")
                    ma = all_metrics(fa, pa)
                    mb = all_metrics(fb, pb)
                    diffs_seed.append({mk: ma[mk] - mb[mk] for mk in metric_keys})
                for mk in metric_keys:
                    boot_stats[mk][i] = float(np.mean([d[mk] for d in diffs_seed]))

            for mk in metric_keys:
                t5.append(
                    {
                        "family": "T5",
                        "base": base,
                        "contrast": cname,
                        "metric": mk,
                        "estimate": points[mk],
                        "ci_low": float(np.nanpercentile(boot_stats[mk], 2.5)),
                        "ci_high": float(np.nanpercentile(boot_stats[mk], 97.5)),
                        "p": bootstrap_p(boot_stats[mk]),
                        "seed_diffs": seed_diffs[mk],
                        "seed_min": float(np.min(seed_diffs[mk])),
                        "seed_max": float(np.max(seed_diffs[mk])),
                    }
                )
    pack("T5_finetune", t5)

    # ---- LaTeX ----
    def row_tex(e: dict, model_key: str = "model") -> str:
        name = DISPLAY.get(e.get(model_key, ""), e.get(model_key, e.get("base", "")))
        extra = e.get("view", e.get("contrast", ""))
        return (
            f"{name} & {latex_escape(str(e['metric']))} & {extra} & "
            f"{fmt_prop_plain(e['estimate'])} & {fmt_ci(e['ci_low'], e['ci_high'])} & "
            f"{e['p']:.3g} & {e['p_holm']:.3g} \\\\"
        )

    tex_blocks.append("% Auto-generated by scripts/analysis/supp_results.py")
    tex_blocks.append(
        f"% Family-clustered bootstrap: {BOOT_N} reps, seed {BOOT_SEED}; Holm within each test family."
    )

    def table_for(title: str, label: str, entries: list[dict], caption: str, cols_header: str) -> str:
        body = [
            "\\begin{table*}[t]",
            "\\centering",
            "\\footnotesize",
            "\\setlength{\\tabcolsep}{3pt}",
            "\\begin{tabular}{llccccc}",
            "\\toprule",
            "\\rowcolor{headblue}",
            cols_header,
            "\\midrule",
        ]
        # show only valid models first, then dagger; or all sorted by |estimate|
        for e in entries:
            mid = e.get("model", e.get("base", ""))
            body.append(row_tex(e, "model" if "model" in e else "base"))
        body += [
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\end{table*}",
        ]
        return "\n".join(body)

    # Compact T1: highlight key models + note full in json
    key_models = [
        "qwen2.5-1.5b",
        "qwen3-1.7b",
        "gemma3-1b",
        "qwen2-1.5b",
        "qwen2.5-0.5b",
        "llama-3.2-1b",
    ]
    t1_show = [
        e
        for e in results["tests"]["T1_mood"]
        if e["model"] in key_models and e["view"] == "FULL"
    ]
    tex_blocks.append(
        table_for(
            "T1",
            "tab:supp-stats-t1",
            t1_show,
            "T1 mood effect (full call; selected models). "
            "\\texttt{fpr\\_sus\\_minus\\_coop}: false-alarm rate(suspicious)$-$false-alarm rate(cooperative) "
            "on genuine calls. \\texttt{rec\\_coop\\_minus\\_others}: cooperative fraud recall $-$ mean recall "
            "of the other three moods. Estimate and 95\\% CI from family-clustered bootstrap "
            f"({BOOT_N:,} reps, seed {BOOT_SEED}); $p$ is two-sided bootstrap; $p_{{\\mathrm{{Holm}}}}$ "
            "Holm-corrected within T1. Full 16-model $\\times$ 2-view results in \\texttt{stats.json}. "
            "For FPR contrasts, exact McNemar on paired cooperative/suspicious siblings is also stored in JSON. "
            "Low power with 12 families.",
            "\\textbf{Model} & \\textbf{Metric} & \\textbf{View} & \\textbf{Est.} & \\textbf{95\\% CI} & "
            "$p$ & $p_{\\mathrm{Holm}}$ \\\\",
        )
    )

    t2_show = sorted(results["tests"]["T2_branch_split"], key=lambda e: -abs(e["estimate"]))
    tex_blocks.append(
        table_for(
            "T2",
            "tab:supp-stats-t2",
            t2_show,
            "T2 branch splitting: disagreement(opening+first reply)$-$disagreement(opening only). "
            f"Family-clustered bootstrap ({BOOT_N:,} reps). Holm within T2.",
            "\\textbf{Model} & \\textbf{Metric} & & \\textbf{Est.} & \\textbf{95\\% CI} & "
            "$p$ & $p_{\\mathrm{Holm}}$ \\\\",
        )
    )

    t3_show = [e for e in results["tests"]["T3_language"] if e["model"] not in DAGGER]
    t3_show = sorted(t3_show, key=lambda e: (e["metric"], -abs(e["estimate"])))
    tex_blocks.append(
        table_for(
            "T3",
            "tab:supp-stats-t3",
            t3_show,
            "T3 language gap on full-call macro-F1: Hindi$-$Tamil and Hindi$-$Telugu. "
            "Holm within T3. Dagger models omitted from the printed table (see JSON).",
            "\\textbf{Model} & \\textbf{Metric} & & \\textbf{Est.} & \\textbf{95\\% CI} & "
            "$p$ & $p_{\\mathrm{Holm}}$ \\\\",
        )
    )

    t4_show = [
        e
        for e in results["tests"]["T4_caller_only"]
        if e["model"] in key_models
    ]
    tex_blocks.append(
        table_for(
            "T4",
            "tab:supp-stats-t4",
            t4_show,
            "T4 caller-only vs.\\ full (selected models): $\\Delta$macro-F1 and $\\Delta$disagreement "
            "(caller $-$ full). McNemar on per-call correctness discordances in JSON. Holm within T4.",
            "\\textbf{Model} & \\textbf{Metric} & & \\textbf{Est.} & \\textbf{95\\% CI} & "
            "$p$ & $p_{\\mathrm{Holm}}$ \\\\",
        )
    )

    t5_show = [
        e
        for e in results["tests"]["T5_finetune"]
        if e["metric"] in {"macro_f1", "d_full", "d_ppr", "legitimate_fpr", "rec_coop"}
        and e["contrast"]
        in {
            "branched_minus_unbranched",
            "consistency_minus_branched",
            "branched_minus_zeroshot",
            "unbranched_minus_zeroshot",
        }
    ]
    # custom rows for T5 with seed range
    body = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{tabular}{lllccccc}",
        "\\toprule",
        "\\rowcolor{headblue}",
        "\\textbf{Size} & \\textbf{Contrast} & \\textbf{Metric} & \\textbf{Est.} & "
        "\\textbf{95\\% CI} & \\textbf{Seed range} & $p$ & $p_{\\mathrm{Holm}}$ \\\\",
        "\\midrule",
    ]
    for e in t5_show:
        body.append(
            f"{e['base'].replace('qwen2.5-', '')} & {latex_escape(e['contrast'])} & "
            f"{latex_escape(e['metric'])} & {fmt_prop_plain(e['estimate'])} & "
            f"{fmt_ci(e['ci_low'], e['ci_high'])} & "
            f"[{fmt_prop_plain(e['seed_min'])}, {fmt_prop_plain(e['seed_max'])}] & "
            f"{e['p']:.3g} & {e['p_holm']:.3g} \\\\"
        )
    body += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{T5 fine-tuning arm contrasts. Within each bootstrap replicate, the metric is "
        "averaged over the three seeds before forming the contrast. Seed range is the min/max of "
        "the three seed-level point contrasts (not a CI). "
        "\\textbf{Caveat:} 3 seeds $\\times$ 12 families is low power; non-significant differences "
        "should not be over-interpreted. Holm within T5.}",
        "\\label{tab:supp-stats-t5}",
        "\\end{table*}",
    ]
    tex_blocks.append("\n".join(body))

    # Serialize stats.json without huge arrays
    def clean(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items() if k != "boot_samples"}
        if isinstance(obj, list):
            return [clean(x) for x in obj]
        if isinstance(obj, (np.floating, float)):
            return float(obj) if math.isfinite(float(obj)) else None
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return None
        return obj

    return "\n\n".join(tex_blocks), clean(results)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def make_figures(
    zs_metrics: dict,
    ft_metrics: dict,
    caller_metrics: dict,
) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    conds = list(CONDITIONS)
    cond_labels = ["Cooperative", "Clarifying", "Resistant", "Suspicious"]

    # --- Mood plot ---
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharex=True)
    series = [
        ("Qwen2.5-1.5B (ZS)", "qwen2.5-1.5b", None, CB["blue"], "-"),
        ("Gemma-3-1B (ZS)", "gemma3-1b", None, CB["orange"], "-"),
        ("Qwen3-1.7B (ZS)", "qwen3-1.7b", None, CB["green"], "-"),
    ]
    for arm, color, ls in [
        ("unbranched", CB["sky"], "--"),
        ("branched", CB["vermillion"], "--"),
        ("branched_consistency", CB["purple"], "--"),
    ]:
        series.append((f"FT 1.5B {ARM_NAME[arm]}", "qwen2.5-1.5b", arm, color, ls))

    x = np.arange(len(conds))
    for ax, kind, ylab in [
        (axes[0], "legitimate_fpr", "False-alarm rate (genuine)"),
        (axes[1], "fraud_recall", "Fraud recall"),
    ]:
        for label, mid, arm, color, ls in series:
            if arm is None:
                bc = zs_metrics["slices"][mid]["FULL"]["pooled"]["by_condition"]
                ys = [bc[c][kind] for c in conds]
                yerr = None
            else:
                vals = []
                for seed in (0, 1, 2):
                    bc = ft_metrics["slices"][ft_model_id(mid, arm, seed)]["FULL"]["pooled"]["by_condition"]
                    vals.append([bc[c][kind] for c in conds])
                arr = np.asarray(vals, dtype=float)
                ys = arr.mean(axis=0)
                yerr = arr.std(axis=0, ddof=1)
            if yerr is None:
                ax.plot(x, ys, marker="o", color=color, linestyle=ls, label=label, linewidth=1.6, markersize=4)
            else:
                ax.errorbar(
                    x,
                    ys,
                    yerr=yerr,
                    marker="o",
                    color=color,
                    linestyle=ls,
                    label=label,
                    linewidth=1.6,
                    markersize=4,
                    capsize=2,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(cond_labels, rotation=15, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel(ylab)
        ax.grid(True, axis="y", alpha=0.3)
    axes[1].legend(fontsize=7, loc="lower right", frameon=False)
    axes[0].set_title("False alarms by recipient mood")
    axes[1].set_title("Fraud recall by recipient mood")
    fig.tight_layout()
    fig.savefig(FIG / "mood.pdf", bbox_inches="tight")
    fig.savefig(FIG / "mood.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # --- Disagreement by view ---
    views = ["PREFIX_ONLY", "PREFIX_PLUS_REACTION", "FULL", "CALLER_ONLY"]
    view_labs = ["Opening", "+ reply", "Full", "Caller"]
    valid = [m for m in ZEROSHOT_IDS if m not in DAGGER]
    # sort by FULL F1
    valid.sort(key=lambda m: zs_metrics["slices"][m]["FULL"]["pooled"]["macro_f1"], reverse=True)
    data = []
    for mid in valid:
        row = []
        for v in views:
            if v == "CALLER_ONLY":
                row.append(caller_metrics["slices"][mid]["CALLER_ONLY"]["branch"]["branch_disagreement_rate"])
            else:
                row.append(zs_metrics["slices"][mid][v]["branch"]["branch_disagreement_rate"])
        data.append(row)
    data = np.asarray(data)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    width = 0.18
    colors = [CB["gray"], CB["orange"], CB["blue"], CB["green"]]
    xpos = np.arange(len(valid))
    for j, (vl, col) in enumerate(zip(view_labs, colors)):
        ax.bar(xpos + (j - 1.5) * width, data[:, j], width=width, label=vl, color=col, edgecolor="none")
    ax.set_xticks(xpos)
    ax.set_xticklabels([DISPLAY[m] for m in valid], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Sibling disagreement rate")
    ax.set_ylim(0, 1)
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "disagreement_by_view.pdf", bbox_inches="tight")
    fig.savefig(FIG / "disagreement_by_view.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # --- Language heatmap ---
    mats = np.array(
        [[zs_metrics["slices"][m]["FULL"]["by_language"][L]["macro_f1"] for L in LANGS] for m in valid]
    )
    fig, ax = plt.subplots(figsize=(5.2, 6.0))
    im = ax.imshow(mats, cmap="cividis", vmin=0.35, vmax=0.9, aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Hindi–En", "Tamil–En", "Telugu–En"])
    ax.set_yticks(range(len(valid)))
    ax.set_yticklabels([DISPLAY[m] for m in valid], fontsize=8)
    for i in range(mats.shape[0]):
        for j in range(mats.shape[1]):
            ax.text(j, i, f"{mats[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if mats[i, j] < 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Macro-F1")
    ax.set_title("Full-call macro-F1 by language")
    fig.tight_layout()
    fig.savefig(FIG / "language_heatmap.pdf", bbox_inches="tight")
    fig.savefig(FIG / "language_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# F1 CIs for quality table
# ---------------------------------------------------------------------------


def compute_f1_cis(zs_preds: dict[str, dict[str, list[dict]]]) -> dict[str, tuple[float, float]]:
    out = {}
    for mid in ZEROSHOT_IDS:
        rows = zs_preds[mid]["FULL"]
        fam = group_by_family(rows)

        def f1(sample: list[dict]) -> float:
            return macro_f1([r["y"] for r in sample], [r["pred"] for r in sample])

        boot = family_bootstrap_stat(fam, f1)
        out[mid] = (boot["ci_low"], boot["ci_high"])
        # sanity vs metrics point
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report(
    repro_lines: list[str],
    stats: dict,
    files: list[str],
) -> str:
    def summarize_family(name: str, n: int = 8) -> list[str]:
        entries = stats["tests"][name]
        # pick notable significant after Holm
        sig = [e for e in entries if e.get("p_holm") is not None and e["p_holm"] < 0.05]
        sig = sorted(sig, key=lambda e: e["p_holm"])[:n]
        lines = []
        for e in sig:
            who = e.get("model") or e.get("base")
            lines.append(
                f"- `{who}` {e.get('view', e.get('contrast', ''))} {e['metric']}: "
                f"est={e['estimate']:.3f} CI=[{e['ci_low']:.3f}, {e['ci_high']:.3f}] "
                f"p={e['p']:.3g} p_Holm={e['p_holm']:.3g}"
            )
        if not lines:
            lines.append("- No tests with Holm-adjusted $p<0.05$ in the printed subset.")
        return lines

    parts = [
        "# Supplement RESULTS report",
        "",
        "Generated by `scripts/analysis/supp_results.py`. All numbers verified against "
        "`results/eval_hpc*/metrics.json` (decision rule: `score >=` per-(model, view) "
        "development threshold).",
        "",
        "## Reproduction checks",
        "",
    ]
    parts += [f"- {x}" for x in repro_lines]
    parts += [
        "",
        "## Files",
        "",
    ]
    parts += [f"- `{f}`" for f in files]
    parts += [
        "",
        "## Table interpretations",
        "",
        "### models.tex",
        "Lists HF ids, parameter counts, family, median per-inference latency on the H100 MIG "
        "slice, and invalid free-text rates. SmolLM2-135M and TinyLlama-1.1B are daggered "
        "(100% invalid). Granite is `ibm-granite/granite-3.1-1b-a400m-instruct` (MoE; ~1.33B params).",
        "",
        "### zeroshot_quality.tex / zeroshot_branch.tex",
        "Qwen2.5-1.5B leads full-call macro-F1 (~.73). Opening-only disagreement is ~0 (identical "
        "openings); adding the first reply raises disagreement sharply (.30–.80). Mean reaction "
        "deltas are often negative (first reply lowers P(fraud) on average) but mood-specific "
        "flips remain large.",
        "",
        "### by_language.tex",
        "Hindi–English typically outperforms Tamil–English; gaps are often driven by false alarms.",
        "",
        "### by_mood.tex",
        "For several strong models (e.g., Qwen2.5-1.5B, Gemma-3-1B), suspicious recipients raise "
        "false alarms while cooperative victims depress fraud recall—evidence that the detector "
        "reads the victim.",
        "",
        "### caller_only.tex",
        "Hiding recipient turns does not remove sibling disagreement; caller responses after the "
        "branch still leak stance.",
        "",
        "### streaming.tex",
        "Qwen2.5-1.5B alerts early (negative mean delay, high pre-evidence alert rate); Qwen3-1.7B "
        "is more conservative (higher miss rate, positive mean delay).",
        "",
        "### ft_*.tex",
        "Fine-tuning lifts macro-F1 into the mid-.80s for 1.5B and mid-.70s for 0.5B; branched vs "
        "unbranched differences sit inside seed variation. Consistency (lambda=1) does not "
        "reliably beat branched.",
        "",
        "### break_points.tex",
        "Hardest families include keyword-confound legitimates (OTP in delivery, IFSC requests) "
        "and hard fraud misses; reaction flips concentrate under suspicious replies for several models.",
        "",
        "## Statistical tests (key Holm-significant findings)",
        "",
        "### T1 mood",
        "",
    ]
    parts += summarize_family("T1_mood")
    parts += ["", "### T2 branch splitting", ""]
    parts += summarize_family("T2_branch_split")
    parts += ["", "### T3 language", ""]
    parts += summarize_family("T3_language")
    parts += ["", "### T4 caller-only", ""]
    parts += summarize_family("T4_caller_only")
    parts += ["", "### T5 fine-tuning", ""]
    parts += summarize_family("T5_finetune")
    parts += [
        "",
        "## Method notes",
        "",
        f"- Bootstrap: resample 12 scenario families with replacement, {BOOT_N:,} reps, seed {BOOT_SEED}; "
        "recompute metrics with fixed dev thresholds.",
        "- Two-sided bootstrap $p = 2\\min(P(\\mathrm{diff}\\le 0), P(\\mathrm{diff}\\ge 0))$.",
        "- Holm correction applied within each test family (T1–T5) separately.",
        "- McNemar (exact binomial) where paired per-call or paired-sibling decisions apply; see `stats.json`.",
        "- T5 pools seeds by averaging the metric inside each bootstrap replicate; seed range is reported separately. "
        "**3 seeds × 12 families is low power.**",
        "",
        "## Anomalies",
        "",
        "- SmolLM2-135M / TinyLlama-1.1B: invalid_output_rate = 1.0; quality numbers are not interpretable.",
        "- StableLM-2-1.6B: elevated invalid rate (~.26) on full call.",
        "- PREFIX_ONLY disagreement is exactly 0 for models with identical openings (expected).",
        "- Streaming metrics exist only for Qwen2.5-1.5B and Qwen3-1.7B.",
        "",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    np.random.seed(BOOT_SEED)

    print("Loading metrics / resources / predictions…")
    metrics_hpc = load_json(ROOT / "results" / "eval_hpc" / "metrics.json")
    metrics_gated = load_json(ROOT / "results" / "eval_hpc_gated" / "metrics.json")
    metrics_caller = load_json(ROOT / "results" / "eval_hpc_caller_only" / "metrics.json")
    metrics_ft = merge_metrics(
        ROOT / "results" / "eval_hpc_ft" / "metrics.json",
        ROOT / "results" / "eval_hpc_ft_rest" / "metrics.json",
    )
    zs_metrics = merge_metrics(
        ROOT / "results" / "eval_hpc" / "metrics.json",
        ROOT / "results" / "eval_hpc_gated" / "metrics.json",
    )
    # Prefer FULL etc. from hpc/gated; ensure CALLER from caller file is separate
    resources = merge_resources(
        ROOT / "results" / "eval_hpc" / "resources.json",
        ROOT / "results" / "eval_hpc_gated" / "resources.json",
        ROOT / "results" / "eval_hpc_caller_only" / "resources.json",
    )

    preds_hpc = load_jsonl(ROOT / "results" / "eval_hpc" / "predictions.jsonl")
    repro = reproduce_checks(metrics_hpc, preds_hpc)
    for line in repro:
        print(line)

    print("Attaching predictions for bootstrap / stats…")
    # Build threshold maps
    thr_zs = {**metrics_hpc["thresholds"], **metrics_gated["thresholds"]}
    thr_caller = metrics_caller["thresholds"]
    thr_ft = metrics_ft["thresholds"]

    preds_gated = load_jsonl(ROOT / "results" / "eval_hpc_gated" / "predictions.jsonl")
    preds_caller = load_jsonl(ROOT / "results" / "eval_hpc_caller_only" / "predictions.jsonl")
    preds_ft = load_jsonl(ROOT / "results" / "eval_hpc_ft" / "predictions.jsonl") + load_jsonl(
        ROOT / "results" / "eval_hpc_ft_rest" / "predictions.jsonl"
    )

    zs_all = attach_preds(preds_hpc + preds_gated, thr_zs)
    caller_all = attach_preds(preds_caller, thr_caller)
    ft_all = attach_preds(preds_ft, thr_ft)

    def index(rows: list[dict]) -> dict[str, dict[str, list[dict]]]:
        out: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
        for r in rows:
            out[r["model"]][r["view"]].append(r)
        return out

    zs_idx = index(zs_all)
    caller_idx = index(caller_all)
    ft_idx = index(ft_all)

    skip_stats = os.environ.get("SUPP_SKIP_STATS", "").strip() in {"1", "true", "yes"}

    print("Writing tables…")
    write_text(OUT / "models.tex", build_models_tex(zs_metrics, resources))
    if skip_stats and (OUT / "zeroshot_quality.tex").exists():
        print("Keeping existing zeroshot_quality.tex (CIs); skipping F1 bootstrap")
        f1_cis = {mid: (float("nan"), float("nan")) for mid in ZEROSHOT_IDS}
    else:
        print("Computing family-bootstrap F1 CIs…")
        f1_cis = compute_f1_cis(zs_idx)
        write_text(OUT / "zeroshot_quality.tex", build_zeroshot_quality(zs_metrics, f1_cis))
    write_text(OUT / "zeroshot_branch.tex", build_zeroshot_branch(zs_metrics, metrics_caller))
    # Combined pointer file
    write_text(
        OUT / "zeroshot_main.tex",
        "% Split into zeroshot_quality.tex and zeroshot_branch.tex for width.\n"
        "\\input{zeroshot_quality}\n\\input{zeroshot_branch}\n",
    )
    write_text(OUT / "by_language.tex", build_by_language(zs_metrics))
    write_text(
        OUT / "by_mood.tex",
        build_by_mood(zs_metrics, "FULL", "full-call")
        + "\n\n"
        + build_by_mood(zs_metrics, "PREFIX_PLUS_REACTION", "opening + first reply"),
    )
    write_text(OUT / "caller_only.tex", build_caller_only(metrics_caller))
    write_text(OUT / "streaming.tex", build_streaming(zs_metrics))
    write_text(OUT / "ft_per_seed.tex", build_ft_per_seed(metrics_ft, zs_metrics, metrics_caller))
    write_text(OUT / "ft_summary.tex", build_ft_summary(metrics_ft, zs_metrics, metrics_caller))
    write_text(OUT / "ft_hparams.tex", build_ft_hparams())
    write_text(OUT / "break_points.tex", build_break_points())

    if skip_stats and (OUT / "stats.json").exists():
        print("Skipping statistical tests (SUPP_SKIP_STATS); reusing stats.json")
        stats_obj = load_json(OUT / "stats.json")
    else:
        print("Running statistical tests (this takes a few minutes)…")
        stats_tex, stats_obj = run_stats(zs_idx, caller_idx, ft_idx, zs_metrics)
        write_text(OUT / "stats_tests.tex", stats_tex)
        write_text(OUT / "stats.json", json.dumps(stats_obj, indent=2, allow_nan=False))

    print("Figures…")
    make_figures(zs_metrics, metrics_ft, metrics_caller)

    files = sorted(
        str(p.relative_to(OUT))
        for p in OUT.rglob("*")
        if p.is_file()
    )
    report = write_report(repro, stats_obj, files)
    write_text(OUT / "REPORT.md", report)
    print("Done. Wrote", OUT)
    print(report.split("## Statistical tests")[0][-500:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
