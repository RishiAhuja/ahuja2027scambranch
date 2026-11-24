#!/usr/bin/env python3
"""Supplement dataset statistics for AAAI-27 student abstract.

Reuses generator English-share tagging from scambranch.checks / render.local_lexicon
(WORD_RE + wordfreq zipf>=3.5 -> en; else local_lexicon -> local; else untagged).

Writes results/supplement/dataset/{stats.json,REPORT.md,tables/*.tex,figures/*.pdf}.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from scambranch.checks import WORD_RE, _zipf_en  # noqa: E402
from scambranch.render import load_profile, local_lexicon  # noqa: E402

PAIR_NAMES = {
    "hi_en": "Hindi--English",
    "ta_en": "Tamil--English",
    "te_en": "Telugu--English",
}
COND_NAMES = {
    "cooperative": "Cooperative",
    "clarifying": "Clarifying",
    "resistant": "Resistant",
    "suspicious": "Suspicious",
}
CONDS = ["cooperative", "clarifying", "resistant", "suspicious"]
LABEL_NAMES = {"fraud_attempt": "Fraud", "legitimate": "Genuine"}
PROBE_NAMES = {
    "opening": "Opening only",
    "first_reply": "First reply",
    "recipient_turns": "Recipient turns",
    "caller_turns": "Caller turns",
    "full_call": "Full call",
    "condition_first_reply": "Condition from first reply",
}
SEED = 42
BOOT_REPS = 2000
TTR_SAMPLE_TOKENS = 10_000
TTR_N_SAMPLES = 20
MATTR_WINDOW = 50
ZIPF_MAX_RANK = 1000

# Colorblind-safe Okabe–Ito-ish
COLORS = {
    "hi_en": "#0072B2",
    "ta_en": "#D55E00",
    "te_en": "#009E73",
    "synthetic": "#0072B2",
    "reference": "#D55E00",
    "cooperative": "#0072B2",
    "clarifying": "#E69F00",
    "resistant": "#009E73",
    "suspicious": "#CC79A7",
}


# ---------------------------------------------------------------------------
# Tagging (exact generator method)
# ---------------------------------------------------------------------------


def tag_word(tok: str, lexicon: set[str]) -> str:
    """Return 'en', 'local', or 'unk' using generator checks.py rule."""
    low = tok.lower()
    if _zipf_en(low) >= 3.5:
        return "en"
    if low in lexicon:
        return "local"
    return "unk"


def tokenize_words(text: str) -> list[str]:
    return WORD_RE.findall(text or "")


def tag_tokens(tokens: list[str], lexicon: set[str]) -> list[str]:
    return [tag_word(t, lexicon) for t in tokens]


def mix_metrics_from_tags(tags: list[str]) -> dict[str, Any]:
    """Utterance-level mixing metrics (Gambäck & Das 2016 CMI, 2-lang form).

    CMI = 100 * (1 - max(n_lang) / (n - n_u)), with n_u = untagged ('unk').
    English share = n_en / (n_en + n_local). Switches count adjacent tagged-label changes.
    """
    n_en = sum(1 for t in tags if t == "en")
    n_loc = sum(1 for t in tags if t == "local")
    n_u = sum(1 for t in tags if t == "unk")
    n = len(tags)
    den = n_en + n_loc  # == n - n_u among {en,local,unk}
    switches = 0
    prev = None
    for t in tags:
        if t == "unk":
            continue
        if prev is not None and t != prev:
            switches += 1
        prev = t
    if den == 0:
        return {
            "english_share": None,
            "cmi": None,
            "n_switches": switches,
            "switches_per_word": None,
            "monolingual": True,
            "defined": False,
            "n_tokens": n,
            "n_tagged": 0,
            "n_unk": n_u,
        }
    return {
        "english_share": n_en / den,
        "cmi": 100.0 * (1.0 - (max(n_en, n_loc) / den)),
        "n_switches": switches,
        "switches_per_word": switches / den,
        "monolingual": n_en == 0 or n_loc == 0,
        "defined": True,
        "n_tokens": n,
        "n_tagged": den,
        "n_unk": n_u,
    }


def english_share_turns(turns: list[dict[str, Any]], lexicon: set[str]) -> float:
    """Match scambranch.checks.english_share exactly."""
    n_en = n_local = 0
    for turn in turns:
        for tok in tokenize_words(turn.get("text") or ""):
            tag = tag_word(tok, lexicon)
            if tag == "en":
                n_en += 1
            elif tag == "local":
                n_local += 1
    denom = n_en + n_local
    return 0.0 if denom == 0 else n_en / denom


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def summarize(xs: Iterable[float | None]) -> dict[str, Any]:
    arr = np.asarray([float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))], dtype=float)
    if arr.size == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "iqr": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    q = np.percentile(arr, [10, 25, 50, 75, 90])
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(q[2]),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "iqr": float(q[3] - q[1]),
        "p10": float(q[0]),
        "p25": float(q[1]),
        "p75": float(q[3]),
        "p90": float(q[4]),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def pearson_spearman(x: list[float], y: list[float]) -> dict[str, Any]:
    if len(x) < 3:
        return {"pearson_r": None, "pearson_p": None, "spearman_rho": None, "spearman_p": None, "n": len(x)}
    pr = scipy_stats.pearsonr(x, y)
    sr = scipy_stats.spearmanr(x, y)
    return {
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
        "spearman_rho": float(sr.statistic),
        "spearman_p": float(sr.pvalue),
        "n": len(x),
    }


def ks_wasserstein(a: list[float], b: list[float]) -> dict[str, Any]:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.size == 0 or bb.size == 0:
        return {"ks_statistic": None, "ks_pvalue": None, "wasserstein": None, "n_a": int(aa.size), "n_b": int(bb.size)}
    ks = scipy_stats.ks_2samp(aa, bb, method="auto")
    return {
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "wasserstein": float(scipy_stats.wasserstein_distance(aa, bb)),
        "n_a": int(aa.size),
        "n_b": int(bb.size),
    }


def fmt_prop(x: float | None, digits: int = 2) -> str:
    if x is None:
        return "---"
    return f"{x:.{digits}f}"


def fmt_ci(lo: float, hi: float, digits: int = 2) -> str:
    return f"[{lo:.{digits}f}, {hi:.{digits}f}]"


def latex_escape(s: str) -> str:
    return s.replace("&", "\\&").replace("%", "\\%")


# ---------------------------------------------------------------------------
# Lexical diversity
# ---------------------------------------------------------------------------


def word_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def mattr(tokens: list[str], window: int = MATTR_WINDOW) -> float | None:
    if len(tokens) < window:
        if not tokens:
            return None
        return len(set(tokens)) / len(tokens)
    scores = []
    for i in range(0, len(tokens) - window + 1):
        w = tokens[i : i + window]
        scores.append(len(set(w)) / window)
    return float(np.mean(scores))


def ttr_fixed_samples(tokens: list[str], n_tok: int, n_samples: int, rng: np.random.Generator) -> float | None:
    if len(tokens) < n_tok:
        return None
    vals = []
    for _ in range(n_samples):
        if len(tokens) == n_tok:
            sample = tokens
        else:
            start = int(rng.integers(0, len(tokens) - n_tok + 1))
            sample = tokens[start : start + n_tok]
        vals.append(len(set(sample)) / n_tok)
    return float(np.mean(vals))


def distinct_n(tokens: list[str], n: int) -> float | None:
    grams = word_ngrams(tokens, n)
    if not grams:
        return None
    return len(set(grams)) / len(grams)


def hapax_ratio(tokens: list[str]) -> float | None:
    if not tokens:
        return None
    c = Counter(tokens)
    return sum(1 for v in c.values() if v == 1) / len(c)


def zipf_fit(tokens: list[str], max_rank: int = ZIPF_MAX_RANK) -> dict[str, Any]:
    if not tokens:
        return {"slope": None, "r2": None, "n_ranks": 0, "vocab": 0}
    counts = Counter(tokens)
    freqs = np.asarray(sorted(counts.values(), reverse=True), dtype=float)
    v = len(freqs)
    m = min(max_rank, v)
    ranks = np.arange(1, m + 1, dtype=float)
    y = np.log(freqs[:m])
    x = np.log(ranks)
    # OLS
    slope, intercept, r, _, _ = scipy_stats.linregress(x, y)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r**2),
        "n_ranks": int(m),
        "vocab": int(v),
        "freqs": freqs[:m].tolist(),
    }


def heaps_fit(tokens: list[str], n_points: int = 50) -> dict[str, Any]:
    """Fit V = K * N^beta on cumulative vocabulary growth."""
    if len(tokens) < 20:
        return {"beta": None, "k": None, "r2": None, "n_points": 0}
    seen: set[str] = set()
    ns: list[int] = []
    vs: list[int] = []
    step = max(1, len(tokens) // n_points)
    for i, tok in enumerate(tokens, start=1):
        seen.add(tok)
        if i % step == 0 or i == len(tokens):
            ns.append(i)
            vs.append(len(seen))
    x = np.log(np.asarray(ns, dtype=float))
    y = np.log(np.asarray(vs, dtype=float))
    slope, intercept, r, _, _ = scipy_stats.linregress(x, y)
    return {
        "beta": float(slope),
        "k": float(math.exp(intercept)),
        "r2": float(r**2),
        "n_points": len(ns),
        "final_N": len(tokens),
        "final_V": len(seen),
    }


def lexical_bundle(tokens: list[str], rng: np.random.Generator) -> dict[str, Any]:
    low = [t.lower() for t in tokens]
    z = zipf_fit(low)
    return {
        "n_tokens": len(low),
        "vocab_size": len(set(low)),
        "ttr_raw": (len(set(low)) / len(low)) if low else None,
        "ttr_10k_mean": ttr_fixed_samples(low, TTR_SAMPLE_TOKENS, TTR_N_SAMPLES, rng),
        "mattr_50": mattr(low, MATTR_WINDOW),
        "distinct_1": distinct_n(low, 1),
        "distinct_2": distinct_n(low, 2),
        "distinct_3": distinct_n(low, 3),
        "hapax_ratio": hapax_ratio(low),
        "zipf_slope": z["slope"],
        "zipf_r2": z["r2"],
        "zipf_n_ranks": z["n_ranks"],
        "heaps": heaps_fit(low),
        "_zipf_freqs": z.get("freqs"),
    }


# ---------------------------------------------------------------------------
# Distributional similarity
# ---------------------------------------------------------------------------


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    # base-2 JSD in [0,1]
    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def ngram_jsd(tokens_a: list[str], tokens_b: list[str], n: int) -> dict[str, Any]:
    ca = Counter(word_ngrams([t.lower() for t in tokens_a], n))
    cb = Counter(word_ngrams([t.lower() for t in tokens_b], n))
    vocab = sorted(set(ca) | set(cb))
    if not vocab:
        return {"jsd": None, "vocab": 0}
    pa = np.asarray([ca[g] for g in vocab], dtype=float)
    pb = np.asarray([cb[g] for g in vocab], dtype=float)
    return {"jsd": jensen_shannon(pa, pb), "vocab": len(vocab), "n_a": int(pa.sum()), "n_b": int(pb.sum())}


def oov_rate(synth_tokens: list[str], ref_vocab: set[str]) -> float | None:
    if not synth_tokens:
        return None
    low = [t.lower() for t in synth_tokens]
    return sum(1 for t in low if t not in ref_vocab) / len(low)


@dataclass
class CharLM:
    """Interpolated character 3–5-gram model with add-k smoothing."""

    counts: dict[int, Counter] = field(default_factory=dict)
    context_totals: dict[int, Counter] = field(default_factory=dict)
    charset: list[str] = field(default_factory=list)
    k: float = 0.01
    lambdas: tuple[float, float, float] = (0.5, 0.3, 0.2)  # 5,4,3

    def fit(self, texts: list[str]) -> None:
        self.counts = {3: Counter(), 4: Counter(), 5: Counter()}
        self.context_totals = {3: Counter(), 4: Counter(), 5: Counter()}
        chars: set[str] = set()
        for text in texts:
            s = text.replace("\n", " ")
            chars.update(s)
            for n in (3, 4, 5):
                if len(s) < n:
                    continue
                for i in range(len(s) - n + 1):
                    gram = s[i : i + n]
                    ctx = gram[:-1]
                    self.counts[n][gram] += 1
                    self.context_totals[n][ctx] += 1
        self.charset = sorted(chars) or [" "]
        self._V = len(self.charset)

    def _p_addk(self, n: int, gram: str) -> float:
        ctx = gram[:-1]
        c = self.counts[n][gram]
        tot = self.context_totals[n][ctx]
        return (c + self.k) / (tot + self.k * self._V)

    def log2prob_char(self, text: str) -> list[float]:
        s = text.replace("\n", " ")
        # pad so we can score from first char with short contexts
        pad = " " * 4
        full = pad + s
        out: list[float] = []
        uni = 1.0 / self._V
        l5, l4, l3 = self.lambdas
        l0 = max(0.0, 1.0 - (l5 + l4 + l3))
        for i in range(4, len(full)):
            c = full[i]
            # map unknown chars to uniform mass via add-k over full V (already)
            g5 = full[i - 4 : i + 1]
            g4 = full[i - 3 : i + 1]
            g3 = full[i - 2 : i + 1]
            p = l5 * self._p_addk(5, g5) + l4 * self._p_addk(4, g4) + l3 * self._p_addk(3, g3) + l0 * uni
            p = max(p, 1e-15)
            out.append(math.log2(p))
        return out

    def bits_per_char(self, texts: list[str]) -> float | None:
        logs: list[float] = []
        for t in texts:
            if not t:
                continue
            logs.extend(self.log2prob_char(t))
        if not logs:
            return None
        return float(-np.mean(logs))


# ---------------------------------------------------------------------------
# Weighted log-odds (Monroe et al. 2008)
# ---------------------------------------------------------------------------


def weighted_log_odds(
    counts_i: Counter,
    counts_j: Counter,
    prior: Counter,
    alpha0: float | None = None,
    top_k: int = 15,
) -> list[dict[str, Any]]:
    """One-vs-rest style: i vs j with informative Dirichlet prior from prior counts."""
    n_i = sum(counts_i.values())
    n_j = sum(counts_j.values())
    a0 = float(alpha0 if alpha0 is not None else sum(prior.values()) or 1.0)
    p_tot = sum(prior.values()) or 1.0
    vocab = set(counts_i) | set(counts_j) | set(prior)
    rows = []
    for w in vocab:
        yw_i = counts_i[w]
        yw_j = counts_j[w]
        aw = a0 * (prior[w] / p_tot)
        # avoid zero prior mass words dominating: skip if aw==0 and both zero
        if aw == 0 and yw_i == 0 and yw_j == 0:
            continue
        aw = max(aw, 1e-6)
        delta = math.log((yw_i + aw) / (n_i + a0 - yw_i - aw)) - math.log(
            (yw_j + aw) / (n_j + a0 - yw_j - aw)
        )
        # variance approx from Monroe
        var = 1.0 / (yw_i + aw) + 1.0 / (yw_j + aw)
        z = delta / math.sqrt(var) if var > 0 else 0.0
        rows.append({"word": w, "log_odds": delta, "z": z, "count_i": yw_i, "count_j": yw_j})
    rows.sort(key=lambda r: -r["z"])
    return rows[:top_k]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class ConvView:
    cid: str
    split: str
    pair: str
    family: str
    case_id: str
    history: str
    branch: str
    label: str  # fraud_attempt | legitimate
    condition: str
    branch_after: int
    turns: list[dict[str, Any]]
    target_en: float | None
    realized_en_stored: float | None
    lang_review: dict[str, Any] | None


def load_frozen_conversations(root: Path) -> list[ConvView]:
    frozen = json.loads((root / "runs/full_v1/FROZEN.json").read_text(encoding="utf-8"))
    ids = list(frozen["conversations"].keys())
    out: list[ConvView] = []
    for cid in ids:
        rec = json.loads((root / "runs/full_v1/conversations" / f"{cid}.json").read_text(encoding="utf-8"))
        pa = rec["private_annotations"]
        gs = (rec.get("provenance") or {}).get("generation_settings") or {}
        checks = gs.get("checks") or {}
        out.append(
            ConvView(
                cid=cid,
                split=rec["split"],
                pair=rec["language_pair"],
                family=rec["scenario_family_id"],
                case_id=rec["case_id"],
                history=rec["history_id"],
                branch=rec["branch_id"],
                label=pa["scenario_label"],
                condition=pa["recipient_condition"],
                branch_after=int(pa["branch_after_turn"]),
                turns=rec["turns"],
                target_en=safe_float(gs.get("language_target_english_share")),
                realized_en_stored=safe_float(checks.get("english_share")),
                lang_review=gs.get("reviews", {}).get("language") if isinstance(gs.get("reviews"), dict) else None,
            )
        )
    return out


def load_reference_pool(root: Path, pair: str) -> list[dict[str, Any]]:
    path = root / "data/calibration/processed" / pair / "utterances.jsonl"
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_reference_200(root: Path, pair: str) -> list[dict[str, Any]]:
    profile = load_profile(pair)
    return list(profile.get("reference_utterances") or [])


def opening_turns(conv: ConvView) -> list[dict[str, Any]]:
    return [t for t in conv.turns if int(t["turn_id"]) <= conv.branch_after]


def post_turns(conv: ConvView) -> list[dict[str, Any]]:
    return [t for t in conv.turns if int(t["turn_id"]) > conv.branch_after]


def first_recipient_reply(conv: ConvView) -> dict[str, Any] | None:
    for t in post_turns(conv):
        if t.get("speaker") == "recipient":
            return t
    return None


def text_of(turns: list[dict[str, Any]], speaker: str | None = None) -> str:
    parts = []
    for t in turns:
        if speaker is not None and t.get("speaker") != speaker:
            continue
        parts.append(t.get("text") or "")
    return " ".join(parts)


def words_of(turns: list[dict[str, Any]], speaker: str | None = None) -> list[str]:
    toks: list[str] = []
    for t in turns:
        if speaker is not None and t.get("speaker") != speaker:
            continue
        toks.extend(tokenize_words(t.get("text") or ""))
    return toks


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------


def complete_four_branch_sets(convs: list[ConvView]) -> dict[tuple, dict[str, ConvView]]:
    groups: dict[tuple, dict[str, ConvView]] = defaultdict(dict)
    for c in convs:
        key = (c.split, c.family, c.case_id, c.pair, c.history)
        groups[key][c.branch] = c
    return {k: v for k, v in groups.items() if set(CONDS).issubset(v.keys())}


def analysis_size_structure(convs: list[ConvView], complete: dict) -> dict[str, Any]:
    counts: dict[str, Any] = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
    # split x lang x label x condition
    flat_counts = []
    for c in convs:
        counts[c.split][c.pair][c.label][c.condition] += 1
        flat_counts.append(
            {
                "split": c.split,
                "language_pair": c.pair,
                "label": c.label,
                "condition": c.condition,
            }
        )

    complete_by_split = Counter(k[0] for k in complete)
    complete_by_pair = Counter(k[3] for k in complete)
    complete_by_pair_split: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for k in complete:
        complete_by_pair_split[k[3]][k[0]] += 1

    turns_per_call = [len(c.turns) for c in convs]
    words_per_call = [len(words_of(c.turns)) for c in convs]
    words_per_turn = [len(tokenize_words(t.get("text") or "")) for c in convs for t in c.turns]
    opening_turns_n = [c.branch_after for c in convs]
    opening_words = [len(words_of(opening_turns(c))) for c in convs]
    post_turns_n = [len(post_turns(c)) for c in convs]

    by_pair: dict[str, Any] = {}
    for pair in PAIR_NAMES:
        sub = [c for c in convs if c.pair == pair]
        by_pair[pair] = {
            "n": len(sub),
            "turns_per_call": summarize([len(c.turns) for c in sub]),
            "words_per_call": summarize([len(words_of(c.turns)) for c in sub]),
            "words_per_turn": summarize(
                [len(tokenize_words(t.get("text") or "")) for c in sub for t in c.turns]
            ),
            "opening_turns": summarize([c.branch_after for c in sub]),
            "opening_words": summarize([len(words_of(opening_turns(c))) for c in sub]),
            "post_branch_turns": summarize([len(post_turns(c)) for c in sub]),
        }

    # nested counts to plain dict
    counts_plain: dict[str, Any] = {}
    for split, d1 in counts.items():
        counts_plain[split] = {}
        for pair, d2 in d1.items():
            counts_plain[split][pair] = {}
            for label, d3 in d2.items():
                counts_plain[split][pair][label] = dict(d3)

    return {
        "n_usable": len(convs),
        "counts_split_lang_label_condition": counts_plain,
        "complete_four_branch_sets": {
            "n_total": len(complete),
            "by_split": dict(complete_by_split),
            "by_language_pair": dict(complete_by_pair),
            "by_language_pair_split": {p: dict(complete_by_pair_split[p]) for p in PAIR_NAMES},
        },
        "overall": {
            "turns_per_call": summarize(turns_per_call),
            "words_per_call": summarize(words_per_call),
            "words_per_turn": summarize(words_per_turn),
            "opening_turns": summarize(opening_turns_n),
            "opening_words": summarize(opening_words),
            "post_branch_turns": summarize(post_turns_n),
        },
        "by_language_pair": by_pair,
    }


def turn_mix(text: str, lexicon: set[str]) -> dict[str, Any]:
    tags = tag_tokens(tokenize_words(text), lexicon)
    return mix_metrics_from_tags(tags)


def analysis_code_mixing(
    convs: list[ConvView],
    lexicons: dict[str, set[str]],
    ref_pool: dict[str, list[dict]],
    ref200: dict[str, list[dict]],
) -> dict[str, Any]:
    per_turn: dict[str, list[float]] = defaultdict(list)
    per_call_en: dict[str, list[float]] = defaultdict(list)
    per_call_cmi_turnmean: dict[str, list[float]] = defaultdict(list)
    cmi_all_turns: dict[str, list[float]] = defaultdict(list)
    cmi_mixed_turns: dict[str, list[float]] = defaultdict(list)
    switches_pw: dict[str, list[float]] = defaultdict(list)
    mono_flags: dict[str, list[int]] = defaultdict(list)
    by_speaker_en: dict[str, dict[str, list[float]]] = {
        "caller": defaultdict(list),
        "recipient": defaultdict(list),
    }

    # corpus-level tag counts
    corpus_tags: dict[str, Counter] = defaultdict(Counter)

    target_realized = []

    for c in convs:
        lex = lexicons[c.pair]
        call_turn_cmis = []
        for t in c.turns:
            m = turn_mix(t.get("text") or "", lex)
            tags = tag_tokens(tokenize_words(t.get("text") or ""), lex)
            corpus_tags[c.pair].update(tags)
            if m["defined"] and m["english_share"] is not None:
                per_turn[c.pair].append(m["english_share"])
                by_speaker_en[t.get("speaker") or "?"][c.pair].append(m["english_share"])
            if m["defined"] and m["cmi"] is not None:
                cmi_all_turns[c.pair].append(m["cmi"])
                call_turn_cmis.append(m["cmi"])
                if m["cmi"] > 0:
                    cmi_mixed_turns[c.pair].append(m["cmi"])
            if m["defined"] and m["switches_per_word"] is not None:
                switches_pw[c.pair].append(m["switches_per_word"])
            if m["defined"]:
                mono_flags[c.pair].append(1 if m["monolingual"] else 0)
        share = english_share_turns(c.turns, lex)
        per_call_en[c.pair].append(share)
        if call_turn_cmis:
            per_call_cmi_turnmean[c.pair].append(float(np.mean(call_turn_cmis)))
        if c.target_en is not None:
            target_realized.append((c.pair, c.target_en, share, c.realized_en_stored))

    def corpus_cmi(tag_counts: Counter) -> dict[str, Any]:
        n_en = tag_counts.get("en", 0)
        n_loc = tag_counts.get("local", 0)
        n_u = tag_counts.get("unk", 0)
        den = n_en + n_loc
        if den == 0:
            return {"cmi": None, "english_share": None, "n_en": n_en, "n_local": n_loc, "n_unk": n_u}
        return {
            "cmi": 100.0 * (1.0 - max(n_en, n_loc) / den),
            "english_share": n_en / den,
            "n_en": n_en,
            "n_local": n_loc,
            "n_unk": n_u,
        }

    def ref_metrics(rows: list[dict], lexicon: set[str], use_gold: bool = False) -> dict[str, Any]:
        en_shares, cmis, cmis_mixed, sw_pw, mono = [], [], [], [], []
        tag_tot: Counter = Counter()
        for row in rows:
            text = row.get("text") or ""
            if use_gold and row.get("lang_tags") and row.get("tokens"):
                # map gold tags to en/local/unk
                local_labels = {"hi", "hin", "ta", "tam", "te", "tel", "mixed", "other"}
                tags = []
                for g in row["lang_tags"]:
                    g = (g or "").lower()
                    if g in {"en", "eng", "english"}:
                        tags.append("en")
                    elif g in local_labels or g in {"hi", "ta", "te"}:
                        tags.append("local")
                    else:
                        tags.append("unk")
            else:
                tags = tag_tokens(tokenize_words(text), lexicon)
            tag_tot.update(tags)
            m = mix_metrics_from_tags(tags)
            if m["defined"] and m["english_share"] is not None:
                en_shares.append(m["english_share"])
            if m["defined"] and m["cmi"] is not None:
                cmis.append(m["cmi"])
                if m["cmi"] > 0:
                    cmis_mixed.append(m["cmi"])
            if m["defined"] and m["switches_per_word"] is not None:
                sw_pw.append(m["switches_per_word"])
            if m["defined"]:
                mono.append(1 if m["monolingual"] else 0)
        return {
            "n_utterances": len(rows),
            "english_share": summarize(en_shares),
            "cmi_all_turns": summarize(cmis),
            "cmi_mixed_turns": summarize(cmis_mixed),
            "switches_per_word": summarize(sw_pw),
            "monolingual_turn_share": float(np.mean(mono)) if mono else None,
            "corpus_level": corpus_cmi(tag_tot),
            "english_share_values": en_shares,
            "cmi_values": cmis,
        }

    out: dict[str, Any] = {
        "method": (
            "Word tokens via scambranch.checks.WORD_RE; tag en if wordfreq zipf_frequency(w,'en')>=3.5, "
            "else local if in render.local_lexicon(profile), else untagged. "
            "English share = n_en/(n_en+n_local). "
            "CMI (Gambäck & Das 2016) = 100*(1-max(n_lang)/(n-n_u)) with n_u=untagged. "
            "Switch points = adjacent tagged-label changes; normalized by n_tagged."
        ),
        "by_language_pair": {},
        "target_vs_realized": {},
        "synthetic_vs_reference": {},
    }

    for pair in PAIR_NAMES:
        mono = mono_flags[pair]
        out["by_language_pair"][pair] = {
            "per_turn_english_share": summarize(per_turn[pair]),
            "per_call_english_share": summarize(per_call_en[pair]),
            "per_call_mean_turn_cmi": summarize(per_call_cmi_turnmean[pair]),
            "cmi_all_turns": summarize(cmi_all_turns[pair]),
            "cmi_mixed_turns_only": summarize(cmi_mixed_turns[pair]),
            "corpus_level": corpus_cmi(corpus_tags[pair]),
            "switches_per_word": summarize(switches_pw[pair]),
            "monolingual_turn_share": float(np.mean(mono)) if mono else None,
            "by_speaker_english_share": {
                sp: summarize(by_speaker_en[sp][pair]) for sp in ("caller", "recipient")
            },
        }

    # correlations overall and per pair
    tx = [t for _, t, _, _ in target_realized]
    ry = [r for _, _, r, _ in target_realized]
    out["target_vs_realized"]["overall"] = pearson_spearman(tx, ry)
    # also vs stored
    stored = [s for _, _, _, s in target_realized if s is not None]
    recomputed = [r for _, _, r, s in target_realized if s is not None]
    out["target_vs_realized"]["recomputed_vs_stored_max_abs_diff"] = (
        float(max(abs(a - b) for a, b in zip(recomputed, stored))) if stored else None
    )
    for pair in PAIR_NAMES:
        xs = [t for p, t, _, _ in target_realized if p == pair]
        ys = [r for p, _, r, _ in target_realized if p == pair]
        out["target_vs_realized"][pair] = pearson_spearman(xs, ys)

    for pair in PAIR_NAMES:
        syn_en = per_turn[pair]
        syn_cmi = cmi_all_turns[pair]
        ref_auto = ref_metrics(ref_pool[pair], lexicons[pair], use_gold=False)
        ref_gold = ref_metrics(
            [r for r in ref_pool[pair] if r.get("tag_origin") == "gold" and r.get("lang_tags")],
            lexicons[pair],
            use_gold=True,
        )
        ref200_auto = ref_metrics(ref200[pair], lexicons[pair], use_gold=False)
        out["synthetic_vs_reference"][pair] = {
            "synthetic": {
                "english_share": summarize(syn_en),
                "cmi": summarize(syn_cmi),
            },
            "reference_full_pool_auto_tag": {
                k: v
                for k, v in ref_auto.items()
                if k not in {"english_share_values", "cmi_values"}
            },
            "reference_gold_tagged_subset": {
                k: v
                for k, v in ref_gold.items()
                if k not in {"english_share_values", "cmi_values"}
            },
            "reference_200_profile_auto_tag": {
                k: v
                for k, v in ref200_auto.items()
                if k not in {"english_share_values", "cmi_values"}
            },
            "compare_vs_full_pool_auto": {
                "english_share": ks_wasserstein(syn_en, ref_auto["english_share_values"]),
                "cmi": ks_wasserstein(syn_cmi, ref_auto["cmi_values"]),
            },
            "compare_vs_200_auto": {
                "english_share": ks_wasserstein(syn_en, ref200_auto["english_share_values"]),
                "cmi": ks_wasserstein(syn_cmi, ref200_auto["cmi_values"]),
            },
        }
        # stash values for plots
        out["synthetic_vs_reference"][pair]["_plot"] = {
            "syn_en": syn_en,
            "syn_cmi": syn_cmi,
            "ref_en": ref_auto["english_share_values"],
            "ref_cmi": ref_auto["cmi_values"],
            "targets": [t for p, t, _, _ in target_realized if p == pair],
            "realized": [r for p, _, r, _ in target_realized if p == pair],
        }

    return out


def analysis_lexical_diversity(
    convs: list[ConvView],
    ref_pool: dict[str, list[dict]],
    ref200: dict[str, list[dict]],
    rng: np.random.Generator,
) -> dict[str, Any]:
    out: dict[str, Any] = {"by_language_pair": {}, "method": (
        f"Tokens via WORD_RE lowercased. TTR on contiguous {TTR_SAMPLE_TOKENS}-token windows "
        f"(mean of {TTR_N_SAMPLES} random starts). MATTR window={MATTR_WINDOW}. "
        f"Zipf: OLS log freq ~ log rank for ranks 1..min({ZIPF_MAX_RANK},V). "
        "Heaps: OLS log V ~ log N on cumulative growth."
    )}
    for pair in PAIR_NAMES:
        sub = [c for c in convs if c.pair == pair]
        all_toks: list[str] = []
        caller_toks: list[str] = []
        recip_toks: list[str] = []
        for c in sub:
            for t in c.turns:
                w = tokenize_words(t.get("text") or "")
                all_toks.extend(w)
                if t.get("speaker") == "caller":
                    caller_toks.extend(w)
                elif t.get("speaker") == "recipient":
                    recip_toks.extend(w)
        ref_toks = []
        for row in ref_pool[pair]:
            ref_toks.extend(tokenize_words(row.get("text") or ""))
        ref200_toks = []
        for row in ref200[pair]:
            ref200_toks.extend(tokenize_words(row.get("text") or ""))

        syn = lexical_bundle(all_toks, rng)
        syn_caller = lexical_bundle(caller_toks, rng)
        syn_recip = lexical_bundle(recip_toks, rng)
        ref = lexical_bundle(ref_toks, rng)
        ref2 = lexical_bundle(ref200_toks, rng)

        # strip plot helpers from nested for JSON later — keep freqs at top for plotting
        out["by_language_pair"][pair] = {
            "synthetic": {k: v for k, v in syn.items() if not k.startswith("_")},
            "synthetic_caller": {k: v for k, v in syn_caller.items() if not k.startswith("_")},
            "synthetic_recipient": {k: v for k, v in syn_recip.items() if not k.startswith("_")},
            "reference_full_pool": {k: v for k, v in ref.items() if not k.startswith("_")},
            "reference_200": {k: v for k, v in ref2.items() if not k.startswith("_")},
            "_zipf_plot": {
                "synthetic_freqs": syn.get("_zipf_freqs"),
                "reference_freqs": ref.get("_zipf_freqs"),
            },
        }
    return out


def analysis_distributional(
    convs: list[ConvView],
    ref_pool: dict[str, list[dict]],
    rng: np.random.Generator,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "caveat": (
            "Reference sources are tweets, QA, and service dialogues — not phone-call transcripts. "
            "JSD / OOV / bits-per-char are upper-bound-of-mismatch indicators, not naturalness scores."
        ),
        "by_language_pair": {},
    }
    for pair in PAIR_NAMES:
        syn_toks: list[str] = []
        syn_texts: list[str] = []
        for c in convs:
            if c.pair != pair:
                continue
            for t in c.turns:
                syn_toks.extend(tokenize_words(t.get("text") or ""))
                syn_texts.append(t.get("text") or "")
        ref_rows = ref_pool[pair]
        ref_toks: list[str] = []
        ref_texts = [r.get("text") or "" for r in ref_rows]
        for text in ref_texts:
            ref_toks.extend(tokenize_words(text))

        ref_vocab = {t.lower() for t in ref_toks}
        uni = ngram_jsd(syn_toks, ref_toks, 1)
        bi = ngram_jsd(syn_toks, ref_toks, 2)
        oov = oov_rate(syn_toks, ref_vocab)

        # hold out 10% reference utterances
        idx = np.arange(len(ref_texts))
        rng.shuffle(idx)
        n_hold = max(1, int(0.1 * len(idx)))
        hold_idx = set(idx[:n_hold].tolist())
        train_texts = [ref_texts[i] for i in idx if i not in hold_idx]
        hold_texts = [ref_texts[i] for i in idx if i in hold_idx]
        lm = CharLM(k=0.01)
        lm.fit(train_texts)
        bpc_hold = lm.bits_per_char(hold_texts)
        # subsample synthetic turns if huge for speed parity
        if len(syn_texts) > 50000:
            syn_eval = [syn_texts[i] for i in rng.choice(len(syn_texts), size=50000, replace=False)]
        else:
            syn_eval = syn_texts
        bpc_syn = lm.bits_per_char(syn_eval)

        out["by_language_pair"][pair] = {
            "unigram_jsd": uni,
            "bigram_jsd": bi,
            "oov_rate_synthetic_vs_ref_vocab": oov,
            "char_lm": {
                "order": "interpolated 3-5 gram, add-k=0.01, lambdas=(0.5,0.3,0.2)",
                "n_ref_train_utts": len(train_texts),
                "n_ref_holdout_utts": len(hold_texts),
                "n_synth_eval_utts": len(syn_eval),
                "bits_per_char_holdout_reference": bpc_hold,
                "bits_per_char_synthetic": bpc_syn,
            },
        }
    return out


def analysis_romanization(convs: list[ConvView], ref_pool: dict[str, list[dict]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "method": (
            "Clusters from language_profile romanization_variants with kind=corpus_cluster "
            "and >=2 variants. For each corpus (synthetic turns / reference utterances), "
            "count how many variants of each cluster appear; report share of clusters with >1 "
            "attested variant and mean Shannon entropy over variant counts within cluster."
        ),
        "by_language_pair": {},
    }
    for pair in PAIR_NAMES:
        profile = load_profile(pair)
        clusters = [
            r
            for r in (profile.get("romanization_variants") or [])
            if r.get("kind") == "corpus_cluster" and len(r.get("variants") or []) >= 2
        ]
        syn_tok_counts = Counter(
            tok.lower()
            for c in convs
            if c.pair == pair
            for t in c.turns
            for tok in tokenize_words(t.get("text") or "")
        )
        ref_tok_counts = Counter(
            tok.lower() for r in ref_pool[pair] for tok in tokenize_words(r.get("text") or "")
        )

        def cluster_stats(tok_counts: Counter) -> dict[str, Any]:
            multi = 0
            entropies = []
            used = 0
            for cl in clusters:
                variants = [v.lower() for v in cl.get("variants") or []]
                counts = [tok_counts.get(v, 0) for v in variants]
                present = sum(1 for c in counts if c > 0)
                if present == 0:
                    continue
                used += 1
                if present > 1:
                    multi += 1
                total = sum(counts)
                probs = [c / total for c in counts if c > 0]
                ent = -sum(p * math.log(p, 2) for p in probs)
                entropies.append(ent)
            return {
                "n_clusters_considered": len(clusters),
                "n_clusters_attested": used,
                "share_multi_variant": (multi / used) if used else None,
                "mean_variant_entropy": float(np.mean(entropies)) if entropies else None,
                "median_variant_entropy": float(np.median(entropies)) if entropies else None,
            }

        out["by_language_pair"][pair] = {
            "synthetic": cluster_stats(syn_tok_counts),
            "reference_full_pool": cluster_stats(ref_tok_counts),
        }
    return out


def analysis_branches(complete: dict, lexicons: dict[str, set[str]]) -> dict[str, Any]:
    # first-reply length by condition
    reply_words: dict[str, list[int]] = defaultdict(list)
    reply_words_by_pair: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))

    # pairwise jaccard of post-branch text
    pair_jac_uni: list[float] = []
    pair_jac_bi: list[float] = []

    # log-odds corpora
    first_reply_by_cond: dict[str, Counter] = defaultdict(Counter)
    opening_by_label: dict[str, Counter] = defaultdict(Counter)
    background_reply = Counter()
    background_open = Counter()

    seen_openings: set[tuple] = set()

    for key, branches in complete.items():
        pair = key[3]
        # openings once per set
        any_c = next(iter(branches.values()))
        open_key = (key[1], key[2], key[3], key[4])  # family, case, pair, history
        if open_key not in seen_openings:
            seen_openings.add(open_key)
            otoks = [t.lower() for t in words_of(opening_turns(any_c))]
            opening_by_label[any_c.label].update(otoks)
            background_open.update(otoks)

        post_ng_uni: dict[str, set] = {}
        post_ng_bi: dict[str, set] = {}
        for cond, c in branches.items():
            fr = first_recipient_reply(c)
            if fr is not None:
                n = len(tokenize_words(fr.get("text") or ""))
                reply_words[cond].append(n)
                reply_words_by_pair[pair][cond].append(n)
                toks = [t.lower() for t in tokenize_words(fr.get("text") or "")]
                first_reply_by_cond[cond].update(toks)
                background_reply.update(toks)
            post_toks = [t.lower() for t in words_of(post_turns(c))]
            post_ng_uni[cond] = set(post_toks)
            post_ng_bi[cond] = set(word_ngrams(post_toks, 2))

        for i, a in enumerate(CONDS):
            for b in CONDS[i + 1 :]:
                if a not in post_ng_uni or b not in post_ng_uni:
                    continue
                ua, ub = post_ng_uni[a], post_ng_uni[b]
                ba, bb = post_ng_bi[a], post_ng_bi[b]
                if ua or ub:
                    pair_jac_uni.append(len(ua & ub) / len(ua | ub) if (ua | ub) else 1.0)
                if ba or bb:
                    pair_jac_bi.append(len(ba & bb) / len(ba | bb) if (ba | bb) else 1.0)

    # log-odds per condition vs rest
    distinguishing = {}
    for cond in CONDS:
        rest = Counter()
        for c2, cnt in first_reply_by_cond.items():
            if c2 != cond:
                rest.update(cnt)
        distinguishing[cond] = weighted_log_odds(
            first_reply_by_cond[cond], rest, background_reply, top_k=15
        )

    fraud_vs_legit = weighted_log_odds(
        opening_by_label.get("fraud_attempt", Counter()),
        opening_by_label.get("legitimate", Counter()),
        background_open,
        top_k=15,
    )
    legit_vs_fraud = weighted_log_odds(
        opening_by_label.get("legitimate", Counter()),
        opening_by_label.get("fraud_attempt", Counter()),
        background_open,
        top_k=15,
    )

    return {
        "n_complete_sets": len(complete),
        "first_reply_words_by_condition": {c: summarize(reply_words[c]) for c in CONDS},
        "first_reply_words_by_pair_condition": {
            p: {c: summarize(reply_words_by_pair[p][c]) for c in CONDS} for p in PAIR_NAMES
        },
        "pairwise_postbranch_jaccard": {
            "unigram_mean": float(np.mean(pair_jac_uni)) if pair_jac_uni else None,
            "unigram": summarize(pair_jac_uni),
            "bigram_mean": float(np.mean(pair_jac_bi)) if pair_jac_bi else None,
            "bigram": summarize(pair_jac_bi),
            "n_pairs": len(pair_jac_uni),
        },
        "weighted_log_odds_first_reply_by_condition": distinguishing,
        "weighted_log_odds_opening_fraud_vs_genuine": {
            "fraud_vs_genuine_top15": fraud_vs_legit,
            "genuine_vs_fraud_top15": legit_vs_fraud,
        },
        "_plot_reply_lengths": {c: reply_words[c] for c in CONDS},
    }


def analysis_probes(convs: list[ConvView], rng_seed: int = SEED) -> dict[str, Any]:
    """TF-IDF 1-2gram logistic probes with family-level bootstrap CI on test."""

    def view_text(c: ConvView, kind: str) -> str:
        if kind == "opening":
            return text_of(opening_turns(c))
        if kind == "first_reply":
            fr = first_recipient_reply(c)
            return (fr.get("text") or "") if fr else ""
        if kind == "recipient_turns":
            return text_of(post_turns(c), speaker="recipient")
        if kind == "caller_turns":
            return text_of(c.turns, speaker="caller")
        if kind == "full_call":
            return text_of(c.turns)
        if kind == "condition_first_reply":
            fr = first_recipient_reply(c)
            return (fr.get("text") or "") if fr else ""
        raise ValueError(kind)

    train = [c for c in convs if c.split == "train"]
    dev = [c for c in convs if c.split == "dev"]
    test = [c for c in convs if c.split == "test"]
    test_families = sorted({c.family for c in test})
    assert len(test_families) == 12, f"expected 12 test families, got {len(test_families)}"

    Cs = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    probe_kinds_label = ["opening", "first_reply", "recipient_turns", "caller_turns", "full_call"]
    results: dict[str, Any] = {
        "method": (
            "sklearn TfidfVectorizer(word 1-2grams, min_df=2) + LogisticRegression "
            "(liblinear for binary label probes; lbfgs for 4-way condition; class_weight=balanced). "
            "C chosen on dev by macro-F1. "
            f"Test macro-F1 with 95% bootstrap CI over {BOOT_REPS} resamples of the "
            f"{len(test_families)} test scenario families with replacement (seed {rng_seed})."
        ),
        "test_families": test_families,
        "probes": {},
    }

    def fit_select(kind: str, y_fn) -> dict[str, Any]:
        X_train = [view_text(c, kind) for c in train]
        y_train = [y_fn(c) for c in train]
        X_dev = [view_text(c, kind) for c in dev]
        y_dev = [y_fn(c) for c in dev]
        X_test = [view_text(c, kind) for c in test]
        y_test = [y_fn(c) for c in test]

        n_classes = len(set(y_train))
        # liblinear is fine for binary; lbfgs for multiclass (condition probe).
        solver = "liblinear" if n_classes <= 2 else "lbfgs"
        best_C = None
        best_dev = -1.0
        best_pipe = None
        for C in Cs:
            pipe = Pipeline(
                [
                    (
                        "tfidf",
                        TfidfVectorizer(
                            analyzer="word",
                            ngram_range=(1, 2),
                            min_df=2,
                            token_pattern=r"[A-Za-z]+(?:'[A-Za-z]+)?",
                        ),
                    ),
                    (
                        "clf",
                        LogisticRegression(
                            C=C,
                            max_iter=4000,
                            class_weight="balanced",
                            solver=solver,
                            random_state=rng_seed,
                        ),
                    ),
                ]
            )
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_dev)
            f1 = f1_score(y_dev, pred, average="macro")
            if f1 > best_dev:
                best_dev = float(f1)
                best_C = C
                best_pipe = pipe
        assert best_pipe is not None
        # refit already on train; evaluate test
        pred_test = best_pipe.predict(X_test)
        point = float(f1_score(y_test, pred_test, average="macro"))

        # map test index by family
        fam_to_idx: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(test):
            fam_to_idx[c.family].append(i)
        rng = np.random.default_rng(rng_seed)
        boots = []
        y_test_arr = np.asarray(y_test)
        pred_arr = np.asarray(pred_test)
        for _ in range(BOOT_REPS):
            sampled = rng.choice(test_families, size=len(test_families), replace=True)
            idxs = [i for f in sampled for i in fam_to_idx[f]]
            boots.append(float(f1_score(y_test_arr[idxs], pred_arr[idxs], average="macro")))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return {
            "best_C": best_C,
            "dev_macro_f1": best_dev,
            "test_macro_f1": point,
            "test_macro_f1_ci95": [float(lo), float(hi)],
            "n_train": len(train),
            "n_dev": len(dev),
            "n_test": len(test),
            "n_classes": len(set(y_train)),
        }

    label_fn = lambda c: c.label
    for kind in probe_kinds_label:
        results["probes"][kind] = fit_select(kind, label_fn)

    results["probes"]["condition_first_reply"] = fit_select(
        "condition_first_reply", lambda c: c.condition
    )
    return results


# ---------------------------------------------------------------------------
# Tables & figures & report
# ---------------------------------------------------------------------------


def write_table(path: Path, caption: str, label: str, header: list[str], rows: list[list[str]], group_rows: set[int] | None = None) -> None:
    group_rows = group_rows or set()
    cols = "l" + "c" * (len(header) - 1)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        "\\rowcolor{headblue}",
        " & ".join(f"\\textbf{{{h}}}" for h in header) + " \\\\",
        "\\midrule",
    ]
    for i, row in enumerate(rows):
        if i in group_rows:
            lines.append("\\rowcolor{groupgray}")
            # group label spanning
            lines.append(f"\\multicolumn{{{len(header)}}}{{l}}{{\\emph{{{row[0]}}}}} \\\\")
        else:
            lines.append(" & ".join(row) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def make_tables(stats: dict[str, Any], out_dir: Path) -> list[str]:
    files = []
    size = stats["size_structure"]
    # Table: counts by split x language
    rows = []
    group_rows = set()
    for split in ("train", "dev", "test"):
        group_rows.add(len(rows))
        rows.append([split.capitalize()] + [""] * 3)
        for pair in PAIR_NAMES:
            n = 0
            d = size["counts_split_lang_label_condition"].get(split, {}).get(pair, {})
            for lab in d.values():
                n += sum(lab.values())
            # fraud / legit
            nf = sum(size["counts_split_lang_label_condition"].get(split, {}).get(pair, {}).get("fraud_attempt", {}).values())
            nl = sum(size["counts_split_lang_label_condition"].get(split, {}).get(pair, {}).get("legitimate", {}).values())
            rows.append([PAIR_NAMES[pair], str(n), str(nf), str(nl)])
    p = out_dir / "size_counts.tex"
    write_table(
        p,
        "Usable conversation counts by split and language pair (fraud vs genuine).",
        "tab:ds-size-counts",
        ["Language", "N", "Fraud", "Genuine"],
        rows,
        group_rows,
    )
    files.append(str(p))

    # length table
    rows = []
    o = size["overall"]
    rows.append(
        [
            "Overall",
            f"{o['turns_per_call']['mean']:.1f}",
            f"{o['turns_per_call']['median']:.0f}",
            f"{o['words_per_call']['mean']:.0f}",
            f"{o['words_per_turn']['mean']:.1f}",
            f"{o['opening_turns']['mean']:.1f}",
            f"{o['post_branch_turns']['mean']:.1f}",
        ]
    )
    for pair in PAIR_NAMES:
        b = size["by_language_pair"][pair]
        rows.append(
            [
                PAIR_NAMES[pair],
                f"{b['turns_per_call']['mean']:.1f}",
                f"{b['turns_per_call']['median']:.0f}",
                f"{b['words_per_call']['mean']:.0f}",
                f"{b['words_per_turn']['mean']:.1f}",
                f"{b['opening_turns']['mean']:.1f}",
                f"{b['post_branch_turns']['mean']:.1f}",
            ]
        )
    p = out_dir / "lengths.tex"
    write_table(
        p,
        "Call length statistics (means unless noted). Opening length is shared prefix turns.",
        "tab:ds-lengths",
        ["Language", "Turns mean", "Turns med.", "Words/call", "Words/turn", "Opening turns", "Post-branch turns"],
        rows,
    )
    files.append(str(p))

    # complete sets
    cs = size["complete_four_branch_sets"]
    # prefer cross tab if present
    cross = cs.get("by_language_pair_split")
    rows = [
        [
            "Overall",
            str(cs["n_total"]),
            str(cs["by_split"].get("train", 0)),
            str(cs["by_split"].get("dev", 0)),
            str(cs["by_split"].get("test", 0)),
        ]
    ]
    for pair in PAIR_NAMES:
        if cross and pair in cross:
            rows.append(
                [
                    PAIR_NAMES[pair],
                    str(sum(cross[pair].values())),
                    str(cross[pair].get("train", 0)),
                    str(cross[pair].get("dev", 0)),
                    str(cross[pair].get("test", 0)),
                ]
            )
        else:
            rows.append([PAIR_NAMES[pair], str(cs["by_language_pair"].get(pair, 0)), "---", "---", "---"])
    p = out_dir / "complete_sets.tex"
    write_table(
        p,
        "Complete four-branch sets among usable conversations.",
        "tab:ds-complete-sets",
        ["Language", "N sets", "Train", "Dev", "Test"],
        rows,
    )
    files.append(str(p))

    # code-mixing synthetic vs ref
    cm = stats["code_mixing"]
    rows = []
    group_rows = set()
    for pair in PAIR_NAMES:
        group_rows.add(len(rows))
        rows.append([PAIR_NAMES[pair]] + [""] * 5)
        syn = cm["by_language_pair"][pair]
        ref = cm["synthetic_vs_reference"][pair]["reference_full_pool_auto_tag"]
        cmp_ = cm["synthetic_vs_reference"][pair]["compare_vs_full_pool_auto"]
        rows.append(
            [
                "Synthetic",
                fmt_prop(syn["per_turn_english_share"]["median"]),
                fmt_prop(syn["cmi_all_turns"]["median"], 1) if syn["cmi_all_turns"]["median"] is not None else "---",
                fmt_prop(syn["monolingual_turn_share"]),
                "---",
                "---",
            ]
        )
        rows.append(
            [
                "Reference",
                fmt_prop(ref["english_share"]["median"]),
                fmt_prop(ref["cmi_all_turns"]["median"], 1) if ref["cmi_all_turns"]["median"] is not None else "---",
                fmt_prop(ref["monolingual_turn_share"]),
                fmt_prop(cmp_["english_share"]["ks_statistic"]),
                fmt_prop(cmp_["english_share"]["wasserstein"]),
            ]
        )
    p = out_dir / "code_mixing.tex"
    write_table(
        p,
        "Code-mixing: per-turn English-word share and CMI (auto-tagging). KS/W1 compare synthetic vs full reference pool.",
        "tab:ds-code-mixing",
        ["Corpus", "EN share med.", "CMI med.", "Mono. share", "KS (EN)", "W1 (EN)"],
        rows,
        group_rows,
    )
    files.append(str(p))

    # target vs realized
    rows = []
    for scope in ["overall"] + list(PAIR_NAMES):
        d = cm["target_vs_realized"][scope]
        name = "Overall" if scope == "overall" else PAIR_NAMES[scope]
        rows.append(
            [
                name,
                str(d["n"]),
                fmt_prop(d["pearson_r"]),
                fmt_prop(d["spearman_rho"]),
            ]
        )
    p = out_dir / "target_realized.tex"
    write_table(
        p,
        "Correlation between target and realized English-word share (per call).",
        "tab:ds-target-realized",
        ["Language", "N", "Pearson $r$", "Spearman $\\rho$"],
        rows,
    )
    files.append(str(p))

    # lexical diversity
    ld = stats["lexical_diversity"]
    rows = []
    group_rows = set()
    for pair in PAIR_NAMES:
        group_rows.add(len(rows))
        rows.append([PAIR_NAMES[pair]] + [""] * 5)
        for key, name in [("synthetic", "Synthetic"), ("reference_full_pool", "Reference")]:
            d = ld["by_language_pair"][pair][key]
            rows.append(
                [
                    name,
                    f"{d['vocab_size']:,}",
                    fmt_prop(d["ttr_10k_mean"]) if d["ttr_10k_mean"] is not None else "---",
                    fmt_prop(d["mattr_50"]),
                    fmt_prop(d["distinct_2"]),
                    fmt_prop(d["zipf_slope"]),
                ]
            )
    p = out_dir / "lexical_diversity.tex"
    write_table(
        p,
        "Lexical diversity (Synthetic vs Reference). TTR on 10k-token samples; MATTR window 50; Zipf slope of $\\log$ freq on $\\log$ rank.",
        "tab:ds-lexical",
        ["Corpus", "Vocab", "TTR$_{10k}$", "MATTR", "Distinct-2", "Zipf slope"],
        rows,
        group_rows,
    )
    files.append(str(p))

    # distributional
    dist = stats["distributional"]
    rows = []
    for pair in PAIR_NAMES:
        d = dist["by_language_pair"][pair]
        rows.append(
            [
                PAIR_NAMES[pair],
                fmt_prop(d["unigram_jsd"]["jsd"]),
                fmt_prop(d["bigram_jsd"]["jsd"]),
                fmt_prop(d["oov_rate_synthetic_vs_ref_vocab"]),
                fmt_prop(d["char_lm"]["bits_per_char_holdout_reference"]),
                fmt_prop(d["char_lm"]["bits_per_char_synthetic"]),
            ]
        )
    p = out_dir / "distributional.tex"
    write_table(
        p,
        "Distributional similarity to the human reference pool (genre mismatch caveat applies).",
        "tab:ds-distributional",
        ["Language", "JSD uni", "JSD bi", "OOV rate", "BPC holdout", "BPC synth."],
        rows,
    )
    files.append(str(p))

    # romanization
    rom = stats["romanization"]
    rows = []
    for pair in PAIR_NAMES:
        s = rom["by_language_pair"][pair]["synthetic"]
        r = rom["by_language_pair"][pair]["reference_full_pool"]
        rows.append(
            [
                PAIR_NAMES[pair],
                fmt_prop(s["share_multi_variant"]),
                fmt_prop(s["mean_variant_entropy"]),
                fmt_prop(r["share_multi_variant"]),
                fmt_prop(r["mean_variant_entropy"]),
            ]
        )
    p = out_dir / "romanization.tex"
    write_table(
        p,
        "Romanization spelling variation: share of attested clusters with $>1$ variant and mean variant entropy.",
        "tab:ds-romanization",
        ["Language", "Synth multi", "Synth $H$", "Ref multi", "Ref $H$"],
        rows,
    )
    files.append(str(p))

    # branches
    br = stats["branches"]
    rows = []
    for cond in CONDS:
        d = br["first_reply_words_by_condition"][cond]
        rows.append([COND_NAMES[cond], f"{d['mean']:.1f}", f"{d['median']:.0f}", f"{d['iqr']:.1f}"])
    p = out_dir / "first_reply_length.tex"
    write_table(
        p,
        "First recipient reply length (words) by condition, over complete four-branch sets.",
        "tab:ds-first-reply",
        ["Condition", "Mean", "Median", "IQR"],
        rows,
    )
    files.append(str(p))

    jac = br["pairwise_postbranch_jaccard"]
    rows = [
        ["Unigram", fmt_prop(jac["unigram_mean"]), fmt_prop(jac["unigram"]["median"]), str(jac["n_pairs"])],
        ["Bigram", fmt_prop(jac["bigram_mean"]), fmt_prop(jac["bigram"]["median"]), str(jac["n_pairs"])],
    ]
    p = out_dir / "branch_jaccard.tex"
    write_table(
        p,
        "Pairwise Jaccard similarity of post-branch text between conditions (mean over pairs in complete sets).",
        "tab:ds-branch-jaccard",
        ["N-gram", "Mean Jaccard", "Median", "N pairs"],
        rows,
    )
    files.append(str(p))

    # log-odds top words (compact)
    rows = []
    group_rows = set()
    for cond in CONDS:
        group_rows.add(len(rows))
        rows.append([COND_NAMES[cond]] + [""] * 2)
        for item in br["weighted_log_odds_first_reply_by_condition"][cond][:8]:
            rows.append([item["word"], f"{item['z']:.2f}", str(item["count_i"])])
    p = out_dir / "logodds_condition.tex"
    write_table(
        p,
        "Top distinguishing words in first replies by recipient condition (weighted log-odds $z$, Monroe et al.\\ 2008).",
        "tab:ds-logodds-cond",
        ["Word", "$z$", "Count"],
        rows,
        group_rows,
    )
    files.append(str(p))

    rows = []
    group_rows = set()
    group_rows.add(0)
    rows.append(["Fraud vs genuine"] + [""] * 2)
    for item in br["weighted_log_odds_opening_fraud_vs_genuine"]["fraud_vs_genuine_top15"][:10]:
        rows.append([item["word"], f"{item['z']:.2f}", str(item["count_i"])])
    group_rows.add(len(rows))
    rows.append(["Genuine vs fraud"] + [""] * 2)
    for item in br["weighted_log_odds_opening_fraud_vs_genuine"]["genuine_vs_fraud_top15"][:10]:
        rows.append([item["word"], f"{item['z']:.2f}", str(item["count_i"])])
    p = out_dir / "logodds_label.tex"
    write_table(
        p,
        "Top distinguishing words in shared openings: fraud vs genuine (weighted log-odds $z$).",
        "tab:ds-logodds-label",
        ["Word", "$z$", "Count"],
        rows,
        group_rows,
    )
    files.append(str(p))

    # probes
    pr = stats["probes"]["probes"]
    rows = []
    for kind, name in PROBE_NAMES.items():
        d = pr[kind]
        lo, hi = d["test_macro_f1_ci95"]
        rows.append(
            [
                name,
                fmt_prop(d["test_macro_f1"]),
                fmt_ci(lo, hi),
                fmt_prop(d["dev_macro_f1"]),
                str(d["best_C"]),
            ]
        )
    p = out_dir / "lexical_probes.tex"
    write_table(
        p,
        "Lexical shortcut probes (TF--IDF 1--2grams, logistic regression). Macro-F1 on test with family-level 95\\% bootstrap CI.",
        "tab:ds-probes",
        ["Probe", "Test F1", "95\\% CI", "Dev F1", "$C$"],
        rows,
    )
    files.append(str(p))

    return files


def make_figures(stats: dict[str, Any], out_dir: Path) -> list[str]:
    files = []
    cm = stats["code_mixing"]

    # CMI / EN share distributions
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2), sharey="row")
    for j, pair in enumerate(PAIR_NAMES):
        plot = cm["synthetic_vs_reference"][pair]["_plot"]
        ax = axes[0, j]
        ax.hist(plot["syn_en"], bins=20, density=True, alpha=0.55, color=COLORS["synthetic"], label="Synthetic")
        ax.hist(plot["ref_en"], bins=20, density=True, alpha=0.45, color=COLORS["reference"], label="Reference")
        ax.set_title(PAIR_NAMES[pair], fontsize=9)
        if j == 0:
            ax.set_ylabel("Density (EN share)")
        if j == 2:
            ax.legend(fontsize=7, frameon=False)
        ax = axes[1, j]
        ax.hist(plot["syn_cmi"], bins=20, density=True, alpha=0.55, color=COLORS["synthetic"], label="Synthetic")
        ax.hist(plot["ref_cmi"], bins=20, density=True, alpha=0.45, color=COLORS["reference"], label="Reference")
        if j == 0:
            ax.set_ylabel("Density (CMI)")
        ax.set_xlabel("Value")
    fig.tight_layout()
    p = out_dir / "cmi_en_share_dist.pdf"
    fig.savefig(p)
    plt.close(fig)
    files.append(str(p))

    # Zipf
    ld = stats["lexical_diversity"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for j, pair in enumerate(PAIR_NAMES):
        zp = ld["by_language_pair"][pair]["_zipf_plot"]
        ax = axes[j]
        for name, freqs, color in [
            ("Synthetic", zp["synthetic_freqs"], COLORS["synthetic"]),
            ("Reference", zp["reference_freqs"], COLORS["reference"]),
        ]:
            if not freqs:
                continue
            ranks = np.arange(1, len(freqs) + 1)
            ax.loglog(ranks, freqs, label=name, color=color, lw=1.5)
        ax.set_title(PAIR_NAMES[pair], fontsize=9)
        ax.set_xlabel("Rank")
        if j == 0:
            ax.set_ylabel("Frequency")
        if j == 2:
            ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    p = out_dir / "zipf.pdf"
    fig.savefig(p)
    plt.close(fig)
    files.append(str(p))

    # target vs realized scatter
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharex=True, sharey=True)
    for j, pair in enumerate(PAIR_NAMES):
        plot = cm["synthetic_vs_reference"][pair]["_plot"]
        ax = axes[j]
        ax.scatter(plot["targets"], plot["realized"], s=8, alpha=0.35, c=COLORS[pair], edgecolors="none")
        ax.plot([0, 1], [0, 1], ls="--", c="gray", lw=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(PAIR_NAMES[pair], fontsize=9)
        ax.set_xlabel("Target EN share")
        if j == 0:
            ax.set_ylabel("Realized EN share")
    fig.tight_layout()
    p = out_dir / "target_vs_realized.pdf"
    fig.savefig(p)
    plt.close(fig)
    files.append(str(p))

    # first-reply length by condition
    br = stats["branches"]
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    data = [br["_plot_reply_lengths"][c] for c in CONDS]
    bp = ax.boxplot(
        data,
        tick_labels=[COND_NAMES[c] for c in CONDS],
        patch_artist=True,
        showfliers=False,
    )
    for patch, cond in zip(bp["boxes"], CONDS):
        patch.set_facecolor(COLORS[cond])
        patch.set_alpha(0.7)
    ax.set_ylabel("Words in first reply")
    ax.set_xlabel("Recipient condition")
    fig.tight_layout()
    p = out_dir / "first_reply_length.pdf"
    fig.savefig(p)
    plt.close(fig)
    files.append(str(p))

    return files


def strip_plot_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: strip_plot_keys(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [strip_plot_keys(x) for x in obj]
    return obj


def write_report(stats: dict[str, Any], path: Path) -> None:
    cm = stats["code_mixing"]
    ld = stats["lexical_diversity"]
    dist = stats["distributional"]
    pr = stats["probes"]["probes"]
    br = stats["branches"]
    size = stats["size_structure"]

    def probe_line(k: str) -> str:
        d = pr[k]
        lo, hi = d["test_macro_f1_ci95"]
        return f"{PROBE_NAMES[k]}: test macro-F1={d['test_macro_f1']:.3f} CI=[{lo:.3f},{hi:.3f}] (dev={d['dev_macro_f1']:.3f}, C={d['best_C']})"

    open_f1 = pr["opening"]["test_macro_f1"]
    fr_f1 = pr["first_reply"]["test_macro_f1"]
    cond_f1 = pr["condition_first_reply"]["test_macro_f1"]
    tr = cm["target_vs_realized"]["overall"]

    lines = [
        "# Supplement dataset statistics report",
        "",
        f"Usable conversations: **{size['n_usable']}** (from `runs/full_v1/FROZEN.json`).",
        f"Complete four-branch sets: **{size['complete_four_branch_sets']['n_total']}**.",
        "",
        "## Tagging method (shared with generator checks)",
        "",
        cm["method"],
        "",
        "Reference corpora: full mined pools in `data/calibration/processed/{pair}/utterances.jsonl`; "
        "the **200 reference utterances** are `reference_utterances` in each `configs/language_profiles/{pair}.yaml` "
        "(service-oriented calibration subset).",
        "",
        "## 1. Size and structure",
        "",
        f"- Overall turns/call mean={size['overall']['turns_per_call']['mean']:.2f}, median={size['overall']['turns_per_call']['median']:.0f}, IQR={size['overall']['turns_per_call']['iqr']:.1f}.",
        f"- Words/call mean={size['overall']['words_per_call']['mean']:.1f}; words/turn mean={size['overall']['words_per_turn']['mean']:.2f}.",
        f"- Opening turns mean={size['overall']['opening_turns']['mean']:.2f}; post-branch turns mean={size['overall']['post_branch_turns']['mean']:.2f}.",
        "- Tables: `tables/size_counts.tex`, `tables/lengths.tex`, `tables/complete_sets.tex`.",
        "",
        "## 2. Code-mixing",
        "",
    ]
    for pair in PAIR_NAMES:
        syn = cm["by_language_pair"][pair]
        cmp_ = cm["synthetic_vs_reference"][pair]["compare_vs_full_pool_auto"]
        lines.append(
            f"- **{PAIR_NAMES[pair]}**: synth EN-share median={syn['per_turn_english_share']['median']:.3f}, "
            f"CMI median={syn['cmi_all_turns']['median']:.2f}, mono share={syn['monolingual_turn_share']:.3f}; "
            f"vs full ref KS(EN)={cmp_['english_share']['ks_statistic']:.3f}, W1={cmp_['english_share']['wasserstein']:.3f}."
        )
    tr = cm["target_vs_realized"]["overall"]
    lines += [
        f"- Target vs realized EN share: Pearson r={tr['pearson_r']:.3f}, Spearman ρ={tr['spearman_rho']:.3f} (n={tr['n']}).",
        f"- Recomputed vs stored english_share max abs diff: {cm['target_vs_realized']['recomputed_vs_stored_max_abs_diff']}.",
        "- Figures: `figures/cmi_en_share_dist.pdf`, `figures/target_vs_realized.pdf`. Tables: `tables/code_mixing.tex`, `tables/target_realized.tex`.",
        "",
        "### Caveats",
        "- Auto-tagging uses English zipf + a profile lexicon; many Romanized content words remain `unk` and are excluded from share/CMI denominators.",
        "- Full reference pools are much larger and different in genre (tweets/QA/dialogs) than phone calls; the 200 profile refs are a service-biased subset.",
        "",
        "## 3. Lexical diversity",
        "",
        ld["method"],
        "",
    ]
    for pair in PAIR_NAMES:
        s = ld["by_language_pair"][pair]["synthetic"]
        r = ld["by_language_pair"][pair]["reference_full_pool"]
        lines.append(
            f"- **{PAIR_NAMES[pair]}**: synth V={s['vocab_size']}, TTR10k={s['ttr_10k_mean']}, MATTR={s['mattr_50']:.3f}, "
            f"Zipf slope={s['zipf_slope']:.3f}; ref V={r['vocab_size']}, TTR10k={r['ttr_10k_mean']}, Zipf={r['zipf_slope']:.3f}."
        )
    lines += [
        "- Figure: `figures/zipf.pdf`. Table: `tables/lexical_diversity.tex`.",
        "",
        "## 4. Distributional similarity",
        "",
        dist["caveat"],
        "",
    ]
    for pair in PAIR_NAMES:
        d = dist["by_language_pair"][pair]
        lines.append(
            f"- **{PAIR_NAMES[pair]}**: JSD uni={d['unigram_jsd']['jsd']:.3f}, bi={d['bigram_jsd']['jsd']:.3f}, "
            f"OOV={d['oov_rate_synthetic_vs_ref_vocab']:.3f}, "
            f"BPC holdout={d['char_lm']['bits_per_char_holdout_reference']:.3f}, "
            f"BPC synth={d['char_lm']['bits_per_char_synthetic']:.3f}."
        )
    lines += [
        "- Table: `tables/distributional.tex`.",
        "",
        "## 5. Romanization variation",
        "",
        stats["romanization"]["method"],
        "",
    ]
    for pair in PAIR_NAMES:
        s = stats["romanization"]["by_language_pair"][pair]["synthetic"]
        r = stats["romanization"]["by_language_pair"][pair]["reference_full_pool"]
        lines.append(
            f"- **{PAIR_NAMES[pair]}**: synth multi-variant share={s['share_multi_variant']}, H={s['mean_variant_entropy']}; "
            f"ref multi={r['share_multi_variant']}, H={r['mean_variant_entropy']}."
        )
    lines += [
        "- Table: `tables/romanization.tex`.",
        "",
        "## 6. Branch analysis",
        "",
        f"- Complete sets used: {br['n_complete_sets']}.",
        f"- Pairwise post-branch Jaccard (unigram mean)={br['pairwise_postbranch_jaccard']['unigram_mean']:.3f}, "
        f"bigram mean={br['pairwise_postbranch_jaccard']['bigram_mean']:.3f}.",
        "- Tables: `tables/first_reply_length.tex`, `tables/branch_jaccard.tex`, `tables/logodds_condition.tex`, `tables/logodds_label.tex`.",
        "- Figure: `figures/first_reply_length.pdf`.",
        "",
        "## 7. Lexical shortcut probes",
        "",
        stats["probes"]["method"],
        "",
    ]
    for k in PROBE_NAMES:
        lines.append(f"- {probe_line(k)}")
    lines += [
        "",
        "### Interpretation (from these numbers)",
        f"- Condition-from-first-reply is very high (F1={cond_f1:.3f}): recipient conditions leave clear lexical fingerprints.",
        f"- Opening-only label probe is also strong (F1={open_f1:.3f}): fraud vs genuine openings are lexically separable.",
        f"- First-reply and recipient-only label probes are **not** near chance (F1={fr_f1:.3f} / "
        f"{pr['recipient_turns']['test_macro_f1']:.3f}). Victim text alone still predicts the label well on this split — "
        "a lexical shortcut risk for detectors that might attend to recipient wording.",
        f"- Full call / caller-only are highest (F1={pr['full_call']['test_macro_f1']:.3f} / "
        f"{pr['caller_turns']['test_macro_f1']:.3f}).",
        "- Table: `tables/lexical_probes.tex`.",
        "",
        "## Anomalies / flags",
        "",
        f"- **Target vs realized English share is essentially uncorrelated** "
        f"(Pearson r={tr['pearson_r']:.3f}, Spearman ρ={tr['spearman_rho']:.3f}). "
        "Calls do not hit their sampled mixing targets tightly; band checks only enforce p10–p90.",
        f"- Opening label probe F1={open_f1:.3f}; first-reply label F1={fr_f1:.3f}; condition F1={cond_f1:.3f}.",
    ]
    for pair in PAIR_NAMES:
        cmp_ = cm["synthetic_vs_reference"][pair]["compare_vs_full_pool_auto"]
        if cmp_["english_share"]["ks_statistic"] and cmp_["english_share"]["ks_statistic"] > 0.3:
            lines.append(
                f"- Large EN-share distribution shift for {PAIR_NAMES[pair]} "
                f"(KS={cmp_['english_share']['ks_statistic']:.3f})."
            )
    lines.append(
        "- Synthetic romanization is far less variant-diverse than the large reference pools "
        "(multi-variant cluster share ≪ 1); reference multi-variant share ≈ 1.0 is expected for huge pools."
    )
    if fr_f1 >= 0.7:
        lines.append(
            "- SURPRISING: first-reply / recipient-only label probes are strong, not near chance — "
            "do not claim victim words are label-safe without further controls (e.g., family-held-out semantics)."
        )
    if cond_f1 > 0.7:
        lines.append("- Condition probe is strong: conditions leave clear lexical fingerprints in first replies.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--skip-probes", action="store_true")
    ap.add_argument("--skip-charlm", action="store_true", help="Skip slow char LM (for dry runs)")
    args = ap.parse_args()
    root = args.root
    out = args.out or (root / "results/supplement/dataset")
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    print("Loading conversations...", flush=True)
    convs = load_frozen_conversations(root)
    print(f"  n={len(convs)}", flush=True)

    print("Loading lexicons and reference corpora...", flush=True)
    lexicons = {pair: local_lexicon(load_profile(pair)) for pair in PAIR_NAMES}
    ref_pool = {pair: load_reference_pool(root, pair) for pair in PAIR_NAMES}
    ref200 = {pair: load_reference_200(root, pair) for pair in PAIR_NAMES}
    for pair in PAIR_NAMES:
        print(f"  {pair}: lexicon={len(lexicons[pair])} ref_pool={len(ref_pool[pair])} ref200={len(ref200[pair])}", flush=True)

    complete = complete_four_branch_sets(convs)
    print(f"Complete four-branch sets: {len(complete)}", flush=True)

    print("1. Size/structure...", flush=True)
    size = analysis_size_structure(convs, complete)

    print("2. Code-mixing...", flush=True)
    code_mix = analysis_code_mixing(convs, lexicons, ref_pool, ref200)

    print("3. Lexical diversity...", flush=True)
    lexdiv = analysis_lexical_diversity(convs, ref_pool, ref200, rng)

    print("4. Distributional similarity...", flush=True)
    if args.skip_charlm:
        dist = {"caveat": "charlm skipped", "by_language_pair": {}}
        for pair in PAIR_NAMES:
            syn_toks = []
            for c in convs:
                if c.pair != pair:
                    continue
                for t in c.turns:
                    syn_toks.extend(tokenize_words(t.get("text") or ""))
            ref_toks = []
            for row in ref_pool[pair]:
                ref_toks.extend(tokenize_words(row.get("text") or ""))
            ref_vocab = {t.lower() for t in ref_toks}
            dist["by_language_pair"][pair] = {
                "unigram_jsd": ngram_jsd(syn_toks, ref_toks, 1),
                "bigram_jsd": ngram_jsd(syn_toks, ref_toks, 2),
                "oov_rate_synthetic_vs_ref_vocab": oov_rate(syn_toks, ref_vocab),
                "char_lm": {
                    "skipped": True,
                    "bits_per_char_holdout_reference": None,
                    "bits_per_char_synthetic": None,
                },
            }
    else:
        dist = analysis_distributional(convs, ref_pool, rng)

    print("5. Romanization...", flush=True)
    roman = analysis_romanization(convs, ref_pool)

    print("6. Branches...", flush=True)
    branches = analysis_branches(complete, lexicons)

    if args.skip_probes:
        probes = {"method": "skipped", "probes": {}}
    else:
        print("7. Lexical probes (this is the slow step)...", flush=True)
        probes = analysis_probes(convs, rng_seed=args.seed)

    stats = {
        "meta": {
            "seed": args.seed,
            "n_usable": len(convs),
            "n_complete_four_branch": len(complete),
            "reference_pool_sizes": {p: len(ref_pool[p]) for p in PAIR_NAMES},
            "reference_200_sizes": {p: len(ref200[p]) for p in PAIR_NAMES},
            "tagging": code_mix["method"],
        },
        "size_structure": size,
        "code_mixing": code_mix,
        "lexical_diversity": lexdiv,
        "distributional": dist,
        "romanization": roman,
        "branches": branches,
        "probes": probes,
    }

    print("Writing figures...", flush=True)
    fig_files = make_figures(stats, out / "figures")
    print("Writing tables...", flush=True)
    tab_files = make_tables(stats, out / "tables")
    print("Writing REPORT.md...", flush=True)
    write_report(stats, out / "REPORT.md")

    # JSON without plot buffers
    print("Writing stats.json...", flush=True)
    clean = strip_plot_keys(stats)
    (out / "stats.json").write_text(json.dumps(clean, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("Done.")
    print("Figures:", *fig_files, sep="\n  ")
    print("Tables:", *tab_files, sep="\n  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
