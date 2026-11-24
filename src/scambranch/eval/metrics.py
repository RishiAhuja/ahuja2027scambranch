"""Classification, branch, reaction, streaming, and bootstrap metrics (plan §14)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from scambranch.eval.constants import CONDITIONS, FRAUD_LABELS

LabelLike = int | str | bool


def as_binary_label(value: LabelLike) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return 1 if int(value) == 1 else 0
    text = str(value).strip().lower()
    if text in FRAUD_LABELS or text == "fraud":
        return 1
    if text in {"legitimate", "legit", "0"}:
        return 0
    raise ValueError(f"unrecognized label {value!r}")


def _arrays(y_true: Sequence[LabelLike], y_pred: Sequence[LabelLike]) -> tuple[np.ndarray, np.ndarray]:
    yt = np.array([as_binary_label(v) for v in y_true], dtype=int)
    yp = np.array([as_binary_label(v) for v in y_pred], dtype=int)
    return yt, yp


def macro_f1(y_true: Sequence[LabelLike], y_pred: Sequence[LabelLike]) -> float:
    yt, yp = _arrays(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    return float(f1_score(yt, yp, average="macro", labels=[0, 1], zero_division=0.0))


def fraud_recall(y_true: Sequence[LabelLike], y_pred: Sequence[LabelLike]) -> float:
    yt, yp = _arrays(y_true, y_pred)
    n_pos = int((yt == 1).sum())
    if n_pos == 0:
        return float("nan")
    return float(((yt == 1) & (yp == 1)).sum() / n_pos)


def legitimate_fpr(y_true: Sequence[LabelLike], y_pred: Sequence[LabelLike]) -> float:
    """False-positive rate on legitimate calls: P(pred=fraud | label=legitimate)."""
    yt, yp = _arrays(y_true, y_pred)
    n_neg = int((yt == 0).sum())
    if n_neg == 0:
        return float("nan")
    return float(((yt == 0) & (yp == 1)).sum() / n_neg)


def roc_auc(y_true: Sequence[LabelLike], scores: Sequence[float]) -> float:
    yt = np.array([as_binary_label(v) for v in y_true], dtype=int)
    sc = np.asarray(scores, dtype=float)
    if yt.size == 0 or len(np.unique(yt)) < 2:
        return float("nan")
    return float(roc_auc_score(yt, sc))


def fit_threshold(
    scores: Sequence[float],
    labels: Sequence[LabelLike],
    *,
    grid: int = 101,
) -> float:
    """Choose a score threshold on DEV only by maximising macro-F1.

    Ties break toward 0.5. Default 0.5 if the split is empty.
    """
    sc = np.asarray(scores, dtype=float)
    yt = np.array([as_binary_label(v) for v in labels], dtype=int)
    if sc.size == 0:
        return 0.5
    finite = np.isfinite(sc)
    sc = sc[finite]
    yt = yt[finite]
    if sc.size == 0:
        return 0.5
    unique = np.unique(sc)
    midpoints = (unique[:-1] + unique[1:]) / 2.0 if unique.size > 1 else np.array([])
    lin = np.linspace(0.0, 1.0, grid)
    candidates = np.unique(np.concatenate([unique, midpoints, lin, np.array([0.5])]))
    best_t = 0.5
    best_f1 = -1.0
    for threshold in candidates:
        pred = (sc >= threshold).astype(int)
        score = macro_f1(yt, pred)
        if score > best_f1 or (
            score == best_f1 and abs(float(threshold) - 0.5) < abs(best_t - 0.5)
        ):
            best_f1 = score
            best_t = float(threshold)
    return best_t


def apply_threshold(score: float, threshold: float) -> int:
    return 1 if float(score) >= float(threshold) else 0


def classification_metrics(
    y_true: Sequence[LabelLike],
    y_pred: Sequence[LabelLike],
    scores: Sequence[float] | None = None,
    invalid: Sequence[bool] | None = None,
) -> dict[str, float]:
    yt, yp = _arrays(y_true, y_pred)
    out = {
        "n": float(yt.size),
        "n_fraud": float((yt == 1).sum()),
        "n_legitimate": float((yt == 0).sum()),
        "macro_f1": macro_f1(yt, yp),
        "fraud_recall": fraud_recall(yt, yp),
        "legitimate_fpr": legitimate_fpr(yt, yp),
        "accuracy": float((yt == yp).mean()) if yt.size else float("nan"),
        "roc_auc": roc_auc(yt, scores) if scores is not None else float("nan"),
        "invalid_output_rate": (
            float(np.mean(np.asarray(invalid, dtype=bool))) if invalid is not None and len(invalid) else 0.0
        ),
    }
    return out


def per_condition_metrics(
    y_true: Sequence[LabelLike],
    y_pred: Sequence[LabelLike],
    conditions: Sequence[str],
    scores: Sequence[float] | None = None,
) -> dict[str, Any]:
    by: dict[str, dict[str, list[Any]]] = {
        c: {"y": [], "p": [], "s": []} for c in CONDITIONS
    }
    sc = list(scores) if scores is not None else [None] * len(y_true)
    for y, p, cond, s in zip(y_true, y_pred, conditions, sc):
        key = cond if cond in by else None
        if key is None:
            continue
        by[key]["y"].append(y)
        by[key]["p"].append(p)
        by[key]["s"].append(s)
    out: dict[str, Any] = {}
    f1s: dict[str, float] = {}
    recalls: dict[str, float] = {}
    fprs: dict[str, float] = {}
    for cond in CONDITIONS:
        metrics = classification_metrics(
            by[cond]["y"],
            by[cond]["p"],
            [x for x in by[cond]["s"] if x is not None] or None,
        )
        out[cond] = metrics
        f1s[cond] = metrics["macro_f1"]
        recalls[cond] = metrics["fraud_recall"]
        fprs[cond] = metrics["legitimate_fpr"]
    finite_f1 = {k: v for k, v in f1s.items() if np.isfinite(v)}
    finite_rec = {k: v for k, v in recalls.items() if np.isfinite(v)}
    finite_fpr = {k: v for k, v in fprs.items() if np.isfinite(v)}
    worst_f1_cond = min(finite_f1, key=finite_f1.get) if finite_f1 else None
    worst_rec_cond = min(finite_rec, key=finite_rec.get) if finite_rec else None
    worst_fpr_cond = max(finite_fpr, key=finite_fpr.get) if finite_fpr else None
    out["worst_condition_macro_f1"] = finite_f1.get(worst_f1_cond, float("nan")) if worst_f1_cond else float("nan")
    out["worst_condition_name"] = worst_f1_cond
    out["worst_condition_fraud_recall"] = (
        finite_rec.get(worst_rec_cond, float("nan")) if worst_rec_cond else float("nan")
    )
    out["worst_condition_fraud_recall_name"] = worst_rec_cond
    out["worst_condition_legitimate_fpr"] = (
        finite_fpr.get(worst_fpr_cond, float("nan")) if worst_fpr_cond else float("nan")
    )
    out["worst_condition_legitimate_fpr_name"] = worst_fpr_cond
    return out


def branch_disagreement(preds: Sequence[LabelLike]) -> bool:
    bits = {as_binary_label(p) for p in preds}
    return len(bits) > 1


def branch_disagreement_rate(groups: Sequence[Sequence[LabelLike]]) -> float:
    if not groups:
        return float("nan")
    return float(sum(1 for g in groups if branch_disagreement(g)) / len(groups))


def branch_score_spread(scores: Sequence[float]) -> float:
    arr = np.asarray(list(scores), dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmax(arr) - np.nanmin(arr))


def all_branches_correct(preds: Sequence[LabelLike], labels: Sequence[LabelLike]) -> bool:
    yp = [as_binary_label(p) for p in preds]
    yt = [as_binary_label(y) for y in labels]
    return bool(yp) and yp == yt


def all_branches_correct_rate(
    pred_groups: Sequence[Sequence[LabelLike]],
    label_groups: Sequence[Sequence[LabelLike]],
) -> float:
    if not pred_groups:
        return float("nan")
    n = 0
    ok = 0
    for preds, labels in zip(pred_groups, label_groups):
        n += 1
        if all_branches_correct(preds, labels):
            ok += 1
    return float(ok / n) if n else float("nan")


def sibling_groups(
    rows: Sequence[Mapping[str, Any]],
    *,
    require_complete: bool = True,
) -> list[list[Mapping[str, Any]]]:
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = f"{row['case_id']}|{row['language_pair']}|{row['history_id']}"
        buckets[key].append(row)
    groups: list[list[Mapping[str, Any]]] = []
    wanted = set(CONDITIONS)
    for key in sorted(buckets):
        members = buckets[key]
        conds = {str(m.get("recipient_condition")) for m in members}
        if require_complete and not wanted <= conds:
            continue
        if require_complete:
            # one row per condition (first by conversation_id)
            by_cond: dict[str, Mapping[str, Any]] = {}
            for member in sorted(members, key=lambda m: str(m["conversation_id"])):
                cond = str(member.get("recipient_condition"))
                by_cond.setdefault(cond, member)
            if set(by_cond) >= wanted:
                groups.append([by_cond[c] for c in CONDITIONS])
        else:
            groups.append(members)
    return groups


def branch_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    groups = sibling_groups(rows, require_complete=True)
    pred_groups = [[g["pred"] for g in grp] for grp in groups]
    label_groups = [[g["label"] for g in grp] for grp in groups]
    spreads = [branch_score_spread([g["score"] for g in grp]) for grp in groups]
    return {
        "n_complete_histories": float(len(groups)),
        "branch_disagreement_rate": branch_disagreement_rate(pred_groups),
        "branch_score_spread_mean": float(np.nanmean(spreads)) if spreads else float("nan"),
        "branch_score_spread_median": float(np.nanmedian(spreads)) if spreads else float("nan"),
        "all_branches_correct_rate": all_branches_correct_rate(pred_groups, label_groups),
    }


def reaction_score_deltas(
    prefix_only: Sequence[Mapping[str, Any]],
    prefix_plus: Sequence[Mapping[str, Any]],
    *,
    bootstrap_reps: int = 1000,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    """PREFIX_PLUS_REACTION − PREFIX_ONLY, by condition and label (RQ1 / E3)."""
    by_id = {r["conversation_id"]: r for r in prefix_only}
    deltas: list[dict[str, Any]] = []
    for row in prefix_plus:
        base = by_id.get(row["conversation_id"])
        if base is None:
            continue
        deltas.append(
            {
                "conversation_id": row["conversation_id"],
                "delta": float(row["score"]) - float(base["score"]),
                "condition": row.get("recipient_condition"),
                "label": as_binary_label(row["label"]),
                "scenario_family_id": row.get("scenario_family_id") or base.get("scenario_family_id"),
            }
        )
    if not deltas:
        return {
            "n_paired": 0.0,
            "mean_delta": float("nan"),
            "by_condition": {},
            "by_label": {},
            "by_condition_label": {},
            "by_condition_label_ci": {},
        }

    def _mean(items: list[dict[str, Any]]) -> float:
        if not items:
            return float("nan")
        return float(np.mean([d["delta"] for d in items]))

    by_cond = {c: _mean([d for d in deltas if d["condition"] == c]) for c in CONDITIONS}
    by_label = {
        "fraud": _mean([d for d in deltas if d["label"] == 1]),
        "legitimate": _mean([d for d in deltas if d["label"] == 0]),
    }
    by_both: dict[str, float] = {}
    by_both_ci: dict[str, Any] = {}
    for cond in CONDITIONS:
        for name, bit in (("fraud", 1), ("legitimate", 0)):
            key = f"{cond}|{name}"
            subset = [d for d in deltas if d["condition"] == cond and d["label"] == bit]
            by_both[key] = _mean(subset)

            def _metric(sample: list[Mapping[str, Any]], *, _cond=cond, _bit=bit) -> float:
                vals = [
                    float(d["delta"])
                    for d in sample
                    if d.get("condition") == _cond and d.get("label") == _bit
                ]
                return float(np.mean(vals)) if vals else float("nan")

            by_both_ci[key] = family_bootstrap(
                deltas, _metric, n_reps=bootstrap_reps, seed=bootstrap_seed
            )
    return {
        "n_paired": float(len(deltas)),
        "mean_delta": _mean(deltas),
        "by_condition": by_cond,
        "by_label": by_label,
        "by_condition_label": by_both,
        "by_condition_label_ci": by_both_ci,
    }


def evidence_relative_delay(
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """First alert turn minus evidence turn. Misses are counted, not dropped.

    Each item: ``evidence_turn`` (int), ``alert_turn`` (int | None).
    Optional ``pre_evidence_alert`` is ignored; derived from the two turns.
    """
    eligible = 0
    misses = 0
    delays: list[int] = []
    pre = 0
    for item in items:
        ev = item.get("evidence_turn")
        if ev is None:
            continue
        eligible += 1
        alert = item.get("alert_turn")
        if alert is None:
            misses += 1
            continue
        delay = int(alert) - int(ev)
        delays.append(delay)
        if int(alert) < int(ev):
            pre += 1
    # Conversations that never alerted cannot have a pre-evidence alert.
    return {
        "n_eligible": float(eligible),
        "n_detected": float(len(delays)),
        "n_miss": float(misses),
        "miss_rate": float(misses / eligible) if eligible else float("nan"),
        "delays": delays,
        "mean_delay_detected": float(np.mean(delays)) if delays else float("nan"),
        "median_delay_detected": float(np.median(delays)) if delays else float("nan"),
        "pre_evidence_alert_rate": float(pre / eligible) if eligible else float("nan"),
        "n_pre_evidence_alert": float(pre),
    }


def family_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    metric_fn: Callable[[list[Mapping[str, Any]]], float],
    *,
    n_reps: int = 1000,
    seed: int = 42,
    family_key: str = "scenario_family_id",
) -> dict[str, float]:
    """Resample scenario families (all languages/branches of a family stay together)."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[family_key])].append(row)
    families = sorted(grouped)
    if not families:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_reps": float(n_reps)}
    rng = np.random.default_rng(seed)
    stats = np.empty(n_reps, dtype=float)
    n_fam = len(families)
    for i in range(n_reps):
        chosen = rng.choice(families, size=n_fam, replace=True)
        sample: list[Mapping[str, Any]] = []
        for fam in chosen:
            sample.extend(grouped[str(fam)])
        stats[i] = metric_fn(sample)
    return {
        "mean": float(np.nanmean(stats)),
        "ci_low": float(np.nanpercentile(stats, 2.5)),
        "ci_high": float(np.nanpercentile(stats, 97.5)),
        "n_reps": float(n_reps),
        "n_families": float(n_fam),
    }


def attach_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    point: Mapping[str, float],
    *,
    n_reps: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Point estimates plus family-level CIs for the main quality metrics."""

    def _f1(sample: list[Mapping[str, Any]]) -> float:
        return macro_f1([r["label"] for r in sample], [r["pred"] for r in sample])

    def _rec(sample: list[Mapping[str, Any]]) -> float:
        return fraud_recall([r["label"] for r in sample], [r["pred"] for r in sample])

    def _fpr(sample: list[Mapping[str, Any]]) -> float:
        return legitimate_fpr([r["label"] for r in sample], [r["pred"] for r in sample])

    out = dict(point)
    out["macro_f1_ci"] = family_bootstrap(rows, _f1, n_reps=n_reps, seed=seed)
    out["fraud_recall_ci"] = family_bootstrap(rows, _rec, n_reps=n_reps, seed=seed)
    out["legitimate_fpr_ci"] = family_bootstrap(rows, _fpr, n_reps=n_reps, seed=seed)
    return out
