"""Orchestrate pretrained detector evaluation and write result artifacts."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from scambranch.eval.constants import (
    ALL_VIEWS,
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    DEFAULT_MODELS,
    LABEL_FRAUD,
    LABEL_LEGITIMATE,
    STREAMING_MODELS,
    VIEW_FULL,
    VIEW_PREFIX_ONLY,
    VIEW_PREFIX_PLUS_REACTION,
    VIEW_STREAMING,
)
from scambranch.eval.data import (
    Conversation,
    complete_sibling_keys,
    load_usable_conversations,
    mean_turns,
    sample_conversations,
    view_turn_sets,
)
from scambranch.eval.metrics import (
    as_binary_label,
    attach_bootstrap,
    branch_metrics,
    classification_metrics,
    evidence_relative_delay,
    fit_threshold,
    per_condition_metrics,
    reaction_score_deltas,
)
from scambranch.eval.resources import hardware_info, summarize_latencies


N_MODELS_FULL_PANEL = 19


@dataclass
class EvalConfig:
    run_dir: Path
    out_dir: Path
    eval_run_id: str
    models: tuple[str, ...] = DEFAULT_MODELS
    views: tuple[str, ...] = ALL_VIEWS
    split: str = "test"
    max_test: int | None = None
    max_dev: int | None = None
    sample_seed: int = 0
    complete_sets_path: Path | None = None
    use_complete_sets_if_present: bool = True
    bootstrap_reps: int = BOOTSTRAP_REPS
    bootstrap_seed: int = BOOTSTRAP_SEED
    device: str | None = None
    dtype: str | None = None
    n_test_universe: int | None = None
    n_dev_universe: int | None = None


@dataclass
class PredRow:
    conversation_id: str
    model: str
    view: str
    score: float
    pred: str
    split: str
    label: str
    language_pair: str
    recipient_condition: str
    scenario_family_id: str
    case_id: str
    history_id: str
    branch_id: str
    parsed: str | None
    invalid: bool
    latency_s: float
    prefix_turn: int | None
    generated: str
    evidence_turn_ids: list[int] = field(default_factory=list)

    def as_jsonl(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "model": self.model,
            "view": self.view,
            "score": _num(self.score),
            "pred": self.pred,
            "split": self.split,
            "prefix_turn": self.prefix_turn,
            "parsed": self.parsed,
            "invalid": self.invalid,
            "latency_s": _num(self.latency_s),
            "label": self.label,
            "language_pair": self.language_pair,
            "recipient_condition": self.recipient_condition,
            "scenario_family_id": self.scenario_family_id,
        }

    def as_metric_row(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "model": self.model,
            "view": self.view,
            "score": self.score,
            "pred": as_binary_label(self.pred),
            "label": as_binary_label(self.label),
            "language_pair": self.language_pair,
            "recipient_condition": self.recipient_condition,
            "scenario_family_id": self.scenario_family_id,
            "case_id": self.case_id,
            "history_id": self.history_id,
            "branch_id": self.branch_id,
            "invalid": self.invalid,
            "prefix_turn": self.prefix_turn,
        }


def _num(value: float) -> float | None:
    if value is None or not isinstance(value, (int, float)):
        return None
    if isinstance(value, bool):
        return float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return float(value)


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, float):
        return _num(obj)
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return jsonable(obj.item())
        except Exception:
            return None
    return obj


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + "\n")


def classification_rows(rows: list[PredRow], view: str) -> list[PredRow]:
    if view != VIEW_STREAMING:
        return [r for r in rows if r.view == view]
    by_id: dict[str, PredRow] = {}
    for row in rows:
        if row.view != VIEW_STREAMING:
            continue
        prev = by_id.get(row.conversation_id)
        if prev is None or (row.prefix_turn or 0) >= (prev.prefix_turn or 0):
            by_id[row.conversation_id] = row
    return list(by_id.values())


def slice_metrics(
    rows: list[PredRow],
    *,
    bootstrap_reps: int,
    bootstrap_seed: int,
    include_bootstrap: bool = True,
) -> dict[str, Any]:
    metric_rows = [r.as_metric_row() for r in rows]
    y = [m["label"] for m in metric_rows]
    p = [m["pred"] for m in metric_rows]
    s = [m["score"] for m in metric_rows]
    inv = [m["invalid"] for m in metric_rows]
    cond = [m["recipient_condition"] for m in metric_rows]
    point = classification_metrics(y, p, s, inv)
    cond_block = per_condition_metrics(y, p, cond, s)
    point["by_condition"] = {c: cond_block[c] for c in cond_block if isinstance(cond_block[c], dict)}
    point["worst_condition_macro_f1"] = cond_block["worst_condition_macro_f1"]
    point["worst_condition_name"] = cond_block["worst_condition_name"]
    point["worst_condition_fraud_recall"] = cond_block["worst_condition_fraud_recall"]
    point["worst_condition_legitimate_fpr"] = cond_block["worst_condition_legitimate_fpr"]
    if include_bootstrap:
        return attach_bootstrap(metric_rows, point, n_reps=bootstrap_reps, seed=bootstrap_seed)
    return point


def metrics_by_language(
    rows: list[PredRow],
    *,
    bootstrap_reps: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    by_lang: dict[str, list[PredRow]] = defaultdict(list)
    for row in rows:
        by_lang[row.language_pair].append(row)
    return {
        lang: slice_metrics(
            by_lang[lang],
            bootstrap_reps=bootstrap_reps,
            bootstrap_seed=bootstrap_seed,
        )
        for lang in sorted(by_lang)
    }


def streaming_delay_metrics(rows: list[PredRow], threshold: float) -> dict[str, Any]:
    by_id: dict[str, list[PredRow]] = defaultdict(list)
    for row in rows:
        if row.view == VIEW_STREAMING:
            by_id[row.conversation_id].append(row)
    items: list[dict[str, Any]] = []
    for cid, group in by_id.items():
        group = sorted(group, key=lambda r: r.prefix_turn or 0)
        evidence_ids = group[0].evidence_turn_ids
        if not evidence_ids:
            continue
        evidence_turn = min(evidence_ids)
        alert_turn = None
        for row in group:
            if row.score >= threshold:
                alert_turn = row.prefix_turn
                break
        items.append(
            {
                "conversation_id": cid,
                "evidence_turn": evidence_turn,
                "alert_turn": alert_turn,
                "label": group[0].label,
            }
        )
    stats = evidence_relative_delay(items)
    stats["n_conversations_with_prefixes"] = float(len(by_id))
    return stats


def project_runtime(
    *,
    median_s: float,
    n_test: int,
    n_dev: int,
    mean_turn_count: float,
    n_models: int = N_MODELS_FULL_PANEL,
) -> dict[str, Any]:
    """Project full-panel time from per-inference latency.

    FULL / PREFIX_ONLY / PREFIX_PLUS_REACTION: one generate each.
    STREAMING: one generate per turn (``mean_turn_count``).
    Threshold fitting uses the same DEV inferences (already included).
    """
    if not math.isfinite(median_s) or median_s <= 0:
        return {"note": "no latency samples"}
    single_views = 3.0
    streaming_calls = max(mean_turn_count, 1.0)
    calls_per_conversation = single_views + streaming_calls
    n_convos = n_test + n_dev
    calls_one_model = n_convos * calls_per_conversation
    seconds_one_model = calls_one_model * median_s
    seconds_all = seconds_one_model * n_models
    naive_four_views = n_convos * 4.0 * median_s * n_models
    return {
        "median_per_inference_s": median_s,
        "mean_turns": mean_turn_count,
        "n_test": n_test,
        "n_dev": n_dev,
        "n_models": n_models,
        "inferences_per_conversation": calls_per_conversation,
        "inferences_one_model": calls_one_model,
        "seconds_one_model": seconds_one_model,
        "hours_one_model": seconds_one_model / 3600.0,
        "seconds_five_models_four_views": seconds_all,
        "hours_five_models_four_views": seconds_all / 3600.0,
        "naive_four_flat_views_hours": naive_four_views / 3600.0,
        "note": (
            "STREAMING costs one inference per turn; the other three views cost one each. "
            "Naive 4×views undercounts STREAMING."
        ),
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number) or math.isinf(number):
        return "—"
    return f"{number:.{digits}f}"


def _ci(block: dict[str, Any] | None) -> str:
    if not block:
        return ""
    lo, hi = block.get("ci_low"), block.get("ci_high")
    if lo is None or hi is None:
        return ""
    return f" [{_fmt(lo)}, {_fmt(hi)}]"


def render_tables(metrics: dict[str, Any]) -> str:
    lines = [
        f"# Detector eval `{metrics.get('eval_run_id')}`",
        "",
        f"- run_dir: `{metrics.get('run_dir')}`",
        f"- n_dev_scored: {metrics.get('n_dev')}",
        f"- n_test_scored: {metrics.get('n_test')}",
        f"- views: {', '.join(metrics.get('views') or [])}",
        f"- models: {', '.join(metrics.get('models') or [])}",
        "",
    ]
    skipped = metrics.get("skipped_models") or []
    if skipped:
        lines.append("## Skipped models")
        lines.append("")
        for item in skipped:
            lines.append(f"- `{item.get('short_id')}` ({item.get('hf_id')}): {item.get('skip_reason')}")
        lines.append("")

    lines.extend(
        [
            "## Thresholds (fit on DEV, max macro-F1)",
            "",
            "| model | view | threshold |",
            "|---|---|---|",
        ]
    )
    for model, views in (metrics.get("thresholds") or {}).items():
        for view, thr in views.items():
            lines.append(f"| {model} | {view} | {_fmt(thr, 4)} |")
    lines.append("")

    lines.extend(
        [
            "## Test quality (pooled)",
            "",
            "| model | view | n | macro-F1 | fraud recall | legit FPR | ROC-AUC | invalid | worst-cond F1 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    slices = metrics.get("slices") or {}
    for model in slices:
        for view, block in slices[model].items():
            pooled = block.get("pooled") or {}
            lines.append(
                "| {model} | {view} | {n} | {f1}{f1ci} | {rec}{recci} | {fpr}{fprci} | {auc} | {inv} | {worst} |".format(
                    model=model,
                    view=view,
                    n=int(pooled.get("n") or 0),
                    f1=_fmt(pooled.get("macro_f1")),
                    f1ci=_ci(pooled.get("macro_f1_ci")),
                    rec=_fmt(pooled.get("fraud_recall")),
                    recci=_ci(pooled.get("fraud_recall_ci")),
                    fpr=_fmt(pooled.get("legitimate_fpr")),
                    fprci=_ci(pooled.get("legitimate_fpr_ci")),
                    auc=_fmt(pooled.get("roc_auc")),
                    inv=_fmt(pooled.get("invalid_output_rate")),
                    worst=_fmt(pooled.get("worst_condition_macro_f1")),
                )
            )
    lines.append("")

    lines.extend(
        [
            "## Per language (test, pooled views listed separately)",
            "",
            "| model | view | language | n | macro-F1 | fraud recall | legit FPR |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for model in slices:
        for view, block in slices[model].items():
            for lang, lang_block in (block.get("by_language") or {}).items():
                lines.append(
                    f"| {model} | {view} | {lang} | {int(lang_block.get('n') or 0)} | "
                    f"{_fmt(lang_block.get('macro_f1'))} | {_fmt(lang_block.get('fraud_recall'))} | "
                    f"{_fmt(lang_block.get('legitimate_fpr'))} |"
                )
    lines.append("")

    lines.extend(
        [
            "## Per recipient condition (test)",
            "",
            "| model | view | condition | n | fraud recall | legit FPR | macro-F1 |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for model in slices:
        for view, block in slices[model].items():
            for cond, cblock in (block.get("pooled", {}).get("by_condition") or {}).items():
                lines.append(
                    f"| {model} | {view} | {cond} | {int(cblock.get('n') or 0)} | "
                    f"{_fmt(cblock.get('fraud_recall'))} | {_fmt(cblock.get('legitimate_fpr'))} | "
                    f"{_fmt(cblock.get('macro_f1'))} |"
                )
    lines.append("")

    lines.extend(
        [
            "## Branch metrics (complete sibling sets, test)",
            "",
            "| model | view | n histories | disagreement | score spread (mean) | all-correct |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for model in slices:
        for view, block in slices[model].items():
            b = block.get("branch") or {}
            lines.append(
                f"| {model} | {view} | {int(b.get('n_complete_histories') or 0)} | "
                f"{_fmt(b.get('branch_disagreement_rate'))} | {_fmt(b.get('branch_score_spread_mean'))} | "
                f"{_fmt(b.get('all_branches_correct_rate'))} |"
            )
    lines.append("")

    lines.extend(
        [
            "## PREFIX_PLUS_REACTION − PREFIX_ONLY (mean Δ P(fraud), test)",
            "",
            "| model | condition / label | mean Δ | 95% CI |",
            "|---|---|---:|---|",
        ]
    )
    for model in slices:
        # reaction lives on PPR slice
        block = (slices[model].get(VIEW_PREFIX_PLUS_REACTION) or {}).get("reaction_delta") or {}
        if not block:
            continue
        lines.append(f"| {model} | all | {_fmt(block.get('mean_delta'))} | |")
        for cond, val in (block.get("by_condition") or {}).items():
            lines.append(f"| {model} | {cond} | {_fmt(val)} | |")
        for lab, val in (block.get("by_label") or {}).items():
            lines.append(f"| {model} | label={lab} | {_fmt(val)} | |")
        cis = block.get("by_condition_label_ci") or {}
        for key, val in (block.get("by_condition_label") or {}).items():
            lines.append(f"| {model} | {key} | {_fmt(val)} | {_ci(cis.get(key))} |")
    lines.append("")
    complete_slices = metrics.get("slices_complete_sets")
    if complete_slices:
        lines.extend(
            [
                "## Complete-set slice (test)",
                "",
                "| model | view | n | macro-F1 | disagreement | all-correct |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for model in complete_slices:
            for view, block in complete_slices[model].items():
                pooled = block.get("pooled") or {}
                b = block.get("branch") or {}
                lines.append(
                    f"| {model} | {view} | {int(pooled.get('n') or 0)} | "
                    f"{_fmt(pooled.get('macro_f1'))} | {_fmt(b.get('branch_disagreement_rate'))} | "
                    f"{_fmt(b.get('all_branches_correct_rate'))} |"
                )
        lines.append("")

    lines.extend(
        [
            "## Streaming delay (test; misses counted)",
            "",
            "| model | eligible | detected | miss rate | mean delay (hits) | pre-evidence alert rate |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model in slices:
        st = (slices[model].get(VIEW_STREAMING) or {}).get("streaming") or {}
        if not st:
            continue
        lines.append(
            f"| {model} | {int(st.get('n_eligible') or 0)} | {int(st.get('n_detected') or 0)} | "
            f"{_fmt(st.get('miss_rate'))} | {_fmt(st.get('mean_delay_detected'))} | "
            f"{_fmt(st.get('pre_evidence_alert_rate'))} |"
        )
    lines.append("")

    proj = metrics.get("runtime_projection") or {}
    if proj:
        lines.extend(
            [
                "## Runtime projection (full panel)",
                "",
                f"- median per-inference: {_fmt(proj.get('median_per_inference_s'), 3)} s",
                f"- mean turns: {_fmt(proj.get('mean_turns'), 2)}",
                f"- assumed test+dev conversations: {proj.get('n_test')}+{proj.get('n_dev')}",
                f"- one model, 4 views: {_fmt(proj.get('hours_one_model'), 2)} h",
                f"- 5 models × 4 views: {_fmt(proj.get('hours_five_models_four_views'), 2)} h",
                f"- note: {proj.get('note')}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def score_split(
    detector: Any,
    conversations: list[Conversation],
    views: Iterable[str],
    split: str,
) -> list[PredRow]:
    rows: list[PredRow] = []
    view_list = tuple(views)
    n = len(conversations)
    for i, conv in enumerate(conversations, start=1):
        for view, turns, prefix_turn in view_turn_sets(conv, view_list):
            if not turns:
                continue
            result = detector.score_turns(conv.conversation_id, turns)
            rows.append(
                PredRow(
                    conversation_id=conv.conversation_id,
                    model=detector.short_id,
                    view=view,
                    score=result.score,
                    pred=LABEL_LEGITIMATE,  # filled after threshold
                    split=split,
                    label=conv.label,
                    language_pair=conv.language_pair,
                    recipient_condition=conv.recipient_condition,
                    scenario_family_id=conv.scenario_family_id,
                    case_id=conv.case_id,
                    history_id=conv.history_id,
                    branch_id=conv.branch_id,
                    parsed=result.parsed,
                    invalid=result.invalid,
                    latency_s=result.latency_s,
                    prefix_turn=prefix_turn if view == VIEW_STREAMING else None,
                    generated=result.generated,
                    evidence_turn_ids=list(conv.evidence_turn_ids),
                )
            )
        print(
            f"  [{split} {i}/{n}] {conv.conversation_id} "
            f"last_latency={rows[-1].latency_s:.3f}s" if rows else f"  [{split} {i}/{n}] {conv.conversation_id}",
            flush=True,
        )
    return rows


def apply_thresholds(rows: list[PredRow], thresholds: dict[str, dict[str, float]]) -> None:
    for row in rows:
        view_key = row.view
        thr = thresholds.get(row.model, {}).get(view_key, 0.5)
        bit = 1 if row.score >= thr else 0
        row.pred = LABEL_FRAUD if bit == 1 else LABEL_LEGITIMATE


def fit_thresholds_from_dev(dev_rows: list[PredRow], views: Iterable[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = defaultdict(dict)
    by_mv: dict[tuple[str, str], list[PredRow]] = defaultdict(list)
    for row in dev_rows:
        by_mv[(row.model, row.view)].append(row)
    for (model, view), group in by_mv.items():
        class_rows = classification_rows(group, view)
        out[model][view] = fit_threshold(
            [r.score for r in class_rows],
            [r.label for r in class_rows],
        )
    # STREAMING uses last-prefix threshold; already handled via classification_rows.
    return {m: dict(v) for m, v in out.items()}


def per_conversation_latency(rows: list[PredRow], split: str = "test") -> dict[str, Any]:
    by_id: dict[str, list[float]] = defaultdict(list)
    by_view: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.split != split:
            continue
        by_id[row.conversation_id].append(row.latency_s)
        by_view[row.view].append(row.latency_s)
    totals = [sum(v) for v in by_id.values()]
    return {
        "n_conversations": float(len(by_id)),
        "per_conversation_sum": summarize_latencies(totals),
        "per_inference": summarize_latencies([r.latency_s for r in rows if r.split == split]),
        "per_view": {view: summarize_latencies(vals) for view, vals in sorted(by_view.items())},
        "per_conversation": [
            {
                "conversation_id": cid,
                "n_inferences": len(vals),
                "sum_s": float(sum(vals)),
                "mean_s": float(sum(vals) / len(vals)) if vals else None,
            }
            for cid, vals in sorted(by_id.items())
        ],
    }


def build_slices(
    test_rows: list[PredRow],
    views: Iterable[str],
    thresholds: dict[str, dict[str, float]],
    *,
    bootstrap_reps: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    slices: dict[str, Any] = {}
    models = sorted({r.model for r in test_rows})
    view_list = list(views)
    prefix_only = [r for r in test_rows if r.view == VIEW_PREFIX_ONLY]
    prefix_plus = [r for r in test_rows if r.view == VIEW_PREFIX_PLUS_REACTION]
    for model in models:
        slices[model] = {}
        model_rows = [r for r in test_rows if r.model == model]
        for view in view_list:
            class_rows = classification_rows(model_rows, view)
            if not class_rows and view != VIEW_STREAMING:
                continue
            block: dict[str, Any] = {}
            if class_rows:
                block["pooled"] = slice_metrics(
                    class_rows,
                    bootstrap_reps=bootstrap_reps,
                    bootstrap_seed=bootstrap_seed,
                )
                block["by_language"] = metrics_by_language(
                    class_rows,
                    bootstrap_reps=bootstrap_reps,
                    bootstrap_seed=bootstrap_seed,
                )
                block["branch"] = branch_metrics([r.as_metric_row() for r in class_rows])
            if view == VIEW_PREFIX_PLUS_REACTION:
                block["reaction_delta"] = reaction_score_deltas(
                    [r.as_metric_row() for r in prefix_only if r.model == model],
                    [r.as_metric_row() for r in prefix_plus if r.model == model],
                    bootstrap_reps=bootstrap_reps,
                    bootstrap_seed=bootstrap_seed,
                )
            if view == VIEW_STREAMING:
                thr = thresholds.get(model, {}).get(VIEW_STREAMING)
                if thr is None:
                    thr = thresholds.get(model, {}).get(VIEW_FULL, 0.5)
                block["streaming"] = streaming_delay_metrics(model_rows, thr)
            slices[model][view] = block
    return slices


def views_for_model(model_name: str, views: Iterable[str]) -> tuple[str, ...]:
    from scambranch.eval.detectors import parse_model_ref, resolve_model_spec

    base, _adapter = parse_model_ref(model_name)
    try:
        short, _spec = resolve_model_spec(base)
    except Exception:
        short = base.rsplit("/", 1)[-1].lower().replace(".", "-")
    allowed = list(views)
    if short not in STREAMING_MODELS:
        allowed = [v for v in allowed if v != VIEW_STREAMING]
    return tuple(allowed)


def _append_progress(out_dir: Path, line: str) -> None:
    from datetime import datetime, timezone

    path = out_dir / "progress.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    text = f"{stamp} {line.rstrip()}"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text, flush=True)


def run_evaluation(config: EvalConfig) -> dict[str, Any]:
    from scambranch.eval.detectors import try_load_detector

    run_dir = Path(config.run_dir)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Always score all usable conversations; complete-set metrics are a subset.
    dev_all = load_usable_conversations(
        run_dir,
        split="dev",
        complete_sets_path=None,
        use_complete_sets_if_present=False,
    )
    test_all = load_usable_conversations(
        run_dir,
        split=config.split,
        complete_sets_path=None,
        use_complete_sets_if_present=False,
    )
    n_dev_universe = len(dev_all)
    n_test_universe = len(test_all)
    if config.max_dev is not None:
        dev = sample_conversations(dev_all, config.max_dev, seed=config.sample_seed)
    else:
        dev = dev_all
    if config.max_test is not None:
        test = sample_conversations(test_all, config.max_test, seed=config.sample_seed)
    else:
        test = test_all

    complete_test_keys = complete_sibling_keys(test_all)
    _append_progress(
        out_dir,
        f"loaded usable dev={n_dev_universe} test={n_test_universe}; "
        f"scoring dev={len(dev)} test={len(test)}; "
        f"complete_test_histories={len(complete_test_keys)}",
    )

    hw = hardware_info()
    all_rows: list[PredRow] = []
    resources: dict[str, Any] = {"hardware": hw, "models": {}, "eval_run_id": config.eval_run_id}
    skipped: list[dict[str, Any]] = []

    def _write_partial() -> dict[str, Any]:
        thresholds = fit_thresholds_from_dev(
            [r for r in all_rows if r.split == "dev"],
            config.views,
        )
        apply_thresholds(all_rows, thresholds)
        test_rows = [r for r in all_rows if r.split == "test"]
        complete_rows = [r for r in test_rows if f"{r.case_id}|{r.language_pair}|{r.history_id}" in complete_test_keys]
        slices_all = build_slices(
            test_rows,
            config.views,
            thresholds,
            bootstrap_reps=config.bootstrap_reps,
            bootstrap_seed=config.bootstrap_seed,
        )
        slices_complete = build_slices(
            complete_rows,
            config.views,
            thresholds,
            bootstrap_reps=config.bootstrap_reps,
            bootstrap_seed=config.bootstrap_seed,
        )
        latency_block = per_conversation_latency(all_rows, split="test")
        medians = [
            (resources["models"][m].get("latency") or {}).get("median_s")
            for m in resources["models"]
            if not resources["models"][m].get("skipped")
        ]
        median_s = next(
            (x for x in medians if isinstance(x, (int, float)) and math.isfinite(x)),
            float("nan"),
        )
        scored_for_turns = test or dev
        projection = project_runtime(
            median_s=float(median_s) if median_s == median_s else float("nan"),
            n_test=config.n_test_universe or n_test_universe,
            n_dev=config.n_dev_universe or n_dev_universe,
            mean_turn_count=mean_turns(scored_for_turns) or mean_turns(test_all + dev_all),
        )
        seen: list[str] = []
        for row in all_rows:
            if row.model not in seen:
                seen.append(row.model)
        metrics = {
            "eval_run_id": config.eval_run_id,
            "run_dir": str(run_dir),
            "n_dev": len(dev),
            "n_test": len(test),
            "n_dev_universe": n_dev_universe,
            "n_test_universe": n_test_universe,
            "n_dev_complete_histories": len(complete_sibling_keys(dev_all)),
            "n_test_complete_histories": len(complete_test_keys),
            "models": seen,
            "views": list(config.views),
            "skipped_models": skipped,
            "thresholds": thresholds,
            "slices": slices_all,
            "slices_all_usable": slices_all,
            "slices_complete_sets": slices_complete,
            "latency_test": latency_block,
            "runtime_projection": projection,
            "bootstrap": {"reps": config.bootstrap_reps, "seed": config.bootstrap_seed},
            "conversation_ids": {
                "dev": [c.conversation_id for c in dev],
                "test": [c.conversation_id for c in test],
            },
        }
        pred_path = out_dir / "predictions.jsonl"
        atomic_write_text(
            pred_path,
            "".join(json.dumps(r.as_jsonl(), ensure_ascii=False) + "\n" for r in all_rows),
        )
        atomic_write_json(out_dir / "metrics.json", metrics)
        atomic_write_json(out_dir / "resources.json", resources)
        atomic_write_text(out_dir / "tables.md", render_tables(metrics))
        return metrics

    for model_name in config.models:
        model_views = views_for_model(model_name, config.views)
        _append_progress(out_dir, f"loading {model_name} views={','.join(model_views)}")
        detector, meta = try_load_detector(model_name, device=config.device, dtype_name=config.dtype)
        if detector is None:
            skipped.append(meta)
            resources["models"][meta["short_id"]] = meta
            _append_progress(out_dir, f"SKIP {meta['short_id']}: {meta['skip_reason']}")
            _write_partial()
            continue
        _append_progress(
            out_dir,
            f"loaded {detector.short_id} params={detector.n_params} "
            f"device={detector.device} dtype={detector.dtype_name} "
            f"load_s={detector.load_time_s:.1f}",
        )
        _append_progress(out_dir, f"scoring DEV ({len(dev)}) {detector.short_id}")
        all_rows.extend(score_split(detector, dev, model_views, "dev"))
        _append_progress(out_dir, f"scoring TEST ({len(test)}) {detector.short_id}")
        all_rows.extend(score_split(detector, test, model_views, "test"))
        resources["models"][detector.short_id] = {
            **meta,
            "params": detector.n_params,
            "load_time_s": detector.load_time_s,
            "device": detector.device,
            "dtype": detector.dtype_name,
            "peak_mps_allocated_bytes": detector.tracker.peak_mps_bytes,
            "peak_rss_bytes": detector.tracker.peak_rss_bytes,
            "verbalizer_tokenization": detector.verbalizer_doc,
            "latency": summarize_latencies(detector.latencies_s),
        }
        _write_partial()
        _append_progress(out_dir, f"wrote partial results after {detector.short_id}")
        try:
            import torch

            del detector.model
            del detector
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    metrics = _write_partial()
    _append_progress(out_dir, f"wrote {out_dir} n_predictions={len(all_rows)}")
    return {"metrics": metrics, "resources": resources, "out_dir": str(out_dir), "n_predictions": len(all_rows)}
