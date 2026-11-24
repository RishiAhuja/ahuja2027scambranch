"""Offline pipeline tests (MockProvider only)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from scambranch.checks import (
    cooperative_refusal,
    detect_leakage,
    detector_has_private_fields,
    prefix_recipient_reaction,
    should_stop_for_repetition,
    token_jaccard,
    to_detector_export,
)
from scambranch.pipeline import (
    _assert_caller_prompt_clean,
    archive_semantic_v02,
    collect_rejection_reasons,
    collect_repair_targets,
    combine_verdict,
    families_and_splits,
    load_generation_config,
    looks_like_placeholder_language_review,
    run_generation,
    semantic_is_v03,
    store_reviews,
)
from scambranch.placeholders import resolve_placeholders
from scambranch.llm import BudgetExceeded, sanitize_schema_for_anthropic, schema_for
from scambranch.render import UnfilledTemplateError, fill_template

CONDITION_NAMES = ("cooperative", "clarifying", "resistant", "suspicious")


def test_unfilled_template_detection():
    with pytest.raises(UnfilledTemplateError):
        fill_template("hello {{missing}}", {})
    assert fill_template("hello {{name}}", {"name": "x"}) == "hello x"


def test_leakage_detector():
    turns = [
        {"turn_id": 1, "speaker": "caller", "text": "Namaste, bank se call hai."},
        {"turn_id": 2, "speaker": "recipient", "text": "Haan."},
        {"turn_id": 7, "speaker": "caller", "text": "This is a scam, please share OTP."},
        {"turn_id": 8, "speaker": "recipient", "text": "Yeh scam lag raha hai."},
    ]
    fraud_caller = detect_leakage(turns, condition="cooperative", branch_after=7)
    assert fraud_caller["prefix_broken"] is True
    assert any("label_leakage" in i["flags"] for i in fraud_caller["issues"] if i["speaker"] == "caller")

    suspicious_ok = detect_leakage(
        [
            {"turn_id": 8, "speaker": "recipient", "text": "Yeh scam lag raha hai."},
            {"turn_id": 9, "speaker": "caller", "text": "Main help ke liye hoon."},
        ],
        condition="suspicious",
        branch_after=7,
    )
    assert suspicious_ok["ok"] is True

    brackets = detect_leakage(
        [{"turn_id": 3, "speaker": "caller", "text": "[aside] bolta hai"}],
        condition="cooperative",
        branch_after=7,
    )
    assert brackets["ok"] is False


def test_dry_run_end_to_end_and_invariants(tmp_path: Path):
    result = run_generation(
        run_id="test_dry1",
        families=["bp_01"],
        pairs=["hi_en"],
        conditions=list(CONDITION_NAMES),
        dry_run=True,
        sync_review=True,
        budget=3.0,
        workers=2,
        run_root=tmp_path,
        history_ids=["h1"],
    )
    conv_dir = Path(result["run_dir"]) / "conversations"
    records = [json.loads(p.read_text()) for p in sorted(conv_dir.glob("*.json"))]
    assert len(records) == 8  # 2 cases × 4 conditions

    by_case: dict[str, list[dict]] = {}
    for rec in records:
        by_case.setdefault(rec["case_id"], []).append(rec)
    for case_id, sibs in by_case.items():
        hashes = {s["private_annotations"]["prefix_hash"] for s in sibs}
        assert len(hashes) == 1, f"prefix hash mismatch for {case_id}"
        assert all(len(h) == 64 for h in hashes)

    export_path = Path(result["run_dir"]) / "detector_export.jsonl"
    rows = [json.loads(line) for line in export_path.read_text().splitlines() if line.strip()]
    assert rows
    for row in rows:
        assert set(row.keys()) == {"conversation_id", "turns"}
        assert detector_has_private_fields(row) is False
        from scambranch.checks import schema_errors

        assert not schema_errors(row, "detector")

    prompts_dir = Path(result["run_dir"]) / "prompts"
    caller_files = list(prompts_dir.glob("*_caller_*.json"))
    assert caller_files
    for path in caller_files:
        payload = json.loads(path.read_text())
        _assert_caller_prompt_clean(payload.get("system", ""), payload.get("user", ""))

    # Full record → detector export must drop private fields
    sample_export = to_detector_export(records[0])
    assert "private_annotations" not in sample_export
    assert "provenance" not in sample_export


def test_prefix_recipient_reaction_and_cooperative_refusal():
    prefix_bad = [
        {"turn_id": 6, "speaker": "recipient", "text": "Message aaya hai lekin main share nahi karunga."},
        {"turn_id": 7, "speaker": "caller", "text": "OTP bata dijiye."},
        {"turn_id": 8, "speaker": "recipient", "text": "Haan, padh deta hoon."},
    ]
    assert prefix_recipient_reaction(prefix_bad, branch_after=7, pair="hi_en") is True
    prefix_ok = [
        {"turn_id": 6, "speaker": "recipient", "text": "Haan, message aa gaya hai."},
        {"turn_id": 7, "speaker": "caller", "text": "OTP bata dijiye."},
    ]
    assert prefix_recipient_reaction(prefix_ok, branch_after=7, pair="hi_en") is False

    ta_prefix = [
        {"turn_id": 4, "speaker": "recipient", "text": "Naan solla maatten."},
    ]
    assert prefix_recipient_reaction(ta_prefix, branch_after=7, pair="ta_en") is True

    coop_refuse = [
        {"turn_id": 7, "speaker": "caller", "text": "OTP please."},
        {"turn_id": 8, "speaker": "recipient", "text": "Nahi dunga, main branch jaunga."},
    ]
    assert cooperative_refusal(coop_refuse, branch_after=7, pair="hi_en", condition="cooperative") is True
    coop_ok = [
        {"turn_id": 7, "speaker": "caller", "text": "OTP please."},
        {"turn_id": 8, "speaker": "recipient", "text": "Haan, main padh deta hoon."},
    ]
    assert cooperative_refusal(coop_ok, branch_after=7, pair="hi_en", condition="cooperative") is False
    assert cooperative_refusal(coop_refuse, branch_after=7, pair="hi_en", condition="resistant") is False

    te_refuse = [
        {"turn_id": 8, "speaker": "recipient", "text": "Nenu cheppanu, official number ki call chestha."},
    ]
    assert cooperative_refusal(te_refuse, branch_after=7, pair="te_en", condition="cooperative") is True


def _sem(overall: str = "accept", failed: list[str] | None = None, targets: list[dict] | None = None):
    names = (
        "evidence_present",
        "evidence_not_invalidated",
        "facts_consistent",
        "condition_followed",
        "caller_within_limits",
        "no_leakage",
        "plausible_ending",
    )
    failed_set = set(failed or [])
    checks = {}
    for name in names:
        extra = {"turns": [7]} if name == "evidence_present" else {}
        extra["pass"] = name not in failed_set
        extra["note"] = "fail" if name in failed_set else ""
        checks[name] = extra
    return {"overall": overall, "repair_targets": targets or [], "checks": checks}


def _lang(
    overall: str = "repair",
    flags: list[dict] | None = None,
    switching: int = 3,
    grammar: int = 3,
):
    return {
        "overall": overall,
        "turn_flags": flags or [],
        "switching_naturalness": switching,
        "grammatical_acceptability": grammar,
    }


def _auto(**overrides):
    checks = {
        "prefix_broken": False,
        "prefix_script": False,
        "prefix_recipient_reaction": False,
        "cooperative_refusal": False,
        "repair_candidate": False,
        "prefix_hash_ok": True,
        "english_share_ok": True,
        "script_issues": [],
        "leakage": {"ok": True, "issues": [], "prefix_broken": False},
    }
    checks.update(overrides)
    return checks


def test_language_gate_hard_flags_only():
    grammar_flag = {
        "turn_id": 8,
        "type": "grammar",
        "span": "unusual online transaction",
        "note": "purist complaint",
    }
    script_flag = {"turn_id": 9, "type": "script", "span": "रद्द", "note": "devanagari"}
    artifact_flag = {"turn_id": 8, "type": "artifact", "span": "[OTP]", "note": "brackets"}

    soft_lang = _lang(overall="repair", flags=[grammar_flag], switching=3, grammar=3)
    assert (
        combine_verdict(_sem(), soft_lang, _auto(), language_gate="hard_flags_only")
        == "accept"
    )
    assert (
        combine_verdict(_sem(), soft_lang, _auto(), language_gate="scores") == "repair"
    )

    hard_lang = _lang(overall="repair", flags=[script_flag], switching=4, grammar=4)
    assert (
        combine_verdict(_sem(), hard_lang, _auto(), language_gate="hard_flags_only")
        == "repair"
    )

    score_reject = _lang(overall="reject", flags=[], switching=2, grammar=4)
    assert (
        combine_verdict(_sem(), score_reject, _auto(), language_gate="hard_flags_only")
        == "repair"
    )
    assert (
        combine_verdict(
            _sem(), score_reject, _auto(), language_gate="hard_flags_only_no_scores"
        )
        == "accept"
    )
    unconfirmed_script = _lang(
        overall="repair",
        flags=[{"turn_id": 9, "type": "script", "span": "रद्द", "note": "llm only"}],
        switching=4,
        grammar=4,
    )
    assert (
        combine_verdict(
            _sem(), unconfirmed_script, _auto(), language_gate="hard_flags_only_no_scores"
        )
        == "accept"
    )
    confirmed_script = _lang(
        overall="repair",
        flags=[{"turn_id": 9, "type": "script", "span": "रद्द", "note": "agrees"}],
        switching=4,
        grammar=4,
    )
    assert (
        combine_verdict(
            _sem(),
            confirmed_script,
            _auto(script_issues=[{"turn_id": 9, "scripts": ["devanagari"]}]),
            language_gate="hard_flags_only_no_scores",
        )
        == "repair"
    )
    reject_but_soft_score = _lang(overall="reject", flags=[grammar_flag], switching=3, grammar=3)
    assert (
        combine_verdict(
            _sem(), reject_but_soft_score, _auto(), language_gate="hard_flags_only"
        )
        == "accept"
    )

    turns = [
        {"turn_id": 7, "speaker": "caller", "text": "OTP batao."},
        {"turn_id": 8, "speaker": "recipient", "text": "unusual online transaction hai."},
        {"turn_id": 9, "speaker": "caller", "text": "Cancellation kar dijiye."},
    ]
    mixed_lang = _lang(
        overall="repair",
        flags=[grammar_flag, script_flag, artifact_flag],
        switching=3,
        grammar=3,
    )
    sem = _sem(overall="repair", targets=[{"turn_id": 8, "problem": "condition_followed"}])
    targets = collect_repair_targets(
        sem=sem,
        lang=mixed_lang,
        checks=_auto(),
        turns=turns,
        branch_after=7,
        language_gate="hard_flags_only",
    )
    problems = [t["problem"] for t in targets]
    assert any("condition_followed" in p for p in problems)
    assert any(p.startswith("script:") for p in problems)
    assert any(p.startswith("artifact:") for p in problems)
    assert not any(p.startswith("grammar:") for p in problems)

    score_targets = collect_repair_targets(
        sem=_sem(),
        lang=soft_lang,
        checks=_auto(),
        turns=turns,
        branch_after=7,
        language_gate="scores",
    )
    assert any(t["problem"].startswith("grammar:") for t in score_targets)

    reasons = collect_rejection_reasons(
        _sem(overall="repair", failed=["condition_followed"]),
        hard_lang,
        _auto(cooperative_refusal=True, repair_candidate=True),
        language_gate="hard_flags_only",
    )
    assert reasons == ["condition_followed", "cooperative_refusal", "script"]


def test_budget_cap_raises(tmp_path: Path):
    with pytest.raises(BudgetExceeded):
        run_generation(
            run_id="test_budget",
            families=["bp_01"],
            pairs=["hi_en"],
            conditions=["cooperative"],
            dry_run=True,
            sync_review=True,
            budget=0.0000001,
            workers=1,
            run_root=tmp_path,
            history_ids=["h1"],
        )


def test_repetition_stop():
    same = [
        {"turn_id": 8, "speaker": "caller", "text": "Please share the OTP now sir."},
    ]
    repeat = {"turn_id": 10, "speaker": "caller", "text": "Please share the OTP now sir."}
    assert token_jaccard(same[0]["text"], repeat["text"]) >= 0.6
    assert should_stop_for_repetition(same, repeat) is True
    different = {"turn_id": 10, "speaker": "caller", "text": "Theek hai, main official number se confirm karunga."}
    assert should_stop_for_repetition(same, different) is False

    last4 = [
        {"turn_id": 8, "speaker": "caller", "text": "OTP share kijiye please."},
        {"turn_id": 9, "speaker": "recipient", "text": "Haan theek hai main de deta hoon."},
        {"turn_id": 10, "speaker": "caller", "text": "OTP share kijiye please."},
    ]
    pair2 = {"turn_id": 11, "speaker": "recipient", "text": "Haan theek hai main de deta hoon."}
    assert should_stop_for_repetition(last4, pair2) is True


def test_caller_prompt_allows_cooperative_bank_in_transcript():
    system = "You play the caller. Output JSON only."
    user = (
        "## Your role (private)\n"
        '{"caller_objective": "verify Narmada Cooperative Bank merger"}\n\n'
        "## Conversation so far\n"
        "1 caller: Narmada Cooperative Bank se call hai.\n"
        "8 recipient: Yeh call suspicious lag raha hai.\n\n"
        "## Task\n"
        "Write your next turn only.\n"
    )
    _assert_caller_prompt_clean(system, user)
    with pytest.raises(RuntimeError, match="suspicious"):
        _assert_caller_prompt_clean(
            system,
            user + "\nFollow the suspicious recipient condition.\n",
        )


def test_sanitize_schema_for_anthropic_strips_integer_bounds():
    schema = schema_for("language_review")
    assert "minimum" in json.dumps(schema)
    cleaned = sanitize_schema_for_anthropic(schema)
    blob = json.dumps(cleaned)
    assert "minimum" not in blob
    assert "maximum" not in blob
    assert cleaned["properties"]["switching_naturalness"] == {"type": "integer"}
    assert "turn_flags" in cleaned["properties"]


def test_missing_language_review_is_null_and_does_not_gate():
    assert looks_like_placeholder_language_review(None) is True
    assert looks_like_placeholder_language_review(
        {
            "overall": "repair",
            "turn_flags": [],
            "switching_naturalness": 3,
            "grammatical_acceptability": 3,
        }
    )
    assert looks_like_placeholder_language_review(
        {
            "overall": "repair",
            "turn_flags": [{"turn_id": 8, "type": "grammar", "span": "x", "note": "y"}],
            "switching_naturalness": 3,
            "grammatical_acceptability": 3,
        }
    ) is False
    gs: dict = {"reviews": {"semantic_v02": {"overall": "reject"}}}
    store_reviews(gs, _sem(), None)
    assert gs["reviews"]["semantic_v02"]["overall"] == "reject"
    assert gs["reviews"]["language"] is None
    assert gs["review_missing"]["language"] is True
    assert gs["reviews"]["semantic"]["overall"] == "accept"
    assert (
        combine_verdict(_sem(), None, _auto(), language_gate="hard_flags_only")
        == "accept"
    )
    reasons = collect_rejection_reasons(
        _sem(), None, _auto(), language_gate="hard_flags_only"
    )
    assert reasons == []


def test_archive_semantic_v02_and_version():
    gs: dict = {"reviews": {"semantic": _sem(overall="reject"), "language": None}}
    assert archive_semantic_v02(gs) is True
    assert gs["reviews"]["semantic_v02"]["overall"] == "reject"
    store_reviews(gs, _sem(overall="accept"), None)
    assert gs["reviews"]["semantic_v02"]["overall"] == "reject"
    assert gs["reviews"]["semantic"]["overall"] == "accept"
    gs["semantic_review_version"] = "semrev-0.3"
    assert semantic_is_v03(gs) is True
    assert archive_semantic_v02(gs) is False


def test_histories_and_splits_differ():
    card = json.loads(
        (Path(__file__).resolve().parents[1] / "scenarios" / "json" / "bp_01.json").read_text()
    )
    _r1, m1, p1 = resolve_placeholders(
        card, family="bp_01", case="fraud", pair="hi_en", history_id="h1", seed=42
    )
    _r2, m2, p2 = resolve_placeholders(
        card, family="bp_01", case="fraud", pair="hi_en", history_id="h2", seed=42
    )
    assert m1 != m2
    assert (p1["name"], p1["age"], p1["occupation"]) != (p2["name"], p2["age"], p2["occupation"])
    fams, splits = families_and_splits(load_generation_config(), ["bp_01", "bp_05"])
    assert "bp_01" in fams
    assert splits["bp_01"] == "dev"
    assert splits["bp_05"] == "test"
