"""Unit tests for detector-eval loader, leakage, and plan-§14 metrics."""

from __future__ import annotations

from pathlib import Path

import pytest

from scambranch.eval.data import (
    assert_no_private_leak,
    build_detector_input,
    build_scored_prompt,
    conversation_from_record,
    is_usable,
    load_complete_set_keys,
    load_usable_conversations,
    render_transcript,
    select_view_turns,
    view_turn_sets,
)
from scambranch.eval.constants import (
    DETECTOR_INSTRUCTION,
    VIEW_CALLER_ONLY,
    VIEW_FULL,
    VIEW_PREFIX_ONLY,
    VIEW_PREFIX_PLUS_REACTION,
    VIEW_STREAMING,
)
from scambranch.eval.metrics import (
    all_branches_correct_rate,
    branch_disagreement_rate,
    branch_score_spread,
    evidence_relative_delay,
    family_bootstrap,
    legitimate_fpr,
    macro_f1,
)


def _record(**overrides):
    rec = {
        "schema_version": "1.0",
        "conversation_id": "jr_01_F_hi_en_h1_cooperative",
        "scenario_family_id": "jr_01",
        "case_id": "jr_01_F",
        "history_id": "h1",
        "branch_id": "cooperative",
        "language_pair": "hi_en",
        "script_profile": "roman",
        "split": "test",
        "turns": [
            {"turn_id": 1, "speaker": "caller", "text": "Namaste, bank se call hai."},
            {"turn_id": 2, "speaker": "recipient", "text": "Haan, boliye."},
            {"turn_id": 3, "speaker": "caller", "text": "Card block ho gaya hai."},
            {"turn_id": 4, "speaker": "recipient", "text": "Theek hai, aage bataiye."},
        ],
        "private_annotations": {
            "scenario_label": "fraud_attempt",
            "recipient_condition": "cooperative",
            "prefix_hash": "DEADBEEFPRIVATEHASH001",
            "branch_after_turn": 3,
            "evidence_turn_ids": [3],
            "prefix_evidence_states": [],
            "observed_outcome": "continued",
            "invariance_eligible": True,
        },
        "provenance": {
            "generator_model_id": "test",
            "prompt_version": "test",
            "scenario_version": "0.1",
            "language_profile_version": "0.1",
            "generation_settings": {
                "disposition": "accepted",
                "label_rationale": "SECRET_RATIONALE_SHOULD_NOT_LEAK",
            },
            "repair_history": [],
            "review_status": "accepted",
        },
    }
    rec.update(overrides)
    return rec


def test_detector_input_no_private_leak():
    rec = _record()
    conv = conversation_from_record(rec)
    detector_input = build_detector_input(conv.conversation_id, conv.turns)
    assert set(detector_input) == {"conversation_id", "turns"}
    for turn in detector_input["turns"]:
        assert set(turn) == {"speaker", "text"}
    user_content, checked = build_scored_prompt(conv.turns, DETECTOR_INSTRUCTION)
    assert_no_private_leak(user_content, checked)
    assert "Caller:" in user_content
    assert "Recipient:" in user_content
    assert "DEADBEEFPRIVATEHASH001" not in user_content
    assert "SECRET_RATIONALE_SHOULD_NOT_LEAK" not in user_content
    assert "private_annotations" not in user_content
    assert "generation_settings" not in user_content
    assert "scenario_label" not in user_content
    assert "fraud_attempt" not in user_content
    assert "cooperative" not in user_content
    leaked = dict(detector_input)
    leaked["scenario_label"] = "fraud_attempt"
    with pytest.raises(AssertionError):
        assert_no_private_leak(user_content, leaked)
    with pytest.raises(AssertionError):
        assert_no_private_leak(user_content + "\nprivate_annotations: x", detector_input)


def test_usable_status_and_views():
    assert is_usable(_record())
    assert is_usable(
        _record(
            provenance={
                **_record()["provenance"],
                "review_status": "accepted",
                "generation_settings": {"disposition": "repaired"},
            }
        )
    )
    assert not is_usable(
        _record(
            provenance={
                **_record()["provenance"],
                "review_status": "rejected",
                "generation_settings": {"disposition": "rejected"},
            }
        )
    )
    turns = _record()["turns"]
    prefix = select_view_turns(turns, VIEW_PREFIX_ONLY, 3)
    assert [t["turn_id"] for t in prefix] == [1, 2, 3]
    reaction = select_view_turns(turns, VIEW_PREFIX_PLUS_REACTION, 3)
    assert [t["turn_id"] for t in reaction] == [1, 2, 3, 4]
    assert reaction[-1]["speaker"] == "recipient"
    full = select_view_turns(turns, VIEW_FULL, 3)
    assert [t["turn_id"] for t in full] == [1, 2, 3, 4]
    caller_only = select_view_turns(turns, VIEW_CALLER_ONLY, 3)
    assert [t["turn_id"] for t in caller_only] == [1, 3]
    assert all(t["speaker"] == "caller" for t in caller_only)
    streaming = list(view_turn_sets(conversation_from_record(_record()), [VIEW_STREAMING]))
    assert [item[2] for item in streaming] == [1, 2, 3, 4]
    rendered = render_transcript(prefix)
    assert rendered.startswith("Caller:")
    assert "Recipient:" in rendered


def test_caller_only_view_strips_recipients_no_private_leak():
    turns = _record()["turns"]
    caller_only = select_view_turns(turns, VIEW_CALLER_ONLY, 3)
    assert [t["turn_id"] for t in caller_only] == [1, 3]
    assert all(t["speaker"] == "caller" for t in caller_only)
    user_content, checked = build_scored_prompt(caller_only, DETECTOR_INSTRUCTION)
    assert_no_private_leak(user_content, checked)
    assert "Caller:" in user_content
    assert "Recipient:" not in user_content
    assert "DEADBEEFPRIVATEHASH001" not in user_content
    assert "cooperative" not in user_content
    assert "fraud_attempt" not in user_content
    items = list(view_turn_sets(conversation_from_record(_record()), [VIEW_CALLER_ONLY]))
    assert len(items) == 1
    assert items[0][0] == VIEW_CALLER_ONLY
    assert [t["turn_id"] for t in items[0][1]] == [1, 3]


def test_complete_sets_manifest(tmp_path: Path):
    path = tmp_path / "complete_sets.json"
    path.write_text('{"histories": [{"case_id": "jr_01_F", "language_pair": "hi_en", "history_id": "h1"}]}')
    keys = load_complete_set_keys(path)
    assert keys == {"jr_01_F|hi_en|h1"}
    sets_path = tmp_path / "complete_sets_sets.json"
    sets_path.write_text(
        '{"sets": [{"family": "jr_01", "case": "fraud", "pair": "hi_en", "history": "h1"}]}'
    )
    assert load_complete_set_keys(sets_path) == {"jr_01_F|hi_en|h1"}
    conv_dir = tmp_path / "conversations"
    conv_dir.mkdir()
    import json

    json.dump(_record(), (conv_dir / "ok.json").open("w"))
    other = _record()
    other["conversation_id"] = "jr_02_F_hi_en_h1_cooperative"
    other["case_id"] = "jr_02_F"
    other["scenario_family_id"] = "jr_02"
    json.dump(other, (conv_dir / "other.json").open("w"))
    loaded = load_usable_conversations(tmp_path, split="test", complete_sets_path=path)
    assert [c.conversation_id for c in loaded] == ["jr_01_F_hi_en_h1_cooperative"]


def test_macro_f1_and_fpr_hand_computed():
    # y = [fraud, fraud, legit, legit], pred = [fraud, legit, legit, legit]
    # fraud: P=1/1, R=1/2, F1 = 2/3
    # legit: P=2/3, R=1, F1 = 4/5
    # macro-F1 = (2/3 + 4/5) / 2 = 11/15
    y = [1, 1, 0, 0]
    pred = [1, 0, 0, 0]
    assert macro_f1(y, pred) == pytest.approx(11 / 15)
    assert legitimate_fpr(y, pred) == pytest.approx(0.0)
    assert legitimate_fpr([0, 0], [1, 0]) == pytest.approx(0.5)


def test_branch_disagreement_and_score_spread():
    # One history disagrees, one is unanimous.
    assert branch_disagreement_rate([[1, 1, 1, 0], [1, 1, 1, 1]]) == pytest.approx(0.5)
    assert branch_score_spread([0.2, 0.5, 0.9, 0.4]) == pytest.approx(0.7)
    assert all_branches_correct_rate([[1, 1, 1, 1], [1, 1, 1, 0]], [[1, 1, 1, 1], [1, 1, 1, 1]]) == pytest.approx(0.5)


def test_delay_counts_misses():
    # evidence 5, first alert 8 → delay 3; evidence 4, no alert → miss (kept).
    stats = evidence_relative_delay(
        [
            {"evidence_turn": 5, "alert_turn": 8},
            {"evidence_turn": 4, "alert_turn": None},
        ]
    )
    assert stats["n_eligible"] == 2
    assert stats["n_miss"] == 1
    assert stats["n_detected"] == 1
    assert stats["delays"] == [3]
    assert stats["mean_delay_detected"] == pytest.approx(3.0)
    assert stats["miss_rate"] == pytest.approx(0.5)
    # pre-evidence: alert 2, evidence 5
    pre = evidence_relative_delay([{"evidence_turn": 5, "alert_turn": 2}])
    assert pre["n_pre_evidence_alert"] == 1
    assert pre["pre_evidence_alert_rate"] == pytest.approx(1.0)


def test_bootstrap_reproducible_with_seed():
    rows = [
        {"scenario_family_id": "f1", "label": 1, "pred": 1},
        {"scenario_family_id": "f1", "label": 0, "pred": 0},
        {"scenario_family_id": "f2", "label": 1, "pred": 0},
        {"scenario_family_id": "f2", "label": 0, "pred": 0},
        {"scenario_family_id": "f3", "label": 1, "pred": 1},
        {"scenario_family_id": "f3", "label": 0, "pred": 1},
    ]

    def _f1(sample):
        return macro_f1([r["label"] for r in sample], [r["pred"] for r in sample])

    a = family_bootstrap(rows, _f1, n_reps=200, seed=123)
    b = family_bootstrap(rows, _f1, n_reps=200, seed=123)
    c = family_bootstrap(rows, _f1, n_reps=200, seed=999)
    assert a == b
    assert a["ci_low"] <= a["mean"] <= a["ci_high"]
    # Different seed must be allowed to differ; with 3 families it almost always does.
    assert a != c or a["mean"] == c["mean"]
