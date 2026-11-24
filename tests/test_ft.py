"""Unit tests for LoRA arm selection (matched N, no leakage, group constraints)."""

from __future__ import annotations

import pytest

from scambranch.eval.data import conversation_from_record
from scambranch.ft.data import (
    ARM_BRANCHED,
    ARM_BRANCHED_CONSISTENCY,
    ARM_UNBRANCHED,
    FT_ARMS,
    assert_no_split_leakage,
    compute_matched_n,
    select_arm,
)


CONDITIONS = ("cooperative", "clarifying", "resistant", "suspicious")


def _base_record(**overrides):
    rec = {
        "schema_version": "1.0",
        "conversation_id": "fam_01_F_hi_en_h1_cooperative",
        "scenario_family_id": "fam_01",
        "case_id": "fam_01_F",
        "history_id": "h1",
        "branch_id": "cooperative",
        "language_pair": "hi_en",
        "script_profile": "roman",
        "split": "train",
        "turns": [
            {"turn_id": 1, "speaker": "caller", "text": "Bank se call."},
            {"turn_id": 2, "speaker": "recipient", "text": "Haan."},
            {"turn_id": 3, "speaker": "caller", "text": "OTP do."},
            {"turn_id": 4, "speaker": "recipient", "text": "Theek."},
        ],
        "private_annotations": {
            "scenario_label": "fraud_attempt",
            "recipient_condition": "cooperative",
            "prefix_hash": "HASH",
            "branch_after_turn": 3,
            "evidence_turn_ids": [3],
            "invariance_eligible": True,
        },
        "provenance": {
            "generator_model_id": "test",
            "prompt_version": "test",
            "scenario_version": "0.1",
            "language_profile_version": "0.1",
            "generation_settings": {"disposition": "accepted"},
            "repair_history": [],
            "review_status": "accepted",
        },
    }
    rec.update(overrides)
    return rec


def _make_pool(
    *,
    n_complete_fraud: int = 6,
    n_complete_legit: int = 6,
    n_incomplete: int = 4,
    languages: tuple[str, ...] = ("hi_en", "ta_en", "te_en"),
):
    """Build train conversations with complete + incomplete histories."""
    out = []
    fam = 0
    # Complete 4-sibling histories.
    for label, n_hist in (("fraud_attempt", n_complete_fraud), ("legitimate", n_complete_legit)):
        for i in range(n_hist):
            fam += 1
            lang = languages[i % len(languages)]
            letter = "F" if label == "fraud_attempt" else "L"
            case = f"fam_{fam:02d}_{letter}"
            for cond in CONDITIONS:
                rec = _base_record(
                    conversation_id=f"{case}_{lang}_h1_{cond}",
                    scenario_family_id=f"fam_{fam:02d}",
                    case_id=case,
                    history_id="h1",
                    branch_id=cond,
                    language_pair=lang,
                    private_annotations={
                        "scenario_label": label,
                        "recipient_condition": cond,
                        "prefix_hash": f"H{fam}",
                        "branch_after_turn": 3,
                        "evidence_turn_ids": [3],
                        "invariance_eligible": True,
                    },
                )
                out.append(conversation_from_record(rec))
    # Incomplete (single condition) histories for unbranched padding.
    for i in range(n_incomplete):
        fam += 1
        lang = languages[i % len(languages)]
        case = f"fam_{fam:02d}_F"
        rec = _base_record(
            conversation_id=f"{case}_{lang}_h1_cooperative",
            scenario_family_id=f"fam_{fam:02d}",
            case_id=case,
            history_id="h1",
            branch_id="cooperative",
            language_pair=lang,
            private_annotations={
                "scenario_label": "fraud_attempt",
                "recipient_condition": "cooperative",
                "prefix_hash": f"I{fam}",
                "branch_after_turn": 3,
                "evidence_turn_ids": [3],
                "invariance_eligible": True,
            },
        )
        out.append(conversation_from_record(rec))
    return out


def test_matched_n_multiple_of_four_and_capped():
    pool = _make_pool(n_complete_fraud=5, n_complete_legit=5, n_incomplete=3)
    # histories = 10 complete + 3 incomplete = 13; branched max = 40
    caps = compute_matched_n(pool, target_n=432)
    assert caps["n_histories"] == 13
    assert caps["n_complete_histories"] == 10
    assert caps["matched_n"] == 12  # min(432,13,40) floored to *4


def test_unbranched_one_per_history_matched_n():
    pool = _make_pool()
    caps = compute_matched_n(pool, target_n=16)
    sel = select_arm(pool, ARM_UNBRANCHED, seed=0, target_n=16)
    assert sel.n == caps["matched_n"]
    assert len(sel.conversations) == sel.n
    assert len({c.history_key for c in sel.conversations}) == sel.n
    assert_no_split_leakage(sel.conversations)


def test_branched_only_complete_groups():
    pool = _make_pool()
    for arm in (ARM_BRANCHED, ARM_BRANCHED_CONSISTENCY):
        sel = select_arm(pool, arm, seed=1, target_n=16)
        assert sel.n % 4 == 0
        assert sel.sibling_groups is not None
        assert len(sel.sibling_groups) == sel.n // 4
        for grp in sel.sibling_groups:
            assert len(grp) == 4
            assert {c.recipient_condition for c in grp} == set(CONDITIONS)
            assert len({c.history_key for c in grp}) == 1
        # No incomplete-only histories.
        hist_sizes = {}
        by_hist = {}
        for c in pool:
            by_hist.setdefault(c.history_key, []).append(c)
        for c in sel.conversations:
            assert len({x.recipient_condition for x in by_hist[c.history_key]}) == 4


def test_arms_share_matched_n():
    pool = _make_pool()
    ns = []
    for arm in FT_ARMS:
        sel = select_arm(pool, arm, seed=2, target_n=20)
        ns.append(sel.n)
    assert len(set(ns)) == 1


def test_rejects_dev_test_leakage():
    pool = _make_pool(n_complete_fraud=2, n_complete_legit=2, n_incomplete=0)
    rec = _base_record(split="dev", conversation_id="leak_dev")
    pool = list(pool) + [conversation_from_record(rec)]
    with pytest.raises(ValueError, match="non-train"):
        compute_matched_n(pool, target_n=8)


def test_select_arm_rejects_non_train_in_pool():
    pool = _make_pool(n_complete_fraud=2, n_complete_legit=2, n_incomplete=1)
    bad = conversation_from_record(_base_record(split="test", conversation_id="bad_test"))
    with pytest.raises(ValueError):
        select_arm(list(pool) + [bad], ARM_UNBRANCHED, seed=0, target_n=8)
