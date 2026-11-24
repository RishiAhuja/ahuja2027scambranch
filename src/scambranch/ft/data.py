"""Train-split arm selection for matched-size LoRA experiments (RQ3).

Arms share the same number of training chats ``N``:
- unbranched: one usable branch per distinct train history
- branched / branched_consistency: complete 4-sibling train histories only
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from scambranch.eval.constants import CONDITIONS, FRAUD_LABELS
from scambranch.eval.data import Conversation, complete_sibling_keys

ARM_UNBRANCHED = "unbranched"
ARM_BRANCHED = "branched"
ARM_BRANCHED_CONSISTENCY = "branched_consistency"
FT_ARMS = (ARM_UNBRANCHED, ARM_BRANCHED, ARM_BRANCHED_CONSISTENCY)

TARGET_N = 432


def _is_fraud(label: str) -> bool:
    return label in FRAUD_LABELS or label == "fraud"


def _fraud_bit(conv: Conversation) -> int:
    return 1 if _is_fraud(conv.label) else 0


def compute_matched_n(
    train: Sequence[Conversation],
    *,
    target_n: int = TARGET_N,
) -> dict[str, Any]:
    """Largest N ≤ target that all arms can achieve (multiple of 4)."""
    by_hist: dict[str, list[Conversation]] = defaultdict(list)
    for conv in train:
        if conv.split != "train":
            raise ValueError(f"non-train conversation in pool: {conv.conversation_id} split={conv.split}")
        by_hist[conv.history_key].append(conv)
    n_histories = len(by_hist)
    complete = complete_sibling_keys(train)
    n_complete = len(complete)
    n_branched_max = n_complete * 4
    raw = min(int(target_n), n_histories, n_branched_max)
    # B/C require whole sibling groups → N must be divisible by 4.
    matched = (raw // 4) * 4
    return {
        "target_n": int(target_n),
        "n_train_usable": len(train),
        "n_histories": n_histories,
        "n_complete_histories": n_complete,
        "n_branched_max": n_branched_max,
        "matched_n": matched,
    }


def _balance_summary(conversations: Sequence[Conversation]) -> dict[str, Any]:
    return {
        "n": len(conversations),
        "n_histories": len({c.history_key for c in conversations}),
        "by_label": dict(Counter(c.label for c in conversations)),
        "by_language": dict(Counter(c.language_pair for c in conversations)),
        "by_condition": dict(Counter(c.recipient_condition for c in conversations)),
        "conversation_ids": [c.conversation_id for c in conversations],
        "history_keys": sorted({c.history_key for c in conversations}),
    }


def _pick_balanced(
    candidates: list[Conversation],
    n: int,
    rng: random.Random,
    *,
    key_fn,
) -> list[Conversation]:
    """Round-robin across stratified buckets, then fill."""
    if n <= 0:
        return []
    if n >= len(candidates):
        return list(candidates)
    buckets: dict[Any, list[Conversation]] = defaultdict(list)
    for conv in candidates:
        buckets[key_fn(conv)].append(conv)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    keys = sorted(buckets)
    rng.shuffle(keys)
    chosen: list[Conversation] = []
    while len(chosen) < n and any(buckets[k] for k in keys):
        progressed = False
        for key in keys:
            if len(chosen) >= n:
                break
            if buckets[key]:
                chosen.append(buckets[key].pop())
                progressed = True
        if not progressed:
            break
    if len(chosen) < n:
        remaining = [c for c in candidates if c not in chosen]
        rng.shuffle(remaining)
        chosen.extend(remaining[: n - len(chosen)])
    return chosen


def _select_unbranched(
    train: Sequence[Conversation],
    n: int,
    rng: random.Random,
) -> list[Conversation]:
    by_hist: dict[str, list[Conversation]] = defaultdict(list)
    for conv in train:
        by_hist[conv.history_key].append(conv)
    # One chat per history, chosen with preference for label/lang/condition balance.
    picks: list[Conversation] = []
    for members in by_hist.values():
        # Stratify within history by condition then pick one.
        picks.append(rng.choice(members))
    # Rebalance the set of history-level picks toward label × language × condition.
    selected = _pick_balanced(
        picks,
        n,
        rng,
        key_fn=lambda c: (c.language_pair, _fraud_bit(c), c.recipient_condition),
    )
    return sorted(selected, key=lambda c: c.conversation_id)


def _select_branched_histories(
    train: Sequence[Conversation],
    n_chats: int,
    rng: random.Random,
) -> list[Conversation]:
    if n_chats % 4 != 0:
        raise ValueError(f"branched N must be multiple of 4, got {n_chats}")
    n_hist = n_chats // 4
    by_hist: dict[str, list[Conversation]] = defaultdict(list)
    for conv in train:
        by_hist[conv.history_key].append(conv)
    complete_keys = sorted(complete_sibling_keys(train))
    # Build one representative per complete history for balancing (label × language).
    reps: list[Conversation] = []
    for key in complete_keys:
        members = by_hist[key]
        by_cond = {m.recipient_condition: m for m in members}
        if not set(CONDITIONS) <= set(by_cond):
            continue
        # Prefer cooperative as the balance representative (arbitrary stable pick).
        reps.append(by_cond.get("cooperative") or next(iter(by_cond.values())))
    chosen_reps = _pick_balanced(
        reps,
        n_hist,
        rng,
        key_fn=lambda c: (c.language_pair, _fraud_bit(c)),
    )
    out: list[Conversation] = []
    for rep in chosen_reps:
        members = by_hist[rep.history_key]
        by_cond = {m.recipient_condition: m for m in members}
        for cond in CONDITIONS:
            out.append(by_cond[cond])
    return sorted(out, key=lambda c: c.conversation_id)


@dataclass
class SelectionResult:
    arm: str
    seed: int
    n: int
    conversations: list[Conversation]
    summary: dict[str, Any] = field(default_factory=dict)
    sibling_groups: list[list[Conversation]] | None = None

    def to_config(self) -> dict[str, Any]:
        payload = {
            "arm": self.arm,
            "seed": self.seed,
            "n": self.n,
            **{k: v for k, v in self.summary.items() if k != "conversation_ids"},
            "conversation_ids": [c.conversation_id for c in self.conversations],
        }
        if self.sibling_groups is not None:
            payload["sibling_group_ids"] = [
                [c.conversation_id for c in grp] for grp in self.sibling_groups
            ]
        return payload


def select_arm(
    train: Sequence[Conversation],
    arm: str,
    *,
    seed: int,
    n: int | None = None,
    target_n: int = TARGET_N,
) -> SelectionResult:
    """Select training chats for one arm. Raises if any non-train split sneaks in."""
    if arm not in FT_ARMS:
        raise ValueError(f"unknown arm {arm!r}; known={FT_ARMS}")
    for conv in train:
        if conv.split != "train":
            raise ValueError(f"dev/test leakage: {conv.conversation_id} split={conv.split}")
    caps = compute_matched_n(train, target_n=target_n)
    matched = caps["matched_n"]
    if matched <= 0:
        raise RuntimeError(f"no matched N available: {caps}")
    use_n = matched if n is None else int(n)
    if use_n > matched:
        raise ValueError(f"requested n={use_n} exceeds matched_n={matched}")
    if arm != ARM_UNBRANCHED and use_n % 4 != 0:
        raise ValueError(f"arm {arm} requires n%4==0, got {use_n}")

    rng = random.Random(seed)
    sibling_groups: list[list[Conversation]] | None = None
    if arm == ARM_UNBRANCHED:
        chosen = _select_unbranched(train, use_n, rng)
    else:
        chosen = _select_branched_histories(train, use_n, rng)
        # Group into sibling quartets for consistency arm / batching.
        by_hist: dict[str, list[Conversation]] = defaultdict(list)
        for conv in chosen:
            by_hist[conv.history_key].append(conv)
        sibling_groups = []
        for key in sorted(by_hist):
            members = by_hist[key]
            by_cond = {m.recipient_condition: m for m in members}
            if set(CONDITIONS) <= set(by_cond):
                sibling_groups.append([by_cond[c] for c in CONDITIONS])

    # Hard invariants.
    assert len(chosen) == use_n, f"expected {use_n} chats, got {len(chosen)}"
    if arm == ARM_UNBRANCHED:
        assert len({c.history_key for c in chosen}) == use_n
    else:
        assert len({c.history_key for c in chosen}) == use_n // 4
        assert sibling_groups is not None and len(sibling_groups) == use_n // 4

    summary = _balance_summary(chosen)
    summary["matched_caps"] = caps
    return SelectionResult(
        arm=arm,
        seed=seed,
        n=use_n,
        conversations=chosen,
        summary=summary,
        sibling_groups=sibling_groups,
    )


def assert_no_split_leakage(
    selected: Iterable[Conversation],
    *,
    allowed_split: str = "train",
) -> None:
    bad = [c for c in selected if c.split != allowed_split]
    if bad:
        raise AssertionError(
            f"split leakage: {[c.conversation_id for c in bad[:5]]} split≠{allowed_split}"
        )
