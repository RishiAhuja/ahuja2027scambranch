"""Load usable conversations and build detector-facing views.

Private labels live in ``private_annotations`` and
``provenance.generation_settings``. Detector inputs use only turn
``speaker`` + ``text`` (rendered as ``Caller:`` / ``Recipient:``).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from scambranch.checks import detector_has_private_fields
from scambranch.eval.constants import (
    CONDITIONS,
    PRIVATE_PROMPT_MARKERS,
    REJECTED_STATUSES,
    SPEAKER_LABELS,
    USABLE_STATUSES,
    VIEW_CALLER_ONLY,
    VIEW_FULL,
    VIEW_PREFIX_ONLY,
    VIEW_PREFIX_PLUS_REACTION,
    VIEW_STREAMING,
)


@dataclass
class Conversation:
    conversation_id: str
    scenario_family_id: str
    case_id: str
    history_id: str
    branch_id: str
    language_pair: str
    split: str
    turns: list[dict[str, Any]]
    label: str
    recipient_condition: str
    branch_after_turn: int
    evidence_turn_ids: list[int]
    invariance_eligible: bool
    review_status: str
    disposition: str
    prefix_hash: str | None = None
    source_path: str | None = None

    @property
    def history_key(self) -> str:
        return f"{self.case_id}|{self.language_pair}|{self.history_id}"

    @property
    def is_fraud(self) -> bool:
        return self.label in {"fraud_attempt", "fraud"}


def is_usable(record: dict[str, Any]) -> bool:
    """Usable iff review_status or disposition is accepted/repaired, and neither is rejected."""
    provenance = record.get("provenance") or {}
    review = provenance.get("review_status")
    disposition = (provenance.get("generation_settings") or {}).get("disposition")
    if review in REJECTED_STATUSES or disposition in REJECTED_STATUSES:
        return False
    return review in USABLE_STATUSES or disposition in USABLE_STATUSES


def history_key_from_record(record: dict[str, Any]) -> str:
    return f"{record['case_id']}|{record['language_pair']}|{record['history_id']}"


def load_complete_set_keys(path: Path) -> set[str]:
    """Parse an optional complete-sets manifest.

    Accepted shapes:
    - list of history-key strings (``case_id|language_pair|history_id``)
    - list of objects with those fields
    - object wrapping one of the above under ``histories`` / ``history_keys`` / ``complete_sets``
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    items: Any
    if isinstance(raw, dict):
        items = (
            raw.get("histories")
            or raw.get("history_keys")
            or raw.get("complete_sets")
            or raw.get("sets")
            or []
        )
    else:
        items = raw
    keys: set[str] = set()
    for item in items:
        if isinstance(item, str):
            keys.add(item)
        elif isinstance(item, dict):
            if item.get("history_key"):
                keys.add(str(item["history_key"]))
            elif {"case_id", "language_pair", "history_id"} <= set(item):
                keys.add(f"{item['case_id']}|{item['language_pair']}|{item['history_id']}")
            elif {"family", "case", "pair", "history"} <= set(item):
                letter = "F" if item.get("case") == "fraud" else "L"
                keys.add(f"{item['family']}_{letter}|{item['pair']}|{item['history']}")
    return keys


def conversation_from_record(record: dict[str, Any], source_path: Path | None = None) -> Conversation:
    private = record.get("private_annotations") or {}
    provenance = record.get("provenance") or {}
    turns = sorted(list(record.get("turns") or []), key=lambda t: int(t["turn_id"]))
    evidence = [int(x) for x in (private.get("evidence_turn_ids") or [])]
    return Conversation(
        conversation_id=str(record["conversation_id"]),
        scenario_family_id=str(record["scenario_family_id"]),
        case_id=str(record["case_id"]),
        history_id=str(record["history_id"]),
        branch_id=str(record.get("branch_id") or private.get("recipient_condition") or ""),
        language_pair=str(record["language_pair"]),
        split=str(record["split"]),
        turns=turns,
        label=str(private.get("scenario_label") or ""),
        recipient_condition=str(private.get("recipient_condition") or record.get("branch_id") or ""),
        branch_after_turn=int(private.get("branch_after_turn") or 0),
        evidence_turn_ids=evidence,
        invariance_eligible=bool(private.get("invariance_eligible", False)),
        review_status=str(provenance.get("review_status") or ""),
        disposition=str((provenance.get("generation_settings") or {}).get("disposition") or ""),
        prefix_hash=private.get("prefix_hash"),
        source_path=str(source_path) if source_path else None,
    )


def load_usable_conversations(
    run_dir: Path,
    *,
    split: str | None = "test",
    complete_sets_path: Path | None = None,
    use_complete_sets_if_present: bool = True,
) -> list[Conversation]:
    """Load usable conversations from ``run_dir/conversations/*.json``.

    Skips unreadable JSON (another process may be rewriting files). Does not
    write into ``run_dir``.
    """
    conv_dir = Path(run_dir) / "conversations"
    if not conv_dir.is_dir():
        raise FileNotFoundError(f"missing conversations dir: {conv_dir}")

    manifest_path = complete_sets_path
    if manifest_path is None and use_complete_sets_if_present:
        candidate = Path(run_dir) / "complete_sets.json"
        if candidate.is_file():
            manifest_path = candidate
    allowed: set[str] | None = None
    if manifest_path is not None:
        allowed = load_complete_set_keys(Path(manifest_path))

    out: list[Conversation] = []
    for path in sorted(conv_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(record, dict):
            continue
        if split is not None and record.get("split") != split:
            continue
        if not is_usable(record):
            continue
        if allowed is not None and history_key_from_record(record) not in allowed:
            continue
        if not record.get("turns"):
            continue
        private = record.get("private_annotations") or {}
        if not private.get("scenario_label"):
            continue
        out.append(conversation_from_record(record, path))
    return out


def complete_sibling_keys(conversations: Iterable[Conversation]) -> set[str]:
    """Histories that have all four recipient conditions among ``conversations``."""
    by_key: dict[str, set[str]] = {}
    for conv in conversations:
        by_key.setdefault(conv.history_key, set()).add(conv.recipient_condition)
    wanted = set(CONDITIONS)
    return {key for key, conds in by_key.items() if wanted <= conds}


def sample_conversations(
    conversations: list[Conversation],
    n: int,
    *,
    seed: int = 0,
) -> list[Conversation]:
    """Deterministic stratified sample (language × label × condition), then fill."""
    if n >= len(conversations):
        return list(conversations)
    rng = random.Random(seed)
    groups: dict[tuple[str, str, str], list[Conversation]] = {}
    for conv in conversations:
        key = (conv.language_pair, conv.label, conv.recipient_condition)
        groups.setdefault(key, []).append(conv)
    for bucket in groups.values():
        rng.shuffle(bucket)
    keys = sorted(groups)
    rng.shuffle(keys)
    chosen: list[Conversation] = []
    while len(chosen) < n and any(groups[k] for k in keys):
        progressed = False
        for key in keys:
            if len(chosen) >= n:
                break
            if groups[key]:
                chosen.append(groups[key].pop())
                progressed = True
        if not progressed:
            break
    chosen.sort(key=lambda c: c.conversation_id)
    return chosen


def select_view_turns(
    turns: list[dict[str, Any]],
    view: str,
    branch_after_turn: int,
) -> list[dict[str, Any]]:
    """Return the turn list for a non-streaming view (includes ``turn_id``)."""
    ordered = sorted(turns, key=lambda t: int(t["turn_id"]))
    if view == VIEW_FULL:
        return ordered
    if view == VIEW_CALLER_ONLY:
        return [t for t in ordered if t.get("speaker") == "caller"]
    if view == VIEW_PREFIX_ONLY:
        return [t for t in ordered if int(t["turn_id"]) <= int(branch_after_turn)]
    if view == VIEW_PREFIX_PLUS_REACTION:
        prefix = [t for t in ordered if int(t["turn_id"]) <= int(branch_after_turn)]
        reaction = next(
            (
                t
                for t in ordered
                if int(t["turn_id"]) > int(branch_after_turn) and t.get("speaker") == "recipient"
            ),
            None,
        )
        return prefix + ([reaction] if reaction is not None else [])
    raise ValueError(f"select_view_turns does not handle view={view!r}")


def iter_streaming_prefixes(turns: list[dict[str, Any]]) -> Iterator[list[dict[str, Any]]]:
    ordered = sorted(turns, key=lambda t: int(t["turn_id"]))
    for i in range(1, len(ordered) + 1):
        yield ordered[:i]


def view_turn_sets(
    conv: Conversation,
    views: Iterable[str],
) -> list[tuple[str, list[dict[str, Any]], int | None]]:
    """``(view, turns_with_ids, prefix_end_turn_or_None)`` for scoring.

    STREAMING yields one item per cumulative prefix; other views yield one item.
    """
    items: list[tuple[str, list[dict[str, Any]], int | None]] = []
    for view in views:
        if view == VIEW_STREAMING:
            for prefix in iter_streaming_prefixes(conv.turns):
                items.append((view, prefix, int(prefix[-1]["turn_id"])))
        else:
            selected = select_view_turns(conv.turns, view, conv.branch_after_turn)
            end = int(selected[-1]["turn_id"]) if selected else None
            items.append((view, selected, end))
    return items


def build_detector_input(
    conversation_id: str,
    turns: list[dict[str, Any]],
) -> dict[str, Any]:
    """Detector export: conversation_id + turns with speaker+text only."""
    return {
        "conversation_id": conversation_id,
        "turns": [{"speaker": t["speaker"], "text": t["text"]} for t in turns],
    }


def render_transcript(turns: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for turn in turns:
        speaker = turn["speaker"]
        if speaker not in SPEAKER_LABELS:
            raise ValueError(f"unknown speaker {speaker!r}")
        lines.append(f"{SPEAKER_LABELS[speaker]} {turn['text']}")
    return "\n".join(lines)


def assert_no_private_leak(prompt: str, detector_input: dict[str, Any]) -> None:
    """Fail if private record fields are present in the detector input or prompt."""
    if detector_has_private_fields(detector_input):
        raise AssertionError("detector input contains private record fields")
    extra = set(detector_input) - {"conversation_id", "turns"}
    if extra:
        raise AssertionError(f"detector input has unexpected keys: {sorted(extra)}")
    for turn in detector_input.get("turns") or []:
        extra_turn = set(turn) - {"speaker", "text"}
        if extra_turn:
            raise AssertionError(f"turn has unexpected keys: {sorted(extra_turn)}")
        if turn.get("speaker") not in SPEAKER_LABELS:
            raise AssertionError(f"invalid speaker {turn.get('speaker')!r}")
    lowered = prompt.lower()
    for marker in PRIVATE_PROMPT_MARKERS:
        if marker.lower() in lowered:
            raise AssertionError(f"private field {marker!r} leaked into detector prompt")


def build_scored_prompt(turns: list[dict[str, Any]], instruction: str) -> tuple[str, dict[str, Any]]:
    """Render transcript + instruction; return (user_content, detector_input).

    The caller wraps ``user_content`` in a chat template. Leakage is checked on
    the user content (instruction + transcript), which is what the model sees.
    """
    from scambranch.eval.constants import DETECTOR_INSTRUCTION

    text_turns = [{"speaker": t["speaker"], "text": t["text"]} for t in turns]
    detector_input = {"conversation_id": "", "turns": text_turns}
    transcript = render_transcript(text_turns)
    user_content = f"{instruction or DETECTOR_INSTRUCTION}\n\nConversation:\n{transcript}"
    assert_no_private_leak(user_content, detector_input)
    return user_content, detector_input


def mean_turns(conversations: Iterable[Conversation]) -> float:
    lengths = [len(c.turns) for c in conversations]
    if not lengths:
        return 0.0
    return sum(lengths) / len(lengths)
