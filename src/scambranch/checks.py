"""Automatic structural, leakage, script, and mixing checks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
CONVERSATION_SCHEMA = json.loads(
    (ROOT / "schemas" / "conversation.schema.json").read_text(encoding="utf-8")
)
DETECTOR_SCHEMA = json.loads(
    (ROOT / "schemas" / "detector_export.schema.json").read_text(encoding="utf-8")
)

CONVERSATION_VALIDATOR = Draft202012Validator(CONVERSATION_SCHEMA)
DETECTOR_VALIDATOR = Draft202012Validator(DETECTOR_SCHEMA)

LEAKAGE_RE = re.compile(
    r"\b(scam|fraud|legitimate|evidence|cooperative|clarifying|resistant|suspicious)\b"
    r"|F-E\d|L-M\d|\bbp_\d|\bjr_\d|\bec_\d",
    re.IGNORECASE,
)
BRACKET_RE = re.compile(r"\[[^\]]+\]")
STAGE_RE = re.compile(
    r"\((?:sighs?|pauses?|laughs?|whispers?|hangs? up|aside|stage)[^)]*\)"
    r"|^\s*\*[^*]+\*\s*$"
    r"|\b(?:narrator|stage direction)\b",
    re.IGNORECASE,
)
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")
TELUGU_RE = re.compile(r"[\u0C00-\u0C7F]")
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
TOKEN_SET_RE = re.compile(r"[A-Za-z0-9]+")
REPETITION_JACCARD = 0.6

REACTION_MARKERS = {
    "hi_en": re.compile(
        r"nahi dunga|nahi karunga|share nahi|nahi bataunga|scam|fraud|fake|shak|dhokha",
        re.IGNORECASE,
    ),
    "ta_en": re.compile(
        r"tharamaatten|solla maatten|share panna maatten|scam|fraud|fake|doubt",
        re.IGNORECASE,
    ),
    "te_en": re.compile(
        r"cheppanu|ivvanu|share cheyanu|scam|fraud|fake|anumanam",
        re.IGNORECASE,
    ),
}

FALLBACK_EN = {
    "the", "a", "an", "to", "of", "and", "or", "for", "in", "on", "is", "are",
    "was", "were", "be", "been", "being", "this", "that", "it", "you", "your",
    "we", "our", "they", "i", "me", "my", "please", "thank", "thanks", "yes",
    "no", "not", "with", "from", "have", "has", "had", "will", "can", "could",
    "would", "should", "need", "call", "card", "bank", "account", "order",
    "delivery", "message", "number", "team", "sir", "madam", "hello", "ok",
    "okay", "share", "confirm", "transaction", "amount", "blocked", "code",
    "pin", "otp", "upi", "kyc", "emi", "ticket", "parcel", "agent", "customer",
}


def schema_errors(record: dict[str, Any], kind: str = "conversation") -> list[str]:
    validator = CONVERSATION_VALIDATOR if kind == "conversation" else DETECTOR_VALIDATOR
    return [f"{list(e.path)}: {e.message}" for e in validator.iter_errors(record)]


def token_set(text: str) -> set[str]:
    return {tok.lower() for tok in TOKEN_SET_RE.findall(text or "")}


def token_jaccard(a: str, b: str) -> float:
    left, right = token_set(a), token_set(b)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def last_same_speaker_turn(
    turns: list[dict[str, Any]], speaker: str
) -> dict[str, Any] | None:
    for turn in reversed(turns):
        if turn.get("speaker") == speaker:
            return turn
    return None


def near_duplicate_pair_count(turns: list[dict[str, Any]], threshold: float = REPETITION_JACCARD) -> int:
    count = 0
    for i in range(len(turns)):
        for j in range(i + 1, len(turns)):
            if token_jaccard(turns[i].get("text") or "", turns[j].get("text") or "") >= threshold:
                count += 1
    return count


def should_stop_for_repetition(
    turns: list[dict[str, Any]],
    new_turn: dict[str, Any],
    *,
    threshold: float = REPETITION_JACCARD,
) -> bool:
    prev = last_same_speaker_turn(turns, str(new_turn.get("speaker") or ""))
    if prev and token_jaccard(prev.get("text") or "", new_turn.get("text") or "") >= threshold:
        return True
    window = (turns + [new_turn])[-4:]
    if len(window) >= 4 and near_duplicate_pair_count(window, threshold) >= 2:
        return True
    return False


def prefix_hash(turns: list[dict[str, Any]], branch_after: int) -> str:
    import hashlib

    prefix = [
        {"turn_id": t["turn_id"], "speaker": t["speaker"], "text": t["text"]}
        for t in turns
        if int(t["turn_id"]) <= int(branch_after)
    ]
    payload = json.dumps(prefix, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def script_hits(text: str) -> list[str]:
    hits: list[str] = []
    if DEVANAGARI_RE.search(text):
        hits.append("devanagari")
    if TAMIL_RE.search(text):
        hits.append("tamil")
    if TELUGU_RE.search(text):
        hits.append("telugu")
    return hits


def line_leakage(
    text: str,
    *,
    speaker: str,
    condition: str,
) -> list[str]:
    flags: list[str] = []
    if BRACKET_RE.search(text):
        flags.append("brackets")
    if STAGE_RE.search(text):
        flags.append("stage_direction")
    allow_scam_fraud = speaker == "recipient" and condition == "suspicious"
    if allow_scam_fraud:
        scanned = re.sub(r"\b(scam|fraud)\b", "", text, flags=re.IGNORECASE)
        if LEAKAGE_RE.search(scanned):
            flags.append("label_leakage")
    elif LEAKAGE_RE.search(text):
        flags.append("label_leakage")
    return flags


def detect_leakage(
    turns: list[dict[str, Any]],
    *,
    condition: str,
    branch_after: int,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    prefix_broken = False
    for turn in turns:
        flags = line_leakage(
            turn.get("text") or "",
            speaker=turn.get("speaker") or "",
            condition=condition,
        )
        if flags:
            tid = int(turn.get("turn_id") or 0)
            issues.append({"turn_id": tid, "speaker": turn.get("speaker"), "flags": flags})
            if tid <= int(branch_after):
                prefix_broken = True
    return {"ok": not issues, "issues": issues, "prefix_broken": prefix_broken}


def _zipf_en(word: str) -> float:
    try:
        from wordfreq import zipf_frequency

        return float(zipf_frequency(word, "en"))
    except Exception:
        return 4.0 if word.lower() in FALLBACK_EN else 0.0


def reaction_marker_hit(text: str, pair: str) -> bool:
    pat = REACTION_MARKERS.get(pair)
    return bool(pat and pat.search(text or ""))


def prefix_recipient_reaction(
    turns: list[dict[str, Any]],
    *,
    branch_after: int,
    pair: str,
) -> bool:
    for turn in turns:
        if turn.get("speaker") != "recipient":
            continue
        if int(turn.get("turn_id") or 0) > int(branch_after):
            continue
        if reaction_marker_hit(turn.get("text") or "", pair):
            return True
    return False


def cooperative_refusal(
    turns: list[dict[str, Any]],
    *,
    branch_after: int,
    pair: str,
    condition: str,
) -> bool:
    if condition != "cooperative":
        return False
    first = None
    for turn in turns:
        if turn.get("speaker") == "recipient" and int(turn.get("turn_id") or 0) > int(branch_after):
            first = turn
            break
    if first is None:
        return False
    return reaction_marker_hit(first.get("text") or "", pair)


def english_share(turns: list[dict[str, Any]], lexicon: set[str]) -> float:
    n_en = 0
    n_local = 0
    for turn in turns:
        for tok in WORD_RE.findall(turn.get("text") or ""):
            low = tok.lower()
            if _zipf_en(low) >= 3.5:
                n_en += 1
            elif low in lexicon:
                n_local += 1
    denom = n_en + n_local
    if denom == 0:
        return 0.0
    return n_en / denom


def run_automatic_checks(
    *,
    record: dict[str, Any] | None,
    turns: list[dict[str, Any]],
    condition: str,
    branch_after: int,
    sibling_hashes: list[str] | None,
    expected_prefix_hash: str,
    lexicon: set[str],
    p10: float,
    p90: float,
    pair: str = "hi_en",
) -> dict[str, Any]:
    schema_ok = True
    schema_msgs: list[str] = []
    if record is not None:
        schema_msgs = schema_errors(record, "conversation")
        schema_ok = not schema_msgs

    leak = detect_leakage(turns, condition=condition, branch_after=branch_after)
    script_issues: list[dict[str, Any]] = []
    prefix_script = False
    for turn in turns:
        hits = script_hits(turn.get("text") or "")
        if hits:
            tid = int(turn.get("turn_id") or 0)
            script_issues.append({"turn_id": tid, "scripts": hits})
            if tid <= int(branch_after):
                prefix_script = True

    share = english_share(turns, lexicon)
    share_ok = p10 <= share <= p90
    hash_ok = True
    if sibling_hashes:
        hash_ok = all(h == expected_prefix_hash for h in sibling_hashes)

    prefix_rxn = prefix_recipient_reaction(turns, branch_after=branch_after, pair=pair)
    coop_ref = cooperative_refusal(
        turns, branch_after=branch_after, pair=pair, condition=condition
    )
    prefix_broken = leak["prefix_broken"] or prefix_script or not hash_ok
    repair_candidate = prefix_rxn or coop_ref
    needs_repair = (not leak["ok"] and not leak["prefix_broken"]) or (
        bool(script_issues) and not prefix_script
    ) or not share_ok or repair_candidate

    return {
        "schema_ok": schema_ok,
        "schema_errors": schema_msgs,
        "prefix_hash_ok": hash_ok,
        "leakage": leak,
        "script_issues": script_issues,
        "prefix_script": prefix_script,
        "english_share": share,
        "english_share_ok": share_ok,
        "prefix_broken": prefix_broken,
        "prefix_recipient_reaction": prefix_rxn,
        "cooperative_refusal": coop_ref,
        "repair_candidate": repair_candidate,
        "needs_repair": needs_repair,
        "ok": schema_ok and hash_ok and leak["ok"] and not script_issues and share_ok and not repair_candidate,
    }


def to_detector_export(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "conversation_id": record["conversation_id"],
        "turns": [
            {"speaker": t["speaker"], "text": t["text"]} for t in record["turns"]
        ],
    }


def detector_has_private_fields(row: dict[str, Any]) -> bool:
    banned = {
        "private_annotations",
        "provenance",
        "scenario_label",
        "recipient_condition",
        "prefix_hash",
        "case_id",
        "scenario_family_id",
        "branch_id",
        "label_rationale",
    }
    return any(k in row for k in banned)
