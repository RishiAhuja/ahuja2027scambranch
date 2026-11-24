"""Deterministic fictional placeholder resolution."""

from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any

AMOUNT_RANGE_RE = re.compile(
    r"\[AMOUNT_(\d+)\]\s*\(\s*₹?\s*([\d,]+)\s*[–—-]\s*₹?\s*([\d,]+)",
    re.UNICODE,
)
N_RANGE_RE = re.compile(
    r"\[N\](?:[^\d]{0,24}(\d+)\s*[–—-]\s*(\d+))?",
)
AMOUNT_TOKEN_RE = re.compile(r"\[AMOUNT_(\d+)\]")
PHONE_TOKEN_RE = re.compile(r"\[PHONE_(\d+)\]")
UPI_TOKEN_RE = re.compile(r"\[UPI_ID_(\d+)\]")
ANY_TOKEN_RE = re.compile(
    r"\[(AMOUNT_\d+|PHONE_\d+|UPI_ID_\d+|ACCT_LAST4|ORDER_ID|AWB_ID|"
    r"TICKET_ID|LICENCE_ID|PERSON_NAME|AGENT_NAME|CITY|N)\]"
)

ROUNDISH = [
    500, 750, 1000, 1200, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000,
    6000, 7500, 8000, 9000, 10000, 12000, 15000, 18000, 20000, 22000, 25000,
    30000,
]

NAMES = {
    "hi_en": [
        "Aarav", "Rohan", "Priya", "Ananya", "Vikram", "Neha", "Kabir",
        "Isha", "Aditya", "Meera", "Rahul", "Pooja", "Sanjay", "Kavita",
    ],
    "ta_en": [
        "Arjun", "Karthik", "Priya", "Divya", "Suresh", "Lakshmi", "Vijay",
        "Anjali", "Murugan", "Meena", "Senthil", "Kavitha", "Raja", "Deepa",
    ],
    "te_en": [
        "Ravi", "Suresh", "Ananya", "Lakshmi", "Venkat", "Sravani", "Krishna",
        "Padma", "Sai", "Bhavana", "Naveen", "Keerthi", "Prasad", "Sowmya",
    ],
}

CITIES = {
    "hi_en": [
        "Delhi", "Lucknow", "Jaipur", "Kanpur", "Bhopal", "Noida",
        "Ghaziabad", "Agra",
    ],
    "ta_en": [
        "Chennai", "Coimbatore", "Madurai", "Tiruchirappalli", "Salem",
        "Tirunelveli",
    ],
    "te_en": [
        "Hyderabad", "Vijayawada", "Visakhapatnam", "Warangal", "Guntur",
        "Tirupati",
    ],
}

OCCUPATIONS = [
    "teacher",
    "shopkeeper",
    "engineer",
    "student",
    "accountant",
    "nurse",
    "driver",
    "homemaker",
    "clerk",
]


def keyed_rng(*parts: Any, seed: int) -> random.Random:
    hasher = hashlib.sha256()
    hasher.update(str(seed).encode("utf-8"))
    for part in parts:
        hasher.update(b"|")
        hasher.update(str(part).encode("utf-8"))
    return random.Random(int.from_bytes(hasher.digest()[:8], "big"))


def _parse_int(raw: str) -> int:
    return int(raw.replace(",", "").replace(" ", ""))


def _walk_strings(obj: Any) -> list[str]:
    found: list[str] = []
    if isinstance(obj, str):
        found.append(obj)
    elif isinstance(obj, list):
        for item in obj:
            found.extend(_walk_strings(item))
    elif isinstance(obj, dict):
        for item in obj.values():
            found.extend(_walk_strings(item))
    return found


def _replace_in(obj: Any, mapping: dict[str, str]) -> Any:
    if isinstance(obj, str):
        for key, val in mapping.items():
            obj = obj.replace(key, val)
        return obj
    if isinstance(obj, list):
        return [_replace_in(x, mapping) for x in obj]
    if isinstance(obj, dict):
        return {k: _replace_in(v, mapping) for k, v in obj.items()}
    return obj


def _pick_amount(lo: int, hi: int, rng: random.Random) -> int:
    if hi < lo:
        lo, hi = hi, lo
    opts = [c for c in ROUNDISH if lo <= c <= hi]
    if opts:
        return rng.choice(opts)
    span = max(hi - lo, 1)
    raw = rng.randint(lo, hi)
    step = 500 if span >= 1000 else 50
    rounded = int(round(raw / step) * step)
    return min(max(rounded, lo), hi)


def resolve_placeholders(
    card: dict[str, Any],
    *,
    family: str,
    case: str,
    pair: str,
    history_id: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    """Replace placeholders in a scenario card.

    Returns (resolved_card, mapping, persona).
    """
    rng = keyed_rng(family, case, pair, history_id, "placeholders", seed=seed)
    blob = "\n".join(_walk_strings(card))

    amount_ranges: dict[str, tuple[int, int]] = {}
    for match in AMOUNT_RANGE_RE.finditer(blob):
        amount_ranges[match.group(1)] = (
            _parse_int(match.group(2)),
            _parse_int(match.group(3)),
        )

    mapping: dict[str, str] = {}
    for match in AMOUNT_TOKEN_RE.finditer(blob):
        idx = match.group(1)
        token = f"[AMOUNT_{idx}]"
        if token in mapping:
            continue
        lo, hi = amount_ranges.get(idx, (500, 20000))
        mapping[token] = str(_pick_amount(lo, hi, rng))

    if "[ACCT_LAST4]" in blob:
        mapping["[ACCT_LAST4]"] = f"{rng.randint(1000, 9999)}"

    for match in PHONE_TOKEN_RE.finditer(blob):
        token = match.group(0)
        if token not in mapping:
            mapping[token] = f"90000{rng.randint(0, 99999):05d}"

    names = NAMES[pair]
    person = rng.choice(names)
    agent = rng.choice([n for n in names if n != person] or names)
    if "[PERSON_NAME]" in blob:
        mapping["[PERSON_NAME]"] = person
    if "[AGENT_NAME]" in blob:
        mapping["[AGENT_NAME]"] = agent

    upi_name = re.sub(r"[^a-z]", "", person.lower()) or "user"
    for match in UPI_TOKEN_RE.finditer(blob):
        token = match.group(0)
        if token not in mapping:
            mapping[token] = f"{upi_name}.{rng.randint(1000, 9999)}@paysetu"

    if "[ORDER_ID]" in blob:
        mapping["[ORDER_ID]"] = f"SK{rng.randint(10000000, 99999999)}"
    if "[AWB_ID]" in blob:
        mapping["[AWB_ID]"] = f"SS{rng.randint(100000000, 999999999)}"
    if "[TICKET_ID]" in blob:
        mapping["[TICKET_ID]"] = f"KB{rng.randint(100000, 999999)}"
    if "[LICENCE_ID]" in blob:
        mapping["[LICENCE_ID]"] = f"RA{rng.randint(100000, 999999)}"

    n_match = N_RANGE_RE.search(blob)
    if "[N]" in blob:
        if n_match and n_match.group(1) and n_match.group(2):
            lo, hi = int(n_match.group(1)), int(n_match.group(2))
            if hi < lo:
                lo, hi = hi, lo
            mapping["[N]"] = str(rng.randint(lo, hi))
        else:
            mapping["[N]"] = str(rng.randint(3, 5))

    city = rng.choice(CITIES[pair])
    if "[CITY]" in blob:
        mapping["[CITY]"] = city

    resolved = _replace_in(card, mapping)
    leftover = ANY_TOKEN_RE.findall(json.dumps(resolved, ensure_ascii=False))
    # leftover tokens that were not in the original blob are fine to ignore;
    # any still-present token means we missed a generator.
    still = ANY_TOKEN_RE.findall(json.dumps(resolved, ensure_ascii=False))
    for token in still:
        full = f"[{token}]"
        if full not in mapping:
            mapping[full] = "X" + str(rng.randint(1000, 9999))
            resolved = _replace_in(resolved, {full: mapping[full]})

    persona = {
        "name": person,
        "city": city,
        "age": rng.randint(22, 62),
        "occupation": rng.choice(OCCUPATIONS),
        "language_pair": pair,
    }
    _ = leftover
    return resolved, mapping, persona
