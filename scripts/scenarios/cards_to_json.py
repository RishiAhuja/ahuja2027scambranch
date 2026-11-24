#!/usr/bin/env python3
"""Parse scenario markdown cards into JSON conforming to scenario.schema.json."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DRAFT = REPO_ROOT / "scenarios" / "draft"
DEFAULT_OUT = REPO_ROOT / "scenarios" / "json"

FAMILY_ID_RE = re.compile(r"^(bp|jr|ec)_\d{2}$")
SOURCE_ID_RE = re.compile(r"\b([A-Z]{1,3}\d{1,3})\b")
EVIDENCE_TAG_RE = re.compile(r"\((?:F-E|L-M)\d+(?:\s*,\s*(?:F-E|L-M)\d+)*\)")
EVIDENCE_ID_RE = re.compile(r"(?:F-E|L-M)\d+")
NUMBERED_RE = re.compile(r"^(\d+)\.\s+(.*)$")
BOLD_FIELD_RE = re.compile(
    r"^\*\*(Scenario label|Label rationale|Caller objective|Agent objective|"
    r"Coordinator objective|Sender objective|Buyer objective|Seller objective)"
    r"(?:\s*\([^)]*\))?:\*\*\s*(.*)$",
    re.IGNORECASE,
)
CONSTRAINT_KEYS = ("must_preserve", "may_vary", "must_not_emit")
CONTINUATION_KEYS = ("cooperative", "clarifying", "resistant", "suspicious")
ROLE_ALIASES = (
    "recipient",
    "caller",
    "coordinator",
    "sender",
    "buyer",
    "seller",
    "agent",
    "recruiter",
    "candidate",
    "payroll",
    "tutor",
    "joiner",
    "student",
    "hr",
)
CALLER_SURFACES = {
    "caller",
    "agent",
    "coordinator",
    "sender",
    "recruiter",
    "payroll",
    "hr",
}
RECIPIENT_SURFACES = {
    "recipient",
    "candidate",
    "tutor",
    "joiner",
    "student",
}
BRANCH_AFTER_RE = re.compile(
    r"branch_after[`*\s]*:\s*[`*]*\s*(?:turn\s*)?(\d+)",
    re.IGNORECASE,
)
DIALOGUE_HEADING_RE = re.compile(r"dialogue-act plan", re.IGNORECASE)
KNOWN_ORGS = (
    "Narmada Cooperative Bank",
    "Vindhya State Staff Selection Board",
    "Nilgiri Crest Institute of Technology",
    "SafarWays Overseas Manpower",
    "Dunecrest Facilities Management",
    "VeriTrust Screening Services",
    "KuberQuant Technologies",
    "Pragati Career Academy",
    "Kaushal Path Academy",
    "TalentBridge Staffing",
    "Nirmay Diagnostics",
    "SwiftShip Couriers",
    "Deccan Power Distribution",
    "Vistara Infotech",
    "NeelGagan Airways",
    "Meridian Exim Traders",
    "Suraksha Life",
    "ShopKart Seller Hub",
    "ShopKart Plus",
    "Kaveri Bank",
    "NaukriHub",
    "BechoBazaar",
    "PadhaiMitra",
    "ShopKart",
    "PaySetu",
    "KhanaGo",
    "Voltrix",
)

log = logging.getLogger("cards_to_json")


def warn(family_id: str, message: str) -> None:
    log.warning("%s: %s", family_id or "?", message)


def strip_md(text: str) -> str:
    text = text.strip()
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", text)
    return text.strip().strip('"').strip()


def split_sections(text: str) -> dict[str, str]:
    parts = re.split(r"(?m)^##\s+", text)
    sections: dict[str, str] = {"_preamble": parts[0]}
    for part in parts[1:]:
        if not part.strip():
            continue
        first, _, rest = part.partition("\n")
        sections[first.strip()] = rest
    return sections


def parse_md_table(block: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw in block.splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            if rows:
                break
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", c.replace(" ", "")) for c in cells):
            continue
        if cells and cells[0].lower() == "field" and len(rows) == 0:
            continue
        rows.append([strip_md(c) for c in cells])
    return rows


def header_map(preamble: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for row in parse_md_table(preamble):
        if len(row) >= 2:
            values[row[0].lower()] = row[1]
    return values


def parse_title(preamble: str, family_id: str) -> str:
    for line in preamble.splitlines():
        if line.startswith("#"):
            heading = strip_md(re.sub(r"^#\s*", "", line))
            heading = re.sub(rf"^{re.escape(family_id)}\s*[—–-]\s*", "", heading)
            return heading.strip()
    warn(family_id, "could not parse title")
    return ""


def first_token(value: str, choices: tuple[str, ...], default: str, family_id: str, field: str) -> str:
    lowered = value.strip().lower()
    for choice in choices:
        if lowered == choice or lowered.startswith(choice):
            return choice
    if value.strip():
        warn(family_id, f"unrecognised {field} {value!r}; using {default}")
    return default


def parse_channel(raw: str) -> str:
    lowered = raw.lower()
    if "inbound" in lowered:
        return "inbound_call"
    if lowered.startswith("chat") or "chat (" in lowered:
        return "chat"
    return "outbound_call"


def parse_bool(raw: str, family_id: str) -> bool:
    token = raw.strip().lower().split()[0] if raw.strip() else ""
    if token in {"yes", "true", "1"}:
        return True
    if token in {"no", "false", "0"}:
        return False
    if raw.strip():
        warn(family_id, f"unrecognised pilot {raw!r}; using false")
    return False


def section_by_prefix(sections: dict[str, str], *prefixes: str) -> str:
    for key, body in sections.items():
        key_l = key.lower()
        if any(key_l.startswith(p.lower()) for p in prefixes):
            return body
    return ""


def case_section(sections: dict[str, str], suffix: str) -> str:
    needle = f"_{suffix}".lower()
    for key, body in sections.items():
        if key.lower().startswith("case ") and needle in key.lower():
            return body
    return ""


def bullets(text: str) -> list[str]:
    items: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^[-*]\s+", stripped):
            item = strip_md(re.sub(r"^[-*]\s+", "", stripped))
            if item:
                items.append(item)
    return items


def labelled_value(text: str, labels: tuple[str, ...]) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        match = re.match(r"^[-*]\s+\*?\*?([^:*]+)\*?\*?\s*:\s*(.*)$", stripped)
        if not match:
            continue
        label = strip_md(match.group(1)).lower()
        if any(label == want or label.startswith(want) for want in labels):
            return strip_md(match.group(2))
    return ""


def extract_orgs(value: str) -> list[str]:
    found: list[str] = []
    remaining = value
    for org in sorted(KNOWN_ORGS, key=len, reverse=True):
        if org.lower() in remaining.lower():
            found.append(org)
            remaining = re.sub(re.escape(org), " ", remaining, flags=re.IGNORECASE)
    if found:
        return found
    chunks = [c.strip() for c in re.split(r";|,|\band\b", value) if c.strip()]
    cleaned: list[str] = []
    for chunk in chunks:
        chunk = re.sub(r"\([^)]*\)", "", chunk).strip(" .")
        chunk = re.split(r"\b(ending|order|complaint|linked|policy)\b", chunk, maxsplit=1)[0].strip(" .")
        if chunk and len(chunk.split()) <= 6:
            cleaned.append(chunk)
    return cleaned


def parse_shared_setting(body: str, family_id: str) -> dict:
    orgs: list[str] = []
    pretext = ""
    recipient = ""
    other: list[str] = []
    for item in bullets(body):
        match = re.match(r"^([^:]+):\s*(.*)$", item)
        if not match:
            other.append(item)
            continue
        label = match.group(1).strip().lower()
        value = match.group(2).strip()
        if label.startswith("organisation") or label in {"platform", "employer"}:
            extracted = extract_orgs(value)
            if extracted:
                orgs.extend(extracted)
            elif value:
                orgs.append(value)
                warn(family_id, f"organisation line kept as a single value: {value!r}")
        elif label.startswith("pretext") or label.startswith("reason"):
            pretext = value
        elif label.startswith("recipient"):
            recipient = value
        else:
            other.append(item)
    # de-duplicate orgs, preserve order
    seen: set[str] = set()
    unique_orgs: list[str] = []
    for org in orgs:
        if org not in seen:
            seen.add(org)
            unique_orgs.append(org)
    if not unique_orgs:
        warn(family_id, "no organisations parsed from shared setting")
    if not pretext:
        warn(family_id, "no pretext parsed from shared setting")
    if not recipient:
        warn(family_id, "no recipient profile parsed from shared setting")
    return {
        "organisations": unique_orgs,
        "pretext": pretext,
        "recipient_profile": recipient,
        "other_facts": other,
    }


def recipient_surface_role(family_id: str, channel_raw: str, recipient_profile: str) -> str | None:
    profile = recipient_profile.lower()
    channel = channel_raw.lower()
    if family_id in {"ec_02", "ec_10"} or "to seller" in channel or profile.startswith("the seller"):
        return "seller"
    if family_id in {"ec_07", "ec_20"} or profile.startswith("the buyer"):
        return "buyer"
    return None


def map_speaker(
    surface: str,
    family_id: str,
    channel: str,
    channel_raw: str,
    recipient_role: str | None,
) -> str:
    surface = surface.lower()
    if surface in RECIPIENT_SURFACES:
        return "recipient"
    if surface in CALLER_SURFACES:
        return "caller"
    if surface == "buyer":
        return "recipient" if recipient_role == "buyer" else "caller"
    if surface == "seller":
        return "recipient" if recipient_role == "seller" else "caller"
    warn(family_id, f"unmapped surface role {surface!r}; defaulting via channel")
    if channel == "inbound_call" and family_id in {"bp_12", "ec_09"}:
        return "recipient"
    return "caller"


def detect_surface_role(act: str) -> str | None:
    first = re.match(r"^([A-Za-z]+)\b", act.strip())
    if not first:
        return None
    word = first.group(1).lower()
    if word in ROLE_ALIASES:
        return word
    return None


def extract_evidence_ids(act: str) -> tuple[str, list[str]]:
    ids: list[str] = []
    cleaned = act
    for match in EVIDENCE_TAG_RE.finditer(act):
        ids.extend(EVIDENCE_ID_RE.findall(match.group(0)))
        cleaned = cleaned.replace(match.group(0), "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -—.")
    # de-duplicate, preserve order
    seen: set[str] = set()
    unique: list[str] = []
    for eid in ids:
        if eid not in seen:
            seen.add(eid)
            unique.append(eid)
    return cleaned, unique


def dialogue_block(body: str) -> str:
    cut = BRANCH_AFTER_RE.split(body, maxsplit=1)[0]
    parts = DIALOGUE_HEADING_RE.split(cut, maxsplit=1)
    return parts[1] if len(parts) > 1 else cut


def parse_dialogue_plan(
    body: str,
    family_id: str,
    channel: str,
    channel_raw: str,
    recipient_role: str | None,
) -> list[dict]:
    block = dialogue_block(body)
    turns: list[dict] = []
    for line in block.splitlines():
        match = NUMBERED_RE.match(line.strip())
        if not match:
            continue
        number = int(match.group(1))
        raw_act = strip_md(match.group(2))
        act, evidence_ids = extract_evidence_ids(raw_act)
        surface = detect_surface_role(act)
        if surface:
            speaker = map_speaker(surface, family_id, channel, channel_raw, recipient_role)
        else:
            warn(family_id, f"no role word on dialogue turn {number}; inferring from channel")
            opener_is_recipient = family_id in {"bp_12", "ec_09"}
            speaker = "recipient" if (opener_is_recipient == (number % 2 == 1)) else "caller"
        turns.append(
            {
                "turn": number,
                "speaker": speaker,
                "act": act,
                "evidence_ids": evidence_ids,
            }
        )
    if not turns:
        warn(family_id, "no dialogue-plan turns parsed")
    return turns


def parse_branch_after(body: str, family_id: str, case_tag: str) -> int:
    match = BRANCH_AFTER_RE.search(body)
    if not match:
        warn(family_id, f"missing branch_after in {case_tag}")
        return 0
    return int(match.group(1))


def parse_bold_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    lines = body.splitlines()
    i = 0
    current: str | None = None
    buf: list[str] = []

    def flush() -> None:
        nonlocal current, buf
        if current is None:
            return
        fields[current] = " ".join(part for part in buf if part).strip()
        current = None
        buf = []

    while i < len(lines):
        line = lines[i]
        match = BOLD_FIELD_RE.match(line.strip())
        if match:
            flush()
            key = match.group(1).lower()
            if "rationale" in key:
                current = "label_rationale"
            elif "label" in key:
                current = "scenario_label"
            else:
                current = "caller_objective"
            rest = match.group(2).strip()
            buf = [strip_md(rest)] if rest else []
            i += 1
            continue
        if current and line.strip().startswith("**"):
            flush()
            continue
        if current:
            stripped = line.strip()
            if stripped.startswith("|") or stripped.startswith("##") or NUMBERED_RE.match(stripped):
                flush()
                continue
            if stripped.startswith("-") and current != "label_rationale":
                flush()
                continue
            if stripped:
                buf.append(strip_md(stripped))
            elif buf:
                flush()
        i += 1
    flush()
    return fields


def parse_stable_facts(body: str) -> list[str]:
    match = re.search(
        r"\*\*Stable facts:\*\*\s*(.*?)(?=\*\*|\n## |\Z)",
        body,
        re.DOTALL | re.IGNORECASE,
    )
    return bullets(match.group(1)) if match else []


def parse_evidence_table(body: str, kind: str) -> list[dict]:
    heading = (
        r"Required observable evidence"
        if kind == "fraud"
        else r"Legitimacy markers"
    )
    match = re.search(
        rf"\*\*{heading}.*?\*\*.*?\n(\|.*?)(?:\n\n|\n[A-Z]|\n\*\*|\Z)",
        body,
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        # fallback: first table after the heading-like line
        idx = re.search(heading, body, re.IGNORECASE)
        block = body[idx.end() :] if idx else ""
        rows = parse_md_table(block)
    else:
        rows = parse_md_table(match.group(1))
    items: list[dict] = []
    for row in rows:
        if not row or row[0].lower() in {"id", "marker"}:
            continue
        raw_id = row[0]
        supporting = "supporting" in raw_id.lower()
        eid = strip_md(re.sub(r"\(.*?\)", "", raw_id))
        if not eid:
            continue
        text = row[1] if len(row) > 1 else ""
        if kind == "fraud":
            items.append({"id": eid, "text": text, "role": "supporting" if supporting else "required"})
        else:
            items.append({"id": eid, "text": text})
    return items


def parse_case(
    body: str,
    family_id: str,
    kind: str,
    channel: str,
    channel_raw: str,
    recipient_role: str | None,
) -> dict:
    fields = parse_bold_fields(body)
    label_default = "fraud_attempt" if kind == "fraud" else "legitimate"
    raw_label = fields.get("scenario_label", "").replace("`", "").strip()
    if raw_label and raw_label != label_default:
        warn(family_id, f"{kind} scenario_label {raw_label!r} coerced to {label_default}")
    case: dict = {
        "case_id": f"{family_id}_{'F' if kind == 'fraud' else 'L'}",
        "scenario_label": label_default,
        "label_rationale": fields.get("label_rationale", ""),
        "caller_objective": fields.get("caller_objective", ""),
        "stable_facts": parse_stable_facts(body),
        "dialogue_plan": parse_dialogue_plan(body, family_id, channel, channel_raw, recipient_role),
        "branch_after": parse_branch_after(body, family_id, kind),
    }
    if not case["label_rationale"]:
        warn(family_id, f"missing label_rationale in {kind} case")
    if not case["caller_objective"]:
        warn(family_id, f"missing caller_objective in {kind} case")
    if kind == "fraud":
        case["evidence"] = parse_evidence_table(body, "fraud")
        if not case["evidence"]:
            warn(family_id, "no fraud evidence rows parsed")
    else:
        case["markers"] = parse_evidence_table(body, "legitimate")
        if not case["markers"]:
            warn(family_id, "no legitimacy marker rows parsed")
    return case


def parse_continuations(body: str, family_id: str) -> dict:
    rows = parse_md_table(body)
    result = {key: {"fraud": "", "legitimate": ""} for key in CONTINUATION_KEYS}
    for row in rows:
        if len(row) < 3:
            continue
        condition = row[0].strip().lower()
        if condition not in result:
            continue
        result[condition] = {"fraud": row[1].strip(), "legitimate": row[2].strip()}
    for key in CONTINUATION_KEYS:
        if not result[key]["fraud"] or not result[key]["legitimate"]:
            warn(family_id, f"incomplete continuation row for {key}")
    return result


def split_ending_items(text: str) -> list[str]:
    text = re.sub(r"^`?_[FL]`?\s*:\s*", "", text.strip())
    text = re.sub(r"^[_*][FL][_*]\s*:\s*", "", text)
    parts = [strip_md(p) for p in re.split(r";", text) if strip_md(p)]
    return parts or ([text] if text else [])


def parse_allowed_endings(body: str, family_id: str) -> dict:
    fraud: list[str] = []
    legit: list[str] = []
    match = re.search(
        r"\*\*Allowed endings:\*\*\s*(.*?)(?=\*\*Caller continuation|\*\*Generation|\Z)",
        body,
        re.DOTALL | re.IGNORECASE,
    )
    block = match.group(1) if match else body
    for item in bullets(block):
        key = item.lower()
        if key.startswith("_f") or key.startswith("`_f") or key.startswith("f:"):
            fraud.extend(split_ending_items(item))
        elif key.startswith("_l") or key.startswith("`_l") or key.startswith("l:"):
            legit.extend(split_ending_items(item))
    if not fraud or not legit:
        warn(family_id, "allowed endings missing for one or both cases")
    return {"fraud": fraud, "legitimate": legit}


def parse_limits(body: str) -> str:
    match = re.search(
        r"\*\*Caller continuation limits:\*\*\s*(.*?)(?=\n## |\Z)",
        body,
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return ""
    return " ".join(strip_md(line) for line in match.group(1).splitlines() if line.strip())


def split_constraint_items(text: str) -> list[str]:
    text = text.strip().rstrip(".")
    if not text:
        return []
    if ";" in text:
        parts = [strip_md(p).rstrip(".") for p in text.split(";") if strip_md(p)]
        return parts
    parts: list[str] = []
    buf = ""
    depth = 0
    for char in text:
        if char in "([":
            depth += 1
        elif char in ")]":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            item = strip_md(buf).rstrip(".")
            if item:
                parts.append(item)
            buf = ""
        else:
            buf += char
    tail = strip_md(buf).rstrip(".")
    if tail:
        parts.append(tail)
    return parts or [text]


def parse_constraints(body: str, family_id: str) -> dict:
    result = {key: [] for key in CONSTRAINT_KEYS}
    for item in bullets(body):
        match = re.match(r"^\*?\*?(must_preserve|may_vary|must_not_emit)\*?\*?\s*:\s*(.*)$", item, re.I)
        if not match:
            continue
        result[match.group(1).lower()] = split_constraint_items(match.group(2))
    for key in CONSTRAINT_KEYS:
        if not result[key]:
            warn(family_id, f"empty generation constraint {key}")
    return result


def parse_ambiguities(body: str) -> list[str]:
    items = bullets(body)
    if items:
        return items
    paras = [strip_md(p) for p in re.split(r"\n\s*\n", body) if strip_md(p) and not strip_md(p).startswith("|")]
    return paras


def normalize_ref_status(raw: str, family_id: str) -> str:
    text = raw.strip()
    if not text:
        warn(family_id, "empty evidence-reference status; using VERIFY")
        return "VERIFY"
    first = re.split(r"[\s(—-]", text, maxsplit=1)[0]
    mapping = {
        "verified": "verified",
        "partial": "partial",
        "verify": "VERIFY",
        "reported": "reported",
    }
    if first.lower() in mapping:
        return mapping[first.lower()]
    if "verify" in text.lower() and not text.lower().startswith("verified"):
        return "VERIFY"
    if "partial" in text.lower():
        return "partial"
    if "report" in text.lower():
        return "reported"
    if text.lower().startswith("verified"):
        return "verified"
    warn(family_id, f"unrecognised evidence-reference status {text!r}; using VERIFY")
    return "VERIFY"


def parse_evidence_references(body: str, family_id: str) -> list[dict]:
    refs: list[dict] = []
    for row in parse_md_table(body):
        if not row or row[0].lower() == "claim":
            continue
        claim = row[0] if row else ""
        source = row[1] if len(row) > 1 else ""
        retrieved = row[2] if len(row) > 2 else ""
        status_raw = row[3] if len(row) > 3 else ""
        if not any(cell.strip() for cell in row):
            continue
        source_ids = SOURCE_ID_RE.findall(source)
        retrieved = "" if retrieved in {"—", "-", "–"} else retrieved
        refs.append(
            {
                "claim": claim,
                "source_ids": source_ids,
                "retrieved": retrieved,
                "status": normalize_ref_status(status_raw, family_id),
            }
        )
    if not refs:
        warn(family_id, "no evidence references parsed")
    return refs


def parse_signoff(body: str) -> list[dict]:
    entries: list[dict] = []
    for row in parse_md_table(body):
        if not row or row[0].lower() == "role":
            continue
        while len(row) < 4:
            row.append("")
        role, name, date, notes = row[:4]
        if not role and not name and not date and not notes:
            continue
        entries.append({"role": role, "name": name, "date": date, "notes": notes})
    return entries


def parse_card(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    sections = split_sections(text)
    preamble = sections.get("_preamble", "")
    header = header_map(preamble)
    family_id = strip_md(header.get("family id", ""))
    if not FAMILY_ID_RE.match(family_id):
        stem_match = re.search(r"((?:bp|jr|ec)_\d{2})", path.name)
        family_id = stem_match.group(1) if stem_match else ""
        warn(family_id, f"family_id taken from filename {path.name}")
    domain = first_token(
        header.get("domain", ""),
        ("banking_payments", "jobs_recruitment", "ecommerce_delivery"),
        "banking_payments",
        family_id,
        "domain",
    )
    status = first_token(
        header.get("status", "draft"),
        ("draft", "reviewed", "approved", "retired"),
        "draft",
        family_id,
        "status",
    )
    split = first_token(
        header.get("split", "unassigned"),
        ("unassigned", "train", "dev", "test"),
        "unassigned",
        family_id,
        "split",
    )
    channel_raw = header.get("channel", "")
    channel = parse_channel(channel_raw)
    version = header.get("version", "") or "0.1"
    shared_body = section_by_prefix(sections, "shared setting")
    shared = parse_shared_setting(shared_body, family_id)
    recipient_role = recipient_surface_role(family_id, channel_raw, shared["recipient_profile"])
    fraud_body = case_section(sections, "F")
    legit_body = case_section(sections, "L")
    if not fraud_body:
        warn(family_id, "fraud case section missing")
    if not legit_body:
        warn(family_id, "legitimate case section missing")
    continuations_body = section_by_prefix(sections, "recipient continuation")
    constraints_body = section_by_prefix(sections, "generation constraint")
    ambiguities_body = section_by_prefix(sections, "ambiguities")
    refs_body = section_by_prefix(sections, "evidence reference")
    signoff_body = section_by_prefix(sections, "sign-off", "signoff")
    limits_src = continuations_body or text
    return {
        "family_id": family_id,
        "domain": domain,
        "version": version,
        "status": status,
        "pilot": parse_bool(header.get("pilot", ""), family_id),
        "split": split,
        "channel": channel,
        "title": parse_title(preamble, family_id),
        "summary": " ".join(section_by_prefix(sections, "family summary").split()),
        "shared_setting": shared,
        "cases": {
            "fraud": parse_case(fraud_body, family_id, "fraud", channel, channel_raw, recipient_role),
            "legitimate": parse_case(legit_body, family_id, "legitimate", channel, channel_raw, recipient_role),
        },
        "continuations": parse_continuations(continuations_body, family_id),
        "allowed_endings": parse_allowed_endings(limits_src, family_id),
        "caller_continuation_limits": parse_limits(limits_src),
        "generation_constraints": parse_constraints(constraints_body, family_id),
        "ambiguities": parse_ambiguities(ambiguities_body),
        "evidence_references": parse_evidence_references(refs_body, family_id),
        "signoff": parse_signoff(signoff_body),
    }


def discover_cards(draft_dir: Path) -> list[Path]:
    cards = [
        path
        for path in sorted(draft_dir.glob("*/*.md"))
        if path.name.lower() != "readme.md"
    ]
    return cards


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-dir", type=Path, default=DEFAULT_DRAFT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cards = discover_cards(args.draft_dir)
    if not cards:
        log.error("no markdown cards found under %s", args.draft_dir)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for path in cards:
        record = parse_card(path)
        family_id = record.get("family_id") or path.stem[:5]
        dest = args.out_dir / f"{family_id}.json"
        dest.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written += 1
        log.info("wrote %s from %s", dest.relative_to(REPO_ROOT), path.relative_to(REPO_ROOT))
    log.info("converted %d cards", written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
