"""Template rendering, language-block construction, and pair metadata."""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = ROOT / "prompts"
PROFILES_DIR = ROOT / "configs" / "language_profiles"

HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
PLACEHOLDER = re.compile(r"\{\{([^{}]+)\}\}")
VERSION_RE = re.compile(r"<!--\s*prompt_version:\s*(.+?)\s*-->")

PAIR_META = {
    "hi_en": {
        "pair_name": "Hindi-English",
        "local_language": "Hindi",
        "region": "North India",
        "native_script_name": "Devanagari",
    },
    "ta_en": {
        "pair_name": "Tamil-English",
        "local_language": "Tamil",
        "region": "Tamil Nadu",
        "native_script_name": "Tamil",
    },
    "te_en": {
        "pair_name": "Telugu-English",
        "local_language": "Telugu",
        "region": "Andhra Pradesh and Telangana",
        "native_script_name": "Telugu",
    },
}

# Copied from the HTML comment at the bottom of prompts/recipient.md (v0.2).
CONDITION_INSTRUCTIONS = {
    "cooperative": (
        "You go along with what the caller asks: you start doing it, agree to it, "
        "or ask how to do it. You do not refuse, hesitate on principle, or accuse "
        "the caller. You may ask a short practical question."
    ),
    "clarifying": (
        "You are not sure yet. Ask the caller to explain why this is needed or "
        "how it works. Do not refuse, do not accuse the caller, and do not agree yet."
    ),
    "resistant": (
        "You decline or set a boundary (for example, you will handle it yourself, "
        "visit the branch, or do it later). You do not need to give a reason and "
        "you do not accuse the caller of fraud."
    ),
    "suspicious": (
        "You openly doubt that this caller is genuine and say so. You may mention "
        "that you have heard of such calls."
    ),
}

HINT_STRIP_RE = re.compile(
    r"\b(cooperative|clarifying|resistant|suspicious|fraud|scam|legitimate)\b"
    r"|F-E\d+|L-M\d+",
    re.IGNORECASE,
)
PAREN_QUALIFIER_RE = re.compile(r"\s*\([^)]*\)")
CONDITION_NAME_RE = re.compile(
    r"\b(cooperative|clarifying|resistant|suspicious)\b",
    re.IGNORECASE,
)

# Keep these in the profile; do not suggest them to the caller.
CALLER_STILTED_FORMS = {
    "hi_en": {"kripya", "dhanyavaad"},
    "ta_en": set(),
    "te_en": {"dhanyavaadalu"},
}

CONDITION_DEFINITIONS = {
    "cooperative": "engages without an explicit fraud accusation",
    "clarifying": "requests explanation; does not agree yet",
    "resistant": "refuses or sets a boundary without necessarily accusing",
    "suspicious": "explicitly questions legitimacy",
}

FIRST_TURN_ENDING_HINT = (
    "This turn is your first reaction to the caller's last message."
)
LATER_TURN_ENDING_HINT = (
    "Continue in the same attitude; you may soften or firm up naturally based "
    "on what the caller says."
)

ANCHOR_REGISTER_WEIGHTS = {
    "service": 8,
    "service_dialog_user": 4,
    "assistant_command": 2,
    "conversational": 1,
    "question": 1,
}


class UnfilledTemplateError(ValueError):
    """Raised when a rendered template still contains {{...}} placeholders."""


def load_profile(pair: str) -> dict[str, Any]:
    path = PROFILES_DIR / f"{pair}.yaml"
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def prompt_version(raw: str) -> str:
    match = VERSION_RE.search(raw)
    if not match:
        return "unknown"
    return match.group(1).strip()


def strip_html_comments(text: str) -> str:
    return HTML_COMMENT.sub("", text)


def load_template(name: str) -> tuple[str, str]:
    """Return (prompt_version, template_with_comments_stripped)."""
    path = PROMPTS_DIR / f"{name}.md"
    raw = path.read_text(encoding="utf-8")
    version = prompt_version(raw)
    return version, strip_html_comments(raw)


def fill_template(template: str, values: dict[str, Any]) -> str:
    def repl(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        if key not in values:
            raise UnfilledTemplateError(f"unfilled template placeholder {{{{{key}}}}}")
        val = values[key]
        return val if isinstance(val, str) else str(val)

    filled = PLACEHOLDER.sub(repl, template)
    leftover = PLACEHOLDER.findall(filled)
    if leftover:
        keys = ", ".join(sorted(set(leftover)))
        raise UnfilledTemplateError(f"unfilled template placeholders: {keys}")
    return filled


def split_system_user(rendered: str) -> tuple[str, str]:
    """Split a filled prompt on '# System' / '# User' headings."""
    text = rendered.strip()
    sys_idx = re.search(r"^# System\s*$", text, re.MULTILINE)
    user_idx = re.search(r"^# User\s*$", text, re.MULTILINE)
    if not sys_idx or not user_idx:
        return "", text
    system = text[sys_idx.end() : user_idx.start()].strip()
    user = text[user_idx.end() :].strip()
    return system, user


def _pct(value: float) -> str:
    return f"{value * 100:.1f}"


def _join(items: list[str], limit: int | None = None) -> str:
    if limit is not None:
        items = items[:limit]
    return ", ".join(items)


def _address_forms(
    profile: dict[str, Any],
    register: str,
    *,
    pair: str | None = None,
    for_caller: bool = False,
) -> list[str]:
    rows = [
        row
        for row in profile.get("address_forms") or []
        if row.get("register") == register and row.get("form")
    ]
    rows.sort(key=lambda r: int(r.get("count") or 0), reverse=True)
    exclude = CALLER_STILTED_FORMS.get(pair or "", set()) if for_caller else set()
    forms: list[str] = []
    for row in rows:
        form = str(row["form"]).strip()
        if not form or form.lower() in exclude or form in forms:
            continue
        forms.append(form)
    return forms or ["ji"]


def behaviour_hint(card: dict[str, Any], condition: str, case_kind: str) -> str:
    """Card continuation text with condition/label words stripped."""
    raw = (
        ((card.get("continuations") or {}).get(condition) or {}).get(case_kind) or ""
    )
    cleaned = HINT_STRIP_RE.sub("", raw)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;.")
    return cleaned or "respond in line with the attitude above"


def sanitize_allowed_endings(items: list[str]) -> list[str]:
    """Drop parenthetical condition qualifiers and condition names."""
    out: list[str] = []
    for item in items:
        text = PAREN_QUALIFIER_RE.sub("", str(item))
        text = CONDITION_NAME_RE.sub("", text)
        text = re.sub(r"\s+", " ", text).strip(" ,;")
        if text:
            out.append(text)
    return out


def _top_insertions(profile: dict[str, Any], n: int = 15) -> list[str]:
    rows = list(profile.get("lexical_insertions") or [])
    rows.sort(key=lambda r: int(r.get("count") or 0), reverse=True)
    out: list[str] = []
    for row in rows:
        word = str(row.get("en_word") or "").strip()
        if word and word not in out:
            out.append(word)
        if len(out) >= n:
            break
    return out


def _light_verb_examples(profile: dict[str, Any], n: int = 8) -> list[str]:
    out: list[str] = []
    for row in profile.get("light_verb_constructions") or []:
        examples = [e for e in (row.get("examples") or []) if isinstance(e, str)]
        pattern = row.get("pattern") or ""
        if examples:
            snippet = examples[0].replace("\n", " ")
            if len(snippet) > 160:
                snippet = snippet[:157] + "…"
            out.append(f"{pattern}: {snippet}" if pattern else snippet)
        elif pattern:
            out.append(str(pattern))
        if len(out) >= n:
            break
    return out


def _mixed_morphology(profile: dict[str, Any], n: int = 8) -> list[str]:
    out: list[str] = []
    for row in profile.get("mixed_morphology") or []:
        form = row.get("form")
        if form:
            stem = row.get("stem") or ""
            suffix = row.get("suffix") or ""
            out.append(f"{form} ({stem}+{suffix})" if stem else str(form))
        if len(out) >= n:
            break
    return out


def _romanization_variants(profile: dict[str, Any], n: int = 10) -> list[str]:
    out: list[str] = []
    for row in profile.get("romanization_variants") or []:
        kind = row.get("kind") or ""
        if kind == "dakshina_alternation":
            continue
        canonical = row.get("canonical")
        variants = [v for v in (row.get("variants") or []) if isinstance(v, str)]
        if not canonical or not variants:
            continue
        shown = "/".join(variants[:4])
        out.append(f"{canonical}: {shown}")
        if len(out) >= n:
            break
    return out


def _sample_style_anchors(profile: dict[str, Any], rng: random.Random, n: int = 10) -> list[str]:
    buckets: dict[str, list[str]] = {k: [] for k in ANCHOR_REGISTER_WEIGHTS}
    for row in profile.get("reference_utterances") or []:
        register = row.get("register")
        text = (row.get("text") or "").strip()
        if register in buckets and text:
            buckets[register].append(text)
    weighted: list[str] = []
    weights: list[int] = []
    for register, pool in buckets.items():
        w = ANCHOR_REGISTER_WEIGHTS[register]
        for text in pool:
            weighted.append(text)
            weights.append(w)
    if not weighted:
        return ["(no style anchors in profile)"]
    chosen: list[str] = []
    remaining = list(range(len(weighted)))
    rem_w = list(weights)
    while remaining and len(chosen) < n:
        pick = rng.choices(remaining, weights=rem_w, k=1)[0]
        idx = remaining.index(pick)
        chosen.append(weighted[pick])
        del remaining[idx]
        del rem_w[idx]
    return chosen


def _unnatural_examples(profile: dict[str, Any], rng: random.Random, n: int = 3) -> list[dict[str, str]]:
    rows = [r for r in (profile.get("unnatural_examples") or []) if r.get("text")]
    if not rows:
        return []
    sample = rows[:] if len(rows) <= n else rng.sample(rows, n)
    return [{"text": r["text"], "reason": r.get("reason") or ""} for r in sample]


def sample_language_block_vars(
    pair: str,
    profile: dict[str, Any],
    rng: random.Random,
) -> dict[str, str]:
    meta = PAIR_META[pair]
    bands = (profile.get("mixing_bands") or {}).get("english_share") or {}
    p25 = float(bands.get("p25") or 0.2)
    p75 = float(bands.get("p75") or 0.6)
    if p75 < p25:
        p25, p75 = p75, p25
    target = rng.uniform(p25, p75)
    mono = float((profile.get("mixing_bands") or {}).get("monolingual_turn_share") or 0.0)
    caller_forms = _address_forms(profile, "formal", pair=pair, for_caller=True)
    recipient_register = rng.choice(["formal", "neutral"])
    recipient_forms = _address_forms(profile, recipient_register, pair=pair, for_caller=False)
    anchors = _sample_style_anchors(profile, rng, 10)
    unnatural = _unnatural_examples(profile, rng, 3)
    style_anchors = "\n".join(f"- {a}" for a in anchors)
    unnatural_block = "\n".join(
        f"- {row['text']}  [why: {row['reason']}]" for row in unnatural
    ) or "- (none listed)"
    p50 = float(bands.get("p50") or target)
    return {
        "pair": pair,
        "pair_name": meta["pair_name"],
        "local_language": meta["local_language"],
        "region": meta["region"],
        "native_script_name": meta["native_script_name"],
        "target_english_share_pct": _pct(target),
        "english_share_p25_pct": _pct(p25),
        "english_share_p50_pct": _pct(p50),
        "english_share_p75_pct": _pct(p75),
        "monolingual_share_pct": _pct(mono),
        "caller_register": "formal",
        "caller_address_forms": _join(caller_forms),
        "recipient_register": recipient_register,
        "recipient_address_forms": _join(recipient_forms),
        "top_insertions": _join(_top_insertions(profile, 15)),
        "light_verb_examples": _join(_light_verb_examples(profile, 8)),
        "mixed_morphology_examples": _join(_mixed_morphology(profile, 8)),
        "romanization_variants": _join(_romanization_variants(profile, 10)),
        "style_anchors": style_anchors,
        "unnatural_examples": unnatural_block,
        "unnatural_examples_with_reasons": unnatural_block,
        "target_english_share": target,
        "english_share_p10": float(bands.get("p10") or 0.0),
        "english_share_p90": float(bands.get("p90") or 1.0),
    }


def render_language_block(pair: str, profile: dict[str, Any], rng: random.Random) -> tuple[str, dict[str, Any]]:
    version, template = load_template("_language_block")
    vars_ = sample_language_block_vars(pair, profile, rng)
    # template still has {{pair}} inside a stripped comment; only live vars remain
    filled = fill_template(template, {k: v for k, v in vars_.items() if isinstance(v, str)})
    vars_["language_block_version"] = version
    return filled, vars_


def render_module(name: str, values: dict[str, Any]) -> tuple[str, str]:
    """Fill a prompt module. Returns (prompt_version, filled_text)."""
    version, template = load_template(name)
    return version, fill_template(template, values)


def local_lexicon(profile: dict[str, Any]) -> set[str]:
    """Profile-derived local-language tokens for english_share scoring."""
    words: set[str] = set()
    token_re = re.compile(r"[A-Za-z]+")
    for row in profile.get("romanization_variants") or []:
        if row.get("kind") == "dakshina_alternation":
            continue
        if row.get("canonical"):
            words.add(str(row["canonical"]).lower())
        for v in row.get("variants") or []:
            if isinstance(v, str):
                words.add(v.lower())
    for row in profile.get("address_forms") or []:
        if row.get("form"):
            words.add(str(row["form"]).lower())
    for row in profile.get("mixed_morphology") or []:
        for key in ("form", "stem", "suffix"):
            if row.get(key):
                words.add(str(row[key]).lower())
    for row in profile.get("light_verb_constructions") or []:
        pattern = str(row.get("pattern") or "")
        for tok in token_re.findall(pattern):
            if len(tok) > 1:
                words.add(tok.lower())
    return words
