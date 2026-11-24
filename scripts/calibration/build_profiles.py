#!/usr/bin/env python3
"""Build v0.1-auto language profiles for hi_en, ta_en, and te_en.

Reads processed calibration artifacts only. Writes
configs/language_profiles/{hi_en,ta_en,te_en}.yaml and README.md.
Does not modify data/ or scenarios/.

Re-run from the repo root:

    .venv/bin/python scripts/calibration/build_profiles.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml
from jsonschema import Draft202012Validator
from wordfreq import zipf_frequency

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import mine_ta_en as ta_mine  # noqa: E402
import mine_te_en as te_mine  # noqa: E402
import process_service_dialogs as psd  # noqa: E402
import refine_candidates as rc  # noqa: E402

PROC = REPO_ROOT / "data" / "calibration" / "processed"
SCHEMA_PATH = REPO_ROOT / "schemas" / "language_profile.schema.json"
OUT_DIR = REPO_ROOT / "configs" / "language_profiles"

VERSION = "0.1-auto"
STATUS_NOTE = (
    "automatically derived from human-written corpora; "
    "not yet reviewed by a fluent speaker"
)
CHANGE_NOTE = "auto-built by build_profiles.py"
CHANGE_DATE = "2026-09-25"
GOLD_MIN = 2000
SEED = 42
WORD_RE = re.compile(r"^[A-Za-z][A-Za-z']*$")
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
LOCAL = {"hi_en": "hi", "ta_en": "ta", "te_en": "te"}

HI_LIGHT_VERBS = {
    "kar", "karo", "karna", "kiya", "kijiye", "karein", "ho", "hua",
    "hoga", "dena", "dijiye", "lena",
}
HI_LIGHT_VARIANTS = {
    "kro": "karo", "krna": "karna", "kiye": "kiya", "kijie": "kijiye",
    "kijiye": "kijiye", "kare": "karein", "kren": "karein", "krn": "karna",
    "huaa": "hua", "hogaa": "hoga", "dijie": "dijiye",
}
LVC_EXCLUDE_STEMS = frozenset({
    "video", "videos", "upload", "like", "likes", "share", "subscribe",
    "review", "release", "act", "tweet", "out", "win", "play",
})
MORPH_EXCLUDE_STEMS = frozenset({
    "cinema", "movie", "movies", "theatre", "theater", "match",
    "video", "videos", "game", "games",
})
HI_MORPH_SUFFIXES = (
    "wala", "wali", "wale", "iyon", "mein", "se", "ko",
)
HI_MORPH_FALSE = frozenset({
    "realme", "mereko", "chatne", "chipka", "chipke", "justme",
    "theeka", "theeke", "intone", "comeon", "bareme",
})

TE_REQUEST = frozenset({
    "meeru", "garu", "cheyandi", "cheppandi", "please",
    "sir", "madam", "ivvandi", "pampandi",
})
CONV_MARKERS = {
    "hi_en": frozenset({
        "aap", "aapka", "aapko", "ji", "sir", "madam", "kripya",
        "dhanyavaad", "tum", "bhaiya", "didi", "tu", "bhai",
        "please", "plz", "pls",
    }),
    "ta_en": frozenset({
        "neenga", "unga", "sir", "madam", "nandri", "anna", "akka",
        "nee", "bro", "thambi", "please", "plz", "pls",
    }),
    "te_en": TE_REQUEST,
}
ADDRESS_REGISTER = {
    "hi_en": {
        "formal": ("aap", "aapka", "aapko", "ji", "sir", "madam", "kripya", "dhanyavaad"),
        "neutral": ("tum", "bhaiya", "didi"),
        "informal": ("tu", "bhai"),
    },
    "ta_en": {
        "formal": ("neenga", "unga", "sir", "madam", "nandri"),
        "neutral": ("anna", "akka"),
        "informal": ("nee", "bro", "thambi"),
    },
    "te_en": {
        "formal": ("meeru", "mee", "garu", "sir", "madam", "dhanyavaadalu"),
        "neutral": ("anna", "akka"),
        "informal": ("nuvvu", "nee", "bro", "babu"),
    },
}
SERVICE_REGISTERS = frozenset({
    "service",
    "service_dialog_user",
    "service_dialog_system_template",
    "assistant_command",
})

SOURCE_META: dict[str, dict[str, str]] = {
    "lince": {
        "url": "https://huggingface.co/datasets/lince-benchmark/lince",
        "license_status": "restricted Twitter research; stats only, do not redistribute raw tweets",
        "redistributable": "no",
    },
    "lince_pos": {
        "url": "https://huggingface.co/datasets/lince-benchmark/lince",
        "license_status": "restricted Twitter research; stats only, do not redistribute raw tweets",
        "redistributable": "no",
    },
    "phinc": {
        "url": "https://huggingface.co/datasets/LingoIITGN/PHINC",
        "license_status": "CC BY 4.0",
        "redistributable": "yes",
    },
    "hinge": {
        "url": "https://huggingface.co/datasets/LingoIITGN/HinGE",
        "license_status": "CC BY 4.0 (human column only)",
        "redistributable": "yes",
    },
    "sentimix": {
        "url": "https://doi.org/10.5281/zenodo.3974927",
        "license_status": "CC BY 4.0 (Twitter-derived; platform terms may still apply)",
        "redistributable": "yes",
    },
    "tamilmixsentiment": {
        "url": "https://huggingface.co/datasets/community-datasets/tamilmixsentiment",
        "license_status": "UNCLEAR - stats only, do not redistribute",
        "redistributable": "no",
    },
    "offenseval_dravidian": {
        "url": "https://huggingface.co/datasets/community-datasets/offenseval_dravidian",
        "license_status": "CC BY 4.0 (YouTube TOS on comments)",
        "redistributable": "yes",
    },
    "tanglish_corpus_185k": {
        "url": "https://huggingface.co/datasets/vishnu-n/Tanglish-Corpus-185k",
        "license_status": "CC BY 4.0 compilation; YouTube/Reddit TOS on comments",
        "redistributable": "no",
    },
    "coli_dravidian_ta": {
        "url": "https://codalab.lisn.upsaclay.fr/competitions/19357",
        "license_status": "UNCLEAR - stats only, do not redistribute",
        "redistributable": "no",
    },
    "cmqa_tamlish": {
        "url": "https://github.com/khyathiraghavi/code_switched_QA",
        "license_status": "CMU 2018 COPYING — retain copyright notice",
        "redistributable": "yes",
    },
    "cmtet_sentiment": {
        "url": "https://github.com/ksubbu199/cmtet-sentiment",
        "license_status": "UNCLEAR - stats only, do not redistribute",
        "redistributable": "no",
    },
    "cmtet_lid": {
        "url": "https://github.com/ksubbu199/cmtet-lid",
        "license_status": "UNCLEAR - stats only, do not redistribute",
        "redistributable": "no",
    },
    "icon_2016": {
        "url": "https://github.com/LCS2-IIITD/HIT-ACL2021-Codemixed-Representation",
        "license_status": "UNCLEAR - stats only, do not redistribute",
        "redistributable": "no",
    },
    "chandu_cmqa": {
        "url": "https://github.com/khyathiraghavi/code_switched_QA",
        "license_status": "CMU 2018 COPYING — retain copyright notice",
        "redistributable": "yes",
    },
    "cmqa": {
        "url": "https://github.com/khyathiraghavi/code_switched_QA",
        "license_status": "CMU 2018 COPYING — retain copyright notice",
        "redistributable": "yes",
    },
    "code_mixed_dialog": {
        "url": "https://github.com/sumanbanerjee1/Code-Mixed-Dialog",
        "license_status": "CC BY 4.0 (paper) / Apache 2.0 (repo)",
        "redistributable": "yes",
    },
    "hinglish_top": {
        "url": "https://github.com/google-research-datasets/Hinglish-TOP-Dataset",
        "license_status": "Apache 2.0 (human split only)",
        "redistributable": "yes",
    },
    "dakshina": {
        "url": "https://github.com/google-research-datasets/dakshina",
        "license_status": "CC BY-SA 4.0",
        "redistributable": "yes",
    },
}
PAIR_STAT_SOURCES = {
    "hi_en": ("lince", "lince_pos", "phinc", "hinge", "sentimix"),
    "ta_en": (
        "tamilmixsentiment", "offenseval_dravidian", "tanglish_corpus_185k",
        "coli_dravidian_ta", "cmqa_tamlish",
    ),
    "te_en": ("cmtet_sentiment", "cmtet_lid", "icon_2016", "chandu_cmqa"),
}

# Longest phrases first. Placeholders follow scenarios/fictional_entities.md.
_BRAND_PHRASES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(p, re.I), repl)
    for p, repl in (
        (r"\bgoogle\s*pay\b", "PaySetu"),
        (r"\bamazon\s*pay\b", "PaySetu"),
        (r"\bphone\s*pe\b", "PaySetu"),
        (r"\bg\s*pay\b", "PaySetu"),
        (r"\baxis\s+bank\b", "Kaveri Bank"),
        (r"\bhdfc\s+bank\b", "Kaveri Bank"),
        (r"\bicici\s+bank\b", "Kaveri Bank"),
        (r"\byes\s+bank\b", "Kaveri Bank"),
        (r"\bunion\s+bank\b", "Kaveri Bank"),
        (r"\bbank\s+of\s+baroda\b", "Kaveri Bank"),
        (r"\bbank\s+of\s+india\b", "Kaveri Bank"),
        (r"\bindian\s+bank\b", "Kaveri Bank"),
        (r"\bindian\s+overseas\b", "Kaveri Bank"),
        (r"\bswiss\s+bank\b", "Kaveri Bank"),
        (r"\breserve\s+bank\b", "Kaveri Bank"),
        (r"\bblue\s*dart\b", "SwiftShip Couriers"),
        (r"\becom\s*express\b", "SwiftShip Couriers"),
        (r"\bindia\s*post\b", "SwiftShip Couriers"),
        (r"\bjio\s*prime\b", "[TELECOM]"),
        (r"\breliance\s+jio\b", "[TELECOM]"),
        (r"\bvodafone\s+idea\b", "[TELECOM]"),
        (r"\bpizza[_\s-]*hut\b", "[BRAND]"),
        (r"\bfrankie[_\s-]*and[_\s-]*bennys?\b", "[BRAND]"),
        (r"\bdisney\+?\s*hotstar\b", "[BRAND]"),
    )
]
_BRAND_WORDS: dict[str, str] = {}
for word, repl in (
    ("sbi", "Kaveri Bank"), ("hdfc", "Kaveri Bank"), ("icici", "Kaveri Bank"),
    ("pnb", "Kaveri Bank"), ("kotak", "Kaveri Bank"), ("idbi", "Kaveri Bank"),
    ("canara", "Kaveri Bank"), ("rbi", "Kaveri Bank"), ("bob", "Kaveri Bank"),
    ("paytm", "PaySetu"), ("phonepe", "PaySetu"), ("gpay", "PaySetu"),
    ("bhim", "PaySetu"), ("bharatpe", "PaySetu"), ("mobikwik", "PaySetu"),
    ("freecharge", "PaySetu"),
    ("bluedart", "SwiftShip Couriers"), ("dtdc", "SwiftShip Couriers"),
    ("delhivery", "SwiftShip Couriers"), ("fedex", "SwiftShip Couriers"),
    ("indiapost", "SwiftShip Couriers"),
    ("airtel", "[TELECOM]"), ("jio", "[TELECOM]"), ("vodafone", "[TELECOM]"),
    ("bsnl", "[TELECOM]"), ("jioprime", "[TELECOM]"),
    ("amazon", "ShopKart"), ("flipkart", "ShopKart"), ("myntra", "ShopKart"),
    ("snapdeal", "ShopKart"), ("meesho", "ShopKart"), ("ajio", "ShopKart"),
    ("nykaa", "ShopKart"),
    ("swiggy", "[BRAND]"), ("zomato", "[BRAND]"), ("uber", "[BRAND]"),
    ("ola", "[BRAND]"), ("whatsapp", "[BRAND]"), ("facebook", "[BRAND]"),
    ("instagram", "[BRAND]"), ("youtube", "[BRAND]"), ("netflix", "[BRAND]"),
    ("hotstar", "[BRAND]"), ("wazirx", "[BRAND]"), ("gmail", "[BRAND]"),
    ("snapchat", "[BRAND]"), ("twitter", "[BRAND]"), ("samsung", "[BRAND]"),
    ("xiaomi", "[BRAND]"), ("redmi", "[BRAND]"), ("oneplus", "[BRAND]"),
    ("nandos", "[BRAND]"), ("kfc", "[BRAND]"), ("mcdonalds", "[BRAND]"),
    ("starbucks", "[BRAND]"),
):
    _BRAND_WORDS[word] = repl
_WORD_BRAND_RE = re.compile(
    r"\b(" + "|".join(sorted(_BRAND_WORDS, key=len, reverse=True)) + r")\b",
    re.I,
)
_UNDERSCORE_BRAND_RE = re.compile(
    r"\b[\w]*(?:pizza_?hut|nandos|frankie|mcdonald|starbucks|myntra|"
    r"flipkart|amazon|paytm|airtel|vodafone)[\w]*\b",
    re.I,
)
_ZIPF: dict[str, float] = {}


def log(msg: str) -> None:
    print(msg, flush=True)


def zipf_en(tok: str) -> float:
    key = tok.lower()
    if key not in _ZIPF:
        try:
            _ZIPF[key] = float(zipf_frequency(key, "en"))
        except Exception:
            _ZIPF[key] = 0.0
    return _ZIPF[key]


def r4(x: float) -> float:
    return float(f"{x:.4f}")


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0}
    s = sorted(values)
    n = len(s)

    def pct(p: float) -> float:
        if n == 1:
            return r4(s[0])
        idx = (n - 1) * p
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return r4(s[lo] * (1 - frac) + s[hi] * frac)

    return {
        "p10": pct(0.10),
        "p25": pct(0.25),
        "p50": pct(0.50),
        "p75": pct(0.75),
        "p90": pct(0.90),
    }


def mix_metrics(tags: list[str], local_labels: set[str]) -> dict[str, Any]:
    n_en = sum(1 for t in tags if t == "en")
    n_loc = sum(1 for t in tags if t in local_labels)
    den = n_en + n_loc
    # Adjacent unambiguous labels only; do not bridge over excluded tokens (plan §7.3).
    switches = 0
    for a, b in zip(tags, tags[1:]):
        if (a == "en" or a in local_labels) and (b == "en" or b in local_labels) and a != b:
            switches += 1
    if den == 0:
        return {
            "english_share": None,
            "cmi": None,
            "n_switches": switches,
            "monolingual": False,
            "defined": False,
        }
    return {
        "english_share": n_en / den,
        "cmi": 100.0 * (1.0 - (max(n_en, n_loc) / den)),
        "n_switches": switches,
        "monolingual": n_en == 0 or n_loc == 0,
        "defined": True,
    }


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def fivegrams(tokens: list[str]) -> set[tuple[str, ...]]:
    lows = [t.lower() for t in tokens if WORD_RE.match(t)]
    if len(lows) < 5:
        return set()
    return {tuple(lows[i : i + 5]) for i in range(len(lows) - 4)}


def clip(text: str, n: int = 180) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= n:
        return text
    return text[: n - 1].rstrip() + "…"


def redistributable(source_id: str) -> str:
    meta = SOURCE_META.get(source_id)
    if meta:
        return meta["redistributable"]
    lic = rc.LICENSE.get(source_id)
    if lic:
        return lic[1]
    return "no"


def replace_brands(text: str) -> tuple[str, int]:
    n = 0
    out = text

    def _sub(pattern: re.Pattern[str], repl: str, src: str) -> tuple[str, int]:
        hits = 0

        def _repl(_m: re.Match[str]) -> str:
            nonlocal hits
            hits += 1
            return repl

        return pattern.sub(_repl, src), hits

    for pat, repl in _BRAND_PHRASES:
        out, hits = _sub(pat, repl, out)
        n += hits

    def _us(m: re.Match[str]) -> str:
        nonlocal n
        n += 1
        return "[BRAND]"

    out = _UNDERSCORE_BRAND_RE.sub(_us, out)

    def _word(m: re.Match[str]) -> str:
        nonlocal n
        n += 1
        return _BRAND_WORDS[m.group(1).lower()]

    out = _WORD_BRAND_RE.sub(_word, out)
    return out, n


def is_lvc_excluded(pattern: str) -> bool:
    head = pattern.split()[0] if pattern.split() else ""
    stem = head.rstrip("s")
    return head in LVC_EXCLUDE_STEMS or stem in LVC_EXCLUDE_STEMS


def morph_excluded(stem: str) -> bool:
    s = stem.lower().rstrip("s")
    return s in MORPH_EXCLUDE_STEMS or stem.lower() in MORPH_EXCLUDE_STEMS


def hi_morph_split(tok: str) -> tuple[str, str] | None:
    low = tok.lower()
    if not WORD_RE.match(tok) or len(low) < 7 or low in HI_MORPH_FALSE:
        return None
    if zipf_en(low) >= 3.0:
        return None
    for suf in HI_MORPH_SUFFIXES:
        if not low.endswith(suf) or len(low) <= len(suf) + 3:
            continue
        stem = low[: -len(suf)]
        if zipf_en(stem) >= 4.0 and len(stem) >= 4:
            return stem, suf
    return None


def pair_morph_split(pair: str, tok: str, lexicon: set[str] | None) -> tuple[str, str] | None:
    if pair == "hi_en":
        return hi_morph_split(tok)
    if pair == "ta_en":
        return ta_mine.mixed_split(tok, lexicon)
    return te_mine.intra_word_mixed(tok, te_mine.ZIPF_EN)


def is_light_verb(pair: str, tok: str, tag: str) -> str | None:
    low = tok.lower()
    if pair == "hi_en":
        if low in HI_LIGHT_VERBS:
            return low
        if low in HI_LIGHT_VARIANTS and tag != "en":
            return HI_LIGHT_VARIANTS[low]
        return None
    if pair == "ta_en":
        return low if low in ta_mine.LIGHT_VERBS else None
    if low in te_mine.LIGHT_VERBS:
        return low
    return None


def band_block(
    shares: list[float],
    cmis: list[float],
    switches: list[float],
    n_mono: int,
    n_defined: int,
    origin: str,
    n_utts: int,
    note: str,
) -> dict[str, Any]:
    return {
        "origin": origin,
        "n_utterances": n_utts,
        "note": note,
        "english_share": quantiles(shares),
        "cmi": quantiles(cmis),
        "switches_per_utterance": quantiles(switches),
        "monolingual_turn_share": r4((n_mono / n_defined) if n_defined else 0.0),
    }


class BandAcc:
    def __init__(self) -> None:
        self.shares: list[float] = []
        self.cmis: list[float] = []
        self.switches: list[float] = []
        self.n_mono = 0
        self.n_defined = 0
        self.n = 0

    def add(self, mets: dict[str, Any]) -> None:
        self.n += 1
        self.switches.append(float(mets["n_switches"]))
        if not mets.get("defined"):
            return
        self.n_defined += 1
        self.shares.append(float(mets["english_share"]))
        self.cmis.append(float(mets["cmi"]))
        if mets["monolingual"]:
            self.n_mono += 1

    def as_block(self, origin: str, note: str) -> dict[str, Any]:
        return band_block(
            self.shares, self.cmis, self.switches,
            self.n_mono, self.n_defined, origin, self.n, note,
        )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_ta_lexicon() -> set[str]:
    path = PROC / "ta_en" / "lexicon_ta.json"
    if not path.exists():
        return set()
    return {str(x).lower() for x in load_json(path)}


def dialog_lexicon(pair: str) -> set[str]:
    lang = LOCAL[pair]
    lex = psd.load_roman_lexicon(lang)
    if pair == "hi_en":
        lex.update(psd.HI_FUNCTION)
    elif pair == "ta_en":
        lex.update(psd.TA_FUNCTION)
    else:
        te_path = PROC / "te_en" / "te_lexicon.json"
        if te_path.exists():
            lex.update(str(x).lower() for x in load_json(te_path))
    return lex


def auto_tag(text: str, lexicon: set[str]) -> tuple[list[str], list[str]]:
    tokens = tokenize(text)
    tags = [psd.tag_token(t, lexicon) for t in tokens]
    return tokens, tags


def has_markers(tokens: list[str], markers: Iterable[str]) -> bool:
    bag = {t.lower() for t in tokens}
    blob = " ".join(t.lower() for t in tokens)
    for m in markers:
        if " " in m:
            if m in blob:
                return True
        elif m in bag:
            return True
    return False


def load_dakshina_summary(lang: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stats = load_json(PROC / "romanization" / f"{lang}_variant_stats.json")
    alts = []
    for row in stats.get("top_alternations") or []:
        alts.append({
            "canonical": f"dakshina_alternation:{row['alternation']}",
            "variants": [f"count:{int(row['count'])}"],
            "dakshina_attested": "yes",
            "kind": "dakshina_alternation",
            "count": int(row["count"]),
        })
    words: list[dict[str, Any]] = []
    path = PROC / "romanization" / f"{lang}_variants.tsv"
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for rec in reader:
            variants = [v for v in (rec.get("variants") or "").split("|") if v]
            if len(variants) < 2:
                continue
            words.append({
                "canonical": variants[0],
                "variants": variants,
                "dakshina_attested": "yes",
                "kind": "dakshina_word",
                "count": int(rec.get("total_attestations") or 0),
            })
            if len(words) >= 15:
                break
    return alts[:10], words


def load_clusters(pair: str) -> list[dict[str, Any]]:
    path = PROC / pair / "romanization_clusters_clean.tsv"
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for rec in reader:
            parts = []
            for item in (rec.get("spellings") or "").split("|"):
                if not item:
                    continue
                form = item.rsplit(":", 1)[0]
                parts.append(form)
            if not parts:
                continue
            out.append({
                "canonical": parts[0],
                "variants": parts,
                "dakshina_attested": "yes" if rec.get("dakshina_attested") == "yes" else "no",
                "kind": "corpus_cluster",
                "count": int(rec.get("total") or 0),
            })
            if len(out) >= 100:
                break
    return out


def select_diverse(
    scored: list[tuple],
    limit: int,
    text_idx: int = 0,
    tokens_idx: int | None = None,
) -> list[Any]:
    """scored rows are (score_tuple..., payload). Highest score first."""
    selected = []
    used: set[tuple[str, ...]] = set()
    seen_text: set[str] = set()
    for row in scored:
        payload = row[-1]
        text = payload[text_idx] if isinstance(payload, tuple) else payload.get("text")
        key = norm_text(text)
        if not key or key in seen_text:
            continue
        toks = None
        if tokens_idx is not None and isinstance(payload, tuple):
            toks = payload[tokens_idx]
        elif isinstance(payload, dict):
            toks = payload.get("tokens")
        grams = fivegrams(toks or tokenize(text))
        if grams and grams & used:
            continue
        used |= grams
        seen_text.add(key)
        selected.append(payload)
        if len(selected) >= limit:
            break
    return selected


def mine_pair(pair: str) -> dict[str, Any]:
    local = LOCAL[pair]
    local_labels = {local}
    ta_lex = load_ta_lexicon() if pair == "ta_en" else None
    gold = BandAcc()
    allb = BandAcc()
    en_ins: Counter[str] = Counter()
    lvc: Counter[str] = Counter()
    lvc_ex: dict[str, list[str]] = defaultdict(list)
    morph: Counter[tuple[str, str, str]] = Counter()
    address = Counter()
    forms_needed = {
        f for bucket in ADDRESS_REGISTER[pair].values() for f in bucket
    }
    conv_pool: list[dict[str, Any]] = []
    te_req_pool: list[dict[str, Any]] = []
    gold_n = 0
    n_rows = 0
    seen_text: set[str] = set()

    path = PROC / pair / "utterances.jsonl"
    log(f"  streaming {path.name}")
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            n_rows += 1
            # ta_en calibration stats are on unique texts (overlap across YouTube dumps).
            if pair == "ta_en":
                key = norm_text(rec.get("text") or "")
                if not key or key in seen_text:
                    continue
                seen_text.add(key)
            tokens = rec.get("tokens") or tokenize(rec.get("text") or "")
            tags = rec.get("lang_tags") or []
            if len(tags) != len(tokens):
                tags = ["uncertain"] * len(tokens)
            text = rec.get("text") or ""
            source_id = rec.get("source_id") or ""
            origin = rec.get("tag_origin") or "auto"
            mets = mix_metrics(tags, local_labels)
            allb.add(mets)
            if origin == "gold":
                gold_n += 1
                gold.add(mets)
            if mets.get("defined") and mets["english_share"] is not None and mets["english_share"] < 0.5:
                for tok, tag in zip(tokens, tags):
                    if tag == "en" and WORD_RE.match(tok):
                        en_ins[tok.lower()] += 1
            lows = [t.lower() for t in tokens]
            bag = set(lows)
            for form in forms_needed:
                if form in bag:
                    address[form] += lows.count(form)
            for i, tok in enumerate(tokens):
                tag = tags[i] if i < len(tags) else "uncertain"
                hit = pair_morph_split(pair, tok, ta_lex)
                if hit and not morph_excluded(hit[0]):
                    morph[(tok.lower(), hit[0], hit[1])] += 1
                if i == 0:
                    continue
                lv = is_light_verb(pair, tok, tag)
                prev_tag = tags[i - 1] if i - 1 < len(tags) else ""
                if lv and prev_tag == "en" and WORD_RE.match(tokens[i - 1]):
                    pat = f"{tokens[i - 1].lower()} {lv}"
                    lvc[pat] += 1
                    if len(lvc_ex[pat]) < 3:
                        lvc_ex[pat].append(clip(text))
            if pair == "hi_en" and source_id in {"phinc", "hinge"}:
                if mets.get("cmi") and mets["cmi"] > 0 and len(tokens) >= 5:
                    if has_markers(tokens, CONV_MARKERS[pair]):
                        conv_pool.append({
                            "source_id": source_id,
                            "text": text,
                            "tokens": tokens,
                            "cmi": mets["cmi"],
                            "n_switches": mets["n_switches"],
                            "english_share": mets["english_share"],
                        })
            if pair == "ta_en" and source_id == "offenseval_dravidian":
                if mets.get("cmi") and mets["cmi"] > 0 and 5 <= len(tokens) <= 30:
                    if has_markers(tokens, CONV_MARKERS[pair]) and not rc.domain_blocked(tokens, pair):
                        conv_pool.append({
                            "source_id": source_id,
                            "text": text,
                            "tokens": tokens,
                            "cmi": mets["cmi"],
                            "n_switches": mets["n_switches"],
                            "english_share": mets["english_share"],
                        })
            if pair == "te_en" and source_id in {"cmtet_sentiment", "cmtet_lid", "icon_2016"}:
                if 5 <= len(tokens) <= 30 and mets.get("cmi") and mets["cmi"] > 0:
                    if has_markers(tokens, TE_REQUEST) and not rc.domain_blocked(tokens, pair):
                        te_req_pool.append({
                            "source_id": source_id,
                            "text": text,
                            "tokens": tokens,
                            "cmi": mets["cmi"],
                            "n_switches": mets["n_switches"],
                            "english_share": mets["english_share"],
                        })

    log(f"  utterances={n_rows} gold={gold_n} lvc={len(lvc)} morph={len(morph)}")
    return {
        "gold": gold,
        "all": allb,
        "gold_n": gold_n,
        "en_ins": en_ins,
        "lvc": lvc,
        "lvc_ex": lvc_ex,
        "morph": morph,
        "address": address,
        "conv_pool": conv_pool,
        "te_req_pool": te_req_pool,
    }


def lvc_records(counter: Counter[str], examples: dict[str, list[str]], excluded: bool) -> list[dict[str, Any]]:
    rows = []
    for pat, n in counter.most_common():
        if is_lvc_excluded(pat) != excluded:
            continue
        rows.append({
            "pattern": pat,
            "examples": examples.get(pat, [])[:3],
            "count": int(n),
        })
        if len(rows) >= 40:
            break
    return rows


def morph_records(counter: Counter[tuple[str, str, str]]) -> list[dict[str, Any]]:
    rows = []
    for (form, stem, suffix), n in counter.most_common():
        if morph_excluded(stem):
            continue
        rows.append({
            "form": form,
            "stem": stem,
            "suffix": suffix,
            "count": int(n),
        })
        if len(rows) >= 40:
            break
    return rows


def address_records(pair: str, counts: Counter[str]) -> list[dict[str, Any]]:
    out = []
    for register, forms in ADDRESS_REGISTER[pair].items():
        for form in forms:
            out.append({
                "form": form,
                "register": register,
                "count": int(counts.get(form, 0)),
            })
    return out


def make_ref(
    rid: str,
    source_id: str,
    text: str,
    register: str,
    mets: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cleaned, _n = replace_brands(text)
    rec = {
        "id": rid,
        "source_id": source_id,
        "text": cleaned,
        "register": register,
        "accepted": "pending",
        "redistributable": redistributable(source_id),
        "_n_brand": _n,
        "_mets": mets,
    }
    return rec


def count_brands(refs: list[dict[str, Any]]) -> dict[str, Any]:
    n_repl = sum(int(r.pop("_n_brand", 0)) for r in refs)
    n_touch = sum(1 for r in refs if r.get("_n_brand", 0) > 0)
    # _n_brand already popped; recount via stored flag if needed
    return {
        "n_replacements": n_repl,
        "n_utterances_touched": n_touch,
        "note": "original brand/company/app strings are not stored; mapping count only",
    }


def finalize_refs(refs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    n_repl = 0
    n_touch = 0
    service_mets: list[dict[str, Any]] = []
    clean: list[dict[str, Any]] = []
    for r in refs:
        n = int(r.pop("_n_brand", 0))
        mets = r.pop("_mets", None)
        n_repl += n
        if n:
            n_touch += 1
        if r["register"] in SERVICE_REGISTERS and mets:
            service_mets.append(mets)
        clean.append(r)
    summary = {
        "n_replacements": n_repl,
        "n_utterances_touched": n_touch,
        "note": "original brand/company/app strings are not stored; mapping count only",
    }
    return clean, summary, service_mets


def service_bands_from_mets(mets_list: list[dict[str, Any]], note: str) -> dict[str, Any]:
    acc = BandAcc()
    for m in mets_list:
        acc.add(m)
    return acc.as_block("auto", note)


def csv_mets(row: dict[str, str]) -> dict[str, Any]:
    es = row.get("english_share") or ""
    cmi = row.get("cmi") or ""
    sw = row.get("n_switches") or "0"
    defined = bool(es) and bool(cmi)
    es_f = float(es) if es else None
    cmi_f = float(cmi) if cmi else None
    return {
        "english_share": es_f,
        "cmi": cmi_f,
        "n_switches": int(float(sw)) if sw else 0,
        "monolingual": bool(defined and cmi_f == 0.0),
        "defined": defined,
    }


def pick_strict(pair: str, limit: int | None) -> list[dict[str, Any]]:
    rows = load_csv(PROC / pair / "service_candidates_strict.csv")
    if limit is not None:
        rows = rows[:limit]
    out = []
    for i, row in enumerate(rows, 1):
        out.append(make_ref(
            f"{pair}_ref_svc_{i:03d}",
            row["source_id"],
            row["text"],
            "service",
            csv_mets(row),
        ))
    return out


def pick_htop(n: int) -> list[dict[str, Any]]:
    rows = load_csv(PROC / "service_dialogs" / "hinglish_top_service_150.csv")[:n]
    out = []
    for i, row in enumerate(rows, 1):
        out.append(make_ref(
            f"hi_en_ref_htop_{i:03d}",
            "hinglish_top",
            row["text"],
            "assistant_command",
            csv_mets(row),
        ))
    return out


def pick_dialog_turns(pair: str, n_user: int, n_sys: int, lexicon: set[str]) -> list[dict[str, Any]]:
    dialogs = load_jsonl(PROC / "service_dialogs" / f"{pair}_dialogs.jsonl")
    user_scored: list[tuple] = []
    sys_freq: Counter[str] = Counter()
    sys_meta: dict[str, dict[str, Any]] = {}
    for d in dialogs:
        for t in d.get("turns") or []:
            text = t.get("text") or ""
            speaker = t.get("speaker")
            tokens, tags = auto_tag(text, lexicon)
            mets = mix_metrics(tags, {"uncertain"})
            if speaker == "user":
                if len(tokens) >= 5 and mets.get("cmi") and mets["cmi"] > 0:
                    user_scored.append((
                        -float(mets["cmi"]),
                        -mets["n_switches"],
                        norm_text(text),
                        {
                            "text": text,
                            "tokens": tokens,
                            "mets": mets,
                        },
                    ))
            elif speaker == "system":
                key = norm_text(text)
                if len(tokens) >= 5:
                    sys_freq[key] += 1
                    sys_meta.setdefault(key, {"text": text, "tokens": tokens, "mets": mets})
    user_scored.sort(key=lambda r: r[:3])
    users = select_diverse(user_scored, n_user)
    out = []
    for i, u in enumerate(users, 1):
        out.append(make_ref(
            f"{pair}_ref_user_{i:03d}",
            "code_mixed_dialog",
            u["text"],
            "service_dialog_user",
            u["mets"],
        ))
    sys_scored = [
        (-sys_freq[k], k, sys_meta[k])
        for k in sys_freq
    ]
    sys_scored.sort(key=lambda r: r[:2])
    systems = select_diverse(sys_scored, n_sys)
    for i, s in enumerate(systems, 1):
        out.append(make_ref(
            f"{pair}_ref_sys_{i:03d}",
            "code_mixed_dialog",
            s["text"],
            "service_dialog_system_template",
            s["mets"],
        ))
    return out


def pick_pool(pair: str, pool: list[dict[str, Any]], n: int, register: str, prefix: str, used: set[str]) -> list[dict[str, Any]]:
    scored = []
    for rec in pool:
        if norm_text(rec["text"]) in used:
            continue
        scored.append((
            -float(rec.get("cmi") or 0),
            -int(rec.get("n_switches") or 0),
            norm_text(rec["text"]),
            rec,
        ))
    scored.sort(key=lambda r: r[:3])
    chosen = select_diverse(scored, n)
    out = []
    for i, rec in enumerate(chosen, 1):
        out.append(make_ref(
            f"{pair}_ref_{prefix}_{i:03d}",
            rec["source_id"],
            rec["text"],
            register,
            {
                "english_share": rec.get("english_share"),
                "cmi": rec.get("cmi"),
                "n_switches": rec.get("n_switches") or 0,
                "monolingual": bool(rec.get("cmi") == 0),
                "defined": rec.get("cmi") is not None,
            },
        ))
    return out


def pick_cmqa_te(n: int, lexicon: set[str]) -> list[dict[str, Any]]:
    rows = [
        q for q in load_jsonl(PROC / "cmqa" / "questions.jsonl")
        if q.get("pair") == "te_en"
    ]
    scored = []
    for q in rows:
        text = q.get("text") or ""
        tokens, tags = auto_tag(text, lexicon)
        mets = mix_metrics(tags, {"uncertain"})
        scored.append((
            -(mets["cmi"] or 0),
            -len(tokens),
            norm_text(text),
            {"text": text, "tokens": tokens, "mets": mets, "id": q.get("id")},
        ))
    scored.sort(key=lambda r: r[:3])
    chosen = select_diverse(scored, n)
    out = []
    for i, rec in enumerate(chosen, 1):
        out.append(make_ref(
            f"te_en_ref_cmqa_{i:03d}",
            "chandu_cmqa",
            rec["text"],
            "question",
            rec["mets"],
        ))
    return out


def used_texts(refs: list[dict[str, Any]]) -> set[str]:
    return {norm_text(r["text"]) for r in refs}


def unnatural(pair: str, n: int = 50) -> list[dict[str, Any]]:
    rows = load_csv(PROC / pair / "unnatural_negatives.csv")[:n]
    out = []
    for row in rows:
        out.append({
            "id": row["neg_id"],
            "text": row["text"],
            "reason": row["corruption_type"],
            "synthetic": True,
        })
    return out


def build_sources(pair: str, refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    used_for: dict[str, set[str]] = defaultdict(set)
    for sid in PAIR_STAT_SOURCES[pair]:
        used_for[sid].add("stats")
        used_for[sid].add("romanization")
    used_for["dakshina"].add("romanization")
    for r in refs:
        sid = r["source_id"]
        used_for[sid].add("reference_utterances")
        if sid == "code_mixed_dialog":
            used_for[sid].add("dialog_anchors")
        if sid == "hinglish_top":
            used_for[sid].add("dialog_anchors")
    order = list(PAIR_STAT_SOURCES[pair])
    for extra in ("code_mixed_dialog", "hinglish_top", "dakshina", "chandu_cmqa", "cmqa"):
        if extra in used_for and extra not in order:
            order.append(extra)
    for sid in used_for:
        if sid not in order:
            order.append(sid)
    out = []
    for sid in order:
        meta = SOURCE_META.get(sid, {
            "url": "",
            "license_status": "UNCLEAR",
            "redistributable": redistributable(sid),
        })
        purposes = ["stats", "reference_utterances", "romanization", "dialog_anchors"]
        used = [p for p in purposes if p in used_for[sid]]
        out.append({
            "source_id": sid,
            "url": meta["url"],
            "license_status": meta["license_status"],
            "redistributable": meta["redistributable"],
            "used_for": ", ".join(used),
        })
    return out


def rule_entries(pair: str, mined: dict[str, Any], mixing_origin: str) -> list[dict[str, Any]]:
    n_morph = len(mined["morph"])
    notes = {
        "L01": (
            "needs_speaker_review",
            "clause grammar is not measurable from CMI; fluent review required",
        ),
        "L02": (
            "needs_speaker_review",
            "switches_per_utterance exists in mixing_bands but boundary quality is unvalidated",
        ),
        "L03": (
            "supported_by_data" if n_morph else "needs_speaker_review",
            f"mixed_morphology has {min(40, n_morph)} attested stem+suffix types; "
            "Telugu/Tamil within-word mixing is allowed (plan L03)",
        ),
        "L04": (
            "supported_by_data",
            f"mixing_bands origin={mixing_origin}; also service_register_bands for generation",
        ),
        "L05": (
            "supported_by_data",
            "romanization_variants from romanization_clusters_clean.tsv plus Dakshina alternations",
        ),
        "L06": (
            "supported_by_data",
            "address_forms with formal/neutral/informal counts from the pair corpora",
        ),
        "L07": (
            "policy",
            "named entities, payment acronyms, or a lone greeting do not establish bilingual substance",
        ),
        "L08": (
            "policy",
            "semantic fidelity is an experimental invariant, not a corpus statistic",
        ),
        "L09": (
            "policy",
            "mixing, spelling, and register must be distributed across labels (design requirement)",
        ),
        "L10": (
            "needs_speaker_review",
            "Code-Mixed-Dialog user turns show short replies; natural endings still need speaker check"
            if pair != "te_en"
            else "no Roman Telugu service dialogue; turn-length variation is a policy pending speaker review",
        ),
        "L11": (
            "policy",
            "do not enforce one matrix language globally; mixing_bands include monolingual turns",
        ),
        "L12": (
            "policy",
            "exclude English glosses, parenthetical translations, and stage directions from detector input",
        ),
    }
    return [
        {"rule_id": rid, "status": status, "evidence": ev}
        for rid, (status, ev) in notes.items()
    ]


def known_gaps(pair: str, gold_n: int, mixing_origin: str, n_svc: int) -> list[str]:
    gaps = [
        "Calibration sources are social media, QA, or restaurant-bot translations — not authentic Indian fraud or care calls.",
    ]
    if pair == "hi_en":
        gaps.append("LinCE tweet text is not redistributable; strict service-candidate audit precision is 60%.")
        gaps.append("No open human Roman Hinglish phone-call / customer-care corpus.")
    if pair == "ta_en":
        if gold_n < GOLD_MIN:
            gaps.append(
                f"CoLI gold is {gold_n} utterances (<{GOLD_MIN}); mixing_bands are {mixing_origin} all-data quantiles."
            )
        gaps.append("YouTube/social comments dominate; ICON 2015 Tamil–English gold LID was not obtained.")
        gaps.append("No open Roman Tamil–English call-centre dialogue.")
    if pair == "te_en":
        gaps.append(
            "known_gap: no Roman Telugu-English service dialogue exists publicly."
        )
        gaps.append(
            f"Only {n_svc} strict service candidates survived the domain filter; "
            "remaining reference rows are CMTET/ICON request-marked social text and CMQA trivia questions."
        )
        gaps.append("CMTET and ICON have no clear redistribution licence.")
    return gaps


def build_profile(pair: str) -> dict[str, Any]:
    log(f"=== {pair} ===")
    mined = mine_pair(pair)
    gold_n = mined["gold_n"]
    if pair == "ta_en" and gold_n < GOLD_MIN:
        mixing = mined["all"].as_block(
            "auto",
            f"CoLI gold is {gold_n} (<{GOLD_MIN}); using all-data auto quantiles",
        )
        mixing_origin = "auto"
    else:
        mixing = mined["gold"].as_block(
            "gold",
            f"gold-tagged utterances n={gold_n}",
        )
        mixing_origin = "gold"

    refs: list[dict[str, Any]] = []
    if pair == "hi_en":
        refs.extend(pick_strict(pair, None))
        refs.extend(pick_htop(60))
        lex = dialog_lexicon(pair)
        refs.extend(pick_dialog_turns(pair, 50, 15, lex))
        used = used_texts(refs)
        refs.extend(pick_pool(pair, mined["conv_pool"], 20, "conversational", "conv", used))
    elif pair == "ta_en":
        refs.extend(pick_strict(pair, 120))
        lex = dialog_lexicon(pair)
        refs.extend(pick_dialog_turns(pair, 50, 15, lex))
        used = used_texts(refs)
        refs.extend(pick_pool(pair, mined["conv_pool"], 15, "conversational", "conv", used))
    else:
        refs.extend(pick_strict(pair, None))
        used = used_texts(refs)
        refs.extend(pick_pool(pair, mined["te_req_pool"], 100, "conversational", "req", used))
        lex = dialog_lexicon(pair)
        refs.extend(pick_cmqa_te(86, lex))

    refs, brand_summary, svc_mets = finalize_refs(refs)
    n_svc = sum(1 for r in refs if r["register"] == "service")
    service_bands = service_bands_from_mets(
        svc_mets,
        "computed on service / dialog / assistant reference utterances "
        "(not social-media mixing_bands). Selection-biased upward: candidates "
        "required CMI > 0 and English service terms. Upper reference only, not a generation target.",
    )

    alts, dak_words = load_dakshina_summary(LOCAL[pair])
    roman = load_clusters(pair) + dak_words + alts

    profile = {
        "pair": pair,
        "script": "roman",
        "version": VERSION,
        "status_note": STATUS_NOTE,
        "known_gaps": known_gaps(pair, gold_n, mixing_origin, n_svc),
        "brand_placeholder_replacements": brand_summary,
        "sources": build_sources(pair, refs),
        "mixing_bands": mixing,
        "service_register_bands": service_bands,
        "lexical_insertions": [
            {"en_word": w, "count": int(c)}
            for w, c in mined["en_ins"].most_common(100)
        ],
        "light_verb_constructions": lvc_records(mined["lvc"], mined["lvc_ex"], False),
        "excluded_domain_specific": lvc_records(mined["lvc"], mined["lvc_ex"], True),
        "mixed_morphology": morph_records(mined["morph"]),
        "romanization_variants": roman,
        "address_forms": address_records(pair, mined["address"]),
        "reference_utterances": refs,
        "unnatural_examples": unnatural(pair, 50),
        "rules": rule_entries(pair, mined, mixing_origin),
        "reviewer_signoff": [],
        "change_history": [
            {
                "date": CHANGE_DATE,
                "version": VERSION,
                "summary": CHANGE_NOTE,
                "note": CHANGE_NOTE,
            }
        ],
    }
    return profile


def validate_profile(profile: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    validator = Draft202012Validator(schema)
    return [
        f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}"
        for e in validator.iter_errors(profile)
    ]


def write_yaml(path: Path, profile: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(
            profile,
            fh,
            allow_unicode=True,
            sort_keys=False,
            width=1000,
            default_flow_style=False,
        )


def fmt_band(band: dict[str, Any], key: str) -> str:
    q = band[key]
    return f"{q['p50']:.3f} (p10={q['p10']:.3f}, p90={q['p90']:.3f})"


def register_counts(refs: list[dict[str, Any]]) -> Counter[str]:
    return Counter(r["register"] for r in refs)


def write_readme(profiles: dict[str, dict[str, Any]]) -> None:
    lines = [
        "# Language profiles (v0.1-auto)",
        "",
        "Automatically derived from human-written calibration corpora. "
        "**Not yet reviewed by a fluent speaker.** "
        "Generation targets `mixing_bands` (human, mostly gold-tagged). `service_register_bands` are selection-biased upward (candidates were required to have CMI > 0 and English service terms), so use them only as an upper reference, not a target.",
        "",
        "## Pair comparison",
        "",
        "| Field | hi_en | ta_en | te_en |",
        "|---|---|---|---|",
    ]

    def cell(pair: str, fn) -> str:
        return str(fn(profiles[pair])).replace("|", "/")

    def sources_cell(p: dict[str, Any]) -> str:
        return ", ".join(s["source_id"] for s in p["sources"])

    def origin_cell(p: dict[str, Any]) -> str:
        b = p["mixing_bands"]
        return f"{b['origin']} (n={b['n_utterances']})"

    def band_cell(p: dict[str, Any], which: str, metric: str) -> str:
        return fmt_band(p[which], metric)

    rows = [
        ("Sources", sources_cell),
        ("Mixing-band origin", origin_cell),
        ("Social english_share p50", lambda p: band_cell(p, "mixing_bands", "english_share")),
        ("Social CMI p50", lambda p: band_cell(p, "mixing_bands", "cmi")),
        ("Service english_share p50", lambda p: band_cell(p, "service_register_bands", "english_share")),
        ("Service CMI p50", lambda p: band_cell(p, "service_register_bands", "cmi")),
        ("# reference utterances", lambda p: str(len(p["reference_utterances"]))),
    ]
    for label, fn in rows:
        lines.append(
            f"| {label} | {cell('hi_en', fn)} | {cell('ta_en', fn)} | {cell('te_en', fn)} |"
        )

    registers = [
        "service",
        "service_dialog_user",
        "service_dialog_system_template",
        "assistant_command",
        "question",
        "conversational",
    ]
    lines.append("| # by register | | | |")
    for reg in registers:
        def _c(p: dict[str, Any], r=reg) -> str:
            return str(register_counts(p["reference_utterances"]).get(r, 0))
        lines.append(f"| `{reg}` | {cell('hi_en', _c)} | {cell('ta_en', _c)} | {cell('te_en', _c)} |")

    def gaps_cell(p: dict[str, Any]) -> str:
        gaps = p.get("known_gaps") or []
        if any("no Roman Telugu-English service dialogue" in g for g in gaps):
            return "No public Roman TE-EN service dialogue; 14 strict candidates only"
        if any("CoLI gold" in g for g in gaps):
            return "Gold LID <2000 (auto bands); YouTube-heavy; no call-centre CS"
        return "Social-media calibration; no open phone-call corpus; LinCE not redistributable"

    lines.append(f"| Known gaps | {cell('hi_en', gaps_cell)} | {cell('ta_en', gaps_cell)} | {cell('te_en', gaps_cell)} |")
    lines += [
        "",
        "Full `known_gaps` lists live in each YAML.",
        "",
        "## Fluent-reviewer instructions",
        "",
        "1. Open `configs/language_profiles/{hi_en,ta_en,te_en}.yaml`.",
        "2. Read `status_note` and `known_gaps` before judging items.",
        "3. For each `reference_utterances` row, set `accepted` to `yes` or `no`. Leave `pending` only if you skip the item.",
        "4. Check that brand names were replaced with fictional placeholders (Kaveri Bank, ShopKart, SwiftShip Couriers, PaySetu, `[TELECOM]`, `[BRAND]`). Do not restore real brand strings.",
        "5. Skim `light_verb_constructions` and `mixed_morphology`; film/sports leftovers belong in `excluded_domain_specific`, not the main lists.",
        "6. When you finish a pair, append a `reviewer_signoff` object: `role`, `name`, `date` (YYYY-MM-DD), `notes`.",
        "7. Add a `change_history` entry and bump `version` (for example `0.1-reviewed`).",
        "8. Two bilingual reviewers per pair is the plan-§10.3 target; this file is the first-pass auto draft.",
        "",
        "Neither band set is phone-call truth. `mixing_bands` come from social media; `service_register_bands` are inflated by the CMI > 0 selection filter. Sample mixing from `mixing_bands`, and treat values above the service p75 as review flags. Telugu–English has no public Roman service dialogue.",
        "",
    ]
    path = OUT_DIR / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    log(f"wrote {path}")


def summarize(pair: str, profile: dict[str, Any], errors: list[str]) -> None:
    rc_counts = register_counts(profile["reference_utterances"])
    mb = profile["mixing_bands"]
    sb = profile["service_register_bands"]
    status = "VALID" if not errors else f"INVALID ({len(errors)} errors)"
    log(f"  {pair} {status} refs={len(profile['reference_utterances'])} {dict(rc_counts)}")
    log(
        f"    social en_p50={mb['english_share']['p50']:.3f} cmi_p50={mb['cmi']['p50']:.3f} origin={mb['origin']}"
    )
    log(
        f"    service en_p50={sb['english_share']['p50']:.3f} cmi_p50={sb['cmi']['p50']:.3f} n={sb['n_utterances']}"
    )
    if errors:
        for e in errors[:12]:
            log(f"    schema: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", nargs="+", default=["hi_en", "ta_en", "te_en"])
    args = parser.parse_args()
    schema = load_json(SCHEMA_PATH)
    profiles: dict[str, dict[str, Any]] = {}
    all_ok = True
    for pair in args.pairs:
        profile = build_profile(pair)
        errors = validate_profile(profile, schema)
        if errors:
            all_ok = False
        summarize(pair, profile, errors)
        out = OUT_DIR / f"{pair}.yaml"
        write_yaml(out, profile)
        log(f"  wrote {out}")
        profiles[pair] = profile
    if set(args.pairs) >= {"hi_en", "ta_en", "te_en"}:
        write_readme(profiles)
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
