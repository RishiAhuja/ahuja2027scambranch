#!/usr/bin/env python3
"""Build an empirical Telugu-English (Romanized) calibration profile.

Pair ID: te_en. Uses only human-written code-mixed corpora (no LLM-generated
sources). Re-run from the repo root:

    .venv/bin/python scripts/calibration/mine_te_en.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen

import numpy as np
from wordfreq import zipf_frequency

REPO = Path(__file__).resolve().parents[2]
RAW_ROOT = REPO / "data" / "calibration" / "raw" / "te_en"
PROC_ROOT = REPO / "data" / "calibration" / "processed" / "te_en"
LIC_ROOT = REPO / "data" / "calibration" / "licenses"

PAIR_ID = "te_en"
ZIPF_EN = 3.5
ZIPF_MIXED_GOLD = 4.0
TE_LEX_MIN_FREQ = 5
TELUGU_UNICODE_MAX = 0.10
SERVICE_MAX = 400
NEGATIVES_N = 100
RNG_SEED = 42

TE_SUFFIXES = (
    "cheyandi",
    "chestha",
    "chesaru",
    "cheyyi",
    "chesi",
    "avvadi",
    "ayyindi",
    "ante",
    "tho",
    "lo",
    "ki",
    "ni",
    "nu",
    "la",
    "ga",
)
# longest-first already via length sort below
TE_SUFFIXES_SORTED = tuple(sorted(TE_SUFFIXES, key=len, reverse=True))

LIGHT_VERBS = (
    "chey",
    "cheyyi",
    "cheyandi",
    "chesanu",
    "chesaru",
    "chesthanu",
    "chestha",
    "cheyyali",
    "avvu",
    "ayyindi",
    "avuthundi",
    "pettu",
    "pettandi",
    "ivvu",
    "ivvandi",
)

ADDRESS_FORMS = (
    "meeru",
    "mee",
    "miru",
    "nuvvu",
    "nee",
    "sir",
    "madam",
    "garu",
    "anna",
    "akka",
    "babu",
    "bro",
    "please",
    "plz",
    "pls",
    "dhanyavaadalu",
    "thanks",
    "thank you",
    "namaskaram",
    "hello",
    "sare",
    "avunu",
    "ledu",
    "kaadu",
    "okay",
)

SERVICE_MARKERS = (
    "meeru",
    "mee",
    "garu",
    "cheppandi",
    "cheyandi",
    "cheyyandi",
    "pampandi",
    "ivvandi",
    "konchem",
    "please",
    "sir",
    "madam",
    "payment",
    "dabbulu",
    "rs",
    "rupees",
    "account",
    "order",
    "delivery",
    "refund",
    "bank",
    "card",
    "number",
    "call",
    "message",
    "otp",
    "eppudu",
    "entha",
    "ela",
    "problem",
    "issue",
    "help",
    "service",
    "book",
    "cancel",
    "return",
    "confirm",
    "time",
    "repu",
    "ippudu",
)

TRANSACTIONAL = frozenset(
    {
        "payment",
        "dabbulu",
        "rupees",
        "account",
        "otp",
        "refund",
        "bank",
        "delivery",
        "order",
        "card",
        "book",
        "cancel",
        "return",
        "confirm",
        "service",
        "pampandi",
    }
)
REQUEST_FORMS = frozenset(
    {
        "cheppandi",
        "cheyandi",
        "cheyyandi",
        "ivvandi",
        "please",
        "sir",
        "madam",
        "garu",
        "meeru",
    }
)
MOVIE_DOMAIN = frozenset(
    {
        "movie",
        "movies",
        "review",
        "reviews",
        "cinema",
        "cinemalu",
        "cinemalo",
        "trailer",
        "teaser",
        "hero",
        "heroine",
        "song",
        "songs",
        "interval",
        "boxoffice",
    }
)
EN_STEM_BLOCK = frozenset(
    {
        "the",
        "this",
        "that",
        "then",
        "them",
        "they",
        "than",
        "there",
        "these",
        "those",
        "their",
        "what",
        "when",
        "with",
        "from",
        "have",
        "been",
        "were",
        "your",
        "you",
        "not",
        "but",
        "and",
        "for",
        "are",
        "was",
        "can",
        "all",
        "any",
        "our",
        "out",
        "how",
        "who",
        "him",
        "her",
        "his",
        "she",
        "its",
        "don",
        "dan",
        "int",
        "val",
        "cha",
        "tha",
        "ann",
        "man",
        "men",
        "get",
        "got",
        "let",
        "put",
        "set",
        "run",
        "use",
        "new",
        "old",
        "big",
        "few",
        "own",
        "too",
        "very",
        "just",
        "also",
        "only",
        "even",
        "back",
        "over",
        "such",
        "most",
        "some",
        "into",
        "more",
        "less",
        "much",
        "many",
        "like",
        "well",
        "still",
        "being",
        "will",
        "shall",
        "does",
        "did",
        "has",
        "had",
        "here",
        "once",
        "each",
        "same",
        "other",
        "after",
        "before",
        "about",
        "under",
        "again",
        "anna",
        "villa",
        "panda",
        "ivan",
        "chin",
        "danny",
        "hell",
        "hill",
        "mill",
        "doll",
        "ball",
        "fall",
        "tall",
        "wall",
        "will",
        "bill",
        "fill",
        "kill",
        "till",
        "tell",
        "sell",
        "well",
        "come",
        "came",
        "gone",
        "done",
        "make",
        "made",
        "take",
        "took",
        "give",
        "gave",
        "keep",
        "kept",
        "know",
        "knew",
        "want",
        "need",
        "look",
        "looked",
        "good",
        "best",
        "last",
        "next",
        "long",
        "high",
        "small",
        "great",
        "little",
        "right",
        "left",
        "part",
        "case",
        "fact",
        "hand",
        "world",
        "life",
        "day",
        "way",
        "thing",
        "child",
        "woman",
        "man",
        "people",
        "person",
    }
)
GOOD_SHORT_STEMS = frozenset(
    {
        "time",
        "book",
        "call",
        "card",
        "film",
        "show",
        "phone",
        "office",
        "movie",
        "ticket",
        "order",
        "cancel",
        "refund",
        "bank",
        "cash",
        "mail",
        "chat",
        "post",
        "play",
        "work",
        "team",
        "game",
        "room",
        "home",
        "shop",
        "store",
        "class",
        "exam",
        "test",
        "page",
        "link",
        "file",
        "form",
        "plan",
        "seat",
        "date",
        "week",
        "year",
        "city",
        "town",
        "road",
        "park",
        "bill",
        "otp",
        "pin",
        "app",
        "net",
        "web",
        "bus",
        "cab",
        "car",
        "bike",
        "train",
        "flight",
        "hotel",
        "food",
        "item",
        "list",
        "code",
        "otp",
    }
)

TE_FUNCTION_VERBS = frozenset(
    {
        "meeru",
        "miru",
        "mee",
        "meeru",
        "nuvvu",
        "nuvu",
        "nee",
        "neeku",
        "naku",
        "naaku",
        "nenu",
        "nenu",
        "memu",
        "manam",
        "vaadu",
        "atanu",
        "aame",
        "ame",
        "vaaru",
        "vaallu",
        "vallu",
        "idi",
        "adi",
        "avi",
        "ivi",
        "oka",
        "okati",
        "rendu",
        "mundu",
        "tarvata",
        "tharvatha",
        "ippudu",
        "eppudu",
        "ekkada",
        "ikkada",
        "akkada",
        "ela",
        "enduku",
        "enti",
        "entha",
        "enta",
        "emi",
        "em",
        "kaani",
        "kani",
        "kada",
        "kadha",
        "gadha",
        "ante",
        "ani",
        "ani",
        "anukoni",
        "undi",
        "unnai",
        "unnaru",
        "unna",
        "unnadi",
        "undhi",
        "ledu",
        "levu",
        "leru",
        "kaadu",
        "kaadhu",
        "avunu",
        "avuna",
        "sare",
        "sarele",
        "kavali",
        "kaavali",
        "vaddu",
        "oddu",
        "chey",
        "cheyyi",
        "cheyandi",
        "cheyyandi",
        "cheyyali",
        "chesanu",
        "chesaru",
        "chesindi",
        "chesindhi",
        "chestha",
        "chesthanu",
        "chestham",
        "chestharu",
        "chestundi",
        "chesthundi",
        "chesi",
        "chesi",
        "cheppandi",
        "cheppu",
        "cheppanu",
        "chepparu",
        "cheptha",
        "ivvu",
        "ivvandi",
        "icchanu",
        "iccharu",
        "pettu",
        "pettandi",
        "pampandi",
        "pampu",
        "chudu",
        "chudandi",
        "choodu",
        "raa",
        "randi",
        "vellu",
        "vellandi",
        "vastha",
        "vasthanu",
        "ostha",
        "avvu",
        "ayyindi",
        "ayindi",
        "avuthundi",
        "avuthundi",
        "kavadam",
        "gurinchi",
        "gurinchi",
        "kosam",
        "koraku",
        "tho",
        "tho",
        "lo",
        "lopala",
        "nundi",
        "nunchi",
        "ki",
        "ku",
        "ni",
        "nu",
        "ga",
        "gaa",
        "lanti",
        "laga",
        "kuda",
        "kooda",
        "mathram",
        "matram",
        "inka",
        "inkaa",
        "malli",
        "malli",
        "inkaa",
        "kaabatti",
        "kabatti",
        "anduke",
        "endukante",
        "ayithe",
        "aithe",
        "ayina",
        "aina",
        "tappa",
        "thappa",
        "tarwata",
        "mundu",
        "tarvatha",
        "appudu",
        "apudu",
        "asalu",
        "chala",
        "chaala",
        "konchem",
        "koncham",
        "baaga",
        "baga",
        "manchiga",
        "taggaga",
        "ekkuva",
        "thakkuva",
        "takkuva",
        "mari",
        "mari",
        "aithe",
        "gaani",
        "kani",
        "ledu",
        "kaadu",
        "avunu",
        "sare",
        "namaskaram",
        "dhanyavaadalu",
        "garu",
        "andi",
        "ra",
        "rey",
        "orey",
        "babu",
        "anna",
        "akka",
        "chelli",
        "thammudu",
    }
)

PROFANITY = frozenset(
    {
        "fuck",
        "fucking",
        "fucked",
        "fucker",
        "shit",
        "shitty",
        "bitch",
        "bastard",
        "asshole",
        "dick",
        "pussy",
        "cunt",
        "slut",
        "whore",
        "motherfucker",
        "mf",
        "wtf",
        "bullshit",
        "lanja",
        "lanjaakodaka",
        "lanjakodaka",
        "puku",
        "pooku",
        "dengu",
        "dengey",
        "dengeyandi",
        "dengina",
        "gudda",
        "soothu",
        "soothini",
        "boothu",
        "madarchod",
        "bhenchod",
        "behenchod",
        "chutiya",
        "chutiye",
        "harami",
        "randi",
        "gandu",
        "lund",
        "lavde",
        "loda",
        "mc",
        "bc",
    }
)

HINDI_VERBS = ("karo", "kijiye")

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.I)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
HANDLE_RE = re.compile(r"^@\w+")
PHONE_RE = re.compile(r"(?:\+91[\s-]?)?[6-9]\d{9}\b|^\d{10}$")
HASHTAG_ONLY_RE = re.compile(r"^#+")
EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001faff"
    "\U00002700-\U000027bf"
    "\U0001f1e0-\U0001f1ff"
    "\U00002600-\U000026ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa70-\U0001faff"
    "\U0000200d"
    "\U0000fe0f"
    "]+",
    flags=re.UNICODE,
)
TE_UNICODE_RE = re.compile(r"[\u0C00-\u0C7F]")
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+")
SENT_LABEL_RE = re.compile(r"^(NEG|NTL|POS):\s*(.*)$")

GOLD_TAG_MAP = {
    "en": "en",
    "eng": "en",
    "te": "te",
    "tel": "te",
    "ne": "ne",
    "univ": "other",
    "universal": "other",
    "acro": "other",
    "mix": "mixed",
    "mixed": "mixed",
    "undef": "uncertain",
}


@dataclass
class Utterance:
    id: str
    source_id: str
    text: str
    tokens: list[str]
    lang_tags: list[str]
    tag_origin: str
    source_item_id: str = ""


@dataclass
class SourceInfo:
    source_id: str
    url: str
    version: str
    size_note: str
    license: str
    citation: str
    redistribution: str
    obtained: bool
    notes: str = ""


_zipf_cache: dict[str, float] = {}


def zipf_en(tok: str) -> float:
    key = tok.lower()
    if key not in _zipf_cache:
        _zipf_cache[key] = float(zipf_frequency(key, "en"))
    return _zipf_cache[key]


def telugu_unicode_ratio(text: str) -> float:
    if not text:
        return 0.0
    n = sum(1 for ch in text if not ch.isspace())
    if n == 0:
        return 0.0
    te = len(TE_UNICODE_RE.findall(text))
    return te / n


def is_url(tok: str) -> bool:
    return bool(URL_RE.search(tok)) or tok.lower().startswith("http") or "t.co/" in tok.lower()


def is_handle(tok: str) -> bool:
    return bool(HANDLE_RE.match(tok))


def is_email(tok: str) -> bool:
    return bool(EMAIL_RE.search(tok))


def is_phone(tok: str) -> bool:
    digits = re.sub(r"\D", "", tok)
    return bool(PHONE_RE.search(tok)) or (len(digits) >= 10 and tok.replace(" ", "").replace("-", "").isdigit())


def is_emoji_token(tok: str) -> bool:
    stripped = EMOJI_RE.sub("", tok)
    return (not stripped.strip()) and bool(tok)


def drop_or_clean_token(tok: str) -> str | None:
    if not tok or tok.isspace():
        return None
    if is_url(tok) or is_handle(tok) or is_email(tok) or is_phone(tok) or is_emoji_token(tok):
        return None
    tok = EMOJI_RE.sub("", tok)
    tok = tok.replace("#", "")
    tok = tok.strip()
    if not tok or HASHTAG_ONLY_RE.match(tok):
        return None
    return tok


def map_gold_tag(raw: str) -> str:
    return GOLD_TAG_MAP.get(raw.lower().strip(), "uncertain")


def intra_word_mixed(token: str, min_zipf: float) -> tuple[str, str] | None:
    """Return (en_stem, suffix) if token looks like EN stem + Telugu suffix."""
    t = token.lower()
    if not t.isalpha() or len(t) < 6:
        return None
    for suf in TE_SUFFIXES_SORTED:
        stem = t[: -len(suf)] if t.endswith(suf) else ""
        if not stem or len(stem) < 4:
            continue
        if stem in EN_STEM_BLOCK:
            continue
        if len(stem) < 5 and stem not in GOOD_SHORT_STEMS:
            continue
        if zipf_en(stem) >= min_zipf:
            return stem, suf
    return None


def auto_tag_token(tok: str, te_lexicon: set[str]) -> str:
    low = tok.lower()
    if not re.search(r"[A-Za-z]", tok):
        return "other"
    mixed = intra_word_mixed(tok, ZIPF_EN)
    if mixed:
        return "mixed"
    in_te = low in te_lexicon
    en_ok = zipf_en(low) >= ZIPF_EN
    if en_ok and not in_te:
        return "en"
    if in_te:
        return "te"
    return "uncertain"


def skeleton_key(word: str) -> str:
    w = word.lower()
    w = re.sub(r"(.)\1+", r"\1", w)
    w = w.replace("aa", "a").replace("ee", "i").replace("oo", "u")
    w = w.replace("dh", "d").replace("th", "t")
    w = re.sub(r"(?<=[bcdfgjklmnpqrstvwxyz])h", "", w)
    if w.endswith("u") and len(w) > 2:
        w = w[:-1]
    return w


def download(url: str, dest: Path, timeout: int = 60) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "aaai-icfd-te-en-calibration/1.0"})
    with urlopen(req, timeout=timeout) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out)


def ensure_sources(skip_download: bool) -> dict[str, bool]:
    """Download missing raw files. Returns obtained flags."""
    obtained = {}
    files = {
        "cmtet_sentiment": [
            (
                "https://raw.githubusercontent.com/ksubbu199/cmtet-sentiment/master/codemix_sentiment_data.txt",
                RAW_ROOT / "cmtet_sentiment" / "codemix_sentiment_data.txt",
            ),
            (
                "https://raw.githubusercontent.com/ksubbu199/cmtet-sentiment/master/README.md",
                RAW_ROOT / "cmtet_sentiment" / "README.md",
            ),
        ],
        "cmtet_lid": [
            (
                "https://raw.githubusercontent.com/ksubbu199/cmtet-lid/master/Twitter%20Dataset/train_data.txt",
                RAW_ROOT / "cmtet_lid" / "Twitter Dataset" / "train_data.txt",
            ),
            (
                "https://raw.githubusercontent.com/ksubbu199/cmtet-lid/master/Twitter%20Dataset/test_data.txt",
                RAW_ROOT / "cmtet_lid" / "Twitter Dataset" / "test_data.txt",
            ),
            (
                "https://raw.githubusercontent.com/ksubbu199/cmtet-lid/master/Blog%20Dataset/train_data.txt",
                RAW_ROOT / "cmtet_lid" / "Blog Dataset" / "train_data.txt",
            ),
            (
                "https://raw.githubusercontent.com/ksubbu199/cmtet-lid/master/Blog%20Dataset/test_data.txt",
                RAW_ROOT / "cmtet_lid" / "Blog Dataset" / "test_data.txt",
            ),
            (
                "https://raw.githubusercontent.com/ksubbu199/cmtet-lid/master/README.md",
                RAW_ROOT / "cmtet_lid" / "README.md",
            ),
        ],
        "icon_2016": [
            (
                "https://raw.githubusercontent.com/LCS2-IIITD/HIT-ACL2021-Codemixed-Representation/main/data/telegu_POS/FB_TE_EN_FN.txt",
                RAW_ROOT / "icon_2016" / "FB_TE_EN_FN.txt",
            ),
            (
                "https://raw.githubusercontent.com/LCS2-IIITD/HIT-ACL2021-Codemixed-Representation/main/data/telegu_POS/TWT_TE_EN_FN.txt",
                RAW_ROOT / "icon_2016" / "TWT_TE_EN_FN.txt",
            ),
            (
                "https://raw.githubusercontent.com/LCS2-IIITD/HIT-ACL2021-Codemixed-Representation/main/data/telegu_POS/WA_TE_EN_FN.txt",
                RAW_ROOT / "icon_2016" / "WA_TE_EN_FN.txt",
            ),
        ],
        "chandu_cmqa": [
            (
                "https://raw.githubusercontent.com/khyathiraghavi/code_switched_QA/master/code_mixed_qa_train.json",
                RAW_ROOT / "chandu_cmqa" / "code_mixed_qa_train.json",
            ),
            (
                "https://raw.githubusercontent.com/khyathiraghavi/code_switched_QA/master/COPYING",
                RAW_ROOT / "chandu_cmqa" / "COPYING",
            ),
            (
                "https://raw.githubusercontent.com/khyathiraghavi/code_switched_QA/master/README",
                RAW_ROOT / "chandu_cmqa" / "README",
            ),
        ],
    }
    for sid, items in files.items():
        ok = True
        for url, dest in items:
            if dest.exists() and dest.stat().st_size > 0:
                continue
            if skip_download:
                ok = dest.exists() and dest.stat().st_size > 0
                continue
            try:
                print(f"download {sid}: {url}")
                download(url, dest)
            except Exception as exc:  # noqa: BLE001
                print(f"FAILED {sid} {url}: {exc}", file=sys.stderr)
                ok = False
        obtained[sid] = ok and all(d.exists() and d.stat().st_size > 0 for _, d in items)
    return obtained


def write_licenses() -> list[SourceInfo]:
    LIC_ROOT.mkdir(parents=True, exist_ok=True)
    infos = [
        SourceInfo(
            source_id="cmtet_sentiment",
            url="https://github.com/ksubbu199/cmtet-sentiment",
            version="git ae383ee (2021-08-18); paper RANLP 2021",
            size_note="19,868 sentences; ~3.8 MB text file",
            license="No LICENSE file on GitHub. Paper does not state a redistribution licence.",
            citation=(
                "Kusampudi, Siva Subrahamanyam Varma, Preetham Sathineni, and Radhika Mamidi. "
                "2021. Sentiment Analysis in Code-Mixed Telugu-English Text with Unsupervised "
                "Data Normalization. RANLP 2021. https://aclanthology.org/2021.ranlp-1.86/"
            ),
            redistribution="UNCLEAR - stats only, do not redistribute",
            obtained=True,
            notes="Human-annotated Twitter/YouTube comments with word-level EN/TE/NE/UNIV tags.",
        ),
        SourceInfo(
            source_id="cmtet_lid",
            url="https://github.com/ksubbu199/cmtet-lid",
            version="git f0db442 (2021-08-18); paper RANLP 2021",
            size_note="Twitter ~24,406 sents / Blog ~9,659 sents; word-per-line LID",
            license="No LICENSE file on GitHub. Paper says authors 'plan to make' the corpus open-sourced; terms unspecified.",
            citation=(
                "Kusampudi, Siva Subrahamanyam Varma, Anudeep Chaluvadi, and Radhika Mamidi. "
                "2021. Corpus Creation and Language Identification in Low-Resource Code-Mixed "
                "Telugu-English Text. RANLP 2021. https://aclanthology.org/2021.ranlp-1.85/"
            ),
            redistribution="UNCLEAR - stats only, do not redistribute",
            obtained=True,
            notes="Human-annotated Twitter and blog LID (TE/EN/NE/UNIV). May overlap CMTET sentiment Twitter portion.",
        ),
        SourceInfo(
            source_id="icon_2016",
            url="https://github.com/LCS2-IIITD/HIT-ACL2021-Codemixed-Representation/tree/main/data/telegu_POS",
            version="ICON 2016 POSCMISMT TE_EN files as redistributed in HIT-ACL2021 (FB 744, TWT 744, WA 494 sents)",
            size_note="1,982 sentences; ~29k tokens across FB/Twitter/WhatsApp",
            license="Original shared-task terms not posted with the files. HIT-ACL2021 repo has no dataset licence.",
            citation=(
                "Jamatia, Anupam, and Amitava Das. 2016. Task report: Tool contest on POS tagging "
                "for code-mixed Indian social media (Facebook, Twitter, and WhatsApp) text. ICON 2016. "
                "Original host http://www.amitavadas.com/Code-Mixing.html (now blocked). "
                "Also Gundapu & Mamidi PACLIC 2018 for ICON-2015 TE-EN LID counts."
            ),
            redistribution="UNCLEAR - stats only, do not redistribute",
            obtained=True,
            notes="Official amitavadas.com page redirected to a lander; used HIT-ACL2021 copy of TE_EN files.",
        ),
        SourceInfo(
            source_id="chandu_cmqa",
            url="https://github.com/khyathiraghavi/code_switched_QA",
            version="git 3489317; CALCS 2018 / ACL 2018",
            size_note="445 Telugu-labelled questions in released train JSON (paper reports 1,391 Tenglish)",
            license="CMU 2018 custom notice in COPYING: free for any purpose if copyright notice retained.",
            citation=(
                "Chandu, Khyathi, et al. 2018. Code-Mixed Question Answering Challenge: Crowd-sourcing "
                "Data and Techniques. CALCS 2018. https://aclanthology.org/W18-3204/"
            ),
            redistribution="Yes, with copyright notice (COPYING). Stats and examples OK under those terms.",
            obtained=True,
            notes="Human crowd-sourced Tenglish questions; no gold token LID. Auto-tagged.",
        ),
        SourceInfo(
            source_id="ud_tect",
            url="https://github.com/UniversalDependencies/UD_Telugu_English-TECT",
            version="UD v2.14; git 41c9c20",
            size_note="98 romanized sentences",
            license="CC BY-SA 4.0",
            citation=(
                "Vissamsetty, Anishka. UD Telugu_English-TECT. Universal Dependencies v2.14. "
                "https://github.com/UniversalDependencies/UD_Telugu_English-TECT"
            ),
            redistribution="Yes (CC BY-SA 4.0), but excluded from the profile: sentences were romanized "
            "and then altered to insert at least one code-switch (constructed mixing, not naturally written CS).",
            obtained=True,
            notes="EXCLUDED from utterances.jsonl and statistics.",
        ),
        SourceInfo(
            source_id="icon_official_amitavadas",
            url="http://www.amitavadas.com/Code-Mixing.html",
            version="n/a",
            size_note="n/a",
            license="n/a",
            citation="Jamatia & Das 2016 ICON tool contest.",
            redistribution="n/a",
            obtained=False,
            notes="Page redirected to a parking/lander; official zip not downloadable.",
        ),
    ]
    for info in infos:
        path = LIC_ROOT / f"{info.source_id}.md"
        path.write_text(
            "\n".join(
                [
                    f"# {info.source_id}",
                    "",
                    f"- **URL:** {info.url}",
                    f"- **Version:** {info.version}",
                    f"- **Size:** {info.size_note}",
                    f"- **License / stated terms:** {info.license}",
                    f"- **Citation:** {info.citation}",
                    f"- **Redistribution allowed?** {info.redistribution}",
                    f"- **Obtained:** {info.obtained}",
                    f"- **Notes:** {info.notes}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    return infos


def promote_gold_mixed(tokens: list[str], tags: list[str]) -> list[str]:
    out = []
    for tok, tag in zip(tokens, tags):
        if tag == "te" and intra_word_mixed(tok, ZIPF_MIXED_GOLD):
            out.append("mixed")
        else:
            out.append(tag)
    return out


def clean_aligned(tokens: list[str], tags: list[str]) -> tuple[list[str], list[str], str] | None:
    raw_text = " ".join(tokens)
    if telugu_unicode_ratio(raw_text) > TELUGU_UNICODE_MAX:
        return None
    out_t, out_g = [], []
    for tok, tag in zip(tokens, tags):
        cleaned = drop_or_clean_token(tok)
        if cleaned is None:
            continue
        out_t.append(cleaned)
        out_g.append(tag)
    if not out_t:
        return None
    out_g = promote_gold_mixed(out_t, out_g)
    text = " ".join(out_t)
    if telugu_unicode_ratio(text) > TELUGU_UNICODE_MAX:
        return None
    return out_t, out_g, text


def parse_word_tag_file(path: Path, source_id: str, prefix: str) -> list[Utterance]:
    utterances: list[Utterance] = []
    tokens: list[str] = []
    tags: list[str] = []
    sent_i = 0

    def flush() -> None:
        nonlocal sent_i, tokens, tags
        if not tokens:
            tokens, tags = [], []
            return
        mapped = [map_gold_tag(t) for t in tags]
        cleaned = clean_aligned(tokens, mapped)
        if cleaned:
            toks, gs, text = cleaned
            uid = f"{source_id}:{prefix}:{sent_i}"
            utterances.append(
                Utterance(
                    id=uid,
                    source_id=source_id,
                    text=text,
                    tokens=toks,
                    lang_tags=gs,
                    tag_origin="gold",
                    source_item_id=f"{prefix}:{sent_i}",
                )
            )
        sent_i += 1
        tokens, tags = [], []

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            flush()
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        tokens.append(parts[0])
        tags.append(parts[1])
    flush()
    return utterances


def parse_cmtet_sentiment(path: Path) -> list[Utterance]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    out: list[Utterance] = []
    i = 0
    sent_i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = SENT_LABEL_RE.match(line)
        if not m:
            i += 1
            continue
        sent = m.group(2).strip()
        tags_line = ""
        if i + 1 < len(lines):
            tags_line = lines[i + 1].strip()
        tokens = sent.split()
        raw_tags = tags_line.split()
        if len(tokens) != len(raw_tags) or not tokens:
            i += 1
            sent_i += 1
            continue
        mapped = [map_gold_tag(t) for t in raw_tags]
        cleaned = clean_aligned(tokens, mapped)
        if cleaned:
            toks, gs, text = cleaned
            uid = f"cmtet_sentiment:{sent_i}"
            out.append(
                Utterance(
                    id=uid,
                    source_id="cmtet_sentiment",
                    text=text,
                    tokens=toks,
                    lang_tags=gs,
                    tag_origin="gold",
                    source_item_id=str(sent_i),
                )
            )
        sent_i += 1
        i += 2
    return out


def parse_chandu(path: Path, te_lexicon: set[str]) -> list[Utterance]:
    data = json.loads(path.read_text(encoding="utf-8"))
    qs = data.get("questions", [])
    out: list[Utterance] = []
    for q in qs:
        lang = str(q.get("language", "")).lower()
        if lang not in {"telugu", "te", "tenglish", "te_en"}:
            continue
        query = str(q.get("query", "")).strip()
        if not query:
            continue
        if telugu_unicode_ratio(query) > TELUGU_UNICODE_MAX:
            continue
        raw_toks = WORD_RE.findall(query)
        toks, tags = [], []
        for tok in raw_toks:
            cleaned = drop_or_clean_token(tok)
            if cleaned is None:
                continue
            toks.append(cleaned)
            tags.append(auto_tag_token(cleaned, te_lexicon))
        if not toks:
            continue
        text = " ".join(toks)
        qid = str(q.get("id", len(out)))
        out.append(
            Utterance(
                id=f"chandu_cmqa:{qid}",
                source_id="chandu_cmqa",
                text=text,
                tokens=toks,
                lang_tags=tags,
                tag_origin="auto",
                source_item_id=qid,
            )
        )
    return out


def build_te_lexicon(gold: Iterable[Utterance]) -> set[str]:
    freq: Counter[str] = Counter()
    for u in gold:
        for tok, tag in zip(u.tokens, u.lang_tags):
            if tag == "te" and tok.isalpha():
                freq[tok.lower()] += 1
    return {w for w, c in freq.items() if c >= TE_LEX_MIN_FREQ}


def utterance_metrics(u: Utterance) -> dict:
    n_en = sum(1 for t in u.lang_tags if t == "en")
    n_te = sum(1 for t in u.lang_tags if t == "te")
    n_mixed = sum(1 for t in u.lang_tags if t == "mixed")
    n_unc = sum(1 for t in u.lang_tags if t == "uncertain")
    n_ne = sum(1 for t in u.lang_tags if t == "ne")
    n_other = sum(1 for t in u.lang_tags if t == "other")
    denom = n_en + n_te
    if denom == 0:
        eng_share = None
        cmi = None
        mono = True
    else:
        eng_share = n_en / denom
        cmi = 100.0 * (1.0 - max(n_en, n_te) / denom)
        mono = min(n_en, n_te) == 0
    switches = 0
    prev = None
    spans = []
    run = 0
    run_lang = None
    for tag in u.lang_tags:
        if tag in {"en", "te"}:
            if prev in {"en", "te"} and tag != prev:
                switches += 1
            if run_lang == tag:
                run += 1
            else:
                if run_lang in {"en", "te"} and run:
                    spans.append(run)
                run_lang = tag
                run = 1
            prev = tag
        else:
            if run_lang in {"en", "te"} and run:
                spans.append(run)
            run = 0
            run_lang = None
            prev = tag if tag in {"en", "te"} else None
    if run_lang in {"en", "te"} and run:
        spans.append(run)
    mean_span = float(np.mean(spans)) if spans else 0.0
    ntok = max(len(u.lang_tags), 1)
    return {
        "n_en": n_en,
        "n_te": n_te,
        "n_mixed": n_mixed,
        "n_uncertain": n_unc,
        "n_ne": n_ne,
        "n_other": n_other,
        "english_share": eng_share,
        "cmi": cmi,
        "monolingual": mono,
        "n_switches": switches,
        "mean_span": mean_span,
        "share_mixed": n_mixed / ntok,
        "share_uncertain": n_unc / ntok,
        "n_tokens": len(u.tokens),
    }


def quantiles(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {k: None for k in ("p10", "p25", "p50", "p75", "p90")}
    arr = np.array(xs, dtype=float)
    return {
        "p10": float(np.quantile(arr, 0.10)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
    }


def markers_hit(text: str, tokens: list[str]) -> list[str]:
    blob = " " + " ".join(t.lower() for t in tokens) + " "
    low = text.lower()
    hits = []
    for m in SERVICE_MARKERS:
        if m == "thank you":
            if "thank you" in low:
                hits.append(m)
            continue
        if f" {m} " in blob:
            hits.append(m)
    return hits


def contains_profanity(tokens: list[str]) -> bool:
    for t in tokens:
        if t.lower() in PROFANITY:
            return True
    return False


def select_service_candidates(rows: list[tuple[Utterance, dict]]) -> tuple[list[dict], dict]:
    scored = []
    for u, m in rows:
        if not (5 <= m["n_tokens"] <= 30):
            continue
        if m["cmi"] is None or m["cmi"] <= 0:
            continue
        if contains_profanity(u.tokens):
            continue
        hits = markers_hit(u.text, u.tokens)
        if not hits:
            continue
        lows = [t.lower() for t in u.tokens]
        movieish = sum(1 for t in lows if t in MOVIE_DOMAIN)
        trans = [h for h in hits if h in TRANSACTIONAL]
        request = [h for h in hits if h in REQUEST_FORMS]
        # "rs" alone is noisy unless rupees/digits sit nearby
        if "rs" in hits and "rupees" not in hits and not any(ch.isdigit() for ch in u.text):
            hits = [h for h in hits if h != "rs"]
            trans = [h for h in trans if h != "rs"]
        if not hits:
            continue
        genuine = bool(trans) and movieish == 0
        generic = (not trans) or movieish > 0
        score = (len(trans), 0 if movieish else 1, len(request), m["cmi"])
        scored.append((score, u, m, hits, genuine, generic, movieish))
    scored.sort(key=lambda x: x[0], reverse=True)

    selected = []
    pentagrams: set[tuple[str, ...]] = set()
    n_strong = 0
    n_generic = 0
    n_pool_genuine = sum(1 for s in scored if s[4])
    n_pool_generic = len(scored) - n_pool_genuine
    for _score, u, m, hits, is_genuine, _generic, _movieish in scored:
        lows = [t.lower() for t in u.tokens]
        grams = {tuple(lows[i : i + 5]) for i in range(len(lows) - 4)} if len(lows) >= 5 else set()
        if grams & pentagrams:
            continue
        pentagrams |= grams
        genuine = "service_like" if is_genuine else "generic_marker_only"
        if is_genuine:
            n_strong += 1
        else:
            n_generic += 1
        selected.append(
            {
                "cand_id": f"te_en_c{len(selected)+1:04d}",
                "source_id": u.source_id,
                "source_item_id": u.source_item_id or u.id,
                "text": u.text,
                "english_share": "" if m["english_share"] is None else f"{m['english_share']:.4f}",
                "cmi": "" if m["cmi"] is None else f"{m['cmi']:.4f}",
                "n_switches": m["n_switches"],
                "markers_hit": "|".join(hits),
                "license_status": "UNCLEAR - stats only, do not redistribute"
                if u.source_id != "chandu_cmqa"
                else "CMU COPYING - retain notice",
                "accept": "",
                "reject_reason": "",
                "reviewer_note": "",
                "_genuine": genuine,
                "_tokens": u.tokens,
                "_tags": u.lang_tags,
            }
        )
        if len(selected) >= SERVICE_MAX:
            break
    summary = {
        "n_selected": len(selected),
        "n_service_like": n_strong,
        "n_generic_marker_only": n_generic,
        "n_pool": len(scored),
        "n_pool_service_like": n_pool_genuine,
        "n_pool_generic": n_pool_generic,
    }
    return selected, summary


def corrupt_shuffle(tokens: list[str], rng: random.Random) -> str | None:
    if len(tokens) < 4:
        return None
    perm = tokens[:]
    rng.shuffle(perm)
    if perm == tokens:
        return None
    return " ".join(perm)


def corrupt_suffix(tokens: list[str], rng: random.Random) -> str | None:
    idxs = []
    for i, tok in enumerate(tokens):
        hit = intra_word_mixed(tok, ZIPF_EN)
        if hit:
            idxs.append((i, hit[1]))
        else:
            low = tok.lower()
            for suf in TE_SUFFIXES_SORTED:
                if low.endswith(suf) and len(low) > len(suf) + 2:
                    idxs.append((i, suf))
                    break
    if not idxs or len(tokens) < 2:
        return None
    i, suf = rng.choice(idxs)
    others = [j for j in range(len(tokens)) if j != i]
    j = rng.choice(others)
    new = tokens[:]
    donor = new[i]
    if donor.lower().endswith(suf):
        new[i] = donor[: -len(suf)] or donor
    new[j] = new[j] + suf
    if new == tokens:
        return None
    return " ".join(new)


def corrupt_hindi_verb(tokens: list[str], tags: list[str], rng: random.Random) -> str | None:
    cands = []
    for i, (tok, tag) in enumerate(zip(tokens, tags)):
        low = tok.lower()
        if tag == "te" or any(low.startswith(v) or low == v for v in LIGHT_VERBS):
            if any(low.startswith(v) for v in ("chey", "ches", "chest", "avvu", "ayyi", "pett", "ivv")) or low in LIGHT_VERBS:
                cands.append(i)
    if not cands:
        # fall back: any TE token with verb-like ending
        for i, (tok, tag) in enumerate(zip(tokens, tags)):
            if tag == "te" and tok.lower().endswith(("andi", "anu", "aru", "ali", "undi")):
                cands.append(i)
    if not cands:
        return None
    i = rng.choice(cands)
    new = tokens[:]
    new[i] = rng.choice(HINDI_VERBS)
    return " ".join(new)


def make_negatives(cands: list[dict], n: int = NEGATIVES_N) -> list[dict]:
    rng = random.Random(RNG_SEED)
    types = ["shuffle", "suffix_move", "hindi_verb"]
    out = []
    order = list(range(len(cands)))
    rng.shuffle(order)
    ti = 0
    used = set()
    attempts = 0
    while len(out) < n and attempts < n * 20 and order:
        attempts += 1
        idx = order[len(out) % len(order)]
        c = cands[idx]
        ctype = types[ti % 3]
        ti += 1
        toks = c["_tokens"]
        tags = c["_tags"]
        if ctype == "shuffle":
            text = corrupt_shuffle(toks, rng)
        elif ctype == "suffix_move":
            text = corrupt_suffix(toks, rng)
        else:
            text = corrupt_hindi_verb(toks, tags, rng)
        key = (c["cand_id"], ctype, text)
        if not text or key in used:
            continue
        used.add(key)
        out.append(
            {
                "neg_id": f"te_en_n{len(out)+1:04d}",
                "from_cand_id": c["cand_id"],
                "corruption_type": ctype,
                "text": text,
            }
        )
    return out


def slice_stats(items: list[tuple[Utterance, dict]]) -> dict:
    eng = [m["english_share"] for _, m in items if m["english_share"] is not None]
    cmi = [m["cmi"] for _, m in items if m["cmi"] is not None]
    switches = [m["n_switches"] for _, m in items]
    spans = [m["mean_span"] for _, m in items if m["mean_span"] > 0]
    n_mono = sum(1 for _, m in items if m["monolingual"])
    n_tok = sum(m["n_tokens"] for _, m in items)
    n_mixed = sum(m["n_mixed"] for _, m in items)
    n_unc = sum(m["n_uncertain"] for _, m in items)
    n_en = sum(m["n_en"] for _, m in items)
    n_te = sum(m["n_te"] for _, m in items)
    return {
        "n_utterances": len(items),
        "n_tokens": n_tok,
        "n_en_tokens": n_en,
        "n_te_tokens": n_te,
        "n_gold_tagged_tokens": sum(m["n_tokens"] for u, m in items if u.tag_origin == "gold"),
        "english_share_quantiles": quantiles(eng),
        "cmi_quantiles": quantiles(cmi),
        "monolingual_share": (n_mono / len(items)) if items else None,
        "switches_per_utterance_quantiles": quantiles([float(x) for x in switches]),
        "mean_same_language_span": float(np.mean(spans)) if spans else None,
        "share_mixed_tokens": (n_mixed / n_tok) if n_tok else None,
        "share_uncertain_tokens": (n_unc / n_tok) if n_tok else None,
    }


def top_n(counter: Counter, n: int) -> list[list]:
    return [[w, int(c)] for w, c in counter.most_common(n)]


def compute_profile(utterances: list[Utterance]) -> tuple[dict, list[dict], list[dict]]:
    measured = [(u, utterance_metrics(u)) for u in utterances]
    gold_items = [(u, m) for u, m in measured if u.tag_origin == "gold"]
    all_items = measured

    mixed_cand = Counter()
    for u, _ in gold_items:
        for tok, tag in zip(u.tokens, u.lang_tags):
            if tag not in {"te", "mixed"}:
                continue
            hit = intra_word_mixed(tok, ZIPF_MIXED_GOLD)
            if hit:
                mixed_cand[tok.lower()] += 1

    en_in_te_dom = Counter()
    te_fn = Counter()
    light = Counter()
    intra = Counter()
    switch_bi = Counter()
    address = Counter()
    clusters: dict[str, Counter] = defaultdict(Counter)

    def consume(u: Utterance, m: dict) -> None:
        n_en, n_te = m["n_en"], m["n_te"]
        te_dom = (n_en + n_te) > 0 and n_te > n_en
        lows = [t.lower() for t in u.tokens]
        for tok, tag in zip(u.tokens, u.lang_tags):
            low = tok.lower()
            if te_dom and tag == "en" and low.isalpha():
                en_in_te_dom[low] += 1
            if tag == "te" and (low in TE_FUNCTION_VERBS or any(low.startswith(v) for v in LIGHT_VERBS)):
                te_fn[low] += 1
            elif tag == "te" and low.isalpha() and low in TE_FUNCTION_VERBS:
                te_fn[low] += 1
            hit = intra_word_mixed(tok, ZIPF_EN)
            if hit or tag == "mixed":
                intra[low] += 1
            if tag in {"te", "mixed"} and low.isalpha():
                clusters[skeleton_key(low)][low] += 1
        for i, tok in enumerate(u.tokens):
            low = tok.lower()
            if low in LIGHT_VERBS and i > 0:
                prev = u.tokens[i - 1]
                prev_tag = u.lang_tags[i - 1]
                if prev_tag == "en":
                    light[f"{prev.lower()} {low}"] += 1
        for i in range(len(u.lang_tags) - 1):
            a, b = u.lang_tags[i], u.lang_tags[i + 1]
            if a in {"en", "te"} and b in {"en", "te"} and a != b:
                switch_bi[f"{lows[i]}|{lows[i+1]}"] += 1
        blob = " ".join(lows)
        for form in ADDRESS_FORMS:
            if form == "thank you":
                n = blob.count("thank you")
            else:
                n = len(re.findall(rf"\b{re.escape(form)}\b", blob))
            if n:
                address[form] += n

    for u, m in all_items:
        consume(u, m)

    # fill te function list with frequent gold TE tokens if seed is sparse
    gold_te = Counter()
    for u, _ in gold_items:
        for tok, tag in zip(u.tokens, u.lang_tags):
            if tag == "te" and tok.isalpha():
                gold_te[tok.lower()] += 1
    if len(te_fn) < 150:
        for w, c in gold_te.most_common(400):
            if w not in te_fn:
                te_fn[w] += c
            if len(te_fn) >= 200:
                break

    cluster_rows = []
    for key, spellings in clusters.items():
        if len(spellings) < 2:
            continue
        cluster_rows.append(
            {
                "skeleton": key,
                "n_spellings": len(spellings),
                "total": int(sum(spellings.values())),
                "spellings": [[s, int(c)] for s, c in spellings.most_common()],
            }
        )
    cluster_rows.sort(key=lambda r: (-r["total"], -r["n_spellings"], r["skeleton"]))

    cands, cand_summary = select_service_candidates(all_items)
    negs = make_negatives(cands)

    by_source = {}
    for u, m in all_items:
        by_source.setdefault(u.source_id, {"n_utterances": 0, "n_tokens": 0, "gold": 0, "auto": 0})
        by_source[u.source_id]["n_utterances"] += 1
        by_source[u.source_id]["n_tokens"] += m["n_tokens"]
        by_source[u.source_id][u.tag_origin] += 1

    def norm_text(s: str) -> str:
        return re.sub(r"\s+", " ", s.lower()).strip()

    sent_set = {norm_text(u.text) for u, _ in all_items if u.source_id == "cmtet_sentiment"}
    lid_set = {norm_text(u.text) for u, _ in all_items if u.source_id == "cmtet_lid"}
    overlap = {
        "cmtet_sentiment_and_cmtet_lid_exact_text": len(sent_set & lid_set),
        "cmtet_sentiment_unique_texts": len(sent_set),
        "cmtet_lid_unique_texts": len(lid_set),
    }

    stats = {
        "pair_id": PAIR_ID,
        "sources": by_source,
        "gold": slice_stats(gold_items),
        "all": slice_stats(all_items),
        "gold_mixed_candidates": {
            "definition": "gold-TE tokens whose prefix is English zipf>=4 plus a Telugu suffix",
            "n_types": len(mixed_cand),
            "n_tokens": int(sum(mixed_cand.values())),
            "top": top_n(mixed_cand, 200),
        },
        "top_300_english_in_telugu_dominant": top_n(en_in_te_dom, 300),
        "top_150_telugu_function_verbs": top_n(te_fn, 150),
        "mixed_en_light_verb": top_n(light, 200),
        "mixed_intra_word": top_n(intra, 200),
        "top_100_switch_bigrams": top_n(switch_bi, 100),
        "address_politeness": {k: int(address.get(k, 0)) for k in ADDRESS_FORMS},
        "romanization_clusters": cluster_rows[:300],
        "service_candidates": cand_summary,
        "unnatural_negatives": len(negs),
        "source_text_overlap": overlap,
    }
    # strip helper fields before csv write
    csv_cands = [{k: v for k, v in c.items() if not k.startswith("_")} for c in cands]
    return stats, csv_cands, negs


def write_report(stats: dict, infos: list[SourceInfo], examples: dict) -> None:
    g, a = stats["gold"], stats["all"]

    def qline(name: str, block: dict, key: str) -> str:
        q = block[key]
        if q["p50"] is None:
            return f"- {name}: undefined (no en/te tokens)"
        return (
            f"- {name}: p10={q['p10']:.3f} p25={q['p25']:.3f} p50={q['p50']:.3f} "
            f"p75={q['p75']:.3f} p90={q['p90']:.3f}"
        )

    lines = [
        "# Telugu–English (Romanized) calibration profile",
        "",
        f"Pair ID: `{PAIR_ID}`. Built by `scripts/calibration/mine_te_en.py` from human-written corpora only.",
        "",
        "## Sources obtained",
        "",
    ]
    for info in infos:
        status = "OBTAINED" if info.obtained else "FAILED"
        if info.source_id == "ud_tect":
            status = "OBTAINED but EXCLUDED (constructed CS)"
        lines += [
            f"### {info.source_id} — {status}",
            "",
            f"- URL: {info.url}",
            f"- Version: {info.version}",
            f"- Size: {info.size_note}",
            f"- License: {info.license}",
            f"- Citation: {info.citation}",
            f"- Redistribution: {info.redistribution}",
            f"- Notes: {info.notes}",
            "",
        ]
    lines += [
        "## Counts",
        "",
        f"- Gold utterances: {g['n_utterances']}",
        f"- All utterances: {a['n_utterances']}",
        f"- Gold-tagged tokens (after cleaning): {g['n_tokens']}",
        f"- All tokens: {a['n_tokens']}",
        f"- Gold EN / TE tokens: {g['n_en_tokens']} / {g['n_te_tokens']}",
        "",
        "## Mixing quantiles (gold)",
        "",
        qline("english_share", g, "english_share_quantiles"),
        qline("CMI", g, "cmi_quantiles"),
        f"- monolingual share: {g['monolingual_share']:.3f}" if g["monolingual_share"] is not None else "- monolingual share: n/a",
        qline("switches per utterance", g, "switches_per_utterance_quantiles"),
        f"- mean same-language span: {g['mean_same_language_span']:.3f}" if g["mean_same_language_span"] else "- mean same-language span: n/a",
        f"- share mixed tokens: {g['share_mixed_tokens']:.4f}" if g["share_mixed_tokens"] is not None else "",
        f"- share uncertain tokens: {g['share_uncertain_tokens']:.4f}" if g["share_uncertain_tokens"] is not None else "",
        "",
        "## Mixing quantiles (all)",
        "",
        qline("english_share", a, "english_share_quantiles"),
        qline("CMI", a, "cmi_quantiles"),
        f"- monolingual share: {a['monolingual_share']:.3f}" if a["monolingual_share"] is not None else "- monolingual share: n/a",
        qline("switches per utterance", a, "switches_per_utterance_quantiles"),
        f"- mean same-language span: {a['mean_same_language_span']:.3f}" if a["mean_same_language_span"] else "- mean same-language span: n/a",
        f"- share mixed tokens: {a['share_mixed_tokens']:.4f}" if a["share_mixed_tokens"] is not None else "",
        f"- share uncertain tokens: {a['share_uncertain_tokens']:.4f}" if a["share_uncertain_tokens"] is not None else "",
        "",
        "## Gold mixed-morphology candidates",
        "",
        f"- types: {stats['gold_mixed_candidates']['n_types']}; tokens: {stats['gold_mixed_candidates']['n_tokens']}",
        "- top 10: "
        + ", ".join(f"{w} ({c})" for w, c in stats["gold_mixed_candidates"]["top"][:10]),
        "",
        "## 10 service-like candidates",
        "",
    ]
    for t in examples.get("service", []):
        lines.append(f"- `{t}`")
    lines += [
        "",
        (
            f"Selected {stats['service_candidates']['n_selected']}: "
            f"{stats['service_candidates']['n_service_like']} genuinely service/transaction-like, "
            f"{stats['service_candidates']['n_generic_marker_only']} generic (politeness/request or movie-domain). "
            f"Marker pool {stats['service_candidates']['n_pool']} "
            f"({stats['service_candidates'].get('n_pool_service_like', '?')} service-like, "
            f"{stats['service_candidates'].get('n_pool_generic', '?')} generic)."
        ),
        "",
        "## 10 top EN + light-verb patterns",
        "",
    ]
    for w, c in stats["mixed_en_light_verb"][:10]:
        lines.append(f"- {w} ({c})")
    lines += ["", "## 10 top intra-word mixed forms", ""]
    for w, c in stats["mixed_intra_word"][:10]:
        lines.append(f"- {w} ({c})")
    lines += ["", "## 10 top romanization clusters", ""]
    for row in stats["romanization_clusters"][:10]:
        spell = ", ".join(f"{s}×{c}" for s, c in row["spellings"][:6])
        lines.append(f"- `{row['skeleton']}` ({row['n_spellings']} spellings, n={row['total']}): {spell}")
    lines += [
        "",
        "## Problems / caveats",
        "",
        f"- CMTET sentiment and LID exact-text overlap: {stats.get('source_text_overlap', {}).get('cmtet_sentiment_and_cmtet_lid_exact_text', '?')} utterances (kept as separate source_ids).",
        "- Official ICON download from amitavadas.com failed; TE_EN files came from the HIT-ACL2021 redistribution.",
        "- Chandu CMQA released train file has 445 Telugu questions, not the 1,391 Tenglish items reported in the paper.",
        "- UD Telugu_English-TECT was obtained but excluded: mixing was inserted after romanization.",
        "- Gold LID does not mark intra-word mixing; mixed candidates are reported separately and auto-tag assigns `mixed`.",
        "- Sources are social media, blogs, and trivia questions — not authentic service/phone-call transcripts.",
        "- CMTET/ICON have no clear redistribution licence: do not ship raw text outside this calibration folder.",
        "- Auto LID (wordfreq + lexicon) will mis-label borrowings and short tokens.",
        "",
    ]
    (PROC_ROOT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine te_en calibration profile")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    PROC_ROOT.mkdir(parents=True, exist_ok=True)
    obtained = ensure_sources(skip_download=args.skip_download)
    infos = write_licenses()
    for info in infos:
        if info.source_id in obtained:
            info.obtained = obtained[info.source_id] or info.obtained

    utterances: list[Utterance] = []
    if obtained.get("cmtet_sentiment"):
        print("parse cmtet_sentiment")
        utterances.extend(parse_cmtet_sentiment(RAW_ROOT / "cmtet_sentiment" / "codemix_sentiment_data.txt"))
    if obtained.get("cmtet_lid"):
        print("parse cmtet_lid")
        for split, pref in [
            ("Twitter Dataset/train_data.txt", "tw_tr"),
            ("Twitter Dataset/test_data.txt", "tw_te"),
            ("Blog Dataset/train_data.txt", "bl_tr"),
            ("Blog Dataset/test_data.txt", "bl_te"),
        ]:
            utterances.extend(parse_word_tag_file(RAW_ROOT / "cmtet_lid" / split, "cmtet_lid", pref))
    if obtained.get("icon_2016"):
        print("parse icon_2016")
        seen = set()
        for fn, pref in [("FB_TE_EN_FN.txt", "fb"), ("TWT_TE_EN_FN.txt", "twt"), ("WA_TE_EN_FN.txt", "wa")]:
            for u in parse_word_tag_file(RAW_ROOT / "icon_2016" / fn, "icon_2016", pref):
                key = (tuple(t.lower() for t in u.tokens), tuple(u.lang_tags))
                if key in seen:
                    continue
                seen.add(key)
                utterances.append(u)

    gold = [u for u in utterances if u.tag_origin == "gold"]
    te_lex = build_te_lexicon(gold)
    print(f"telugu lexicon size (freq>={TE_LEX_MIN_FREQ}): {len(te_lex)}")
    (PROC_ROOT / "te_lexicon.json").write_text(
        json.dumps(sorted(te_lex), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if obtained.get("chandu_cmqa"):
        print("parse chandu_cmqa (auto-tag)")
        utterances.extend(parse_chandu(RAW_ROOT / "chandu_cmqa" / "code_mixed_qa_train.json", te_lex))

    print(f"utterances: {len(utterances)}")
    stats, cands, negs = compute_profile(utterances)

    jsonl_path = PROC_ROOT / "utterances.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for u in utterances:
            f.write(
                json.dumps(
                    {
                        "id": u.id,
                        "source_id": u.source_id,
                        "text": u.text,
                        "tokens": u.tokens,
                        "lang_tags": u.lang_tags,
                        "tag_origin": u.tag_origin,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    (PROC_ROOT / "profile_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    cand_fields = [
        "cand_id",
        "source_id",
        "source_item_id",
        "text",
        "english_share",
        "cmi",
        "n_switches",
        "markers_hit",
        "license_status",
        "accept",
        "reject_reason",
        "reviewer_note",
    ]
    with (PROC_ROOT / "service_candidates.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cand_fields)
        w.writeheader()
        for row in cands:
            w.writerow(row)

    with (PROC_ROOT / "unnatural_negatives.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["neg_id", "from_cand_id", "corruption_type", "text"])
        w.writeheader()
        for row in negs:
            w.writerow(row)

    write_report(stats, infos, {"service": [c["text"] for c in cands[:10]]})
    print("wrote", PROC_ROOT)
    print("gold utt", stats["gold"]["n_utterances"], "all", stats["all"]["n_utterances"])
    print("service", stats["service_candidates"], "negs", len(negs))


if __name__ == "__main__":
    main()
