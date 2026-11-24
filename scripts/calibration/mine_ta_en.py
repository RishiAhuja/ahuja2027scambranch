#!/usr/bin/env python3
"""Mine an empirical Tamil–English (Romanized Tanglish) calibration profile.

Re-run from the repo root:

    .venv/bin/python scripts/calibration/mine_ta_en.py

Writes only under data/calibration/ and is safe to re-run (downloads skip
when raw files already exist). Auto language tags are unvalidated except
for the CoLI-Dravidian gold word tags.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import sys
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from urllib.request import urlretrieve

import pandas as pd
from wordfreq import zipf_frequency

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = REPO_ROOT / "data" / "calibration" / "raw" / "ta_en"
PROC_ROOT = REPO_ROOT / "data" / "calibration" / "processed" / "ta_en"
LIC_ROOT = REPO_ROOT / "data" / "calibration" / "licenses"

PAIR_ID = "ta_en"
SEED = 42
LEXICON_MIN_COUNT = 20
ZIPF_EN_WORD = 3.5
ZIPF_LEXICON = 2.5
ZIPF_PATTERN_TA = 3.0
TAMIL_UNICODE_MAX = 0.10

HF_TAMILMIX = "community-datasets/tamilmixsentiment"
HF_OFFENSE = "community-datasets/offenseval_dravidian"
HF_TANGLISH = "vishnu-n/Tanglish-Corpus-185k"
TANGLISH_SHA = "927f5b497b9f9e35ffc2a10b621a97f6e02ac9c2"
TAMILMIX_SHA = "429830c110d37fabc3eebf288da37721f50de10f"
OFFENSE_SHA = "4739cb65194d4157841510a7c46a6fc87622b764"

COLI_TAMIL_ZIP = (
    "https://codalab.lisn.upsaclay.fr/my/datasets/download/"
    "d9300110-8f3f-423c-9123-1a3ebca74e68"
)
CMQA_TGZ = (
    "https://github.com/khyathiraghavi/code_switched_QA/archive/"
    "refs/heads/master.tar.gz"
)

TAMIL_CHAR_RE = re.compile(r"[\u0B80-\u0BFF]")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
HANDLE_RE = re.compile(r"@\w+")
PHONE_RE = re.compile(r"(?:\+91[\s-]?)?[6-9]\d{9}\b|\b\d{10,}\b")
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U0001F900-\U0001F9FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "\U0000FE00-\U0000FE0F"
    "\U0001F680-\U0001F6FF"
    "]+"
)
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+")

TA_ENDINGS = (
    "unga",
    "nga",
    "ukku",
    "kku",
    "oda",
    "achu",
    "chu",
    "ttu",
    "ren",
    "ran",
    "rom",
    "dhu",
    "thu",
    "la",
    "ku",
    "um",
    "aa",
    "nu",
    "na",
    "ey",
    "eh",
)

MIXED_SUFFIXES = (
    "pannunga",
    "pannitu",
    "panna",
    "pannu",
    "ukku",
    "kku",
    "oda",
    "la",
    "ku",
    "ah",
    "nu",
    "um",
    "a",
)

SEED_TA = {
    "enna",
    "epdi",
    "eppo",
    "evlo",
    "illa",
    "illai",
    "irukku",
    "iruku",
    "pannu",
    "pannunga",
    "sollu",
    "sollunga",
    "vaanga",
    "ponga",
    "konjam",
    "romba",
    "naan",
    "nee",
    "neenga",
    "unga",
    "enga",
    "avan",
    "aval",
    "idhu",
    "adhu",
    "ku",
    "kku",
    "la",
    "oda",
    "ah",
    "eh",
    "da",
    "daa",
    "nu",
    "na",
    "um",
    "aa",
    "dhan",
    "than",
    "thaan",
    "ithu",
    "athu",
    "indha",
    "andha",
    "intha",
    "antha",
    "namma",
    "ungal",
    "ungalukku",
    "enakku",
    "unakku",
    "avanga",
    "seri",
    "sari",
    "aama",
    "aamaa",
    "venum",
    "vendaam",
    "ippo",
    "ippa",
    "naalaikku",
    "inniku",
    "innaiku",
    "semma",
    "nalla",
    "anna",
    "thala",
    "mattum",
    "vara",
    "mela",
    "innum",
    "varum",
    "panna",
    "solla",
    "poda",
    "pana",
    "enga",
    "unga",
    "avanga",
    "ivanga",
    "namma",
    "adhula",
    "idhula",
    "apram",
    "appuram",
    "ivan",
    "itha",
    "atha",
    "tha",
    "vera",
    "seriya",
    "kooda",
    "mattum",
}

LIGHT_VERBS = {
    "pannu",
    "pannunga",
    "panren",
    "panna",
    "pannitu",
    "pannala",
    "aagudhu",
    "aachu",
    "aagum",
    "podu",
    "pottu",
    "kodu",
    "kudunga",
}

TA_FUNCTION_VERBS = [
    "naan",
    "naa",
    "nee",
    "neenga",
    "neengal",
    "unga",
    "ungal",
    "ungalukku",
    "enga",
    "avan",
    "aval",
    "avanga",
    "idhu",
    "ithu",
    "adhu",
    "athu",
    "indha",
    "andha",
    "enna",
    "entha",
    "epdi",
    "eppadi",
    "eppo",
    "evlo",
    "evvalavu",
    "enge",
    "yenge",
    "yen",
    "en",
    "enakku",
    "unakku",
    "illa",
    "illai",
    "illama",
    "irukku",
    "iruku",
    "irukken",
    "iruken",
    "irukanga",
    "irukinga",
    "irundhuchu",
    "irundha",
    "vanthen",
    "vandhen",
    "vandhuten",
    "poganum",
    "vaanga",
    "vaa",
    "ponga",
    "po",
    "pannu",
    "pannunga",
    "panren",
    "panna",
    "pannitu",
    "pannala",
    "sollu",
    "sollunga",
    "solren",
    "kudu",
    "kudunga",
    "kodu",
    "konjam",
    "romba",
    "rombha",
    "semma",
    "nalla",
    "ketta",
    "seri",
    "sari",
    "aama",
    "aamaa",
    "venum",
    "vendaam",
    "koodathu",
    "mudiyathu",
    "mudiyum",
    "theriyum",
    "theriyala",
    "puriyala",
    "puriyum",
    "paaru",
    "paarunga",
    "kelu",
    "kelunga",
    "saapdu",
    "podu",
    "pottu",
    "aagudhu",
    "aachu",
    "aagum",
    "varum",
    "varala",
    "pochu",
    "poyiduchu",
    "naalaikku",
    "naalai",
    "inniku",
    "innaiku",
    "ippo",
    "ippa",
    "appo",
    "appuram",
    "apram",
    "munnadi",
    "kitta",
    "mela",
    "kela",
    "ulla",
    "veliya",
    "dhan",
    "than",
    "thaan",
    "da",
    "daa",
    "di",
    "ah",
    "eh",
    "nu",
    "na",
    "la",
    "ku",
    "kku",
    "oda",
    "kooda",
    "mattum",
    "innum",
    "namma",
    "namakku",
    "avaru",
    "ivar",
    "adhuva",
    "idhuva",
    "vendiyathu",
    "panrom",
    "panren",
    "panringa",
    "sollunga",
    "solringa",
    "irukanga",
    "vandhuruku",
    "pogattum",
    "kudunga",
    "anuppunga",
    "paarunga",
    "kelunga",
    "saapdungal",
    "thoongunga",
    "varunga",
    "poitu",
    "vandhutu",
    "pannitu",
    "mudinchu",
    "aagala",
    "pannala",
    "sollala",
    "varala",
    "pogala",
    "kuduthu",
    "eduthu",
    "vachu",
    "pathi",
    "kitte",
    "odane",
    "aprama",
    "pinna",
    "munnadiye",
    "romba",
    "konjam",
    "neraya",
    "neraiya",
    "evalo",
    "evlo",
    "eppadi",
    "epdi",
    "yenna",
    "enna",
]

ADDRESS_FORMS = [
    "neenga",
    "nee",
    "unga",
    "ungal",
    "sir",
    "madam",
    "anna",
    "akka",
    "thambi",
    "bro",
    "please",
    "plz",
    "pls",
    "nandri",
    "thanks",
    "thank you",
    "vanakkam",
    "hello",
    "seri",
    "sari",
    "okay",
    "aama",
    "illa",
]

TRANSACTION_SERVICE = {
    "payment",
    "panam",
    "rs",
    "rupees",
    "rupee",
    "account",
    "order",
    "delivery",
    "refund",
    "bank",
    "card",
    "otp",
    "book",
    "cancel",
    "return",
    "confirm",
    "service",
}
SOFT_SERVICE = {
    "problem",
    "issue",
    "help",
    "call",
    "message",
    "number",
}
STRONG_SERVICE = TRANSACTION_SERVICE | SOFT_SERVICE
WEAK_SERVICE = {
    "neenga",
    "unga",
    "sollunga",
    "pannunga",
    "anuppunga",
    "kudunga",
    "konjam",
    "please",
    "plz",
    "pls",
    "sir",
    "madam",
    "eppo",
    "evlo",
    "epdi",
    "time",
    "naalaikku",
    "ippo",
}
ALL_SERVICE_MARKERS = sorted(STRONG_SERVICE | WEAK_SERVICE)

PROFANITY = {
    "fuck",
    "fucking",
    "shit",
    "bitch",
    "asshole",
    "dick",
    "pussy",
    "whore",
    "slut",
    "bastard",
    "motherfucker",
    "otha",
    "othe",
    "punda",
    "pundai",
    "pundaiya",
    "sunni",
    "soothu",
    "thevdiya",
    "thevidiya",
    "kenna",
    "myru",
    "mairu",
    "maire",
    "mayiru",
    "mayir",
    "potta",
    "pottachi",
    "kunji",
    "oombu",
    "wombu",
}

HINDI_VERB_MAP = {
    "pannu": "karo",
    "pannunga": "kijiye",
    "panren": "karta",
    "panna": "karna",
    "pannitu": "karke",
    "pannala": "nahi_kiya",
    "sollu": "bolo",
    "sollunga": "boliye",
    "solren": "bolta",
    "kudu": "do",
    "kudunga": "dijiye",
    "kodu": "do",
    "podu": "daalo",
    "pottu": "daalke",
    "vaanga": "aaiye",
    "ponga": "jaiye",
}

GOLD_TAG_MAP = {
    "tm": "ta",
    "ta": "ta",
    "en": "en",
    "tmen": "mixed",
    "mixed": "mixed",
    "name": "ne",
    "location": "ne",
    "loc": "ne",
    "place": "ne",
    "other": "other",
    "sym": "other",
    "number": "other",
}

LICENSE_STATUS = {
    "tamilmixsentiment": "UNCLEAR - stats only, do not redistribute",
    "offenseval_dravidian": "cc-by-4.0",
    "tanglish_corpus_185k": "cc-by-4.0",
    "coli_dravidian_ta": "UNCLEAR - stats only, do not redistribute",
    "cmqa_tamlish": "cmu-research-notice-required",
}

_ZIPF: dict[str, float] = {}


def zipf_en(tok: str) -> float:
    key = tok.lower()
    if key not in _ZIPF:
        try:
            _ZIPF[key] = float(zipf_frequency(key, "en"))
        except Exception:
            _ZIPF[key] = 0.0
    return _ZIPF[key]


def tamil_unicode_share(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for ch in text if TAMIL_CHAR_RE.match(ch)) / len(text)


def clean_text(text: str) -> str:
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = HANDLE_RE.sub(" ", text)
    text = PHONE_RE.sub(" ", text)
    text = EMOJI_RE.sub(" ", text)
    text = text.replace("#", " ")
    text = re.sub(r"[^\w\s']+", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def matches_ta_ending(tok: str) -> bool:
    low = tok.lower()
    if len(low) < 4:
        return False
    return any(low.endswith(suf) and len(low) - len(suf) >= 2 for suf in TA_ENDINGS)


def mixed_split(tok: str, lexicon: set[str] | None = None) -> tuple[str, str] | None:
    low = tok.lower()
    if lexicon and low in lexicon:
        return None
    for suf in MIXED_SUFFIXES:
        min_stem = 5 if suf == "a" else (4 if len(suf) <= 2 else 3)
        if len(low) < len(suf) + min_stem:
            continue
        if not low.endswith(suf):
            continue
        stem = low[: -len(suf)]
        if lexicon and stem in lexicon:
            continue
        if zipf_en(stem) >= ZIPF_EN_WORD:
            return stem, suf
    return None


def build_lexicon(token_counts: Counter[str]) -> set[str]:
    lex = set(SEED_TA)
    for tok, n in token_counts.items():
        if n < LEXICON_MIN_COUNT or not tok.isalpha() or len(tok) < 2:
            continue
        if zipf_en(tok) < ZIPF_LEXICON:
            lex.add(tok)
        elif matches_ta_ending(tok) and zipf_en(tok) < ZIPF_PATTERN_TA:
            lex.add(tok)
    return lex


def tag_token(tok: str, lexicon: set[str]) -> str:
    if tok.isdigit():
        return "other"
    low = tok.lower()
    if not re.fullmatch(r"[a-z]+(?:'[a-z]+)?", low):
        return "other"
    if low in lexicon:
        return "ta"
    z = zipf_en(low)
    if z >= ZIPF_EN_WORD:
        return "en"
    if mixed_split(low, lexicon):
        return "mixed"
    if matches_ta_ending(low) and z < ZIPF_PATTERN_TA:
        return "ta"
    return "uncertain"


def skeleton_key(tok: str) -> str:
    s = tok.lower()
    s = re.sub(r"(.)\1+", r"\1", s)
    s = s.replace("aa", "a").replace("ee", "i").replace("oo", "u")
    s = s.replace("zh", "l").replace("dh", "d").replace("th", "t")
    s = re.sub(r"(?<=[bcdfgjklmnpqrstvwxyz])h", "", s)
    return s


def utterance_metrics(tags: list[str]) -> dict[str, Any]:
    n_en = sum(1 for t in tags if t == "en")
    n_ta = sum(1 for t in tags if t == "ta")
    both = n_en + n_ta
    english_share = (n_en / both) if both else None
    cmi = (100.0 * (1.0 - (max(n_en, n_ta) / both))) if both else None
    switches = 0
    for a, b in zip(tags, tags[1:]):
        if {a, b} == {"en", "ta"}:
            switches += 1
    spans: list[int] = []
    i = 0
    while i < len(tags):
        if tags[i] in {"en", "ta"}:
            j = i + 1
            while j < len(tags) and tags[j] == tags[i]:
                j += 1
            spans.append(j - i)
            i = j
        else:
            i += 1
    mean_span = (sum(spans) / len(spans)) if spans else None
    return {
        "n_en": n_en,
        "n_ta": n_ta,
        "both": both,
        "english_share": english_share,
        "cmi": cmi,
        "n_switches": switches,
        "mean_span": mean_span,
        "n_mixed": sum(1 for t in tags if t == "mixed"),
        "n_uncertain": sum(1 for t in tags if t == "uncertain"),
        "n_tokens": len(tags),
    }


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {k: None for k in ("p10", "p25", "p50", "p75", "p90")}
    s = sorted(values)
    def q(p: float) -> float:
        if len(s) == 1:
            return s[0]
        idx = (len(s) - 1) * p
        lo = int(idx)
        hi = min(lo + 1, len(s) - 1)
        frac = idx - lo
        return s[lo] * (1 - frac) + s[hi] * frac
    return {k: round(q(p), 4) for k, p in (("p10", 0.10), ("p25", 0.25), ("p50", 0.50), ("p75", 0.75), ("p90", 0.90))}


def snapshot_hf(repo_id: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id, repo_type="dataset", local_dir=str(dest))


def download_sources(skip_download: bool) -> dict[str, Any]:
    log: dict[str, Any] = {"obtained": [], "failed": []}

    def mark(ok: bool, source_id: str, detail: str) -> None:
        bucket = log["obtained"] if ok else log["failed"]
        bucket.append({"source_id": source_id, "detail": detail})
        print(("OK  " if ok else "FAIL"), source_id, detail)

    if skip_download:
        present = {
            "tamilmixsentiment": (
                bool(list((RAW_ROOT / "tamilmixsentiment").glob("data/*.parquet"))),
                f"{HF_TAMILMIX} sha={TAMILMIX_SHA} (15,744 comments)",
            ),
            "offenseval_dravidian": (
                (RAW_ROOT / "offenseval_dravidian" / "tamil" / "train-00000-of-00001.parquet").exists(),
                f"{HF_OFFENSE} sha={OFFENSE_SHA} (39,527 Tamil comments)",
            ),
            "tanglish_corpus_185k": (
                (RAW_ROOT / "tanglish_corpus_185k" / "tanglish_corpus.jsonl").exists(),
                f"{HF_TANGLISH} sha={TANGLISH_SHA} (185,973 sentences)",
            ),
            "coli_dravidian_ta": (
                (RAW_ROOT / "coli_dravidian_ta" / "tamil_train.csv").exists(),
                "CodaLab 19357 Tamil train/val gold word tags",
            ),
            "cmqa_tamlish": (
                (RAW_ROOT / "cmqa_tamlish" / "code_mixed_qa_train.json").exists(),
                "github.com/khyathiraghavi/code_switched_QA (Tamlish questions)",
            ),
        }
        for sid, (ok, detail) in present.items():
            mark(ok, sid, detail if ok else "missing locally")
        log["failed"].extend(
            [
                {
                    "source_id": "icon2015_ta_en",
                    "detail": (
                        "Retrieved ICON 2015 zip from Wayback "
                        "(amitavadas.com/ICON2015/CM_POS_tagged_corpus_training_testing.zip) "
                        "but TA_EN_TRAIN.txt is Telugu–English (nenu/oka/undi), not Tamil. "
                        "Not used in the ta_en profile."
                    ),
                },
                {
                    "source_id": "hf_dravidian_codemix",
                    "detail": (
                        "HF id 'dravidian_codemix' / unprefixed 'offenseval_dravidian' "
                        "was not available; used community-datasets/offenseval_dravidian."
                    ),
                },
                {
                    "source_id": "tngarg_codemix_tamil_english",
                    "detail": (
                        "Inspected HF Tngarg/Codemix_tamil_english (license: other); "
                        "movie-comment CSV overlapping tamilmixsentiment. Skipped."
                    ),
                },
                {
                    "source_id": "tamiltech_qa",
                    "detail": "Skipped: dataset card states 8% GPT-4o-mini synthesis.",
                },
            ]
        )
        return log

    try:
        dest = RAW_ROOT / "tamilmixsentiment"
        if not list(dest.glob("data/*.parquet")):
            snapshot_hf(HF_TAMILMIX, dest)
        mark(True, "tamilmixsentiment", f"{HF_TAMILMIX} sha={TAMILMIX_SHA}")
    except Exception as exc:
        mark(False, "tamilmixsentiment", str(exc))

    try:
        dest = RAW_ROOT / "offenseval_dravidian"
        if not (dest / "tamil" / "train-00000-of-00001.parquet").exists():
            snapshot_hf(HF_OFFENSE, dest)
        mark(True, "offenseval_dravidian", f"{HF_OFFENSE} sha={OFFENSE_SHA}")
    except Exception as exc:
        mark(False, "offenseval_dravidian", str(exc))

    try:
        dest = RAW_ROOT / "tanglish_corpus_185k"
        if not (dest / "tanglish_corpus.jsonl").exists():
            snapshot_hf(HF_TANGLISH, dest)
        mark(True, "tanglish_corpus_185k", f"{HF_TANGLISH} sha={TANGLISH_SHA}")
    except Exception as exc:
        mark(False, "tanglish_corpus_185k", str(exc))

    try:
        dest = RAW_ROOT / "coli_dravidian_ta"
        dest.mkdir(parents=True, exist_ok=True)
        zip_path = dest / "tamil_train_val.zip"
        if not (dest / "tamil_train.csv").exists():
            urlretrieve(COLI_TAMIL_ZIP, zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                for name in zf.namelist():
                    if name.endswith(".csv") and not Path(name).name.startswith("._"):
                        with zf.open(name) as src, open(dest / Path(name).name, "wb") as out:
                            out.write(src.read())
        mark(True, "coli_dravidian_ta", "CodaLab 19357 Tamil train/val gold word tags")
    except Exception as exc:
        mark(False, "coli_dravidian_ta", str(exc))

    try:
        dest = RAW_ROOT / "cmqa_tamlish"
        dest.mkdir(parents=True, exist_ok=True)
        if not (dest / "code_mixed_qa_train.json").exists():
            tgz = dest / "repo.tar.gz"
            urlretrieve(CMQA_TGZ, tgz)
            with tarfile.open(tgz) as tf:
                tf.extractall(dest / "_extract")
            inner = next((dest / "_extract").iterdir())
            for p in inner.iterdir():
                shutil.copy2(p, dest / p.name)
            shutil.rmtree(dest / "_extract", ignore_errors=True)
            tgz.unlink(missing_ok=True)
        mark(True, "cmqa_tamlish", "github.com/khyathiraghavi/code_switched_QA")
    except Exception as exc:
        mark(False, "cmqa_tamlish", str(exc))

    log["failed"].extend(
        [
            {
                "source_id": "icon2015_ta_en",
                "detail": (
                    "Retrieved ICON 2015 zip from Wayback "
                    "(amitavadas.com/ICON2015/CM_POS_tagged_corpus_training_testing.zip) "
                    "but TA_EN_TRAIN.txt is Telugu–English (nenu/oka/undi), not Tamil. "
                    "Not used in the ta_en profile."
                ),
            },
            {
                "source_id": "hf_dravidian_codemix",
                "detail": (
                    "HF id 'dravidian_codemix' / unprefixed 'offenseval_dravidian' "
                    "was not available; used community-datasets/offenseval_dravidian."
                ),
            },
            {
                "source_id": "tngarg_codemix_tamil_english",
                "detail": (
                    "Inspected HF Tngarg/Codemix_tamil_english (license: other); "
                    "movie-comment CSV overlapping tamilmixsentiment. Skipped."
                ),
            },
            {
                "source_id": "tamiltech_qa",
                "detail": "Skipped: dataset card states 8% GPT-4o-mini synthesis.",
            },
        ]
    )
    return log


def load_raw_records() -> tuple[list[dict[str, Any]], set[str]]:
    records: list[dict[str, Any]] = []
    offensive_norm: set[str] = set()

    dest = RAW_ROOT / "tamilmixsentiment"
    for split in ("train", "validation", "test"):
        path = dest / "data" / f"{split}-00000-of-00001.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        for i, row in df.iterrows():
            records.append(
                {
                    "source_id": "tamilmixsentiment",
                    "source_item_id": f"{split}_{i}",
                    "text": str(row.get("text") or ""),
                    "gold_tokens": None,
                    "gold_tags": None,
                    "offensive": False,
                }
            )

    dest = RAW_ROOT / "offenseval_dravidian"
    label_names = {
        0: "Not_offensive",
        1: "Offensive_Untargetede",
        2: "Offensive_Targeted_Insult_Individual",
        3: "Offensive_Targeted_Insult_Group",
        4: "Offensive_Targeted_Insult_Other",
        5: "not-Tamil",
    }
    for split in ("train", "validation"):
        path = dest / "tamil" / f"{split}-00000-of-00001.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        for i, row in df.iterrows():
            text = str(row.get("text") or "")
            lab = row.get("label")
            try:
                lab_i = int(lab)
            except Exception:
                lab_i = 0
            is_off = lab_i in {1, 2, 3, 4}
            if is_off:
                offensive_norm.add(re.sub(r"\s+", " ", text.lower()).strip())
            records.append(
                {
                    "source_id": "offenseval_dravidian",
                    "source_item_id": f"{split}_{i}",
                    "text": text,
                    "gold_tokens": None,
                    "gold_tags": None,
                    "offensive": is_off,
                    "offense_label": label_names.get(lab_i, str(lab)),
                }
            )

    dest = RAW_ROOT / "tanglish_corpus_185k" / "tanglish_corpus.jsonl"
    if dest.exists():
        with dest.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if not line.strip():
                    continue
                rec = json.loads(line)
                records.append(
                    {
                        "source_id": "tanglish_corpus_185k",
                        "source_item_id": f"{i}_{rec.get('source', '')}",
                        "text": str(rec.get("text") or ""),
                        "gold_tokens": None,
                        "gold_tags": None,
                        "offensive": False,
                        "native_source": rec.get("source"),
                    }
                )

    dest = RAW_ROOT / "coli_dravidian_ta"
    for split, fname in (("train", "tamil_train.csv"), ("val", "tamil_val.csv")):
        path = dest / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        words = ["" if not pd.notna(w) else str(w) for w in df["Word"]]
        tags = ["" if not pd.notna(t) else str(t).strip().lower() for t in df["Tag"]]
        buf_w: list[str] = []
        buf_t: list[str] = []
        sent_i = 0

        def flush() -> None:
            nonlocal sent_i, buf_w, buf_t
            if not buf_w:
                return
            records.append(
                {
                    "source_id": "coli_dravidian_ta",
                    "source_item_id": f"{split}_{sent_i}",
                    "text": " ".join(buf_w),
                    "gold_tokens": list(buf_w),
                    "gold_tags": list(buf_t),
                    "offensive": False,
                }
            )
            sent_i += 1
            buf_w, buf_t = [], []

        for w, t in zip(words, tags):
            if not w or w in {"*", "nan"} or (t == "sym" and w in {"*", ".", "•"}):
                flush()
                continue
            mapped = GOLD_TAG_MAP.get(t, "uncertain" if not t else "other")
            buf_w.append(w)
            buf_t.append(mapped)
        flush()

    dest = RAW_ROOT / "cmqa_tamlish"
    for fname in (
        "code_mixed_qa_train.json",
        "code_mixed_qa_test_batch1.json",
        "code_mixed_qa_test_batch2.json",
        "code_mixed_qa_test_batch3.json",
    ):
        path = dest / fname
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for q in data.get("questions", []):
            if not str(q.get("language", "")).lower().startswith("tam"):
                continue
            records.append(
                {
                    "source_id": "cmqa_tamlish",
                    "source_item_id": f"{fname}_{q.get('id')}",
                    "text": str(q.get("query") or ""),
                    "gold_tokens": None,
                    "gold_tags": None,
                    "offensive": False,
                }
            )
    return records, offensive_norm


def process_records(
    raw: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[str], Counter[str], int, int]:
    token_counts: Counter[str] = Counter()
    prepared: list[dict[str, Any]] = []
    dropped_script = 0
    dropped_empty = 0

    for rec in raw:
        text = rec["text"]
        if tamil_unicode_share(text) > TAMIL_UNICODE_MAX:
            dropped_script += 1
            continue
        if rec["gold_tokens"]:
            tokens = []
            tags = []
            for w, t in zip(rec["gold_tokens"], rec["gold_tags"]):
                if URL_RE.search(w) or HANDLE_RE.search(w) or EMAIL_RE.search(w):
                    continue
                w2 = EMOJI_RE.sub("", w).replace("#", "")
                w2 = re.sub(r"[^\w']+", "", w2)
                if not w2:
                    continue
                tokens.append(w2)
                tags.append(t)
            cleaned = " ".join(tokens)
        else:
            cleaned = clean_text(text)
            tokens = tokenize(cleaned)
        if not tokens:
            dropped_empty += 1
            continue
        for tok in tokens:
            token_counts[tok.lower()] += 1
        prepared.append(
            {
                **rec,
                "cleaned": cleaned,
                "tokens": tokens,
                "preset_tags": rec["gold_tags"] if rec["gold_tokens"] else None,
            }
        )

    lexicon = build_lexicon(token_counts)
    utterances: list[dict[str, Any]] = []
    for rec in prepared:
        if rec["preset_tags"] and len(rec["preset_tags"]) == len(rec["tokens"]):
            tags = rec["preset_tags"]
            origin = "gold"
        else:
            tags = [tag_token(t, lexicon) for t in rec["tokens"]]
            origin = "auto"
        uid = f"{rec['source_id']}:{rec['source_item_id']}"
        utterances.append(
            {
                "id": uid,
                "source_id": rec["source_id"],
                "source_item_id": rec["source_item_id"],
                "text": rec["cleaned"],
                "tokens": rec["tokens"],
                "lang_tags": tags,
                "tag_origin": origin,
                "offensive": bool(rec.get("offensive")),
                "native_source": rec.get("native_source"),
            }
        )
    return utterances, lexicon, token_counts, dropped_script, dropped_empty


def unique_for_stats(utterances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rank = {
        "coli_dravidian_ta": 0,
        "offenseval_dravidian": 1,
        "tamilmixsentiment": 2,
        "cmqa_tamlish": 3,
        "tanglish_corpus_185k": 4,
    }
    best: dict[str, dict[str, Any]] = {}
    for u in utterances:
        key = re.sub(r"\s+", " ", u["text"].lower()).strip()
        prev = best.get(key)
        if prev is None:
            best[key] = u
            continue
        pr = (0 if prev["tag_origin"] == "gold" else 1, rank.get(prev["source_id"], 9))
        cr = (0 if u["tag_origin"] == "gold" else 1, rank.get(u["source_id"], 9))
        if cr < pr:
            best[key] = u
    return list(best.values())


def is_profane(tokens: list[str], text: str) -> bool:
    lows = {t.lower() for t in tokens}
    if lows & PROFANITY:
        return True
    blob = f" {text.lower()} "
    return any(f" {w} " in blob for w in PROFANITY)


def markers_in(tokens: list[str], text: str) -> list[str]:
    lows = [t.lower() for t in tokens]
    blob = f" {text.lower()} "
    hit = []
    for m in ALL_SERVICE_MARKERS:
        if " " in m:
            if f" {m} " in blob:
                hit.append(m)
        elif m in lows:
            hit.append(m)
    if "thank you" in blob and "thank you" not in hit:
        pass
    return hit


def select_service_candidates(
    utterances: list[dict[str, Any]],
    offensive_norm: set[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    scored: list[tuple[float, dict[str, Any]]] = []
    n_elig = 0
    for u in utterances:
        mets = utterance_metrics(u["lang_tags"])
        ntok = len(u["tokens"])
        if ntok < 5 or ntok > 30:
            continue
        if not mets["cmi"] or mets["cmi"] <= 0:
            continue
        hits = markers_in(u["tokens"], u["text"])
        if not hits:
            continue
        norm = re.sub(r"\s+", " ", u["text"].lower()).strip()
        if u["offensive"] or norm in offensive_norm or is_profane(u["tokens"], u["text"]):
            continue
        n_elig += 1
        n_txn = sum(1 for h in hits if h in TRANSACTION_SERVICE)
        n_soft = sum(1 for h in hits if h in SOFT_SERVICE)
        n_strong = n_txn + n_soft
        n_weak = sum(1 for h in hits if h in WEAK_SERVICE)
        score = 20 * n_txn + 6 * n_soft + n_weak + 0.05 * mets["cmi"] + 0.02 * mets["n_switches"]
        if u["source_id"] == "cmqa_tamlish":
            score += 0.5
        row = {
            "source_id": u["source_id"],
            "source_item_id": u["source_item_id"],
            "text": u["text"],
            "tokens": u["tokens"],
            "english_share": mets["english_share"],
            "cmi": mets["cmi"],
            "n_switches": mets["n_switches"],
            "markers_hit": hits,
            "n_strong": n_strong,
            "n_transaction": n_txn,
            "license_status": LICENSE_STATUS[u["source_id"]],
        }
        scored.append((score, row))
    scored.sort(key=lambda x: (-x[0], x[1]["text"]))
    chosen: list[dict[str, Any]] = []
    seen_5: set[tuple[str, ...]] = set()
    for _, row in scored:
        grams = [t.lower() for t in row["tokens"]]
        shingles = [tuple(grams[i : i + 5]) for i in range(len(grams) - 4)]
        if any(s in seen_5 for s in shingles):
            continue
        for s in shingles:
            seen_5.add(s)
        chosen.append(row)
        if len(chosen) >= 400:
            break
    info = {
        "eligible_before_diversity": n_elig,
        "selected": len(chosen),
        "genuinely_service_like": sum(1 for r in chosen if r["n_transaction"] >= 1),
        "soft_help_call_only": sum(
            1 for r in chosen if r["n_transaction"] == 0 and r["n_strong"] >= 1
        ),
        "generic_address_only": sum(1 for r in chosen if r["n_strong"] == 0),
    }
    return chosen, info


def corrupt_candidates(cands: list[dict[str, Any]], n: int = 100) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    if not cands:
        return []
    types = ["shuffle_word_order", "move_tamil_suffix", "hindi_verb_swap"]
    out = []
    pool = list(cands)
    rng.shuffle(pool)
    i = 0
    while len(out) < n:
        src = pool[i % len(pool)]
        ctype = types[len(out) % 3]
        toks = list(src["tokens"])
        if ctype == "shuffle_word_order":
            if len(toks) < 3:
                i += 1
                continue
            shuf = toks[:]
            rng.shuffle(shuf)
            if shuf == toks:
                shuf = toks[1:] + toks[:1]
            text = " ".join(shuf)
        elif ctype == "move_tamil_suffix":
            moved = False
            new = toks[:]
            for j, tok in enumerate(toks):
                split = mixed_split(tok) or (
                    (tok.lower()[: max(3, len(tok) - 2)], tok.lower()[-2:])
                    if matches_ta_ending(tok)
                    else None
                )
                if not split:
                    continue
                stem, suf = split
                dest = (j + 1 + rng.randrange(max(1, len(toks) - 1))) % len(toks)
                if dest == j:
                    continue
                new[j] = stem
                new[dest] = new[dest] + suf
                moved = True
                break
            if not moved:
                # attach a Tamil locative to the first English-looking token
                if len(new) >= 2:
                    new[0] = new[0] + "la"
                    new[-1] = re.sub(r"(la|ku|oda)$", "", new[-1], flags=re.I) or new[-1]
                else:
                    i += 1
                    continue
            text = " ".join(new)
        else:
            new = []
            swapped = False
            for tok in toks:
                low = tok.lower()
                if low in HINDI_VERB_MAP:
                    new.append(HINDI_VERB_MAP[low])
                    swapped = True
                else:
                    new.append(tok)
            if not swapped:
                # force a cross-language copy even if no Tamil light verb
                new.append("karo")
            text = " ".join(new)
        out.append(
            {
                "neg_id": f"ta_en_n{len(out)+1:03d}",
                "from_cand_id": f"ta_en_c{(i % len(pool))+1:03d}",
                "corruption_type": ctype,
                "text": text,
            }
        )
        i += 1
        if i > n * 8:
            break
    # fix from_cand_id to the actual selected ids after numbering
    id_of = {id(c): f"ta_en_c{k+1:03d}" for k, c in enumerate(cands)}
    # remap using pool order is messy; recompute from stored source texts
    text_to_id = {c["text"]: f"ta_en_c{k+1:03d}" for k, c in enumerate(cands)}
    # we stored from_cand_id by pool index; rebuild cleanly
    out2 = []
    rng2 = random.Random(SEED)
    pool2 = list(cands)
    rng2.shuffle(pool2)
    j = 0
    while len(out2) < min(n, len(out)):
        src = pool2[j % len(pool2)]
        ctype = types[len(out2) % 3]
        # reuse already generated text of same type if available
        text = out[len(out2)]["text"]
        out2.append(
            {
                "neg_id": f"ta_en_n{len(out2)+1:03d}",
                "from_cand_id": text_to_id[src["text"]],
                "corruption_type": ctype,
                "text": text,
            }
        )
        j += 1
    return out2


def compute_profile(unique_utts: list[dict[str, Any]], lexicon: set[str]) -> dict[str, Any]:
    shares: list[float] = []
    cmis: list[float] = []
    switches: list[int] = []
    spans: list[float] = []
    n_mono = 0
    n_scored = 0
    tok_tag = Counter()
    en_in_ta_dom: Counter[str] = Counter()
    ta_func: Counter[str] = Counter()
    ta_all: Counter[str] = Counter()
    light_patterns: Counter[str] = Counter()
    intra_mixed: Counter[str] = Counter()
    switch_bi: Counter[str] = Counter()
    address = Counter({k: 0 for k in ADDRESS_FORMS})
    cluster_spell: dict[str, Counter[str]] = defaultdict(Counter)
    origin_counts = Counter()
    source_counts = Counter()

    func_set = set(TA_FUNCTION_VERBS)

    for u in unique_utts:
        origin_counts[u["tag_origin"]] += 1
        source_counts[u["source_id"]] += 1
        tags = u["lang_tags"]
        toks = u["tokens"]
        mets = utterance_metrics(tags)
        for t in tags:
            tok_tag[t] += 1
        if mets["both"]:
            n_scored += 1
            shares.append(mets["english_share"])
            cmis.append(mets["cmi"])
            switches.append(mets["n_switches"])
            if mets["mean_span"] is not None:
                spans.append(mets["mean_span"])
            if mets["n_en"] == 0 or mets["n_ta"] == 0:
                n_mono += 1
        if mets["n_ta"] > mets["n_en"]:
            for tok, tag in zip(toks, tags):
                if tag == "en":
                    en_in_ta_dom[tok.lower()] += 1
        for tok, tag in zip(toks, tags):
            low = tok.lower()
            if tag == "ta":
                ta_all[low] += 1
                if low in func_set:
                    ta_func[low] += 1
            if tag in {"ta", "mixed", "uncertain"} and low.isalpha() and len(low) >= 3:
                cluster_spell[skeleton_key(low)][low] += 1
            if tag == "mixed":
                spl = mixed_split(low, lexicon)
                if spl:
                    intra_mixed[f"{spl[0]}+{spl[1]} ({low})"] += 1
                else:
                    intra_mixed[low] += 1
        for (t0, g0), (t1, g1) in zip(zip(toks, tags), zip(toks[1:], tags[1:])):
            if {g0, g1} == {"en", "ta"}:
                switch_bi[f"{t0.lower()}|{t1.lower()}"] += 1
            if g0 == "en" and t1.lower() in LIGHT_VERBS:
                light_patterns[f"{t0.lower()} {t1.lower()}"] += 1
        blob = f" {u['text'].lower()} "
        lows = [t.lower() for t in toks]
        for form in ADDRESS_FORMS:
            if " " in form:
                address[form] += blob.count(f" {form} ")
            else:
                address[form] += lows.count(form)

    n_tok = sum(tok_tag.values()) or 1
    clusters = []
    for key, spellings in cluster_spell.items():
        if len(spellings) < 2:
            continue
        clusters.append(
            {
                "key": key,
                "n_spellings": len(spellings),
                "total": int(sum(spellings.values())),
                "spellings": spellings.most_common(12),
            }
        )
    clusters.sort(key=lambda x: (-x["total"], -x["n_spellings"], x["key"]))

    # fill Tamil function list to 150 with frequent ta tokens if needed
    ta_top = ta_func.most_common()
    seen = {w for w, _ in ta_top}
    for w, c in ta_all.most_common():
        if w in seen:
            continue
        ta_top.append((w, c))
        seen.add(w)
        if len(ta_top) >= 150:
            break

    return {
        "pair_id": PAIR_ID,
        "n_unique_utterances": len(unique_utts),
        "n_scored_en_ta": n_scored,
        "tag_origin_counts": dict(origin_counts),
        "source_counts_unique": dict(source_counts),
        "gold_word_tags_found": bool(origin_counts.get("gold")),
        "auto_tags_unvalidated": True,
        "english_share_quantiles": quantiles(shares),
        "cmi_quantiles": quantiles(cmis),
        "switches_per_utterance_quantiles": quantiles([float(x) for x in switches]),
        "mean_same_language_span_length": round(sum(spans) / len(spans), 4) if spans else None,
        "monolingual_share": round(n_mono / n_scored, 4) if n_scored else None,
        "token_tag_share": {k: round(v / n_tok, 4) for k, v in tok_tag.items()},
        "mixed_token_share": round(tok_tag["mixed"] / n_tok, 4),
        "uncertain_token_share": round(tok_tag["uncertain"] / n_tok, 4),
        "top_300_english_in_tamil_dominant": en_in_ta_dom.most_common(300),
        "top_150_tamil_function_verbs": ta_top[:150],
        "top_200_en_light_verb": light_patterns.most_common(200),
        "top_200_intra_word_mixed": intra_mixed.most_common(200),
        "top_100_switch_bigrams": switch_bi.most_common(100),
        "address_politeness_counts": dict(address),
        "top_300_romanization_clusters": clusters[:300],
        "lexicon_size": len(lexicon),
        "lexicon_min_count": LEXICON_MIN_COUNT,
        "tagging_notes": (
            "All non-CoLI tags are automatic and unvalidated. Rule order: "
            "intra-word EN-stem+Tamil-suffix -> mixed; zipf_en>=3.5 and not in "
            "lexicon -> en; lexicon or Tamil ending with zipf_en<3 -> ta; else uncertain."
        ),
    }


def write_licenses() -> None:
    LIC_ROOT.mkdir(parents=True, exist_ok=True)
    files = {
        "tamilmixsentiment": f"""# tamilmixsentiment (DravidianCodeMix Tamil–English sentiment)

- **Pair ID:** {PAIR_ID}
- **URL:** https://huggingface.co/datasets/community-datasets/tamilmixsentiment
- **Homepage / task page:** https://dravidian-codemix.github.io/2020/datasets.html
- **Paper:** Chakravarthi et al., 2020. Corpus Creation for Sentiment Analysis in Code-Mixed Tamil-English Text. SLTU-CCURL. https://aclanthology.org/2020.sltu-1.28/
- **Version / commit:** Hugging Face `{TAMILMIX_SHA}`
- **Size:** 15,744 YouTube comments (train 11,335 / validation 1,260 / test 3,149)
- **What it is:** Human-written YouTube comments, comment-level sentiment labels. No word-level language tags.
- **Stated license on the HF card:** `license: unknown`
- **Related release:** The broader DravidianCodeMix package (Zenodo 10.5281/zenodo.4750858; LRE 2022) is stated as CC BY 4.0 and “freely available for research purpose.”
- **Redistribution:** **UNCLEAR - stats only, do not redistribute.** Underlying comments are YouTube user content and remain subject to YouTube Terms of Service. Do not republish comment text in a public dataset dump without a confirmed licence.
- **Citation:**
```
@inproceedings{{chakravarthi-etal-2020-corpus,
  title = {{Corpus Creation for Sentiment Analysis in Code-Mixed Tamil-English Text}},
  author = {{Chakravarthi, Bharathi Raja and Muralidaran, Vigneshwaran and Priyadharshini, Ruba and McCrae, John Philip}},
  booktitle = {{SLTU-CCURL}},
  year = {{2020}}
}}
```
""",
        "offenseval_dravidian": f"""# offenseval_dravidian — Tamil config (DravidianCodeMix offensive language)

- **Pair ID:** {PAIR_ID}
- **URL:** https://huggingface.co/datasets/community-datasets/offenseval_dravidian
- **Zenodo (full DravidianCodeMix 1.0):** https://zenodo.org/records/4750858
- **GitHub:** https://github.com/bharathichezhiyan/DravidianCodeMix-Dataset
- **Paper:** Chakravarthi et al., 2022. DravidianCodeMix. Language Resources and Evaluation. https://arxiv.org/abs/2106.09460
- **Version / commit:** Hugging Face `{OFFENSE_SHA}` (community-datasets mirror of the historical `offenseval_dravidian` dataset; config `tamil`)
- **Size:** Tamil split 35,139 train + 4,388 validation = 39,527 comments. Labels: Not_offensive, Offensive_Untargetede, Offensive_Targeted_Insult_Individual, Offensive_Targeted_Insult_Group, Offensive_Targeted_Insult_Other, not-Tamil.
- **What it is:** Human-written YouTube comments with comment-level offensive-language labels. No word-level language tags.
- **Stated license (HF card and LRE/Zenodo family):** Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/
- **CC BY 4.0 terms (summary):** You may share and adapt, including commercially, if you give appropriate credit, link the licence, and note changes. No additional legal terms that restrict others. No warranties.
- **Redistribution:** Allowed under CC BY 4.0 **for this annotation package** with attribution. Comment text is still user-generated YouTube content; keep attribution and do not claim ownership of the comments.
- **Citation:**
```
@article{{chakravarthi-etal-2021-lre,
  title = {{DravidianCodeMix: Sentiment Analysis and Offensive Language Identification Dataset for Dravidian Languages in Code-Mixed Text}},
  author = {{Chakravarthi, Bharathi Raja and Priyadharshini, Ruba and Muralidaran, Vigneshwaran and Jose, Navya and Suryawanshi, Shardul and Sherly, Elizabeth and McCrae, John P}},
  journal = {{Language Resources and Evaluation}},
  year = {{2022}}
}}
```
""",
        "tanglish_corpus_185k": f"""# Tanglish-Corpus-185k

- **Pair ID:** {PAIR_ID}
- **URL:** https://huggingface.co/datasets/vishnu-n/Tanglish-Corpus-185k
- **Canonical citation URL on the card:** https://huggingface.co/datasets/vishnuexe/Tanglish-Corpus-185k
- **Version / commit:** `{TANGLISH_SHA}` (README upload; jsonl earlier in the same repo)
- **Size:** 185,973 Roman-script sentences. Card breakdown: YouTube (~153k clean), Reddit r/TamilNaduDiscussion (~300), DravidianCodeMix subset (~32k). Filtered from ~1.8M raw comments.
- **What it is:** Human-written social-media Tanglish (YouTube + Reddit + a DravidianCodeMix slice). The card states it is **not** synthetic LLM text. No word-level language tags. Overlaps DravidianCodeMix / tamilmixsentiment / offenseval.
- **Stated license:** CC BY 4.0 — “free for research and commercial use with attribution.” YAML `license: cc-by-4.0`.
- **Source-platform caveat (stated on the card):** YouTube public comments and Reddit public posts remain under those platforms’ Terms of Service. DravidianCodeMix portion is CC BY 4.0 (Zenodo).
- **CC BY 4.0 terms (summary):** Share and adapt with attribution; link the licence; indicate changes; no extra restrictions.
- **Redistribution:** The compilation is released CC BY 4.0. **Do not treat YouTube/Reddit comments as copyright-cleared for wholesale republication** beyond the rights the compiler granted; for this project we keep raw files local and publish statistics / small reviewed examples only if a later audit agrees.
- **Citation:**
```
@misc{{vishnu2026tanglishcorpus,
  title = {{Tanglish-Corpus-185k: The Largest Romanised Tamil-English Code-Mixed Sentence Corpus}},
  author = {{Vishnu N}},
  year = {{2026}},
  publisher = {{HuggingFace}},
  url = {{https://huggingface.co/datasets/vishnuexe/Tanglish-Corpus-185k}}
}}
```
""",
        "coli_dravidian_ta": f"""# CoLI-Dravidian 2024 Tamil (word-level LID)

- **Pair ID:** {PAIR_ID}
- **URL:** https://codalab.lisn.upsaclay.fr/competitions/19357
- **Task site:** https://sites.google.com/view/coli-dravidian-2024/home
- **Paper:** Hegde et al., 2024. Overview of CoLI-Dravidian: Word-level Code-Mixed Language Identification in Dravidian Languages. FIRE 2024. https://ceur-ws.org/Vol-4054/T3-1.pdf
- **Version:** CodaLab public “Tamil - Training” bundle (`tamil_train.csv`, `tamil_val.csv`), downloaded as dataset id `d9300110-8f3f-423c-9123-1a3ebca74e68`.
- **Size:** Train 13,555 token rows; validation 1,983 token rows. Gold tags include `tm`, `en`, `tmen` (mixed), `name`, `location`, `other`, `sym`. Sentence boundaries marked `*`.
- **What it is:** Human-written YouTube-comment tokens with **native-speaker word-level language tags**. This is the gold LID source used in the profile.
- **Stated license:** **Not stated** on the CodaLab page or in the overview PDF we retrieved. Shared-task “public data” download, research use implied, commercial/redistribution terms missing.
- **Redistribution:** **UNCLEAR - stats only, do not redistribute.** Keep raw CSVs local; do not republish the annotated token file.
- **Citation:**
```
@inproceedings{{hegde-etal-2024-coli,
  title = {{Overview of CoLI-Dravidian: Word-level Code-Mixed Language Identification in Dravidian Languages}},
  author = {{Hegde, Asha and Balouchzahi, Fazlourrahman and Butt, Sabur and Coelho, Sharal and G, Kavya and Kumar, H. S. and D, Shashirekha and Hosahalli Lakshmaiah, Shashirekha and Agrawal, Apoorva}},
  booktitle = {{FIRE 2024}},
  year = {{2024}}
}}
```
""",
        "cmqa_tamlish": f"""# Code-mixed QA — Tamlish questions (Chandu et al. 2018)

- **Pair ID:** {PAIR_ID}
- **URL:** https://github.com/khyathiraghavi/code_switched_QA
- **Paper:** Chandu et al., 2018. Code-Mixed Question Answering Challenge: Crowd-sourcing Data and Techniques. CALCS @ ACL 2018. https://aclanthology.org/W18-3204/
- **Version:** GitHub `master` snapshot (files dated 2020-05-05 in the tarball).
- **Size:** About 1,033 Tamlish question records in train+test batches; ~996 unique query strings. Human crowd-sourced factoid questions. No word-level language tags.
- **Stated license (COPYING):** Carnegie Mellon University copyright 2018. Quote:

> This data is free for use for any purpose (commercial or otherwise) subject to the pretty light restrictions detailed below.
>
> Permission to use, copy, modify this data and its documentation for any purpose, is hereby granted without fee, subject to the following conditions:
> 1. The data must retain the above copyright notice, this list of conditions and the following disclaimer.
> 2. Any modifications must be clearly marked as such.
> 3. Original authors' names are not deleted.
>
> THE AUTHORS OF THIS WORK DISCLAIM ALL WARRANTIES ...

- **Redistribution:** Allowed **with the copyright notice, conditions, and disclaimer retained.** Modifications must be marked.
- **Citation:**
```
@inproceedings{{chandu-etal-2018-code,
  title = {{Code-Mixed Question Answering Challenge: Crowd-sourcing Data and Techniques}},
  author = {{Chandu, Khyathi and Loginova, Ekaterina and Gupta, Vishal and van Genabith, Josef and Neumann, G\\\"unter and Chinnakotla, Manoj and Nyberg, Eric and Black, Alan W.}},
  booktitle = {{Proceedings of the Third Workshop on Computational Approaches to Linguistic Code-Switching}},
  year = {{2018}}
}}
```
""",
    }
    for sid, body in files.items():
        (LIC_ROOT / f"{sid}.md").write_text(body, encoding="utf-8")


def write_report(
    stats: dict[str, Any],
    download_log: dict[str, Any],
    n_raw: int,
    n_utt: int,
    dropped_script: int,
    dropped_empty: int,
    cand_info: dict[str, int],
    cands: list[dict[str, Any]],
    lexicon: set[str],
) -> None:
    def fmt_pairs(pairs: Iterable, n: int = 10) -> str:
        rows = list(pairs)[:n]
        return "\n".join(f"- `{a}` ({b})" for a, b in rows)

    gold_n = stats["tag_origin_counts"].get("gold", 0)
    ex = [c["text"] for c in cands[:10]]
    clusters = stats["top_300_romanization_clusters"][:10]
    cl_lines = []
    for c in clusters:
        spell = ", ".join(f"{w}={n}" for w, n in c["spellings"][:6])
        cl_lines.append(f"- key `{c['key']}` ({c['n_spellings']} spellings, n={c['total']}): {spell}")

    md = f"""# Tamil–English (ta_en) calibration profile

Empirical profile from **human-written** Roman-script Tamil–English corpora.
**Automatic token tags are unvalidated** except CoLI-Dravidian gold rows.

## Sources obtained

"""
    for item in download_log.get("obtained", []):
        md += f"- **{item['source_id']}:** {item['detail']}\n"
    md += "\n## Sources failed or skipped\n\n"
    for item in download_log.get("failed", []):
        md += f"- **{item['source_id']}:** {item['detail']}\n"

    md += f"""
## Licence (short)

| source_id | licence_status |
|---|---|
| tamilmixsentiment | UNCLEAR - stats only, do not redistribute |
| offenseval_dravidian | CC BY 4.0 (HF/Zenodo family); YouTube TOS on comments |
| tanglish_corpus_185k | CC BY 4.0 compilation; YouTube/Reddit TOS on comments |
| coli_dravidian_ta | UNCLEAR - stats only, do not redistribute |
| cmqa_tamlish | CMU notice licence; redistribute with copyright notice |

Full notes: `data/calibration/licenses/<source_id>.md`.

## Counts

- Raw records loaded: {n_raw}
- Dropped >10% Tamil Unicode: {dropped_script}
- Dropped empty after cleaning: {dropped_empty}
- Processed utterances (all sources, including overlap): {n_utt}
- Unique texts used for statistics: {stats['n_unique_utterances']}
- Unique texts with at least one en/ta token: {stats['n_scored_en_ta']}
- Gold-tagged utterances (CoLI): {gold_n}
- Auto-tagged utterances (unique set): {stats['tag_origin_counts'].get('auto', 0)}
- Lexicon size: {stats['lexicon_size']} (min count {stats['lexicon_min_count']}, zipf_en < {ZIPF_LEXICON}, plus seeds)

Unique-text source mix: {stats['source_counts_unique']}

**Gold word-level tags:** yes, from CoLI-Dravidian 2024 Tamil train/val (`tm`/`en`/`tmen`/`name`/…). ICON 2015 Tamil–English gold LID was **not** obtained. All other sources are auto-tagged.

## Key quantiles

English share n_en/(n_en+n_ta): {stats['english_share_quantiles']}

CMI 100*(1 - max/(n_en+n_ta)): {stats['cmi_quantiles']}

Switches per utterance (adjacent en↔ta only): {stats['switches_per_utterance_quantiles']}

Mean same-language span length: {stats['mean_same_language_span_length']}

Monolingual share (among scored): {stats['monolingual_share']}

Mixed token share: {stats['mixed_token_share']}; uncertain token share: {stats['uncertain_token_share']}

Token-tag distribution: {stats['token_tag_share']}

## Service-like candidates

Eligible (5–30 tokens, CMI>0, ≥1 marker, not offensive): {cand_info['eligible_before_diversity']}

Selected after 5-gram diversity cap 400: {cand_info['selected']}

**Genuinely service/transaction-like** (≥1 of payment, account, order, delivery, refund, bank, card, otp, cancel, return, …): {cand_info['genuinely_service_like']}

**Soft service only** (help / problem / issue / call / number / message, no transaction word): {cand_info.get('soft_help_call_only', 0)}

**Generic / social-media address only** (neenga, sir, please, …): {cand_info['generic_address_only']}

Most comments are movie/YouTube talk. Strong-marker items were ranked first; the remainder are still mixed but not true service dialogues.

## 10 example service candidates

"""
    for i, t in enumerate(ex, 1):
        md += f"{i}. {t}\n"

    md += f"""
## 10 top EN + Tamil light-verb patterns

{fmt_pairs(stats['top_200_en_light_verb'])}

## 10 top intra-word mixed forms

{fmt_pairs(stats['top_200_intra_word_mixed'])}

## 10 top romanization clusters (≥2 spellings)

{chr(10).join(cl_lines)}

## Address / politeness counts

{json.dumps(stats['address_politeness_counts'], indent=2)}

## Problems

- Social-media comments dominate; they are **not** phone-call service dialogues. Use quantiles as sampling bands, not as fraud-call truth.
- Auto LID is lexicon+zipf+suffix heuristics. Expect errors on names, slang, and English words with Tamil-looking endings.
- `tamilmixsentiment` HF licence is **unknown**; we treat it as stats-only.
- CoLI gold is small (~utterances reconstructed from ~15k tokens) and YouTube-domain.
- Tanglish-185k **includes a DravidianCodeMix slice**; statistics are computed on unique texts to reduce double-counting.
- ICON 2015 gold Tamil–English word tags were not recoverable from a live URL.
- Offensive filtering uses offenseval labels (exact-text overlap) plus a small blocklist; it will miss some abuse and may drop non-overlapping rephrasings.
- Romanization clusters follow the specified skeleton key; some intuitive variants (epdi vs eppadi) may split if the key does not collapse them.
"""
    (PROC_ROOT / "REPORT.md").write_text(md, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    PROC_ROOT.mkdir(parents=True, exist_ok=True)
    print("Downloading / verifying sources…")
    download_log = download_sources(skip_download=args.skip_download)
    write_licenses()

    print("Loading raw records…")
    raw, offensive_norm = load_raw_records()
    print(f" raw records: {len(raw)}")
    print("Processing, building lexicon, tagging…")
    utterances, lexicon, _counts, dropped_script, dropped_empty = process_records(raw)
    print(
        f" utterances={len(utterances)} dropped_script={dropped_script} "
        f"dropped_empty={dropped_empty} lexicon={len(lexicon)}"
    )

    utt_path = PROC_ROOT / "utterances.jsonl"
    with utt_path.open("w", encoding="utf-8") as fh:
        for u in utterances:
            fh.write(
                json.dumps(
                    {
                        "id": u["id"],
                        "source_id": u["source_id"],
                        "text": u["text"],
                        "tokens": u["tokens"],
                        "lang_tags": u["lang_tags"],
                        "tag_origin": u["tag_origin"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    unique = unique_for_stats(utterances)
    print(f" unique texts: {len(unique)}")
    stats = compute_profile(unique, lexicon)
    stats["n_processed_utterances_all_sources"] = len(utterances)
    stats["n_raw_records"] = len(raw)
    stats["dropped_tamil_unicode"] = dropped_script
    stats["dropped_empty"] = dropped_empty
    stats["download_log"] = download_log

    print("Selecting service candidates…")
    cands, cand_info = select_service_candidates(unique, offensive_norm)
    stats["service_candidate_info"] = cand_info
    (PROC_ROOT / "profile_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    cand_csv = PROC_ROOT / "service_candidates.csv"
    with cand_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        for i, row in enumerate(cands, 1):
            writer.writerow(
                {
                    "cand_id": f"ta_en_c{i:03d}",
                    "source_id": row["source_id"],
                    "source_item_id": row["source_item_id"],
                    "text": row["text"],
                    "english_share": "" if row["english_share"] is None else round(row["english_share"], 4),
                    "cmi": "" if row["cmi"] is None else round(row["cmi"], 4),
                    "n_switches": row["n_switches"],
                    "markers_hit": "|".join(row["markers_hit"]),
                    "license_status": row["license_status"],
                    "accept": "",
                    "reject_reason": "",
                    "reviewer_note": "",
                }
            )

    print("Writing unnatural negatives…")
    negs = corrupt_candidates(cands, 100)
    # bind from_cand_id to the numbered candidates by matching source text
    text_to_id = {c["text"]: f"ta_en_c{i:03d}" for i, c in enumerate(cands, 1)}
    # regenerate negatives from numbered candidates for clean IDs
    rng = random.Random(SEED)
    types = ["shuffle_word_order", "move_tamil_suffix", "hindi_verb_swap"]
    negs = []
    order = list(range(len(cands)))
    rng.shuffle(order)
    k = 0
    while len(negs) < 100 and cands:
        src = cands[order[k % len(order)]]
        ctype = types[len(negs) % 3]
        toks = list(src["tokens"])
        if ctype == "shuffle_word_order":
            shuf = toks[:]
            rng.shuffle(shuf)
            if shuf == toks and len(toks) > 1:
                shuf = toks[1:] + toks[:1]
            text = " ".join(shuf)
        elif ctype == "move_tamil_suffix":
            new = toks[:]
            moved = False
            for j, tok in enumerate(toks):
                spl = mixed_split(tok)
                if not spl and matches_ta_ending(tok) and len(tok) > 4:
                    spl = (tok[:-2], tok[-2:])
                if not spl:
                    continue
                stem, suf = spl
                dest = (j + 1) % len(toks)
                new[j] = stem
                new[dest] = new[dest] + suf
                moved = True
                break
            if not moved and len(new) >= 2:
                new[0] = new[0] + "ku"
            text = " ".join(new)
        else:
            new = [HINDI_VERB_MAP.get(t.lower(), t) for t in toks]
            if new == toks:
                new = toks + ["kijiye"]
            text = " ".join(new)
        negs.append(
            {
                "neg_id": f"ta_en_n{len(negs)+1:03d}",
                "from_cand_id": text_to_id[src["text"]],
                "corruption_type": ctype,
                "text": text,
            }
        )
        k += 1

    with (PROC_ROOT / "unnatural_negatives.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["neg_id", "from_cand_id", "corruption_type", "text"])
        writer.writeheader()
        writer.writerows(negs)

    (PROC_ROOT / "lexicon_ta.json").write_text(
        json.dumps(sorted(lexicon), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(
        stats,
        download_log,
        len(raw),
        len(utterances),
        dropped_script,
        dropped_empty,
        cand_info,
        cands,
        lexicon,
    )
    print("Wrote", PROC_ROOT)
    print("service candidates", cand_info)
    print("english_share", stats["english_share_quantiles"])
    print("cmi", stats["cmi_quantiles"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
