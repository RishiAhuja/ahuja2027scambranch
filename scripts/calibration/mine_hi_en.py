#!/usr/bin/env python3
"""Build an empirical Hindi-English (Romanized Hinglish) language profile.

Human-written sources only. Pair ID: hi_en.
Re-run from the repo root:

    .venv/bin/python scripts/calibration/mine_hi_en.py
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from wordfreq import zipf_frequency

try:
    from huggingface_hub import hf_hub_download
except ImportError:  # pragma: no cover
    hf_hub_download = None

PAIR_ID = "hi_en"
SEED = 42

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
RAW_ROOT = REPO_ROOT / "data" / "calibration" / "raw" / PAIR_ID
PROC_ROOT = REPO_ROOT / "data" / "calibration" / "processed" / PAIR_ID
LIC_ROOT = REPO_ROOT / "data" / "calibration" / "licenses"

LINCE_PARQUET_REV = "refs/convert/parquet"
LINCE_HF = "lince-benchmark/lince"
PHINC_HF = "LingoIITGN/PHINC"
HINGE_HF = "LingoIITGN/HinGE"
SENTIMIX_ZENODO = (
    "https://zenodo.org/api/records/3974927/files/"
    "Semeval_2020_task9_data.zip/content"
)
PHINC_ZENODO = (
    "https://zenodo.org/api/records/3605597/files/"
    "English-Hindi%20code-mixed%20parallel%20corpus.csv/content"
)

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
URL_RE = re.compile(
    r"(https?://\S+|www\.\S+|t\.co/\S+)",
    re.IGNORECASE,
)
BROKEN_TCO_RE = re.compile(
    r"(https?\s+)?(//\s+)?t\s+\.\s+co\s+/\s+\S+",
    re.IGNORECASE,
)
BROKEN_TCO_LOOSE_RE = re.compile(
    r"(https?\s+)?//\s+t\s+\.?\s*co\s+/\s+\S+",
    re.IGNORECASE,
)
TRAILING_TCO_RE = re.compile(
    r"//\s+t(\s+\.)?\s*c\S*.*$",
    re.IGNORECASE,
)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
HANDLE_RE = re.compile(r"(?<!\w)@[\w_]+")
PHONE_RE = re.compile(r"(?:\+?91[\s-]*)?[6-9]\d{9}\b")
HASHTAG_RE = re.compile(r"#(\w+)")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z']*|\d+|[^\sA-Za-z0-9]")
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "\U0000FE00-\U0000FE0F"
    "\U0001F1E0-\U0001F1FF"
    "\U0000200D"
    "]+",
    flags=re.UNICODE,
)
WORD_RE = re.compile(r"^[A-Za-z][A-Za-z']*$")

LID_MAP = {
    "lang1": "en",
    "eng": "en",
    "en": "en",
    "english": "en",
    "lang2": "hi",
    "hin": "hi",
    "hi": "hi",
    "hindi": "hi",
    "mixed": "mixed",
    "mix": "mixed",
    "ne": "ne",
    "unk": "uncertain",
    "unknown": "uncertain",
    "ambiguous": "uncertain",
    "amb": "uncertain",
    "other": "other",
    "univ": "other",
    "rest": "other",
    "fw": "other",
    "o": "other",
    "emot": "other",
    "emt": "other",
    "x": "other",
}

LIGHT_VERBS = {
    "kar",
    "karo",
    "karna",
    "kiya",
    "kijiye",
    "karein",
    "ho",
    "hua",
    "hoga",
    "dena",
    "dijiye",
    "lena",
}
LIGHT_VERB_VARIANTS = {
    "kro": "karo",
    "krna": "karna",
    "kiye": "kiya",
    "kijie": "kijiye",
    "kijiye": "kijiye",
    "kare": "karein",
    "kren": "karein",
    "krn": "karna",
    "huaa": "hua",
    "hogaa": "hoga",
    "dijie": "dijiye",
}

POSTPOSITIONS = {
    "ka",
    "ki",
    "ke",
    "ko",
    "se",
    "mein",
    "me",
    "mai",
    "par",
    "pe",
    "tak",
    "ne",
    "liye",
}

ADDRESS_FORMS = [
    "aap",
    "aapka",
    "aapko",
    "tum",
    "tu",
    "ji",
    "sir",
    "madam",
    "bhai",
    "bhaiya",
    "didi",
    "sahab",
    "please",
    "plz",
    "pls",
    "kripya",
    "dhanyavaad",
    "thanks",
    "shukriya",
    "namaste",
    "hello",
    "haan",
    "nahi",
]
THANK_YOU = ("thank", "you")

SERVICE_MARKERS = [
    "aap",
    "aapka",
    "aapko",
    "kijiye",
    "dijiye",
    "karein",
    "bataiye",
    "batao",
    "please",
    "sir",
    "madam",
    "ji",
    "payment",
    "paise",
    "rupaye",
    "rs",
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
    "kab",
    "kitne",
    "kitna",
    "kaise",
    "problem",
    "issue",
    "help",
    "service",
    "book",
    "cancel",
    "return",
    "confirm",
    "time",
    "kal",
    "abhi",
]

FUNCTION_SEED = {
    "hai",
    "hain",
    "ha",
    "ho",
    "hun",
    "hu",
    "tha",
    "thi",
    "the",
    "raha",
    "rahi",
    "rahe",
    "gaya",
    "gayi",
    "gaye",
    "kar",
    "karo",
    "karna",
    "karta",
    "karti",
    "karte",
    "kiya",
    "kiye",
    "kijiye",
    "karein",
    "kare",
    "dena",
    "dijiye",
    "do",
    "diya",
    "lena",
    "liya",
    "lo",
    "ka",
    "ki",
    "ke",
    "ko",
    "se",
    "mein",
    "me",
    "mai",
    "par",
    "pe",
    "tak",
    "ne",
    "liye",
    "wala",
    "wali",
    "wale",
    "nahi",
    "nahin",
    "na",
    "mat",
    "bhi",
    "to",
    "toh",
    "hi",
    "ye",
    "yeh",
    "wo",
    "woh",
    "is",
    "us",
    "in",
    "un",
    "aap",
    "tum",
    "tu",
    "main",
    "hum",
    "ham",
    "kya",
    "kyun",
    "kyu",
    "kaise",
    "kab",
    "kahan",
    "kitna",
    "kitne",
    "kaun",
    "jo",
    "ji",
    "aur",
    "ya",
    "lekin",
    "magar",
    "agar",
    "toh",
    "phir",
    "ab",
    "abhi",
    "kal",
    "aj",
    "aaj",
    "bahut",
    "bohot",
    "zyada",
    "sirf",
    "bas",
    "chahiye",
    "sakta",
    "sakti",
    "sakte",
    "hoga",
    "hogi",
    "honge",
    "hua",
    "hui",
    "hue",
    "diya",
    "deti",
    "deta",
    "dete",
    "liye",
    "wali",
}

HI_EN_GLOSS = {
    "hai": "is",
    "hain": "are",
    "ho": "be",
    "hun": "am",
    "hu": "am",
    "tha": "was",
    "thi": "was",
    "the": "were",
    "raha": "been",
    "rahi": "been",
    "gaya": "went",
    "gayi": "went",
    "kar": "do",
    "karo": "do",
    "karna": "do",
    "kiya": "did",
    "kijiye": "please-do",
    "karein": "do",
    "dena": "give",
    "dijiye": "please-give",
    "lena": "take",
    "nahi": "not",
    "nahin": "not",
    "mein": "in",
    "me": "in",
    "mai": "in",
    "ka": "of",
    "ki": "of",
    "ke": "of",
    "ko": "to",
    "se": "from",
    "par": "on",
    "pe": "on",
    "aap": "you",
    "aapka": "your",
    "aapko": "you",
    "tum": "you",
    "tu": "you",
    "main": "I",
    "hum": "we",
    "yeh": "this",
    "ye": "this",
    "woh": "that",
    "wo": "that",
    "kya": "what",
    "kaise": "how",
    "kab": "when",
    "kahan": "where",
    "kitna": "how-much",
    "kitne": "how-many",
    "kaun": "who",
    "aur": "and",
    "lekin": "but",
    "agar": "if",
    "phir": "then",
    "abhi": "now",
    "kal": "tomorrow",
    "aaj": "today",
    "bahut": "very",
    "chahiye": "need",
    "please": "please",
    "ji": "sir",
    "bhai": "brother",
    "haan": "yes",
    "kripya": "please",
    "dhanyavaad": "thanks",
    "shukriya": "thanks",
}

BLOCKLIST = {
    "fuck",
    "fucking",
    "shit",
    "bitch",
    "asshole",
    "bastard",
    "dick",
    "pussy",
    "slut",
    "whore",
    "cunt",
    "nigger",
    "nigga",
    "retard",
    "madarchod",
    "behenchod",
    "bhenchod",
    "bhenchodd",
    "chutiya",
    "chutiye",
    "chutia",
    "harami",
    "randi",
    "bhosdi",
    "bhosdike",
    "bhosadike",
    "lund",
    "lawda",
    "lavda",
    "lauda",
    "loda",
    "gaand",
    "gandu",
    "bakchod",
    "bsdk",
    "mc",
    "bhodu",
    "raand",
    "chod",
    "chode",
    "chut",
}

INSERT_EN = ["the", "please", "actually", "kindly", "really"]


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dirs() -> None:
    for p in (
        RAW_ROOT / "lince_lid_hineng",
        RAW_ROOT / "lince_pos_hineng",
        RAW_ROOT / "phinc",
        RAW_ROOT / "hinge",
        RAW_ROOT / "sentimix",
        PROC_ROOT,
        LIC_ROOT,
    ):
        p.mkdir(parents=True, exist_ok=True)


def download_url(url: str, dest: Path, timeout: int = 120) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "aaai-icfd-calibration/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as fh:
            shutil.copyfileobj(resp, fh)
        tmp.replace(dest)
        return dest.exists() and dest.stat().st_size > 0
    except Exception as exc:
        log(f"  download failed: {url} ({exc})")
        if tmp.exists():
            tmp.unlink()
        return False


def download_hf_file(repo_id: str, filename: str, dest: Path, revision: str | None = None) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    if hf_hub_download is None:
        return False
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            revision=revision,
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        return True
    except Exception as exc:
        log(f"  HF download failed: {repo_id}/{filename} ({exc})")
        return False


def download_sources(force: bool = False) -> dict:
    """Download all sources. Returns a status dict."""
    ensure_dirs()
    status = {}

    if force:
        for folder in [
            RAW_ROOT / "lince_lid_hineng",
            RAW_ROOT / "lince_pos_hineng",
            RAW_ROOT / "phinc",
            RAW_ROOT / "hinge",
            RAW_ROOT / "sentimix",
        ]:
            for child in folder.glob("*"):
                if child.is_file() and child.suffix in {".parquet", ".csv", ".zip", ".txt", ".conll"}:
                    child.unlink()

    log("Downloading LinCE lid_hineng / pos_hineng (HF parquet conversion)...")
    lince_ok = True
    for config in ("lid_hineng", "pos_hineng"):
        for split in ("train", "validation", "test"):
            dest = RAW_ROOT / f"lince_{config}" / f"{split}.parquet"
            ok = download_hf_file(
                LINCE_HF,
                f"{config}/{split}/0000.parquet",
                dest,
                revision=LINCE_PARQUET_REV,
            )
            if not ok:
                lince_ok = False
    status["lince"] = {
        "ok": lince_ok,
        "detail": "HF parquet conversion of lince-benchmark/lince"
        if lince_ok
        else "FAILED: HF parquet and ritual.uh.edu unavailable",
    }

    log("Downloading PHINC...")
    phinc_dest = RAW_ROOT / "phinc" / "PHINC.csv"
    phinc_ok = download_hf_file(PHINC_HF, "PHINC.csv", phinc_dest)
    if not phinc_ok:
        phinc_ok = download_url(PHINC_ZENODO, phinc_dest)
    status["phinc"] = {
        "ok": phinc_ok,
        "detail": "LingoIITGN/PHINC on Hugging Face (CC BY 4.0; Zenodo 3605597 original)"
        if phinc_ok
        else "FAILED",
    }

    log("Downloading HinGE (human-written column only will be used)...")
    hinge_dest = RAW_ROOT / "hinge" / "HinGE.csv"
    hinge_ok = download_hf_file(HINGE_HF, "HinGE.csv", hinge_dest)
    status["hinge"] = {
        "ok": hinge_ok,
        "detail": "LingoIITGN/HinGE on Hugging Face; WAC/PAC rule-based columns ignored"
        if hinge_ok
        else "FAILED",
    }

    log("Downloading SemEval-2020 SentiMix Hinglish (Zenodo 3974927)...")
    sm_zip = RAW_ROOT / "sentimix" / "Semeval_2020_task9_data.zip"
    sm_ok = download_url(SENTIMIX_ZENODO, sm_zip)
    if sm_ok:
        extract_dir = RAW_ROOT / "sentimix" / "extracted"
        if not (extract_dir / "Semeval_2020_task9_data" / "Hinglish").exists():
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(sm_zip) as zf:
                zf.extractall(extract_dir)
    status["sentimix"] = {
        "ok": sm_ok,
        "detail": "Zenodo 10.5281/zenodo.3974927 V1"
        if sm_ok
        else "FAILED",
    }
    return status


def map_lid(tag: str | None) -> str:
    if tag is None:
        return "uncertain"
    key = str(tag).strip().lower()
    if not key:
        return "uncertain"
    return LID_MAP.get(key, "uncertain")


def has_devanagari(text: str) -> bool:
    return bool(DEVANAGARI_RE.search(text or ""))


def is_url_token(tok: str) -> bool:
    t = tok.lower().strip(".,;:!?()[]{}")
    if t in {"http", "https", "www", "t.co", "tco", "bit.ly"}:
        return True
    if t.startswith("http") or t.startswith("www.") or "t.co" in t:
        return True
    return bool(URL_RE.fullmatch(tok))


def broken_url_token_indices(words: list[str]) -> set[int]:
    """Drop tokenized t.co / https leftovers (SentiMix splits URLs)."""
    if not words:
        return set()
    spans = []
    pos = 0
    for i, w in enumerate(words):
        if i:
            pos += 1
        start = pos
        pos += len(w)
        spans.append((start, pos, i))
    text = " ".join(words)
    drop: set[int] = set()
    for pat in (BROKEN_TCO_RE, BROKEN_TCO_LOOSE_RE, TRAILING_TCO_RE, URL_RE):
        for m in pat.finditer(text):
            a, b = m.span()
            for s, e, i in spans:
                if e > a and s < b:
                    drop.add(i)
    return drop


def is_email_token(tok: str) -> bool:
    return bool(EMAIL_RE.fullmatch(tok))


def is_phone_token(tok: str) -> bool:
    digits = re.sub(r"\D", "", tok)
    if len(digits) >= 10 and PHONE_RE.search(tok):
        return True
    return False


def is_emoji_token(tok: str) -> bool:
    if not tok:
        return False
    stripped = EMOJI_RE.sub("", tok)
    return stripped.strip() == ""


def skeleton_key(tok: str) -> str:
    """Romanization variant key.

    lowercase; expand ChV (nhi->nahi); aa->a, ee->i, oo->u; collapse repeats;
    drop h after consonants and leftover h; drop final n after a vowel.
    """
    t = tok.lower()
    t = re.sub(r"[^a-z]", "", t)
    if not t:
        return ""
    t = re.sub(r"([bcdfgjklmnpqrstvwxyz])h([aeiou])", r"\1ah\2", t)
    t = t.replace("aa", "a").replace("ee", "i").replace("oo", "u")
    t = re.sub(r"(.)\1+", r"\1", t)
    t = re.sub(r"([bcdfgjklmnpqrstvwxyz])h", r"\1", t)
    if t != "h":
        t = t.replace("h", "")
    t = re.sub(r"([aeiou])n$", r"\1", t)
    return t


def clean_surface_text(text: str) -> str:
    text = BROKEN_TCO_RE.sub(" ", text)
    text = BROKEN_TCO_LOOSE_RE.sub(" ", text)
    text = TRAILING_TCO_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = HANDLE_RE.sub(" ", text)
    text = PHONE_RE.sub(" ", text)
    text = HASHTAG_RE.sub(r"\1", text)
    text = EMOJI_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_untagged(text: str) -> list[str]:
    cleaned = clean_surface_text(text)
    return [t for t in TOKEN_RE.findall(cleaned) if t.strip()]


def clean_gold_sequence(
    words: list[str],
    tags: list[str],
    pos: list[str] | None = None,
) -> tuple[list[str], list[str], list[str | None]]:
    out_w: list[str] = []
    out_t: list[str] = []
    out_p: list[str | None] = []
    skip_next = False
    url_drop = broken_url_token_indices([str(w) for w in words])
    for i, raw in enumerate(words):
        if skip_next:
            skip_next = False
            continue
        if i in url_drop:
            continue
        w = str(raw)
        tag = map_lid(tags[i] if i < len(tags) else "")
        p = None
        if pos is not None and i < len(pos):
            pv = str(pos[i]).strip()
            p = pv or None
        if w == "@":
            skip_next = True
            continue
        if w.startswith("@") and len(w) > 1:
            continue
        if is_url_token(w) or is_email_token(w) or is_phone_token(w) or is_emoji_token(w):
            continue
        if w.startswith("#") and len(w) > 1:
            w = w[1:]
        if not w.strip():
            continue
        out_w.append(w)
        out_t.append(tag)
        out_p.append(p)
    return out_w, out_t, out_p


def zipf_en(tok: str) -> float:
    if not WORD_RE.match(tok):
        return 0.0
    try:
        return float(zipf_frequency(tok.lower(), "en"))
    except Exception:
        return 0.0


def auto_tag_tokens(tokens: list[str], hindi_list: set[str]) -> list[str]:
    tags = []
    for tok in tokens:
        low = tok.lower()
        if not WORD_RE.match(tok):
            tags.append("other")
            continue
        if low in hindi_list:
            tags.append("hi")
        elif zipf_en(tok) >= 3.5:
            tags.append("en")
        else:
            tags.append("uncertain")
    return tags


def utterance_metrics(tags: list[str]) -> dict:
    n_en = sum(1 for t in tags if t == "en")
    n_hi = sum(1 for t in tags if t == "hi")
    den = n_en + n_hi
    if den == 0:
        english_share = None
        cmi = None
    else:
        english_share = n_en / den
        cmi = 100.0 * (1.0 - (max(n_en, n_hi) / den))
    switches = 0
    for a, b in zip(tags, tags[1:]):
        if a in {"en", "hi"} and b in {"en", "hi"} and a != b:
            switches += 1
    spans: list[int] = []
    cur = None
    n = 0
    for t in tags:
        if t in {"en", "hi"}:
            if t == cur:
                n += 1
            else:
                if cur is not None:
                    spans.append(n)
                cur = t
                n = 1
        else:
            if cur is not None:
                spans.append(n)
            cur = None
            n = 0
    if cur is not None:
        spans.append(n)
    return {
        "n_en": n_en,
        "n_hi": n_hi,
        "english_share": english_share,
        "cmi": cmi,
        "n_switches": switches,
        "span_lengths": spans,
        "monolingual": bool(den > 0 and (n_en == 0 or n_hi == 0)),
    }


def quantiles(values: list[float]) -> dict:
    if not values:
        return {k: None for k in ("p10", "p25", "p50", "p75", "p90", "mean", "n")}
    s = sorted(values)
    n = len(s)

    def pct(p: float) -> float:
        if n == 1:
            return s[0]
        idx = (n - 1) * p
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return s[lo] * (1 - frac) + s[hi] * frac

    return {
        "p10": pct(0.10),
        "p25": pct(0.25),
        "p50": pct(0.50),
        "p75": pct(0.75),
        "p90": pct(0.90),
        "mean": sum(s) / n,
        "n": n,
    }


def parse_lince_lid() -> list[dict]:
    rows = []
    for split in ("train", "validation", "test"):
        path = RAW_ROOT / "lince_lid_hineng" / f"{split}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        for rec in df.to_dict("records"):
            words = [str(w) for w in rec["words"]]
            lids = [str(x) for x in rec["lid"]]
            if not lids or all(x == "" for x in lids):
                continue
            text = " ".join(words)
            if has_devanagari(text):
                continue
            cw, ct, cp = clean_gold_sequence(words, lids)
            if not cw:
                continue
            rows.append(
                {
                    "source_id": "lince",
                    "source_item_id": f"lid_{split}_{rec.get('idx', len(rows))}",
                    "text": " ".join(cw),
                    "tokens": cw,
                    "lang_tags": ct,
                    "pos_tags": cp,
                    "tag_origin": "gold",
                    "offensive_label": False,
                }
            )
    return rows


def parse_lince_pos() -> list[dict]:
    rows = []
    for split in ("train", "validation", "test"):
        path = RAW_ROOT / "lince_pos_hineng" / f"{split}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        for rec in df.to_dict("records"):
            words = [str(w) for w in rec["words"]]
            lids = [str(x) for x in rec["lid"]]
            pos = [str(x) for x in rec["pos"]]
            if not lids or all(x == "" for x in lids):
                continue
            text = " ".join(words)
            if has_devanagari(text):
                continue
            cw, ct, cp = clean_gold_sequence(words, lids, pos)
            if not cw:
                continue
            rows.append(
                {
                    "source_id": "lince_pos",
                    "source_item_id": f"pos_{split}_{rec.get('idx', len(rows))}",
                    "text": " ".join(cw),
                    "tokens": cw,
                    "lang_tags": ct,
                    "pos_tags": cp,
                    "tag_origin": "gold",
                    "offensive_label": False,
                }
            )
    return rows


def parse_sentimix_conll(path: Path, split: str) -> list[dict]:
    rows = []
    text = path.read_text(encoding="utf-8", errors="replace")
    current_id = None
    current_sa = None
    words: list[str] = []
    tags: list[str] = []

    def flush() -> None:
        nonlocal current_id, current_sa, words, tags
        if not words:
            current_id = None
            current_sa = None
            return
        joined = " ".join(words)
        if not has_devanagari(joined):
            cw, ct, cp = clean_gold_sequence(words, tags)
            if cw:
                rows.append(
                    {
                        "source_id": "sentimix",
                        "source_item_id": f"{split}_{current_id or len(rows)}",
                        "text": " ".join(cw),
                        "tokens": cw,
                        "lang_tags": ct,
                        "pos_tags": cp,
                        "tag_origin": "gold",
                        "offensive_label": False,
                    }
                )
        current_id = None
        current_sa = None
        words = []
        tags = []

    for line in text.splitlines():
        line = line.rstrip("\n")
        if not line.strip():
            flush()
            continue
        parts = line.split("\t") if "\t" in line else line.split()
        if parts and parts[0] == "meta":
            flush()
            current_id = parts[1] if len(parts) > 1 else None
            current_sa = parts[2] if len(parts) > 2 else None
            continue
        if len(parts) == 1:
            words.append(parts[0])
            tags.append("O")
        else:
            words.append(parts[0])
            tags.append(parts[1])
    flush()
    return rows


def parse_sentimix() -> list[dict]:
    base = RAW_ROOT / "sentimix" / "extracted" / "Semeval_2020_task9_data" / "Hinglish"
    if not base.exists():
        return []
    rows = []
    mapping = {
        "train": "Hinglish_train_14k_split_conll.txt",
        "dev": "Hinglish_dev_3k_split_conll.txt",
        "test": "Hinglish_test_unalbelled_conll_updated.txt",
    }
    for split, fname in mapping.items():
        path = base / fname
        if path.exists():
            rows.extend(parse_sentimix_conll(path, split))
    return rows


def parse_phinc(hindi_list: set[str]) -> list[dict]:
    path = RAW_ROOT / "phinc" / "PHINC.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    col = "Sentence" if "Sentence" in df.columns else df.columns[0]
    rows = []
    for i, rec in enumerate(df.to_dict("records")):
        text = "" if pd.isna(rec[col]) else str(rec[col])
        if not text.strip() or has_devanagari(text):
            continue
        tokens = tokenize_untagged(text)
        if not tokens:
            continue
        tags = auto_tag_tokens(tokens, hindi_list)
        rows.append(
            {
                "source_id": "phinc",
                "source_item_id": f"phinc_{i}",
                "text": " ".join(tokens),
                "tokens": tokens,
                "lang_tags": tags,
                "pos_tags": [None] * len(tokens),
                "tag_origin": "auto",
                "offensive_label": False,
            }
        )
    return rows


def parse_hinge(hindi_list: set[str]) -> list[dict]:
    path = RAW_ROOT / "hinge" / "HinGE.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    col = "Human-generated Hinglish"
    if col not in df.columns:
        return []
    rows = []
    uid = 0
    for i, rec in enumerate(df.to_dict("records")):
        raw = rec[col]
        if pd.isna(raw):
            continue
        try:
            sents = ast.literal_eval(str(raw))
        except (ValueError, SyntaxError):
            sents = [str(raw)]
        if isinstance(sents, str):
            sents = [sents]
        for j, sent in enumerate(sents):
            text = str(sent).strip()
            if not text or has_devanagari(text):
                continue
            tokens = tokenize_untagged(text)
            if not tokens:
                continue
            tags = auto_tag_tokens(tokens, hindi_list)
            rows.append(
                {
                    "source_id": "hinge",
                    "source_item_id": f"hinge_{i}_{j}",
                    "text": " ".join(tokens),
                    "tokens": tokens,
                    "lang_tags": tags,
                    "pos_tags": [None] * len(tokens),
                    "tag_origin": "auto",
                    "offensive_label": False,
                }
            )
            uid += 1
    return rows


def build_hindi_list(gold_rows: list[dict]) -> Counter:
    counts: Counter = Counter()
    for row in gold_rows:
        if row["source_id"] not in {"lince", "lince_pos"}:
            continue
        for tok, tag in zip(row["tokens"], row["lang_tags"]):
            if tag == "hi" and WORD_RE.match(tok):
                counts[tok.lower()] += 1
    return Counter({w: c for w, c in counts.items() if c >= 20})


def contains_blocklist(tokens: list[str]) -> bool:
    for tok in tokens:
        if tok.lower() in BLOCKLIST:
            return True
    return False


def markers_hit(tokens: list[str]) -> list[str]:
    low = [t.lower() for t in tokens]
    hit = []
    bag = set(low)
    for m in SERVICE_MARKERS:
        if m in bag:
            hit.append(m)
    for i in range(len(low) - 1):
        if (low[i], low[i + 1]) == THANK_YOU:
            hit.append("thank you")
            break
    return hit


def fivegrams(tokens: list[str]) -> set[tuple[str, ...]]:
    low = [t.lower() for t in tokens]
    if len(low) < 5:
        return set()
    return {tuple(low[i : i + 5]) for i in range(len(low) - 4)}


def compute_profile(rows: list[dict]) -> dict:
    shares = []
    cmis = []
    switches = []
    spans = []
    n_mono = 0
    n_with_den = 0
    en_in_hi = Counter()
    hi_func = Counter()
    mixed = Counter()
    switch_bg = Counter()
    switch_bg_pos: dict[tuple, Counter] = defaultdict(Counter)
    address = Counter({k: 0 for k in ADDRESS_FORMS + ["thank you"]})
    cluster_forms: dict[str, Counter] = defaultdict(Counter)
    n_tokens = 0
    n_gold_tokens = 0
    tag_counts = Counter()

    for row in rows:
        tags = row["lang_tags"]
        tokens = row["tokens"]
        pos = row.get("pos_tags") or [None] * len(tokens)
        n_tokens += len(tokens)
        if row["tag_origin"] == "gold":
            n_gold_tokens += len(tokens)
        tag_counts.update(tags)
        m = utterance_metrics(tags)
        if m["english_share"] is not None:
            shares.append(m["english_share"])
            cmis.append(m["cmi"])
            n_with_den += 1
            if m["monolingual"]:
                n_mono += 1
        switches.append(m["n_switches"])
        spans.extend(m["span_lengths"])

        if m["english_share"] is not None and m["english_share"] < 0.5:
            for tok, tag in zip(tokens, tags):
                if tag == "en" and WORD_RE.match(tok):
                    en_in_hi[tok.lower()] += 1

        for tok, tag in zip(tokens, tags):
            low = tok.lower()
            if tag == "hi" and WORD_RE.match(tok) and (low in FUNCTION_SEED or low in LIGHT_VERBS):
                hi_func[low] += 1
            if WORD_RE.match(tok):
                key = skeleton_key(tok)
                if key:
                    cluster_forms[key][low] += 1

        for i in range(len(tokens) - 1):
            t0, t1 = tokens[i], tokens[i + 1]
            g0, g1 = tags[i], tags[i + 1]
            if g0 == "en" and WORD_RE.match(t0):
                low1 = t1.lower()
                is_lv = low1 in LIGHT_VERBS or (
                    low1 in LIGHT_VERB_VARIANTS and g1 != "en"
                )
                if is_lv:
                    mixed[f"{t0} {t1}"] += 1
            if g0 in {"en", "hi"} and g1 in {"en", "hi"} and g0 != g1:
                pair = (t0.lower(), t1.lower(), f"{g0}->{g1}")
                switch_bg[pair] += 1
                p0, p1 = pos[i], pos[i + 1]
                if p0 and p1:
                    switch_bg_pos[pair][f"{p0}->{p1}"] += 1

        low_tokens = [t.lower() for t in tokens]
        bag = Counter(low_tokens)
        for form in ADDRESS_FORMS:
            address[form] += bag.get(form, 0)
        for a, b in zip(low_tokens, low_tokens[1:]):
            if (a, b) == THANK_YOU:
                address["thank you"] += 1

    clusters = []
    for key, forms in cluster_forms.items():
        if len(forms) < 2:
            continue
        clusters.append(
            {
                "skeleton": key,
                "variants": [
                    {"form": f, "count": c} for f, c in forms.most_common()
                ],
                "n_variants": len(forms),
                "total": int(sum(forms.values())),
            }
        )
    clusters.sort(key=lambda x: (-x["total"], -x["n_variants"], x["skeleton"]))

    switch_list = []
    for (prev, nxt, direction), count in switch_bg.most_common(100):
        pos_pat = None
        if switch_bg_pos[(prev, nxt, direction)]:
            pos_pat = switch_bg_pos[(prev, nxt, direction)].most_common(1)[0][0]
        switch_list.append(
            {
                "prev": prev,
                "next": nxt,
                "direction": direction,
                "count": int(count),
                "pos_pattern": pos_pat,
            }
        )

    return {
        "n_utterances": len(rows),
        "n_tokens": n_tokens,
        "n_gold_tokens": n_gold_tokens,
        "tag_counts": dict(tag_counts),
        "english_share": quantiles(shares),
        "cmi": quantiles(cmis),
        "switches": quantiles([float(x) for x in switches]),
        "mean_same_language_span_length": (
            sum(spans) / len(spans) if spans else None
        ),
        "monolingual_utterance_share": (n_mono / n_with_den) if n_with_den else None,
        "n_utterances_with_en_hi": n_with_den,
        "top_english_in_hindi_utts": [
            {"word": w, "count": int(c)} for w, c in en_in_hi.most_common(300)
        ],
        "top_hindi_function_words": [
            {"word": w, "count": int(c)} for w, c in hi_func.most_common(150)
        ],
        "mixed_en_light_verb": [
            {"pattern": p, "count": int(c)} for p, c in mixed.most_common(200)
        ],
        "switch_bigrams": switch_list,
        "address_politeness": {k: int(v) for k, v in address.items()},
        "romanization_clusters": clusters[:300],
    }


def select_service_candidates(rows: list[dict], license_status: dict, limit: int = 400) -> list[dict]:
    scored = []
    for row in rows:
        toks = row["tokens"]
        if not (5 <= len(toks) <= 30):
            continue
        m = utterance_metrics(row["lang_tags"])
        if m["cmi"] is None or m["cmi"] <= 0:
            continue
        if row.get("offensive_label") or contains_blocklist(toks):
            continue
        hits = markers_hit(toks)
        if not hits:
            continue
        scored.append((len(hits), m["cmi"], m["n_switches"], row, hits, m))
    scored.sort(key=lambda x: (-x[0], -x[1], -x[2], x[3]["text"]))

    selected = []
    used_grams: set[tuple[str, ...]] = set()
    for _n_hit, _cmi, _nsw, row, hits, m in scored:
        grams = fivegrams(row["tokens"])
        if grams & used_grams:
            continue
        used_grams |= grams
        selected.append(
            {
                "cand_id": f"hi_en_c{len(selected)+1:03d}",
                "source_id": row["source_id"],
                "source_item_id": row["source_item_id"],
                "text": row["text"],
                "english_share": None if m["english_share"] is None else round(m["english_share"], 4),
                "cmi": None if m["cmi"] is None else round(m["cmi"], 4),
                "n_switches": m["n_switches"],
                "markers_hit": "|".join(hits),
                "license_status": license_status.get(row["source_id"], "UNCLEAR"),
                "accept": "",
                "reject_reason": "",
                "reviewer_note": "",
                "_tokens": row["tokens"],
            }
        )
        if len(selected) >= limit:
            break
    return selected


def corrupt_replace(tokens: list[str], rng: random.Random) -> str:
    if not tokens:
        return ""
    n = max(1, int(round(0.3 * len(tokens))))
    idxs = [i for i, t in enumerate(tokens) if WORD_RE.match(t)]
    if not idxs:
        return " ".join(tokens)
    rng.shuffle(idxs)
    out = list(tokens)
    replaced = 0
    for i in idxs:
        gloss = HI_EN_GLOSS.get(out[i].lower())
        if gloss:
            out[i] = gloss
            replaced += 1
            if replaced >= n:
                break
    if replaced < n:
        dummy = ["the", "is", "please", "item", "service"]
        for i in idxs:
            if out[i] == tokens[i] and WORD_RE.match(out[i]):
                out[i] = rng.choice(dummy)
                replaced += 1
                if replaced >= n:
                    break
    return " ".join(out)


def corrupt_insert_en(tokens: list[str], rng: random.Random) -> str | None:
    sites = []
    for i in range(len(tokens) - 1):
        if WORD_RE.match(tokens[i]) and tokens[i + 1].lower() in POSTPOSITIONS:
            sites.append(i)
    if not sites:
        return None
    i = rng.choice(sites)
    word = rng.choice(INSERT_EN)
    out = tokens[: i + 1] + [word] + tokens[i + 1 :]
    return " ".join(out)


def corrupt_shuffle(tokens: list[str], rng: random.Random) -> str:
    out = list(tokens)
    rng.shuffle(out)
    return " ".join(out)


def _has_postp_site(tokens: list[str]) -> bool:
    return any(
        WORD_RE.match(tokens[i]) and tokens[i + 1].lower() in POSTPOSITIONS
        for i in range(len(tokens) - 1)
    )


def make_negatives(candidates: list[dict], n: int = 100) -> list[dict]:
    rng = random.Random(SEED)
    if not candidates:
        return []
    with_postp = [c for c in candidates if _has_postp_site(c["_tokens"])]
    plan = (
        ["dict_replace_30pct"] * 34
        + ["en_between_noun_postp"] * 33
        + ["shuffle_order"] * 33
    )
    rng.shuffle(plan)
    negs = []
    for ctype in plan:
        text = None
        cand = None
        for _ in range(40):
            pool = with_postp if ctype == "en_between_noun_postp" and with_postp else candidates
            cand = rng.choice(pool)
            tokens = cand["_tokens"]
            if ctype == "dict_replace_30pct":
                text = corrupt_replace(tokens, rng)
            elif ctype == "en_between_noun_postp":
                text = corrupt_insert_en(tokens, rng)
            else:
                text = corrupt_shuffle(tokens, rng)
            if text and text != cand["text"]:
                break
            text = None
        if text is None or cand is None:
            continue
        negs.append(
            {
                "neg_id": f"hi_en_neg_{len(negs)+1:03d}",
                "from_cand_id": cand["cand_id"],
                "corruption_type": ctype,
                "text": text,
            }
        )
        if len(negs) >= n:
            break
    return negs


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def write_license_files(source_status: dict, counts: dict) -> None:
    lince_text = f"""# lince (Hindi-English LID + POS)

- **source_id:** `lince`, `lince_pos`
- **URL:** https://huggingface.co/datasets/lince-benchmark/lince (configs `lid_hineng`, `pos_hineng`); official site https://ritual.uh.edu/lince/ (login-walled; site unreachable at harvest). Data taken from the Hugging Face `refs/convert/parquet` conversion.
- **version:** LinCE benchmark 1.0.0 / HF parquet conversion harvested 2026-09-25. LID splits used: train 4823 + validation 744 (test 1854 has empty LID labels and was skipped). POS splits used: train 1030 + validation 160 + test 299 (test POS labels empty; LID kept).
- **size:** raw parquet {dir_size(RAW_ROOT / 'lince_lid_hineng') + dir_size(RAW_ROOT / 'lince_pos_hineng')} bytes; processed utterances: LID {counts.get('lince', 0)}, POS {counts.get('lince_pos', 0)}.
- **underlying corpora:** LID = Mave, Maharjan & Solorio 2018 (Facebook/Twitter + ICON 2016 reuse). POS = Singh, Sen & Kumaraguru 2018 (Hindi-English Twitter POS).
- **stated terms:** Hugging Face dataset card has no license field ("Licensing Information: More Information Needed"). The official LinCE datasets page states that Twitter-derived data is **distributed for non-commercial and for research purposes only, following Twitter's own Developer Agreement and Policy**.
- **redistribution:** **NOT allowed** for raw tweet text (Twitter Developer Agreement + official non-commercial research restriction). **UNCLEAR - stats only, do not redistribute** the raw utterances outside this local calibration folder.
- **citation:**

```
@inproceedings{{aguilar-etal-2020-lince,
  title = "{{L}}in{{CE}}: A Centralized Benchmark for Linguistic Code-switching Evaluation",
  author = "Aguilar, Gustavo and Kar, Sudipta and Solorio, Thamar",
  booktitle = "Proceedings of The 12th Language Resources and Evaluation Conference",
  year = 2020,
  pages = "1803--1813"
}}

@inproceedings{{mave-etal-2018-language,
  title = "Language Identification and Analysis of Code-Switched Social Media Text",
  author = "Mave, Deepthi and Maharjan, Suraj and Solorio, Thamar",
  booktitle = "Proceedings of the Third Workshop on Computational Approaches to Linguistic Code-Switching",
  year = 2018,
  pages = "51--61"
}}

@inproceedings{{singh-etal-2018-twitter,
  title = "A Twitter Corpus for {{H}}indi-{{E}}nglish Code Mixed {{POS}} Tagging",
  author = "Singh, Kushagra and Sen, Indira and Kumaraguru, Ponnurangam",
  booktitle = "Proceedings of the Sixth International Workshop on Natural Language Processing for Social Media",
  year = 2018,
  pages = "12--17"
}}
```

Download status: {source_status.get('lince', {})}
"""
    (LIC_ROOT / "lince.md").write_text(lince_text, encoding="utf-8")

    phinc_text = f"""# phinc

- **source_id:** `phinc`
- **URL:** https://huggingface.co/datasets/LingoIITGN/PHINC ; original https://doi.org/10.5281/zenodo.3605597
- **version:** Zenodo record 3605597 (published 2020-01-12); HF mirror `LingoIITGN/PHINC` (13,738 pairs). Harvested 2026-09-25.
- **size:** raw CSV {dir_size(RAW_ROOT / 'phinc')} bytes; {counts.get('phinc', 0)} processed Roman-script utterances (Hinglish `Sentence` column only; English translations unused).
- **stated terms (CC BY 4.0):** Hugging Face and the PHINC paper state Creative Commons Attribution 4.0 International. Paper: "The data is released with a clear and accessible data usage license" pointing to https://creativecommons.org/licenses/by/4.0/legalcode . Zenodo record license: Creative Commons Attribution 4.0 International.
- **CC BY 4.0 summary:** You are free to share and adapt the material for any purpose, even commercially, if you give appropriate credit, provide a link to the license, and indicate if changes were made. No additional restrictions.
- **redistribution:** **Allowed** with attribution under CC BY 4.0.
- **citation:**

```
@inproceedings{{srivastava-singh-2020-phinc,
  title = "{{PHINC}}: A Parallel {{H}}inglish Social Media Code-Mixed Corpus for Machine Translation",
  author = "Srivastava, Vivek and Singh, Mayank",
  booktitle = "Proceedings of the Sixth Workshop on Noisy User-generated Text (W-NUT 2020)",
  year = 2020,
  pages = "41--49"
}}
```

Download status: {source_status.get('phinc', {})}
"""
    (LIC_ROOT / "phinc.md").write_text(phinc_text, encoding="utf-8")

    hinge_text = f"""# hinge

- **source_id:** `hinge`
- **URL:** https://huggingface.co/datasets/LingoIITGN/HinGE ; paper https://aclanthology.org/2021.eval4nlp-1.20/ ; author page https://sites.google.com/view/vivek-srivastava/resources
- **version:** Srivastava & Singh 2021; HF card license `cc-by-4.0`; 1,976 source pairs. Harvested 2026-09-25.
- **size:** raw CSV {dir_size(RAW_ROOT / 'hinge')} bytes; {counts.get('hinge', 0)} processed human-written Hinglish utterances.
- **human vs synthetic:** Only the `Human-generated Hinglish` column is used. WAC and PAC columns are rule-based algorithm output and were **dropped**.
- **stated terms:** Hugging Face dataset card: `license: cc-by-4.0` / "Licensed by: cc-by-4.0".
- **CC BY 4.0 summary:** You are free to share and adapt for any purpose, even commercially, with attribution and indication of changes.
- **redistribution:** **Allowed** with attribution under CC BY 4.0. Note: source Hindi/English sentences come from the IIT Bombay parallel corpus, which has its own terms; we store only the human Hinglish strings.
- **citation:**

```
@inproceedings{{srivastava-singh-2021-hinge,
  title = "{{H}}in{{GE}}: A Dataset for Generation and Evaluation of Code-Mixed {{H}}inglish Text",
  author = "Srivastava, Vivek and Singh, Mayank",
  booktitle = "Proceedings of the 2nd Workshop on Evaluation and Comparison of NLP Systems",
  year = 2021,
  pages = "200--208"
}}
```

Download status: {source_status.get('hinge', {})}
"""
    (LIC_ROOT / "hinge.md").write_text(hinge_text, encoding="utf-8")

    sm_text = f"""# sentimix (SemEval-2020 Task 9 Hinglish)

- **source_id:** `sentimix`
- **URL:** https://doi.org/10.5281/zenodo.3974927 ; task site https://ritual-uh.github.io/sentimix2020/ ; paper https://aclanthology.org/2020.semeval-1.100/ . HF `lince` config `sa_hineng` is **not** present on `lince-benchmark/lince` (only `sa_spaeng`); official Zenodo release used instead.
- **version:** Zenodo V1, 2020-08-06, DOI 10.5281/zenodo.3974927. File `Semeval_2020_task9_data.zip` (md5 4d93553c213b3a966b6d55b7a7c03be7), 2.8 MB.
- **size:** raw zip {dir_size(RAW_ROOT / 'sentimix')} bytes; {counts.get('sentimix', 0)} processed utterances (train 14k + dev 3k + test with word-level LID).
- **stated terms:** Zenodo metadata `license.id = cc-by-4.0`. Task paper: "This work is licensed under a Creative Commons Attribution 4.0 International License" (http://creativecommons.org/licenses/by/4.0/).
- **CC BY 4.0 summary:** Share and adapt with attribution. Zenodo: "The Creative Commons Attribution license allows re-distribution and re-use of a licensed work on the condition that the creator is appropriately credited."
- **redistribution:** **Allowed** under the stated CC BY 4.0 terms (with attribution). Content is Twitter-derived; Twitter's Developer Policy may still constrain public tweet-text redistribution even when organizers used CC BY 4.0.
- **citation:**

```
@inproceedings{{patwa-etal-2020-semeval,
  title = "{{S}}em{{E}}val-2020 Task 9: Overview of Sentiment Analysis of Code-Mixed Tweets",
  author = "Patwa, Parth and Aguilar, Gustavo and Kar, Sudipta and Pandey, Suraj and PYKL, Srinivas and Gamb{{\\"a}}ck, Bj{{\\"o}}rn and Chakraborty, Tanmoy and Solorio, Thamar and Das, Amitava",
  booktitle = "Proceedings of the Fourteenth Workshop on Semantic Evaluation",
  year = 2020,
  pages = "774--790"
}}
```

Download status: {source_status.get('sentimix', {})}
"""
    (LIC_ROOT / "sentimix.md").write_text(sm_text, encoding="utf-8")


def fmt_q(q: dict) -> str:
    if not q or q.get("n") in (None, 0):
        return "n/a"
    def f(x):
        return "n/a" if x is None else f"{x:.3f}"
    return (
        f"n={q['n']}; p10={f(q['p10'])}, p25={f(q['p25'])}, p50={f(q['p50'])}, "
        f"p75={f(q['p75'])}, p90={f(q['p90'])}, mean={f(q['mean'])}"
    )


def write_report(
    source_status: dict,
    counts: dict,
    gold_stats: dict,
    all_stats: dict,
    hindi_list: Counter,
    candidates: list[dict],
    problems: list[str],
) -> None:
    mixed = all_stats.get("mixed_en_light_verb", [])[:10]
    clusters = all_stats.get("romanization_clusters", [])[:10]
    cands = candidates[:10]
    lines = [
        "# Hindi-English (Romanized Hinglish) calibration profile",
        "",
        f"Pair ID: `{PAIR_ID}`. Built by `scripts/calibration/mine_hi_en.py` (seed {SEED}).",
        "Human-written corpora only. HinGE WAC/PAC rule-based sentences were not used.",
        "",
        "## Sources",
        "",
        "| source_id | obtained | license / redistribution | processed utts |",
        "|---|---|---|---:|",
        f"| lince (+ lince_pos) | {'yes' if source_status.get('lince', {}).get('ok') else 'NO'} | restricted Twitter research; do not redistribute raw tweets | {counts.get('lince', 0) + counts.get('lince_pos', 0)} |",
        f"| phinc | {'yes' if source_status.get('phinc', {}).get('ok') else 'NO'} | CC BY 4.0; redistribution allowed with attribution | {counts.get('phinc', 0)} |",
        f"| hinge | {'yes' if source_status.get('hinge', {}).get('ok') else 'NO'} | CC BY 4.0; human column only | {counts.get('hinge', 0)} |",
        f"| sentimix | {'yes' if source_status.get('sentimix', {}).get('ok') else 'NO'} | CC BY 4.0 (Zenodo); Twitter-derived | {counts.get('sentimix', 0)} |",
        "",
        "License notes are in `data/calibration/licenses/`.",
        "",
        "## Counts",
        "",
        f"- All utterances: **{all_stats['n_utterances']}**",
        f"- Gold-tagged utterances: **{gold_stats['n_utterances']}**",
        f"- All tokens: **{all_stats['n_tokens']}**",
        f"- Gold-tagged tokens: **{gold_stats['n_gold_tokens']}**",
        f"- LinCE-derived Hindi romanized function-word list (hi, freq≥20): **{len(hindi_list)}** types",
        "",
        "## Quantiles",
        "",
        "### Gold-tagged",
        "",
        f"- english_share: {fmt_q(gold_stats['english_share'])}",
        f"- CMI: {fmt_q(gold_stats['cmi'])}",
        f"- switches / utterance: {fmt_q(gold_stats['switches'])}",
        f"- monolingual share: {gold_stats['monolingual_utterance_share']}",
        f"- mean same-language span length: {gold_stats['mean_same_language_span_length']}",
        "",
        "### All data",
        "",
        f"- english_share: {fmt_q(all_stats['english_share'])}",
        f"- CMI: {fmt_q(all_stats['cmi'])}",
        f"- switches / utterance: {fmt_q(all_stats['switches'])}",
        f"- monolingual share: {all_stats['monolingual_utterance_share']}",
        f"- mean same-language span length: {all_stats['mean_same_language_span_length']}",
        "",
        "## Example service candidates",
        "",
    ]
    for c in cands:
        lines.append(
            f"- `{c['cand_id']}` ({c['source_id']}, CMI={c['cmi']}, markers={c['markers_hit']}): {c['text']}"
        )
    lines += [
        "",
        f"Total service candidates written: {len(candidates)}",
        "",
        "## Top mixed EN + Hindi light-verb patterns",
        "",
    ]
    for item in mixed:
        lines.append(f"- {item['pattern']} ({item['count']})")
    lines += ["", "## Top romanization clusters", ""]
    for cl in clusters:
        variants = ", ".join(f"{v['form']}={v['count']}" for v in cl["variants"][:8])
        lines.append(f"- `{cl['skeleton']}` (n={cl['total']}): {variants}")
    lines += ["", "## Address / politeness (all data)", ""]
    for k, v in all_stats["address_politeness"].items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Problems / caveats", ""]
    if problems:
        for p in problems:
            lines.append(f"- {p}")
    else:
        lines.append("- None recorded.")
    lines.append("")
    (PROC_ROOT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def write_utterances(rows: list[dict]) -> None:
    path = PROC_ROOT / "utterances.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(rows, start=1):
            rec = {
                "id": f"hi_en_u{i:06d}",
                "source_id": row["source_id"],
                "text": row["text"],
                "tokens": row["tokens"],
                "lang_tags": row["lang_tags"],
                "tag_origin": row["tag_origin"],
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_candidates_csv(candidates: list[dict]) -> None:
    path = PROC_ROOT / "service_candidates.csv"
    fields = [
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
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in candidates:
            w.writerow({k: row.get(k, "") for k in fields})


def write_negatives_csv(negs: list[dict]) -> None:
    path = PROC_ROOT / "unnatural_negatives.csv"
    fields = ["neg_id", "from_cand_id", "corruption_type", "text"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in negs:
            w.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    if args.skip_download:
        source_status = {
            "lince": {"ok": (RAW_ROOT / "lince_lid_hineng" / "train.parquet").exists()},
            "phinc": {"ok": (RAW_ROOT / "phinc" / "PHINC.csv").exists()},
            "hinge": {"ok": (RAW_ROOT / "hinge" / "HinGE.csv").exists()},
            "sentimix": {"ok": (RAW_ROOT / "sentimix" / "Semeval_2020_task9_data.zip").exists()},
        }
    else:
        source_status = download_sources(force=args.force_download)

    problems = []
    problems.append(
        "LinCE official site ritual.uh.edu timed out; HF loading scripts are "
        "deprecated in datasets 5.x. Used Hugging Face refs/convert/parquet "
        "for lid_hineng and pos_hineng. lid_hineng test labels are empty and "
        "were skipped. HF lince has no sa_hineng config; SentiMix came from Zenodo. "
        "SentiMix gold often tags frequent English words (the/are/a) as Hin, "
        "so some switch-bigram ranks reflect label noise."
    )
    if not source_status.get("lince", {}).get("ok"):
        problems.append("LinCE parquet download failed.")
    for sid, st in source_status.items():
        if not st.get("ok"):
            problems.append(f"Source {sid} failed to download.")

    log("Parsing gold-tagged LinCE + SentiMix...")
    gold_rows = []
    if source_status.get("lince", {}).get("ok"):
        gold_rows.extend(parse_lince_lid())
        gold_rows.extend(parse_lince_pos())
    if source_status.get("sentimix", {}).get("ok"):
        gold_rows.extend(parse_sentimix())

    hindi_counter = build_hindi_list(gold_rows)
    hindi_list = set(hindi_counter)
    if not hindi_list:
        problems.append(
            "LinCE Hindi list empty or LinCE missing; auto-tag Hindi list "
            "falls back to FUNCTION_SEED only."
        )
        hindi_list = set(FUNCTION_SEED)
        hindi_counter = Counter({w: 0 for w in FUNCTION_SEED})

    log(f"Hindi romanized list from LinCE gold (freq>=20): {len(hindi_list)} types")

    auto_rows = []
    if source_status.get("phinc", {}).get("ok"):
        log("Auto-tagging PHINC...")
        auto_rows.extend(parse_phinc(hindi_list))
    if source_status.get("hinge", {}).get("ok"):
        log("Auto-tagging HinGE human sentences...")
        auto_rows.extend(parse_hinge(hindi_list))

    all_rows = gold_rows + auto_rows
    counts = Counter(r["source_id"] for r in all_rows)
    log(f"Utterances by source: {dict(counts)}")

    log("Computing statistics...")
    gold_stats = compute_profile(gold_rows)
    all_stats = compute_profile(all_rows)

    license_status = {
        "lince": "restricted_twitter_research",
        "lince_pos": "restricted_twitter_research",
        "phinc": "cc-by-4.0",
        "hinge": "cc-by-4.0",
        "sentimix": "cc-by-4.0",
    }
    log("Selecting service-like candidates...")
    candidates = select_service_candidates(all_rows, license_status, limit=400)
    log(f"Service candidates: {len(candidates)}")
    if len(candidates) < 50:
        problems.append(f"Only {len(candidates)} service-like candidates passed filters.")

    log("Writing synthetic negatives...")
    negatives = make_negatives(candidates, n=100)
    if len(negatives) < 100:
        problems.append(f"Only {len(negatives)} synthetic negatives produced.")

    write_utterances(all_rows)
    write_candidates_csv(candidates)
    write_negatives_csv(negatives)
    write_license_files(source_status, dict(counts))

    profile = {
        "pair_id": PAIR_ID,
        "seed": SEED,
        "sources": source_status,
        "utterance_counts_by_source": dict(counts),
        "hindi_list_from_lince": {
            "rule": "tokens tagged hi in LinCE lid/pos gold, alphabetic, freq>=20",
            "n_types": len(hindi_counter),
            "items": [{"word": w, "count": int(c)} for w, c in hindi_counter.most_common()],
        },
        "gold": gold_stats,
        "all": all_stats,
        "n_service_candidates": len(candidates),
        "n_unnatural_negatives": len(negatives),
        "problems": problems,
    }
    (PROC_ROOT / "profile_stats.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (PROC_ROOT / "hindi_function_words.json").write_text(
        json.dumps(
            [{"word": w, "count": int(c)} for w, c in hindi_counter.most_common()],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(source_status, dict(counts), gold_stats, all_stats, hindi_counter, candidates, problems)
    log(f"Wrote outputs under {PROC_ROOT} and {LIC_ROOT}")


if __name__ == "__main__":
    main()
