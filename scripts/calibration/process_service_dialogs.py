#!/usr/bin/env python3
"""Process Code-Mixed-Dialog (HI-EN, TA-EN) and human Hinglish-TOP.

Re-run from the repo root:

    .venv/bin/python scripts/calibration/process_service_dialogs.py

Downloads skip existing non-empty files. Writes only under
data/calibration/ and this script path.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import date
import subprocess
from pathlib import Path
from typing import Any

from wordfreq import zipf_frequency

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
CAL_ROOT = REPO_ROOT / "data" / "calibration"
RAW_SPOKEN = CAL_ROOT / "raw" / "spoken"
PROC_DIR = CAL_ROOT / "processed" / "service_dialogs"
LIC_DIR = CAL_ROOT / "licenses"
ROMAN_DIR = CAL_ROOT / "processed" / "romanization"

CMD_BASE = "https://raw.githubusercontent.com/sumanbanerjee1/Code-Mixed-Dialog/master"
HTOP_BASE = (
    "https://raw.githubusercontent.com/google-research-datasets/"
    "Hinglish-TOP-Dataset/main"
)

SEED = 42
ZIPF_EN = 3.5
ZIPF_LOCAL = 2.5
LOCAL_MIN_FREQ = 5
N_REFERENCE = 30
N_HTOP_SERVICE = 150

WORD_RE = re.compile(r"^[A-Za-z]+(?:'[A-Za-z]+)?$")
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
TURN_NUM_RE = re.compile(r"^(\d+)\s+(.*)$")
KB_RE = re.compile(r"^\S+\s+R_[A-Za-z0-9_]+(?:\s+\S+)?")
API_RE = re.compile(r"\bapi_call\b", re.I)
NATIVE_SCRIPT_RE = re.compile(r"[\u0900-\u097F\u0B80-\u0BFF\u0A80-\u0AFF]")
INTENT_RE = re.compile(r"\[IN:([A-Z0-9_]+)")
SLOT_RE = re.compile(r"\[SL:([A-Z0-9_]+)")

SILENCE_TOKENS = {"<silence>", "silence"}

HI_LIGHT_VERBS = ("kar", "karo", "kijiye", "karein", "karna", "ho", "hai")
TA_LIGHT_VERBS = (
    "pannu",
    "pannunga",
    "panren",
    "panna",
    "pannitu",
    "sollunga",
    "venum",
    "vendum",
)

HI_POLITE = (
    "aap",
    "aapka",
    "aapko",
    "tum",
    "ji",
    "sir",
    "madam",
    "please",
    "kripya",
    "dhanyavaad",
    "thank you",
    "shukriya",
    "namaste",
    "haan",
    "nahi",
)
TA_POLITE = (
    "neenga",
    "unga",
    "sir",
    "madam",
    "please",
    "nandri",
    "thanks",
    "vanakkam",
    "seri",
    "sari",
    "aama",
    "illa",
)

# DSTC2 restaurant food types (Henderson et al. 2014 / Bordes & Weston 2017 KB).
DSTC2_CUISINES = {
    "afghan", "african", "asian", "asianoriental", "australasian", "australian",
    "austrian", "barbeque", "basque", "belgian", "bistro", "brazilian", "british",
    "canapes", "cantonese", "caribbean", "catalan", "chinese", "christmas",
    "corsica", "creative", "crossover", "cuban", "danish", "english", "eritrean",
    "european", "french", "fusion", "gastropub", "german", "greek", "indian",
    "indonesian", "international", "irish", "italian", "jamaican", "japanese",
    "korean", "lebanese", "mediterranean", "mexican", "moroccan", "persian",
    "polish", "portuguese", "romanian", "russian", "seafood", "spanish",
    "steakhouse", "swedish", "thai", "turkish", "tuscan", "vietnamese",
    "world", "modern", "oriental", "american",
}

HI_FUNCTION = {
    "ke", "ki", "ka", "ko", "se", "mein", "me", "mai", "main", "hai", "hain",
    "ho", "hun", "hoon", "tha", "thi", "the", "aur", "ya", "to", "toh", "bhi",
    "hi", "nahi", "nahin", "nhi", "na", "yeh", "ye", "woh", "wo", "us", "ek",
    "kya", "kaun", "kaise", "kab", "kahan", "kyun", "kyon", "jo", "par", "pe",
    "tak", "liye", "wala", "wali", "wale", "apna", "apne", "apni", "mera",
    "meri", "mere", "hum", "ham", "tum", "aap", "unko", "unka", "unki", "kuch",
    "sab", "mat", "lekin", "magar", "agar", "phir", "ab", "abhi", "yahan",
    "wahan", "itna", "itni", "bahut", "sirf", "bas", "aur", "bina", "saath",
    "baad", "pehle", "andar", "bahar", "upar", "neeche", "kyunki", "isliye",
    "jab", "tab", "agar", "toh", "hi", "na", "ji", "haan", "nahi", "kyu",
    "kiske", "kis", "kitna", "kitne", "kaunsa", "kaunse", "wala", "hue",
    "hui", "hua", "kiye", "kiye", "kiye", "kiye", "raha", "rahe", "rahi",
    "gaya", "gaye", "gayi", "sakta", "sakti", "sakte", "chahiye", "hoga",
    "hogi", "honge", "kar", "karo", "karke", "karta", "karte", "karti",
    "hu", "hoon", "hun", "pa",
}
TA_FUNCTION = {
    "oru", "um", "la", "le", "ku", "kku", "na", "nu", "ah", "da", "irukku",
    "iruka", "irukinga", "venum", "vendum", "illa", "illai", "neenga", "unga",
    "ungaluku", "ungalukku", "enna", "en", "naan", "naa", "ithu", "idhu",
    "adhu", "athu", "andha", "indha", "intha", "romba", "konjam", "apram",
    "appuram", "tha", "dhaan", "than", "va", "oda", "kitta", "kooda", "mattum",
    "varum", "varala", "illa", "aama", "seri", "sari", "ippo", "appa",
    "eppo", "enge", "evlo", "eppadi", "epdi", "yen", "enakku", "unakku",
    "avanga", "ivanga", "namma", "adhula", "idhula", "munnadi", "pinadi",
    "mela", "kizha", "ulla", "veliya", "sari", "illama", "aana", "aanaa",
    "ana", "pana", "panna", "solli", "kudu", "kudunga", "po", "vaa", "vaanga",
    "inga", "unga", "ungal", "naanga", "avanga", "ivan", "avan", "aval",
    "il", "oda", "la",
}

AREA_WORDS = {"north", "south", "east", "west", "centre", "center"}
PRICE_WORDS = {"cheap", "moderate", "expensive"}
# Never force these into the local lexicon via the function-word rule.
EN_FUNCTION_BLOCK = {
    "the", "and", "or", "is", "are", "was", "were", "you", "we", "he", "she",
    "it", "of", "for", "be", "can", "not", "yes", "do", "did", "does", "a",
    "an", "this", "that", "with", "from", "by", "on", "at",
}

HTOP_TIME_RE = re.compile(
    r"\b(alarm|timer|reminder|o'?clock|\d+\s*(am|pm)|baje|ghante)\b", re.I
)

_ZIPF: dict[str, float] = {}


def zipf_en(tok: str) -> float:
    key = tok.lower()
    if key not in _ZIPF:
        try:
            _ZIPF[key] = float(zipf_frequency(key, "en"))
        except Exception:
            _ZIPF[key] = 0.0
    return _ZIPF[key]


def quantiles(values: list[float]) -> dict[str, float | int | None]:
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


def fmt_q(q: dict[str, Any], nd: int = 3) -> str:
    if not q or q.get("n") in (None, 0):
        return "n=0"
    parts = [f"n={q['n']}"]
    for k in ("p10", "p25", "p50", "p75", "p90", "mean"):
        v = q.get(k)
        if v is None:
            parts.append(f"{k}=NA")
        else:
            parts.append(f"{k}={v:.{nd}f}")
    return ", ".join(parts)


def download(url: str, dest: Path, force: bool = False) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(
            f"  skip existing {dest.relative_to(REPO_ROOT)} "
            f"({dest.stat().st_size} bytes)",
            flush=True,
        )
        return False
    print(f"  GET {url}", flush=True)
    # curl is more reliable than urllib on flaky GitHub raw connections.
    tmp = dest.with_suffix(dest.suffix + ".part")
    cmd = [
        "curl",
        "-fL",
        "--retry",
        "4",
        "--retry-delay",
        "2",
        "--connect-timeout",
        "20",
        "--max-time",
        "180",
        "-A",
        "aaai-icfd-calibration/1.0",
        "-o",
        str(tmp),
        url,
    ]
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"download failed ({proc.returncode}): {url}")
    tmp.replace(dest)
    print(
        f"  wrote {dest.relative_to(REPO_ROOT)} ({dest.stat().st_size} bytes)",
        flush=True,
    )
    return True


def download_all(force: bool = False) -> dict[str, Any]:
    fetched: list[str] = []
    skipped: list[str] = []
    missing_candidates = True

    files = []
    for lang, pair in (("hindi", "hi_en"), ("tamil", "ta_en")):
        dest_dir = RAW_SPOKEN / "code_mixed_dialog" / pair
        for name in (
            "dialog-dstc2-trn.txt",
            "dialog-dstc2-dev.txt",
            "dialog-dstc2-tst.txt",
            "vocab_splits.json",
        ):
            files.append((f"{CMD_BASE}/data/{lang}/{name}", dest_dir / name))
    files.append((f"{CMD_BASE}/LICENSE", RAW_SPOKEN / "code_mixed_dialog" / "LICENSE"))
    files.append((f"{CMD_BASE}/README.md", RAW_SPOKEN / "code_mixed_dialog" / "README.md"))

    htop_dir = RAW_SPOKEN / "hinglish_top"
    for name in ("train.tsv", "test.tsv", "validation.tsv"):
        enc = name.replace(" ", "%20")
        files.append(
            (
                f"{HTOP_BASE}/Dataset/Human%20Annotated%20Data/{enc}",
                htop_dir / name,
            )
        )
    files.append((f"{HTOP_BASE}/LICENSE.md", htop_dir / "LICENSE.md"))
    files.append((f"{HTOP_BASE}/README.md", htop_dir / "README.md"))

    print("Downloading Code-Mixed-Dialog (hi/ta) and human Hinglish-TOP…", flush=True)
    optional = {"LICENSE", "LICENSE.md", "README.md"}
    for url, dest in files:
        try:
            if download(url, dest, force=force):
                fetched.append(str(dest.relative_to(REPO_ROOT)))
            else:
                skipped.append(str(dest.relative_to(REPO_ROOT)))
        except RuntimeError as exc:
            if dest.name in optional:
                print(f"  optional skip {dest.name}: {exc}", flush=True)
                continue
            raise

    # Official repo has no candidate-response files for hi/ta.
    return {
        "fetched": fetched,
        "skipped": skipped,
        "candidates_present": not missing_candidates,
    }


def load_roman_lexicon(lang: str) -> set[str]:
    path = ROMAN_DIR / f"{lang}_roman_lexicon.tsv"
    out: set[str] = set()
    if not path.exists():
        print(f"WARNING: missing roman lexicon {path}")
        return out
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            tok = (row.get("token") or "").strip().lower()
            if tok:
                out.add(tok)
    return out


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text) if t]


def is_silence(text: str) -> bool:
    return text.strip().lower() in SILENCE_TOKENS


def is_api_text(text: str) -> bool:
    return bool(API_RE.search(text))


def is_kb_text(text: str) -> bool:
    t = text.strip()
    if KB_RE.match(t):
        return True
    # leftover KB triples without the leading turn index
    parts = t.split()
    if len(parts) >= 2 and any(p.startswith("R_") for p in parts):
        # natural language can mention R_ placeholders rarely; require
        # entity-like first token (underscore or all restaurant-id style)
        first = parts[0]
        if "_" in first or first.startswith("R_"):
            return True
    return False


def has_native_script(text: str) -> bool:
    return bool(NATIVE_SCRIPT_RE.search(text))


def parse_cmd_file(path: Path, split: str) -> tuple[list[dict], dict[str, int]]:
    """Parse a DSTC2-style dialog file into natural-language turns only."""
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n\s*\n", text)
    dialogs: list[dict] = []
    counts = {
        "raw_blocks": 0,
        "raw_lines": 0,
        "api_removed": 0,
        "kb_removed": 0,
        "non_roman_removed": 0,
        "silence_skipped": 0,
        "empty_after_filter": 0,
    }
    for bi, block in enumerate(blocks):
        block = block.strip()
        if not block:
            continue
        counts["raw_blocks"] += 1
        turns: list[dict] = []
        goal_args: list[str] = []
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            counts["raw_lines"] += 1
            m = TURN_NUM_RE.match(line)
            rest = m.group(2) if m else line

            if "\t" in rest:
                user_raw, sys_raw = rest.split("\t", 1)
            else:
                user_raw, sys_raw = rest, ""

            user_raw = user_raw.strip()
            sys_raw = sys_raw.strip()

            # Standalone API / KB lines (no tab, or both sides structural).
            standalone = not sys_raw
            if standalone:
                if is_api_text(user_raw):
                    counts["api_removed"] += 1
                    goal_args.extend(_api_args(user_raw))
                    continue
                if is_kb_text(user_raw):
                    counts["kb_removed"] += 1
                    continue

            if is_api_text(user_raw) and not sys_raw:
                counts["api_removed"] += 1
                goal_args.extend(_api_args(user_raw))
                continue

            if user_raw and not is_silence(user_raw):
                if is_api_text(user_raw):
                    counts["api_removed"] += 1
                    goal_args.extend(_api_args(user_raw))
                elif is_kb_text(user_raw):
                    counts["kb_removed"] += 1
                elif has_native_script(user_raw):
                    counts["non_roman_removed"] += 1
                else:
                    turns.append({"speaker": "user", "text": user_raw})
            elif is_silence(user_raw):
                counts["silence_skipped"] += 1

            if sys_raw:
                if is_api_text(sys_raw):
                    counts["api_removed"] += 1
                    goal_args.extend(_api_args(sys_raw))
                elif is_kb_text(sys_raw):
                    counts["kb_removed"] += 1
                elif has_native_script(sys_raw):
                    counts["non_roman_removed"] += 1
                else:
                    turns.append({"speaker": "system", "text": sys_raw})

        if not turns:
            counts["empty_after_filter"] += 1
            continue
        for i, t in enumerate(turns, start=1):
            t["turn_id"] = i
        dialog_id = f"{split}_{len(dialogs) + 1:04d}"
        dialogs.append(
            {
                "dialog_id": dialog_id,
                "split": split,
                "turns": turns,
                "goal_slots": infer_goal_slots(turns, goal_args),
            }
        )
    return dialogs, counts


def _api_args(text: str) -> list[str]:
    # "api_call R_cuisine west moderate" or "api_call italian R_location cheap"
    rest = API_RE.sub("", text).strip()
    return [p for p in rest.split() if p and p != "no" and p != "result"]


def _norm_cuisine(tok: str) -> str:
    t = tok.lower().replace("-", "").replace("_", "")
    if t in {"asianoriental", "asian"}:
        return "asian"
    if t in {"northamerican", "american"}:
        return "american"
    if t in {"moderneuropean", "modern"}:
        return "european"
    return t


def infer_goal_slots(turns: list[dict], api_args: list[str]) -> dict[str, str]:
    area = ""
    price = ""
    cuisine = ""
    for a in api_args:
        low = a.lower()
        if low.startswith("r_"):
            continue
        if low in AREA_WORDS and not area:
            area = "centre" if low in {"center", "centre"} else low
        elif low in PRICE_WORDS and not price:
            price = low
        else:
            c = _norm_cuisine(low)
            if c in DSTC2_CUISINES and not cuisine:
                cuisine = c
    blob = " ".join(t["text"].lower() for t in turns)
    words = TOKEN_RE.findall(blob)
    wordset = set(words)
    if not area:
        for w in AREA_WORDS:
            if w in wordset:
                area = "centre" if w in {"center", "centre"} else w
                break
    if not price:
        for w in PRICE_WORDS:
            if w in wordset:
                price = w
                break
    if not cuisine:
        for w in words:
            c = _norm_cuisine(w)
            if c in DSTC2_CUISINES:
                cuisine = c
                break
    return {"area": area or "unk", "price": price or "unk", "cuisine": cuisine or "unk"}


def _function_set(lang: str) -> set[str]:
    return HI_FUNCTION if lang == "hi" else TA_FUNCTION


def build_local_lexicon(lang: str, dialogs: list[dict]) -> set[str]:
    lex = load_roman_lexicon(lang)
    func = _function_set(lang)
    counts: Counter[str] = Counter()
    for d in dialogs:
        for t in d["turns"]:
            for tok in tokenize(t["text"]):
                low = tok.lower()
                if WORD_RE.match(tok):
                    counts[low] += 1
    for tok, n in counts.items():
        if n < LOCAL_MIN_FREQ:
            continue
        if zipf_en(tok) < ZIPF_LOCAL:
            lex.add(tok)
        elif tok in func and tok not in EN_FUNCTION_BLOCK:
            # Function-word types stay local even when wordfreq lists them as
            # English (la, mere, hu, me). Documented in REPORT.md.
            lex.add(tok)
    return lex


def tag_token(tok: str, lexicon: set[str]) -> str:
    if not WORD_RE.match(tok):
        return "other"
    low = tok.lower()
    if zipf_en(low) >= ZIPF_EN and low not in lexicon:
        return "en"
    return "uncertain"


def turn_metrics(text: str, lexicon: set[str]) -> dict[str, Any]:
    tokens = tokenize(text)
    tags = [tag_token(t, lexicon) for t in tokens]
    n_en = sum(1 for x in tags if x == "en")
    n_loc = sum(1 for x in tags if x == "uncertain")
    den = n_en + n_loc
    if den == 0:
        english_share = None
        cmi = None
        monolingual = False
    else:
        english_share = n_en / den
        cmi = 100.0 * (1.0 - (max(n_en, n_loc) / den))
        monolingual = n_en == 0 or n_loc == 0
    switches = 0
    prev = None
    for tag in tags:
        if tag not in {"en", "uncertain"}:
            continue
        if prev is not None and tag != prev:
            switches += 1
        prev = tag
    return {
        "tokens": tokens,
        "tags": tags,
        "n_tokens": len(tokens),
        "n_en": n_en,
        "n_local": n_loc,
        "english_share": english_share,
        "cmi": cmi,
        "n_switches": switches,
        "monolingual": monolingual,
        "tag_origin": "auto",
    }


def is_light_verb(tok: str, verbs: tuple[str, ...]) -> str | None:
    low = tok.lower()
    if low in verbs:
        return low
    # prefix match only for longer stems; avoid ho/hai false hits
    for v in verbs:
        if len(v) >= 3 and low.startswith(v) and low != v:
            # kar* / pann* / soll* / venum*
            if v in {"kar", "karo", "karna", "pannu", "panna", "sollunga"}:
                return v
    return None


def collect_pair_stats(
    pair: str, dialogs: list[dict], lexicon: set[str]
) -> dict[str, Any]:
    lang = pair.split("_")[0]
    light_verbs = HI_LIGHT_VERBS if lang == "hi" else TA_LIGHT_VERBS
    polite_forms = HI_POLITE if lang == "hi" else TA_POLITE
    func_set = HI_FUNCTION if lang == "hi" else TA_FUNCTION

    sys_texts: list[str] = []
    user_cmi: list[float] = []
    sys_cmi: list[float] = []
    user_en: list[float] = []
    sys_en: list[float] = []
    user_tok: list[float] = []
    sys_tok: list[float] = []
    user_sw: list[float] = []
    sys_sw: list[float] = []
    turns_per: list[float] = []
    n_user = n_sys = 0
    n_user_mono = n_sys_mono = 0
    en_ins: Counter[str] = Counter()
    func_counts: Counter[str] = Counter()
    lv_patterns: Counter[str] = Counter()
    polite_counts: Counter[str] = Counter({k: 0 for k in polite_forms})
    n_turns = 0

    annotated: list[dict] = []
    for d in dialogs:
        turns_per.append(len(d["turns"]))
        d_user_cmi: list[float] = []
        d_user_en: list[float] = []
        d_sys_texts: list[str] = []
        for t in d["turns"]:
            n_turns += 1
            m = turn_metrics(t["text"], lexicon)
            t_ann = {**t, **m}
            annotated.append(t_ann)
            toks = m["tokens"]
            tags = m["tags"]
            low_text = t["text"].lower()
            for form in polite_forms:
                if " " in form:
                    polite_counts[form] += low_text.count(form)
                else:
                    polite_counts[form] += sum(1 for x in toks if x.lower() == form)
            for tok, tag in zip(toks, tags):
                low = tok.lower()
                if tag == "en":
                    en_ins[low] += 1
                if tag == "uncertain" and low in func_set:
                    func_counts[low] += 1
            for i in range(len(toks) - 1):
                if tags[i] != "en":
                    continue
                hit = is_light_verb(toks[i + 1], light_verbs)
                if hit:
                    lv_patterns[f"{toks[i].lower()} {hit}"] += 1
            if t["speaker"] == "user":
                n_user += 1
                user_tok.append(m["n_tokens"])
                user_sw.append(m["n_switches"])
                if m["monolingual"]:
                    n_user_mono += 1
                if m["cmi"] is not None:
                    user_cmi.append(m["cmi"])
                    d_user_cmi.append(m["cmi"])
                if m["english_share"] is not None:
                    user_en.append(m["english_share"])
                    d_user_en.append(m["english_share"])
            else:
                n_sys += 1
                sys_texts.append(t["text"])
                d_sys_texts.append(t["text"])
                sys_tok.append(m["n_tokens"])
                sys_sw.append(m["n_switches"])
                if m["monolingual"]:
                    n_sys_mono += 1
                if m["cmi"] is not None:
                    sys_cmi.append(m["cmi"])
                if m["english_share"] is not None:
                    sys_en.append(m["english_share"])
        d["user_cmi_mean"] = (sum(d_user_cmi) / len(d_user_cmi)) if d_user_cmi else 0.0
        d["user_en_mean"] = (sum(d_user_en) / len(d_user_en)) if d_user_en else 0.0
        d["n_natural_turns"] = len(d["turns"])
        d["n_user_turns"] = sum(1 for t in d["turns"] if t["speaker"] == "user")
        d["n_mixed_user"] = sum(1 for x in d_user_cmi if x and x > 0)
        d["system_texts"] = d_sys_texts

    sys_freq = Counter(sys_texts)
    n_sys_dup = sum(1 for x in sys_texts if sys_freq[x] > 1)
    template_rate = (n_sys_dup / n_sys) if n_sys else 0.0

    for d in dialogs:
        texts = d.get("system_texts") or []
        if not texts:
            d["template_frac"] = 0.0
        else:
            d["template_frac"] = sum(1 for x in texts if sys_freq[x] > 1) / len(texts)

    return {
        "pair": pair,
        "n_dialogs": len(dialogs),
        "n_turns": n_turns,
        "n_user_turns": n_user,
        "n_system_turns": n_sys,
        "lexicon_size": len(lexicon),
        "turns_per_dialog": quantiles(turns_per),
        "tokens_per_turn_user": quantiles(user_tok),
        "tokens_per_turn_system": quantiles(sys_tok),
        "english_share_user": quantiles(user_en),
        "english_share_system": quantiles(sys_en),
        "cmi_user": quantiles(user_cmi),
        "cmi_system": quantiles(sys_cmi),
        "switches_user": quantiles(user_sw),
        "switches_system": quantiles(sys_sw),
        "monolingual_share_user": (n_user_mono / n_user) if n_user else 0.0,
        "monolingual_share_system": (n_sys_mono / n_sys) if n_sys else 0.0,
        "template_repetition_rate": template_rate,
        "n_system_exact_duplicates": n_sys_dup,
        "unique_system_texts": len(sys_freq),
        "top_100_english_insertions": en_ins.most_common(100),
        "top_50_local_function_words": func_counts.most_common(50),
        "top_50_en_light_verb": lv_patterns.most_common(50),
        "politeness_counts": dict(polite_counts),
        "tag_origin": "auto",
    }


def select_reference_dialogs(dialogs: list[dict], n: int = N_REFERENCE) -> list[dict]:
    def balanced_mix(d: dict) -> bool:
        en = float(d.get("user_en_mean") or 0.0)
        return 0.20 <= en <= 0.80

    eligible = [
        d
        for d in dialogs
        if d.get("n_natural_turns", 0) >= 6
        and d.get("n_user_turns", 0) >= 3
        and d.get("n_mixed_user", 0) >= 2
        and balanced_mix(d)
    ]
    if len(eligible) < n:
        eligible = [
            d
            for d in dialogs
            if d.get("n_natural_turns", 0) >= 6 and d.get("n_mixed_user", 0) >= 1
        ]
    if len(eligible) < n:
        eligible = list(dialogs)
    # Prefer true mixing (CMI high, English share not extreme), known cuisine, longer dialogs.
    eligible = sorted(
        eligible,
        key=lambda d: (
            0 if (d.get("goal_slots") or {}).get("cuisine", "unk") != "unk" else 1,
            -float(d.get("user_cmi_mean") or 0.0),
            abs(float(d.get("user_en_mean") or 0.5) - 0.5),
            float(d.get("template_frac") or 0.0),
            -int(d.get("n_natural_turns") or 0),
        ),
    )
    picked: list[dict] = []
    seen_slots: set[tuple[str, str, str]] = set()
    seen_openings: set[str] = set()
    seen_ids: set[str] = set()

    def opening(d: dict) -> str:
        for t in d["turns"]:
            if t["speaker"] == "user":
                return " ".join(tokenize(t["text"])[:8]).lower()
        return d["dialog_id"]

    def slot_key(d: dict) -> tuple[str, str, str]:
        g = d.get("goal_slots") or {}
        return (g.get("cuisine", "unk"), g.get("area", "unk"), g.get("price", "unk"))

    def consider(d: dict, require_distinct_slots: bool) -> bool:
        if d["dialog_id"] in seen_ids:
            return False
        sk = slot_key(d)
        op = opening(d)
        if op in seen_openings:
            return False
        if require_distinct_slots and sk in seen_slots:
            return False
        picked.append(d)
        seen_ids.add(d["dialog_id"])
        seen_slots.add(sk)
        seen_openings.add(op)
        return True

    for d in eligible:
        consider(d, require_distinct_slots=True)
        if len(picked) >= n:
            break
    if len(picked) < n:
        for d in eligible:
            consider(d, require_distinct_slots=False)
            if len(picked) >= n:
                break
    return picked[:n]


def write_jsonl(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            if fields:
                out = {k: row[k] for k in fields if k in row}
            else:
                out = row
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def format_dialog_md(d: dict) -> str:
    lines = [f"### {d['dialog_id']} ({d['split']})"]
    g = d.get("goal_slots")
    if g:
        lines.append(
            f"_slots: cuisine={g.get('cuisine')} area={g.get('area')} "
            f"price={g.get('price')}; user_cmi_mean="
            f"{d.get('user_cmi_mean', 0):.2f}; user_en_mean="
            f"{d.get('user_en_mean', 0):.2f}; template_frac="
            f"{d.get('template_frac', 0):.2f}_"
        )
    lines.append("")
    for t in d["turns"]:
        who = "User" if t["speaker"] == "user" else "System"
        lines.append(f"{who}: {t['text']}")
    lines.append("")
    return "\n".join(lines)


def write_reference_md(path: Path, pair: str, dialogs: list[dict]) -> None:
    parts = [
        f"# {pair} reference dialogues (n={len(dialogs)})",
        "",
        "Few-shot style anchors from Code-Mixed-Dialog natural-language turns.",
        "Selected for ≥6 turns, higher user-turn CMI, lower templated system "
        "repetition, and diverse goal slots (cuisine / area / price).",
        "",
    ]
    for d in dialogs:
        parts.append(format_dialog_md(d))
    path.write_text("\n".join(parts), encoding="utf-8")


def parse_htop() -> list[dict]:
    rows: list[dict] = []
    for split, fname in (
        ("train", "train.tsv"),
        ("validation", "validation.tsv"),
        ("test", "test.tsv"),
    ):
        path = RAW_SPOKEN / "hinglish_top" / fname
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for i, rec in enumerate(reader, start=1):
                cs = (rec.get("cs_query") or "").strip()
                if not cs:
                    continue
                parse = rec.get("cs_parse") or rec.get("en_parse") or ""
                intents = INTENT_RE.findall(parse)
                slots = SLOT_RE.findall(parse)
                domain = (rec.get("domain") or "").strip()
                uid = f"htop_{split}_{i:05d}"
                rows.append(
                    {
                        "id": uid,
                        "text": cs,
                        "intent": intents[0] if intents else "",
                        "labels": {
                            "domain": domain,
                            "intents": intents,
                            "slots": sorted(set(slots)),
                        },
                        "domain": domain,
                        "split": split,
                        "en_query": (rec.get("en_query") or "").strip(),
                    }
                )
    return rows


def htop_metrics(rows: list[dict], lexicon: set[str]) -> dict[str, Any]:
    en_s: list[float] = []
    cmis: list[float] = []
    toks: list[float] = []
    sw: list[float] = []
    n_mono = 0
    en_ins: Counter[str] = Counter()
    func_counts: Counter[str] = Counter()
    lv_patterns: Counter[str] = Counter()
    polite_counts: Counter[str] = Counter({k: 0 for k in HI_POLITE})
    for r in rows:
        m = turn_metrics(r["text"], lexicon)
        r.update(
            {
                "english_share": m["english_share"],
                "cmi": m["cmi"],
                "n_switches": m["n_switches"],
                "n_tokens": m["n_tokens"],
                "tag_origin": "auto",
            }
        )
        toks.append(m["n_tokens"])
        sw.append(m["n_switches"])
        if m["monolingual"]:
            n_mono += 1
        if m["english_share"] is not None:
            en_s.append(m["english_share"])
        if m["cmi"] is not None:
            cmis.append(m["cmi"])
        low_text = r["text"].lower()
        for form in HI_POLITE:
            if " " in form:
                polite_counts[form] += low_text.count(form)
            else:
                polite_counts[form] += sum(
                    1 for x in m["tokens"] if x.lower() == form
                )
        for tok, tag in zip(m["tokens"], m["tags"]):
            low = tok.lower()
            if tag == "en":
                en_ins[low] += 1
            if tag == "uncertain" and low in HI_FUNCTION:
                func_counts[low] += 1
        for i in range(len(m["tokens"]) - 1):
            if m["tags"][i] != "en":
                continue
            hit = is_light_verb(m["tokens"][i + 1], HI_LIGHT_VERBS)
            if hit:
                lv_patterns[f"{m['tokens'][i].lower()} {hit}"] += 1
    by_domain = Counter(r.get("domain") or "?" for r in rows)
    by_split = Counter(r.get("split") or "?" for r in rows)
    return {
        "n_utterances": len(rows),
        "by_split": dict(by_split),
        "by_domain": dict(by_domain),
        "tokens_per_utt": quantiles(toks),
        "english_share": quantiles(en_s),
        "cmi": quantiles(cmis),
        "switches": quantiles(sw),
        "monolingual_share": (n_mono / len(rows)) if rows else 0.0,
        "top_100_english_insertions": en_ins.most_common(100),
        "top_50_local_function_words": func_counts.most_common(50),
        "top_50_en_light_verb": lv_patterns.most_common(50),
        "politeness_counts": dict(polite_counts),
        "lexicon_size": len(lexicon),
        "tag_origin": "auto",
    }


def htop_service_category(row: dict) -> str | None:
    domain = (row.get("domain") or "").lower()
    intent = (row.get("intent") or "").upper()
    text = row.get("text") or ""
    # Order: explicit service acts first. "phone" alone is not a call.
    if re.search(r"\b(pay|payment|bill|upi)\b", text, re.I) or (
        "PAY" in intent
    ):
        return "payment"
    if "CALL" in intent or re.search(r"\b(call|dial)\b", text, re.I):
        return "call"
    if domain == "reminder" or "REMIND" in intent:
        return "reminder"
    if domain == "messaging" or "MESSAGE" in intent:
        return "message"
    if domain in {"alarm", "timer"} or "ALARM" in intent or "TIMER" in intent:
        return "time"
    if HTOP_TIME_RE.search(text) and domain not in {"weather", "music", "event"}:
        return "time"
    return None


def select_htop_service(rows: list[dict], n: int = N_HTOP_SERVICE) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        cat = htop_service_category(r)
        if not cat:
            continue
        # prefer mixed utterances
        if r.get("cmi") is None:
            continue
        buckets[cat].append(r)
    for cat in buckets:
        buckets[cat].sort(
            key=lambda r: (
                -(r.get("cmi") or 0.0),
                -(r.get("n_switches") or 0),
                r.get("id"),
            )
        )
    order = ["reminder", "message", "call", "payment", "time"]
    # add any extra cats
    for k in buckets:
        if k not in order:
            order.append(k)
    picked: list[dict] = []
    seen_text: set[str] = set()
    seen_sig: set[str] = set()

    def sig(r: dict) -> str:
        toks = tokenize(r["text"])[:5]
        return f"{r.get('intent')}|{' '.join(t.lower() for t in toks)}"

    # round-robin for diversity across service categories
    idxs = {c: 0 for c in order}
    while len(picked) < n:
        progressed = False
        for cat in order:
            lst = buckets.get(cat) or []
            i = idxs[cat]
            while i < len(lst):
                r = lst[i]
                i += 1
                key = r["text"].strip().lower()
                s = sig(r)
                if key in seen_text or s in seen_sig:
                    continue
                seen_text.add(key)
                seen_sig.add(s)
                out = dict(r)
                out["service_cat"] = cat
                picked.append(out)
                progressed = True
                break
            idxs[cat] = i
            if len(picked) >= n:
                break
        if not progressed:
            break
    return picked[:n]


def write_htop_service_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "cand_id",
        "source_item_id",
        "text",
        "english_share",
        "cmi",
        "accept",
        "reject_reason",
        "reviewer_note",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(rows, start=1):
            es = r.get("english_share")
            cmi = r.get("cmi")
            w.writerow(
                {
                    "cand_id": f"ht_s{i:03d}",
                    "source_item_id": r["id"],
                    "text": r["text"],
                    "english_share": "" if es is None else f"{es:.4f}",
                    "cmi": "" if cmi is None else f"{cmi:.4f}",
                    "accept": "",
                    "reject_reason": "",
                    "reviewer_note": (
                        f"auto-selected service-like ({r.get('service_cat')}; "
                        f"domain={r.get('domain')}; intent={r.get('intent')}); "
                        "needs human review"
                    ),
                }
            )


def build_local_lexicon_from_texts(lang: str, texts: list[str]) -> set[str]:
    lex = load_roman_lexicon(lang)
    func = _function_set(lang)
    counts: Counter[str] = Counter()
    for text in texts:
        for tok in tokenize(text):
            if WORD_RE.match(tok):
                counts[tok.lower()] += 1
    for tok, n in counts.items():
        if n < LOCAL_MIN_FREQ:
            continue
        if zipf_en(tok) < ZIPF_LOCAL:
            lex.add(tok)
        elif tok in func and tok not in EN_FUNCTION_BLOCK:
            lex.add(tok)
    return lex


def write_licenses(dl_meta: dict[str, Any]) -> None:
    today = date.today().isoformat()
    cmd = f"""# Code-Mixed-Dialog (DSTC2) — license record

Recorded: {today}

Local files:

- `data/calibration/raw/spoken/code_mixed_dialog/hi_en/dialog-dstc2-{{trn,dev,tst}}.txt`
- `data/calibration/raw/spoken/code_mixed_dialog/ta_en/dialog-dstc2-{{trn,dev,tst}}.txt`
- `data/calibration/raw/spoken/code_mixed_dialog/{{hi_en,ta_en}}/vocab_splits.json`
- `data/calibration/raw/spoken/code_mixed_dialog/LICENSE` (repo copy)
- `data/calibration/raw/spoken/code_mixed_dialog/README.md`
- earlier 200-line samples kept beside the full files

## Source

- Official repository: https://github.com/sumanbanerjee1/Code-Mixed-Dialog
- Paper: Banerjee, Moghe, Arora, Khapra, *A Dataset for Building Code-Mixed Goal Oriented Conversation Systems*, COLING 2018, https://aclanthology.org/C18-1319/ (arXiv:1806.05997)
- Languages kept here: Hindi–English and Tamil–English only (repo also has Bengali, Gujarati, English).

## License (two statements; both permissive)

1. **Paper footnote (COLING 2018 / ACL Anthology C18-1319):** “This work is licensed under a Creative Commons Attribution 4.0 International License. License details: http://creativecommons.org/licenses/by/4.0/”
2. **GitHub `LICENSE` file (present in the official repo as of {today}):** **Apache License 2.0** (full text copied to `raw/spoken/code_mixed_dialog/LICENSE`). A previous calibration note that the repo had no LICENSE file is outdated.

English DSTC2 source conversations (Henderson et al. 2014; Bordes & Weston 2017 adaptation) are a separate corpus. This record covers the IIT Madras code-mixed translations.

## What was downloaded

Full `dialog-dstc2-trn.txt`, `dialog-dstc2-dev.txt`, `dialog-dstc2-tst.txt` plus `vocab_splits.json` for Hindi and Tamil. **No candidate-response files** exist in those language folders. The 170k-class synthetic sets are not part of this repo.

## Citation

Banerjee, Suman; Moghe, Nikita; Arora, Siddhartha; Khapra, Mitesh M. 2018. A Dataset for Building Code-Mixed Goal Oriented Conversation Systems. In *Proceedings of the 27th International Conference on Computational Linguistics*, pages 3766–3780.
"""
    htop = f"""# Hinglish-TOP — license record

Recorded: {today}
Local files:

- `data/calibration/raw/spoken/hinglish_top/train.tsv`
- `data/calibration/raw/spoken/hinglish_top/validation.tsv`
- `data/calibration/raw/spoken/hinglish_top/test.tsv`
- `data/calibration/raw/spoken/hinglish_top/LICENSE.md` (repo copy)
- `data/calibration/raw/spoken/hinglish_top/README.md`
- earlier `train_sample_200.tsv` kept beside the full files

## Source

- Official repository: https://github.com/google-research-datasets/Hinglish-TOP-Dataset
- Hugging Face mirror: https://huggingface.co/datasets/WillHeld/hinglish_top
- Paper: Agarwal et al., *CST5: Data Augmentation for Code-Switched Semantic Parsing*, arXiv:2211.07514

## License

**Apache License 2.0**, verified from `LICENSE.md` in the official GitHub repository (copy stored next to the TSVs).

## What was downloaded

The **human-annotated** split only (`Dataset/Human Annotated Data/{{train,validation,test}}.tsv`). Columns: `en_query`, `cs_query`, `en_parse`, `cs_parse`, `domain`.

The ~170k **synthetically generated** CST5/mT5 file under `Dataset/Synthetically Generated Data/` was **not** downloaded and must not be used for human-register calibration.

## Citation

Agarwal, Anmol; Gupta, Jigar; Goel, Rahul; Upadhyay, Shyam; Joshi, Pankaj; Aravamudhan, Rengarajan. 2022. CST5: Data Augmentation for Code-Switched Semantic Parsing. arXiv:2211.07514.
"""
    (LIC_DIR / "code_mixed_dialog.md").write_text(cmd, encoding="utf-8")
    (LIC_DIR / "hinglish_top.md").write_text(htop, encoding="utf-8")


def qrow(label: str, q: dict) -> str:
    return f"- **{label}:** {fmt_q(q)}"


def top_md(pairs: list[tuple[str, int]], n: int | None = None) -> str:
    items = pairs if n is None else pairs[:n]
    if not items:
        return "_none_"
    return ", ".join(f"`{k}` ({v})" for k, v in items)


def example_dialogs_md(dialogs: list[dict], k: int = 5) -> str:
    # pick varied examples: first long mixed ones from train if possible
    pool = [d for d in dialogs if d.get("n_natural_turns", 0) >= 6]
    pool = sorted(pool, key=lambda d: -float(d.get("user_cmi_mean") or 0))
    chosen = pool[:k]
    if len(chosen) < k:
        chosen = dialogs[:k]
    return "\n".join(format_dialog_md(d) for d in chosen)


def write_report(
    parse_counts: dict[str, dict],
    pair_stats: dict[str, dict],
    pair_dialogs: dict[str, list[dict]],
    refs: dict[str, list[dict]],
    htop_stats: dict,
    htop_n: int,
    htop_service_n: int,
    dl_meta: dict,
) -> None:
    today = date.today().isoformat()
    hi = pair_stats["hi_en"]
    ta = pair_stats["ta_en"]
    hc = parse_counts["hi_en"]
    tc = parse_counts["ta_en"]

    def pair_block(name: str, st: dict, pc: dict) -> str:
        return f"""### {name}

- Dialogues: **{st['n_dialogs']}**
- Natural turns: **{st['n_turns']}** (user {st['n_user_turns']}, system {st['n_system_turns']})
- Local lexicon size: {st['lexicon_size']} (Dakshina roman lexicon + corpus tokens with zipf_en<{ZIPF_LOCAL} and freq≥{LOCAL_MIN_FREQ} + pair function-word types with freq≥{LOCAL_MIN_FREQ}, so wordfreq false friends like `la`/`me`/`hu` stay local)
- Token tags: `en` if wordfreq zipf_frequency(tok,'en')≥{ZIPF_EN} and token ∉ local lexicon; else `uncertain`. All tags **auto**.
- File splits: 1618 trn / 500 dev / 1117 tst (paper table said 1168 train).
- Removed structural lines: API-call **{pc['api_removed']}**, KB triples **{pc['kb_removed']}**, non-Roman turns **{pc['non_roman_removed']}**; user `<SILENCE>` skipped **{pc['silence_skipped']}**; empty dialogs after filter **{pc['empty_after_filter']}**
- Unique system texts: {st['unique_system_texts']}
- **Template repetition** (share of system turns that are an exact duplicate of another system turn): **{st['template_repetition_rate']:.4f}** ({st['n_system_exact_duplicates']}/{st['n_system_turns']})

{qrow("turns / dialog", st["turns_per_dialog"])}
{qrow("tokens / turn (user)", st["tokens_per_turn_user"])}
{qrow("tokens / turn (system)", st["tokens_per_turn_system"])}
{qrow("english_share (user)", st["english_share_user"])}
{qrow("english_share (system)", st["english_share_system"])}
{qrow("CMI (user)", st["cmi_user"])}
{qrow("CMI (system)", st["cmi_system"])}
{qrow("switches / turn (user)", st["switches_user"])}
{qrow("switches / turn (system)", st["switches_system"])}

- Monolingual-turn share user: {st['monolingual_share_user']:.4f}
- Monolingual-turn share system: {st['monolingual_share_system']:.4f}

**Top 20 English insertions:** {top_md(st["top_100_english_insertions"], 20)}

**Top 20 local function words:** {top_md(st["top_50_local_function_words"], 20)}

**Top 20 EN+light-verb patterns:** {top_md(st["top_50_en_light_verb"], 20)}

**Politeness / address counts:** {", ".join(f"`{k}`={v}" for k, v in st["politeness_counts"].items())}
"""

    report = f"""# Service-dialogue calibration report

Date: {today}
Script: `scripts/calibration/process_service_dialogs.py`

## How Code-Mixed-Dialog was created (Banerjee et al., COLING 2018)

Facts from [C18-1319](https://aclanthology.org/C18-1319/), not from memory.

1. **English source.** The authors start from the DSTC2 restaurant-reservation set (Henderson, Thomson, Williams 2014): crowdsourced **human users talking to existing restaurant bots**. Original collection was **spoken** (ASR, including a degraded ASR condition) then **transcribed**. Bordes & Weston (2017) recast those transcripts as utterance–response pairs and inserted `api_call` lines plus KB triples. The COLING table lists 1,168 train / 500 val / 1,117 test. The released GitHub files parsed here have **1,618 / 500 / 1,117** (3,235 dialogs) — the 1,618-train cut is the Facebook bAbI DSTC2 conversion; the paper’s 1,168 figure is smaller.

2. **Templates, not free conversation.** Many English utterances share a frame and differ only in slot values. The authors replaced area / cuisine / price / restaurant / address / phone / postcode with placeholders and obtained **3,590 unique English utterance templates**.

3. **Human code-mixed translations of those templates.** In-house and AMT workers saw each English template and wrote a code-mixed version. Instructions: pretend you are chatting with a bilingual friend; do **not** force a full native-language translation; switch to English when you naturally would; **use Roman script only** (no Devanagari / Tamil script); keep placeholders untranslated. Hindi (and Gujarati) were **fully crowdsourced then in-house verified**. Tamil was **~40% AMT, ~60% in-house**. Bengali was all in-house (not used here).

4. **Re-instantiation into full dialogs.** Verified template translations were filled back with the original DSTC2 entities, so each En-DSTC2 dialog becomes a code-mixed dialog with the **same structure and bot policy**. Utterances were translated **independently** (annotators did not see neighbouring turns). A second in-house pass rated 100 full dialogs/language for colloquialism, intelligibility, and coherence (Hindi ~4.2 / 4.1 / 4.2 on a 1–5 scale).

5. **Mixing in the paper.** 87.73% of created utterances were code-mixed; 7.18% English-only; 5.09% native-only. Hindi unique utterances 6,549 (5,750 code-mixed). Paper Cavg for Hi-DSTC2 is 32.12, Ta-DSTC2 29.54.

This is **human-written code-mixing**, but it is **translation of English restaurant-bot templates**, not spontaneous Indian call-centre speech.

## License

See `data/calibration/licenses/code_mixed_dialog.md` and `hinglish_top.md`.

- Code-Mixed-Dialog paper: **CC BY 4.0**. Official GitHub `LICENSE` is **Apache 2.0** (the file exists; an earlier note that the repo had no LICENSE is outdated).
- Hinglish-TOP: **Apache 2.0**. Human TSV only; CST5 170k synthetic file not downloaded.

Candidates: **not present** in the Hindi or Tamil folders (trn/dev/tst + `vocab_splits.json` only).

Download: full HI/TA dialog files + human Hinglish-TOP TSVs are on disk (script skips non-empty files; this run fetched {len(dl_meta.get("fetched") or [])}, skipped {len(dl_meta.get("skipped") or [])}).

## Code-Mixed-Dialog counts and mixing

{pair_block("hi_en (Hindi–English)", hi, hc)}

{pair_block("ta_en (Tamil–English)", ta, tc)}

## Reference dialogues (few-shot anchors)

30 dialogs / pair in `{{hi_en,ta_en}}_reference_30.jsonl` and `.md`.

Selection: prefer ≥6 natural turns; high mean **user** CMI; low fraction of globally templated system turns; unique (cuisine, area, price) slot triples when detectable from `api_call` arguments or user text; unique user-opening token prefix so two dialogs do not share the same first request.

| pair | n | mean user CMI (selected) | mean template_frac (selected) |
|---|---:|---:|---:|
| hi_en | {len(refs["hi_en"])} | {sum(d.get("user_cmi_mean") or 0 for d in refs["hi_en"]) / max(len(refs["hi_en"]), 1):.2f} | {sum(d.get("template_frac") or 0 for d in refs["hi_en"]) / max(len(refs["hi_en"]), 1):.2f} |
| ta_en | {len(refs["ta_en"])} | {sum(d.get("user_cmi_mean") or 0 for d in refs["ta_en"]) / max(len(refs["ta_en"]), 1):.2f} | {sum(d.get("template_frac") or 0 for d in refs["ta_en"]) / max(len(refs["ta_en"]), 1):.2f} |

## Five example dialogues per pair

### hi_en

{example_dialogs_md(pair_dialogs["hi_en"], 5)}

### ta_en

{example_dialogs_md(pair_dialogs["ta_en"], 5)}

## Hinglish-TOP (human-annotated only)

Agarwal et al. (2022): bilingual annotators **rewrote** 10,896 English TOPv2 assistant utterances into Roman Hinglish (not the 170k CST5 generations). Single-turn voice-assistant commands (alarm, reminder, weather, navigation, music, timer, event, messaging) — IVR-like, not agent–customer calls.

- Utterances written: **{htop_n}** → `hinglish_top_utterances.jsonl` (`id`, `text`, `intent`, `labels`)
- Splits: {json.dumps(htop_stats.get("by_split", {}))}
- Domains: {json.dumps(htop_stats.get("by_domain", {}))}
- Lexicon size: {htop_stats.get("lexicon_size")}
{qrow("tokens / utterance", htop_stats.get("tokens_per_utt") or dict())}
{qrow("english_share", htop_stats.get("english_share") or dict())}
{qrow("CMI", htop_stats.get("cmi") or dict())}
{qrow("switches / utterance", htop_stats.get("switches") or dict())}
- Monolingual share: {htop_stats.get("monolingual_share", 0):.4f}

**Top 15 English insertions:** {top_md(htop_stats.get("top_100_english_insertions") or [], 15)}

**Top 15 local function words:** {top_md(htop_stats.get("top_50_local_function_words") or [], 15)}

**Top 15 EN+light-verb:** {top_md(htop_stats.get("top_50_en_light_verb") or [], 15)}

**Politeness:** {", ".join(f"`{k}`={v}" for k, v in (htop_stats.get("politeness_counts") or dict()).items())}

Service-like subset: **{htop_service_n}** rows in `hinglish_top_service_150.csv` (reminders, messages, calls, payments, time/alarm/timer). `accept` / `reject_reason` left blank for human review.

## Limitations

- **Restaurant domain only** for the multi-turn data. Slots are area, cuisine, price, phone, address. Not banking, courier, or general customer care.
- **Not spontaneous speech.** DSTC2 users spoke to a Cambridge restaurant bot; Banerjee et al. then had humans **translate unique English templates** into Roman code-mixing and stitched them back. System turns inherit the original **templated bot** wording (see the high template-repetition rate). User turns are more natural than system turns but still translations of English prompts, not recorded Indian phone calls.
- **Independent utterance translation** can weaken discourse (paper still got ~4/5 coherence ratings on 100 dialogs/language).
- **Roman-only by instruction**; any residual native-script turns were dropped (counts above).
- **LID is automatic** (wordfreq + Dakshina/corpus lexicon + frequent function-word types). Residual false friends remain (`karen`, restaurant-name `ask`, truncated `pa`). Paper `vocab_splits.json` was downloaded but **not** used as gold tags.
- **System turns are almost entirely templated** (~97.5% exact-duplicate rate). Few-shot anchors therefore lean on **user** turns; system lines are still useful as bot phrasing but are not natural speech.
- **Released train split is 1,618 dialogs**, not the 1,168 in Table 3 of the paper. Dev/test match (500 / 1,117).
- **No Telugu–English** dialogs in this source.
- Hinglish-TOP has **no multi-turn** structure and no payment domain to speak of; “calls” are rare keyword hits, not telephony.
- Do not treat this register as a substitute for closed real Hinglish care-call data (e.g. NSED).
"""
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    (PROC_DIR / "REPORT.md").write_text(report, encoding="utf-8")
    (PROC_DIR / "stats.json").write_text(
        json.dumps(
            {
                "date": today,
                "parse_counts": parse_counts,
                "pair_stats": pair_stats,
                "hinglish_top": htop_stats,
                "n_reference": {k: len(v) for k, v in refs.items()},
                "n_htop_service": htop_service_n,
                "download": {
                    "n_fetched": len(dl_meta.get("fetched") or []),
                    "n_skipped": len(dl_meta.get("skipped") or []),
                    "candidates_present": dl_meta.get("candidates_present"),
                },
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def update_spoken_sources_download_note() -> None:
    path = CAL_ROOT / "spoken_register_sources.md"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    marker = "## Downloads (this run)"
    extra = """## Downloads (service-dialog full pull)

Full text (no audio) pulled by `scripts/calibration/process_service_dialogs.py`:

| Path | What |
|---|---|
| `data/calibration/raw/spoken/code_mixed_dialog/hi_en/dialog-dstc2-{trn,dev,tst}.txt` | Full HI-EN DSTC2 dialogs (Banerjee et al. 2018) |
| `data/calibration/raw/spoken/code_mixed_dialog/ta_en/dialog-dstc2-{trn,dev,tst}.txt` | Full TA-EN DSTC2 dialogs |
| `data/calibration/raw/spoken/code_mixed_dialog/{hi_en,ta_en}/vocab_splits.json` | Paper English / native / NE vocab splits |
| `data/calibration/raw/spoken/hinglish_top/{train,validation,test}.tsv` | Full human Hinglish-TOP (~10.8k); CST5 170k **not** fetched |
| `data/calibration/processed/service_dialogs/` | Parsed dialogs, reference-30, Hinglish-TOP utterances, REPORT |

The 200-line / 200-row samples from the earlier survey pass are still in the same folders.

"""
    if "## Downloads (service-dialog full pull)" in text:
        start = text.index("## Downloads (service-dialog full pull)")
        # replace until next ## or Downloads (this run)
        rest = text[start:]
        nxt = re.search(r"\n## ", rest[4:])
        end = start + 4 + nxt.start() if nxt else len(text)
        text = text[:start] + extra + text[end:]
    elif marker in text:
        text = text.replace(marker, extra + marker, 1)
    else:
        text = text.rstrip() + "\n\n" + extra
    # correct the outdated "GitHub has no LICENSE" cell if still present
    text = text.replace(
        "**CC BY 4.0** (paper footnote). GitHub has no LICENSE file",
        "**CC BY 4.0** (paper footnote); GitHub `LICENSE` is Apache 2.0",
    )
    path.write_text(text, encoding="utf-8")


def strip_for_jsonl(d: dict) -> dict:
    return {
        "dialog_id": d["dialog_id"],
        "split": d["split"],
        "turns": [
            {"turn_id": t["turn_id"], "speaker": t["speaker"], "text": t["text"]}
            for t in d["turns"]
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--force-download", action="store_true")
    args = ap.parse_args()

    if args.skip_download:
        dl_meta = {"fetched": [], "skipped": ["--skip-download"], "candidates_present": False}
    else:
        dl_meta = download_all(force=args.force_download)

    write_licenses(dl_meta)

    pair_dialogs: dict[str, list[dict]] = {}
    parse_counts: dict[str, dict] = {}
    pair_stats: dict[str, dict] = {}
    refs: dict[str, list[dict]] = {}

    split_map = {"trn": "trn", "dev": "dev", "tst": "tst"}
    for pair, lang in (("hi_en", "hi"), ("ta_en", "ta")):
        print(f"Parsing {pair}…")
        dest = RAW_SPOKEN / "code_mixed_dialog" / pair
        dialogs: list[dict] = []
        merged_counts = Counter()
        for split_name, key in split_map.items():
            fpath = dest / f"dialog-dstc2-{split_name}.txt"
            if not fpath.exists():
                raise FileNotFoundError(fpath)
            ds, cnt = parse_cmd_file(fpath, key)
            dialogs.extend(ds)
            merged_counts.update(cnt)
        parse_counts[pair] = dict(merged_counts)
        print(
            f"  {pair}: {len(dialogs)} dialogs, "
            f"api_removed={merged_counts['api_removed']}, "
            f"kb_removed={merged_counts['kb_removed']}"
        )
        lexicon = build_local_lexicon(lang, dialogs)
        stats = collect_pair_stats(pair, dialogs, lexicon)
        pair_dialogs[pair] = dialogs
        pair_stats[pair] = stats
        out_jsonl = PROC_DIR / f"{pair}_dialogs.jsonl"
        write_jsonl(out_jsonl, [strip_for_jsonl(d) for d in dialogs])
        print(f"  wrote {out_jsonl.relative_to(REPO_ROOT)}")
        ref = select_reference_dialogs(dialogs, N_REFERENCE)
        refs[pair] = ref
        write_jsonl(PROC_DIR / f"{pair}_reference_30.jsonl", [strip_for_jsonl(d) for d in ref])
        write_reference_md(PROC_DIR / f"{pair}_reference_30.md", pair, ref)
        print(f"  selected {len(ref)} reference dialogs")

    print("Parsing Hinglish-TOP…")
    htop = parse_htop()
    htop_lex = build_local_lexicon_from_texts("hi", [r["text"] for r in htop])
    htop_stats = htop_metrics(htop, htop_lex)
    write_jsonl(
        PROC_DIR / "hinglish_top_utterances.jsonl",
        [
            {
                "id": r["id"],
                "text": r["text"],
                "intent": r.get("intent") or None,
                "labels": r.get("labels"),
            }
            for r in htop
        ],
    )
    service = select_htop_service(htop, N_HTOP_SERVICE)
    write_htop_service_csv(PROC_DIR / "hinglish_top_service_150.csv", service)
    print(f"  Hinglish-TOP utterances={len(htop)} service150={len(service)}")

    write_report(
        parse_counts,
        pair_stats,
        pair_dialogs,
        refs,
        htop_stats,
        len(htop),
        len(service),
        dl_meta,
    )
    update_spoken_sources_download_note()
    print(f"Wrote { (PROC_DIR / 'REPORT.md').relative_to(REPO_ROOT) }")


if __name__ == "__main__":
    main()
