#!/usr/bin/env python3
"""Re-select strict service candidates and cleaned romanization clusters.

Reads existing processed utterances only. Does not modify mine_* outputs.
Re-run from the repo root:

    .venv/bin/python scripts/calibration/refine_candidates.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from wordfreq import zipf_frequency

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PROC_ROOT = REPO_ROOT / "data" / "calibration" / "processed"
ROMAN_DIR = PROC_ROOT / "romanization"

SEED = 42
PAIRS = ("hi_en", "ta_en", "te_en")
# After the first full-pool pass, hi_en and te_en self-audits were <60%
# service_like. Tightened: ambiguous service unigrams need a nearby
# transaction neighbour; Telugu 'emi' cannot self-match (it is also "what");
# extra domain tokens (opener/tweet/acting/…). Romanization vowel-merge is
# 1:1 and only for a medial vowel between consonants.
TIGHTENED = (
    "ambiguous service terms require a nearby transaction neighbour "
    "(window excludes the term itself); pay/paid need money context "
    "(blocks Hindi postposition pay); te_en emi needs loan/money context; "
    "romanization vowel-merge is 1:1 CVC-medial only"
)
LOCAL = {"hi_en": "hi", "ta_en": "ta", "te_en": "te"}
CMQA_SOURCES = frozenset({"cmqa_tamlish", "chandu_cmqa", "cmqa"})

URL_RE = re.compile(r"(https?://\S+|www\.\S+|t\.co/\S+)", re.I)
BROKEN_URL_RE = re.compile(
    r"(https?\s+)?(//\s+)?t\s*\.\s*co\s*/|\bhttps?\b|\bwww\b",
    re.I,
)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
HANDLE_RE = re.compile(r"(?<!\w)@[\w_]+")
PHONE_RE = re.compile(r"(?:\+?91[\s-]*)?[6-9]\d{9}\b")
WORD_RE = re.compile(r"^[A-Za-z][A-Za-z']*$")
VOWELS = set("aeiou")
CONSONANTS = set("bcdfgjklmnpqrstvwxyz")

# Strong transactional unigrams that can stand alone.
SERVICE_UNIGRAMS = frozenset(
    {
        "payment",
        "refund",
        "courier",
        "atm",
        "upi",
        "otp",
        "emi",
        "bill",
        "recharge",
        "helpline",
        "network",
        "kyc",
        "appointment",
    }
)
# Contextual unigrams: (window, required neighbours).
# book/return/number are from the task brief. The rest were added after a
# first-pass audit (<60% service_like on hi_en and te_en): ambiguous words
# such as job/call/order/account/emi(te) fired on jokes, cricket, and film talk.
MONEY_NEIGHBORS = frozenset(
    {
        "pay",
        "paid",
        "payment",
        "refund",
        "bill",
        "recharge",
        "loan",
        "bank",
        "account",
        "ticket",
        "booking",
        "order",
        "delivery",
        "salary",
        "emi",
        "credit",
        "debit",
        "transfer",
        "upi",
        "atm",
        "wallet",
        "deduct",
        "balance",
        "bal",
        "rent",
        "fee",
        "google",
        "gpay",
        "paytm",
    }
)
SERVICE_CONTEXT = {
    "book": (2, frozenset({"ticket", "seat", "order", "slot", "appointment", "cab", "room"})),
    "return": (3, frozenset({"order", "product", "parcel", "item", "refund"})),
    "number": (3, frozenset({"phone", "mobile", "contact", "call"})),
    "order": (
        3,
        frozenset(
            {
                "place",
                "item",
                "product",
                "cancel",
                "delivery",
                "deliver",
                "food",
                "dish",
                "menu",
                "amazon",
                "flipkart",
                "myntra",
                "swiggy",
                "zomato",
                "cart",
                "refund",
                "parcel",
                "booking",
                "book",
            }
        ),
    ),
    "account": (
        3,
        frozenset(
            {
                "bank",
                "login",
                "password",
                "hack",
                "refund",
                "payment",
                "kyc",
                "aadhaar",
                "aadhar",
                "recover",
                "disabled",
                "deposit",
                "savings",
                "gmail",
                "upi",
                "otp",
            }
        ),
    ),
    "card": (
        3,
        frozenset(
            {
                "atm",
                "credit",
                "debit",
                "bank",
                "sim",
                "aadhaar",
                "aadhar",
                "pan",
                "otp",
                "pin",
                "visa",
                "rupay",
                "graphics",
                "grafics",
            }
        ),
    ),
    "job": (
        3,
        frozenset(
            {
                "salary",
                "interview",
                "apply",
                "offer",
                "vacancy",
                "opening",
                "letter",
                "hr",
                "company",
            }
        ),
    ),
    "call": (
        3,
        frozenset(
            {
                "number",
                "numbers",
                "phone",
                "mobile",
                "customer",
                "helpline",
                "missed",
                "incoming",
                "care",
                "bank",
            }
        ),
    ),
    "pay": (3, MONEY_NEIGHBORS),
    "paid": (3, MONEY_NEIGHBORS),
    "cancel": (
        3,
        frozenset(
            {"order", "ticket", "booking", "recharge", "subscription", "plan", "delivery", "refund", "product"}
        ),
    ),
    "replace": (3, frozenset({"order", "product", "item", "phone", "screen", "defective", "damaged"})),
    "exchange": (3, frozenset({"return", "product", "phone", "item", "currency", "old"})),
    "deliver": (3, frozenset({"order", "amazon", "product", "parcel", "food", "package"})),
    "delivery": (
        3,
        frozenset({"order", "charges", "address", "amazon", "food", "cash", "late", "slot", "partner"}),
    ),
    "complaint": (
        3,
        frozenset(
            {
                "customer",
                "care",
                "service",
                "product",
                "order",
                "bill",
                "bank",
                "refund",
                "scam",
                "fraud",
                "app",
                "light",
                "power",
            }
        ),
    ),
    "aadhaar": (3, frozenset({"update", "kyc", "card", "bank", "uidai", "enroll", "biometric", "verify"})),
    "aadhar": (3, frozenset({"update", "kyc", "card", "bank", "uidai", "enroll", "biometric", "verify"})),
    "bank": (
        3,
        frozenset(
            {"account", "atm", "loan", "branch", "deduct", "balance", "card", "transfer", "sbi", "deposit"}
        ),
    ),
    "salary": (3, frozenset({"job", "month", "credit", "account", "slip", "hike", "credited", "company"})),
    "sim": (3, frozenset({"card", "recharge", "network", "phone", "mobile", "jio", "airtel", "prepaid"})),
    "loan": (3, frozenset({"bank", "emi", "interest", "apply", "home", "car", "student", "monthly"})),
    "parcel": (3, frozenset({"order", "delivery", "courier", "amazon", "track", "ship"})),
    "booking": (3, frozenset({"ticket", "book", "cab", "room", "slot", "appointment", "hotel", "flight", "train"})),
    "interview": (3, frozenset({"job", "hr", "salary", "offer", "campus", "appointment"})),
    "office": (3, frozenset({"branch", "appointment", "hours", "visit", "document", "kyc"})),
    "document": (3, frozenset({"kyc", "aadhaar", "aadhar", "pan", "upload", "submit", "verify"})),
    "pin": (3, frozenset({"atm", "card", "otp", "debit", "forgot"})),
    "ticket": (
        3,
        frozenset({"book", "booking", "cancel", "train", "bus", "flight", "seat", "show"}),
    ),
    "transfer": (
        3,
        frozenset({"bank", "account", "money", "amount", "neft", "upi", "imps", "refund"}),
    ),
    "balance": (3, frozenset({"account", "bank", "low", "recharge", "check", "enquiry", "bal"})),
    "branch": (3, frozenset({"bank", "office", "atm", "visit"})),
    "rs": (4, MONEY_NEIGHBORS),
    "rupees": (4, MONEY_NEIGHBORS),
    "rupaye": (4, MONEY_NEIGHBORS),
    "paise": (4, MONEY_NEIGHBORS),
    "panam": (4, MONEY_NEIGHBORS),
    "dabbulu": (4, MONEY_NEIGHBORS),
    "amount": (4, MONEY_NEIGHBORS),
}
# Telugu 'emi' is also the WH-word "what"; require loan/money context.
EMI_TE_NEED = frozenset(
    {"loan", "bank", "card", "salary", "month", "rs", "amount", "dabbulu", "installment"}
)
ORDER_NEG = frozenset({"middle", "descending", "batting", "opener"})
SERVICE_PHRASES = (
    (("customer", "care"), "customer care"),
    (("customer", "service"), "customer service"),
    (("offer", "letter"), "offer letter"),
    (("pan", "card"), "pan card"),
)
REVIEW_SAFE = frozenset({"product", "order", "app"})

REQUEST = {
    "hi_en": frozenset(
        {
            "aap",
            "aapka",
            "aapko",
            "aapke",
            "kijiye",
            "dijiye",
            "karein",
            "kariye",
            "bataiye",
            "batao",
            "bolo",
            "boliye",
            "please",
            "pls",
            "plz",
            "sir",
            "madam",
            "ji",
            "bhaiya",
            "kripya",
            "kya",
            "kab",
            "kitna",
            "kitne",
            "kaise",
            "kyun",
            "haan",
            "nahi",
        }
    ),
    "ta_en": frozenset(
        {
            "neenga",
            "unga",
            "ungaluku",
            "sollunga",
            "pannunga",
            "kudunga",
            "anuppunga",
            "paarunga",
            "please",
            "pls",
            "sir",
            "madam",
            "anna",
            "enna",
            "epdi",
            "eppadi",
            "eppo",
            "evlo",
            "yen",
            "aama",
            "illa",
            "venum",
            "vendum",
        }
    ),
    "te_en": frozenset(
        {
            "meeru",
            "mee",
            "miru",
            "garu",
            "cheppandi",
            "cheyandi",
            "cheyyandi",
            "pampandi",
            "ivvandi",
            "chudandi",
            "please",
            "pls",
            "sir",
            "madam",
            "anna",
            "enti",
            "ela",
            "eppudu",
            "entha",
            "enduku",
            "avunu",
            "ledu",
            "kavali",
        }
    ),
}

DOMAIN_UNIGRAMS = frozenset(
    {
        "movie",
        "movies",
        "film",
        "cinema",
        "cinemalo",
        "song",
        "songs",
        "trailer",
        "teaser",
        "hero",
        "heroine",
        "actor",
        "actress",
        "director",
        "music",
        "bgm",
        "video",
        "videos",
        "youtube",
        "channel",
        "subscribe",
        "subscribers",
        "views",
        "like",
        "likes",
        "comment",
        "comments",
        "episode",
        "serial",
        "match",
        "cricket",
        "wicket",
        "bowling",
        "batting",
        "bowler",
        "batsman",
        "team",
        "ipl",
        "fans",
        "fan",
        "election",
        "vote",
        "bjp",
        "congress",
        "party",
        "minister",
        "modi",
        "god",
        "record",
        "records",
        "guinness",
        "collection",
        "release",
        "released",
        "review",
        "opener",
        "innings",
        "century",
        "stadium",
        "auction",
        "tweet",
        "tweets",
        "acting",
        "descending",
        "confession",
    }
)
DOMAIN_PHRASES = (("box", "office"),)

# Reused from mine_hi_en.py / mine_ta_en.py / mine_te_en.py.
PROFANITY = {
    "hi_en": frozenset(
        {
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
    ),
    "ta_en": frozenset(
        {
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
    ),
    "te_en": frozenset(
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
    ),
}

LICENSE = {
    "lince": ("restricted_twitter_research", "no"),
    "lince_pos": ("restricted_twitter_research", "no"),
    "phinc": ("cc-by-4.0", "yes"),
    "hinge": ("cc-by-4.0", "yes"),
    "sentimix": ("cc-by-4.0", "yes"),
    "tamilmixsentiment": ("UNCLEAR - stats only, do not redistribute", "no"),
    "offenseval_dravidian": ("cc-by-4.0", "yes"),
    "tanglish_corpus_185k": ("cc-by-4.0", "no"),
    "coli_dravidian_ta": ("UNCLEAR - stats only, do not redistribute", "no"),
    "cmqa_tamlish": ("cmu-research-notice-required", "yes"),
    "cmtet_sentiment": ("UNCLEAR - stats only, do not redistribute", "no"),
    "cmtet_lid": ("UNCLEAR - stats only, do not redistribute", "no"),
    "icon_2016": ("UNCLEAR - stats only, do not redistribute", "no"),
    "chandu_cmqa": ("CMU COPYING - retain notice", "yes"),
}

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


def norm_tok(tok: str) -> str:
    return re.sub(r"[^a-z0-9]", "", tok.lower())


def is_roman_script(text: str) -> bool:
    for ch in text:
        if ch.isalpha() and not ("a" <= ch.lower() <= "z"):
            return False
    return True


def has_contact_remnant(text: str, tokens: list[str]) -> bool:
    if URL_RE.search(text) or BROKEN_URL_RE.search(text):
        return True
    if EMAIL_RE.search(text) or HANDLE_RE.search(text) or PHONE_RE.search(text):
        return True
    for tok in tokens:
        low = tok.lower().strip(".,;:!?()[]{}")
        if low in {"http", "https", "www", "t.co", "tco", "bit.ly"}:
            return True
        if low.startswith("http") or low.startswith("www.") or "t.co" in low:
            return True
        if low.startswith("@") and len(low) > 1:
            return True
        digits = re.sub(r"\D", "", tok)
        if len(digits) >= 10 and PHONE_RE.search(tok):
            return True
    return False


def utterance_metrics(tags: list[str], local: str) -> dict:
    n_en = sum(1 for t in tags if t == "en")
    n_loc = sum(1 for t in tags if t == local)
    den = n_en + n_loc
    if den == 0:
        return {"english_share": None, "cmi": None, "n_switches": 0}
    switches = 0
    for a, b in zip(tags, tags[1:]):
        if a in {"en", local} and b in {"en", local} and a != b:
            switches += 1
    return {
        "english_share": n_en / den,
        "cmi": 100.0 * (1.0 - (max(n_en, n_loc) / den)),
        "n_switches": switches,
    }


def window_has(lows: list[str], i: int, radius: int, need: frozenset[str]) -> bool:
    lo = max(0, i - radius)
    hi = min(len(lows), i + radius + 1)
    others = lows[lo:i] + lows[i + 1 : hi]
    return bool(need.intersection(others))


def context_table(pair: str) -> dict[str, tuple[int, frozenset[str]]]:
    ctx = dict(SERVICE_CONTEXT)
    if pair == "te_en":
        ctx["emi"] = (3, EMI_TE_NEED)
    return ctx


def service_terms(tokens: list[str], pair: str) -> list[str]:
    lows = [norm_tok(t) for t in tokens]
    ctx = context_table(pair)
    hits: list[str] = []
    consumed: set[int] = set()
    for i, _tok in enumerate(lows):
        for parts, label in SERVICE_PHRASES:
            n = len(parts)
            if i + n <= len(lows) and tuple(lows[i : i + n]) == parts:
                hits.append(label)
                consumed.update(range(i, i + n))
                break
    for i, tok in enumerate(lows):
        if i in consumed or not tok:
            continue
        if tok == "order" and window_has(lows, i, 3, ORDER_NEG):
            continue
        if tok in ctx:
            radius, need = ctx[tok]
            if window_has(lows, i, radius, need):
                hits.append(tok)
            continue
        if tok in SERVICE_UNIGRAMS:
            if pair == "te_en" and tok == "emi":
                continue
            hits.append(tok)
    # distinct, stable order
    seen: set[str] = set()
    out: list[str] = []
    for h in hits:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


def request_markers(tokens: list[str], pair: str) -> list[str]:
    bag = {norm_tok(t) for t in tokens}
    return sorted(m for m in REQUEST[pair] if m in bag)


def domain_blocked(tokens: list[str], pair: str) -> bool:
    lows = [norm_tok(t) for t in tokens]
    for i, tok in enumerate(lows):
        if tok == "review":
            if window_has(lows, i, 3, REVIEW_SAFE):
                continue
            return True
        if tok in DOMAIN_UNIGRAMS:
            return True
        if tok in PROFANITY[pair]:
            return True
    for a, b in DOMAIN_PHRASES:
        for i in range(len(lows) - 1):
            if lows[i] == a and lows[i + 1] == b:
                return True
    return False


def has_amount(tokens: list[str]) -> bool:
    for tok in tokens:
        if re.search(r"\d", tok):
            return True
        if "₹" in tok or "rs" in tok.lower() and re.search(r"\d", tok):
            return True
    return False


def fivegrams(tokens: list[str]) -> set[tuple[str, ...]]:
    lows = [norm_tok(t) for t in tokens if norm_tok(t)]
    if len(lows) < 5:
        return set()
    return {tuple(lows[i : i + 5]) for i in range(len(lows) - 4)}


def source_item_id(rec: dict, source_id: str) -> str:
    if rec.get("source_item_id"):
        return str(rec["source_item_id"])
    uid = str(rec.get("id") or "")
    prefix = source_id + ":"
    if uid.startswith(prefix):
        return uid[len(prefix) :]
    return uid


def item_cluster(source_id: str, item_id: str) -> str:
    if "youtube/" in item_id:
        return "yt:" + item_id.split("youtube/", 1)[-1]
    return f"{source_id}:{item_id}"


def load_utterances(pair: str) -> list[dict]:
    path = PROC_ROOT / pair / "utterances.jsonl"
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def select_service(
    pair: str,
    rows: list[dict],
    min_service_terms: int,
    limit: int = 200,
) -> tuple[list[dict], int]:
    local = LOCAL[pair]
    scored: list[tuple] = []
    n_qual = 0
    for rec in rows:
        source_id = rec.get("source_id") or ""
        if source_id in CMQA_SOURCES:
            continue
        tokens = rec.get("tokens") or []
        text = rec.get("text") or ""
        tags = rec.get("lang_tags") or []
        if not (5 <= len(tokens) <= 30):
            continue
        if not is_roman_script(text):
            continue
        if has_contact_remnant(text, tokens):
            continue
        mets = utterance_metrics(tags, local)
        if mets["cmi"] is None or mets["cmi"] <= 0:
            continue
        if domain_blocked(tokens, pair):
            continue
        svc = service_terms(tokens, pair)
        req = request_markers(tokens, pair)
        if len(svc) < min_service_terms or not req:
            continue
        n_qual += 1
        bonus = 1 if has_amount(tokens) else 0
        score = len(svc) + len(req) + bonus
        item_id = source_item_id(rec, source_id)
        scored.append((score, mets["cmi"], mets["n_switches"], rec, svc, req, mets, item_id))

    scored.sort(key=lambda x: (-x[0], -x[1], -x[2], x[3].get("text") or ""))
    selected: list[dict] = []
    used_grams: set[tuple[str, ...]] = set()
    per_cluster: Counter[str] = Counter()
    for score, _cmi, _nsw, rec, svc, req, mets, item_id in scored:
        source_id = rec.get("source_id") or ""
        cluster = item_cluster(source_id, item_id)
        if per_cluster[cluster] >= 3:
            continue
        grams = fivegrams(rec.get("tokens") or [])
        if grams and grams & used_grams:
            continue
        used_grams |= grams
        per_cluster[cluster] += 1
        lic, redis = LICENSE.get(source_id, ("UNCLEAR", "no"))
        selected.append(
            {
                "cand_id": f"{pair}_s{len(selected) + 1:03d}",
                "source_id": source_id,
                "source_item_id": item_id,
                "text": rec.get("text") or "",
                "english_share": ""
                if mets["english_share"] is None
                else f"{mets['english_share']:.4f}",
                "cmi": "" if mets["cmi"] is None else f"{mets['cmi']:.4f}",
                "n_switches": mets["n_switches"],
                "service_terms": "|".join(svc),
                "request_markers": "|".join(req),
                "score": score,
                "license_status": lic,
                "redistributable": redis,
                "accept": "",
                "reject_reason": "",
                "reviewer_note": "",
            }
        )
        if len(selected) >= limit:
            break
    return selected, n_qual


def write_candidates(pair: str, rows: list[dict]) -> None:
    path = PROC_ROOT / pair / "service_candidates_strict.csv"
    fields = [
        "cand_id",
        "source_id",
        "source_item_id",
        "text",
        "english_share",
        "cmi",
        "n_switches",
        "service_terms",
        "request_markers",
        "score",
        "license_status",
        "redistributable",
        "accept",
        "reject_reason",
        "reviewer_note",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def skeleton_key(tok: str) -> str:
    """Improved romanization key.

    Vowel-length mappings are applied before collapsing remaining repeats so
    ee->i and oo->u are not wiped by the collapse step.
    """
    t = tok.lower()
    t = re.sub(r"[^a-z]", "", t)
    if not t:
        return ""
    for _ in range(4):
        nxt = t.replace("aa", "a").replace("ee", "i").replace("ii", "i")
        nxt = nxt.replace("oo", "u").replace("uu", "u")
        if nxt == t:
            break
        t = nxt
    t = re.sub(r"(.)\1+", r"\1", t)
    t = re.sub(r"([bcdfgjklmnpqrstvwxyz])h", r"\1", t)
    return t


def load_dakshina_sets(lang: str) -> list[set[str]]:
    path = ROMAN_DIR / f"{lang}_variants.tsv"
    sets: list[set[str]] = []
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for rec in reader:
            variants = rec.get("variants") or ""
            forms = {re.sub(r"[^a-z]", "", v.lower()) for v in variants.split("|")}
            forms.discard("")
            if len(forms) >= 2:
                sets.append(forms)
    return sets


def dakshina_attested(spellings: list[str], dak_sets: list[set[str]]) -> bool:
    forms = {re.sub(r"[^a-z]", "", s.lower()) for s in spellings}
    forms.discard("")
    if len(forms) < 2:
        return False
    for dset in dak_sets:
        if len(forms & dset) >= 2:
            return True
    return False


def vowel_deletions(key: str) -> list[str]:
    """Medial vowel between two consonants only (blocks hain/haan, kya/kiya chains)."""
    out = []
    for i, ch in enumerate(key):
        if ch not in VOWELS or i == 0 or i == len(key) - 1:
            continue
        if key[i - 1] in CONSONANTS and key[i + 1] in CONSONANTS:
            out.append(key[:i] + key[i + 1 :])
    return out


def medial_a_drops(key: str) -> list[str]:
    out = []
    for i in range(1, len(key) - 1):
        if key[i] != "a":
            continue
        if key[i - 1] in CONSONANTS and key[i + 1] in CONSONANTS:
            out.append(key[:i] + key[i + 1 :])
    return out


class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if ra < rb:
            self.parent[rb] = ra
        else:
            self.parent[ra] = rb


def build_clusters(pair: str, rows: list[dict], top_n: int = 300) -> list[dict]:
    local = LOCAL[pair]
    forms: dict[str, Counter[str]] = defaultdict(Counter)
    for rec in rows:
        tokens = rec.get("tokens") or []
        tags = rec.get("lang_tags") or []
        origin = rec.get("tag_origin") or "auto"
        for tok, tag in zip(tokens, tags):
            if not WORD_RE.match(tok) or len(tok) < 2:
                continue
            if tag != local:
                continue
            # gold: trust gold local tag; auto: already lexicon-tagged local
            if origin != "gold" and tag != local:
                continue
            if zipf_en(tok) >= 3.0:
                continue
            key = skeleton_key(tok)
            if len(key) < 2:
                continue
            forms[key][tok.lower()] += 1

    totals = {k: int(sum(c.values())) for k, c in forms.items()}
    keys = list(forms)
    uf = UnionFind(keys)

    if pair == "ta_en":
        present = set(keys)
        for key in keys:
            for dropped in medial_a_drops(key):
                if dropped in present:
                    uf.union(key, dropped)

    # Merge A/B only when B is A with one vowel removed, both freq>=20, and
    # the short key is not a shared stem of several long keys (avoids chaining
    # kya/kiya, epdi/ipdi/apdi, aapka/aapke through apk/pdi).
    parents: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        if totals[key] < 20 or len(key) < 4:
            continue
        for dropped in vowel_deletions(key):
            if dropped in forms and totals[dropped] >= 20 and len(dropped) >= 3:
                parents[dropped].append(key)
    short_to_long = {
        short: list(set(longs))[0]
        for short, longs in parents.items()
        if len(set(longs)) == 1
    }
    long_to_shorts: dict[str, list[str]] = defaultdict(list)
    for short, long in short_to_long.items():
        long_to_shorts[long].append(short)
    for long, shorts in long_to_shorts.items():
        if len(shorts) == 1:
            uf.union(long, shorts[0])

    merged: dict[str, Counter[str]] = defaultdict(Counter)
    for key, counter in forms.items():
        merged[uf.find(key)] += counter

    clusters = []
    for key, counter in merged.items():
        if len(counter) < 2:
            continue
        clusters.append(
            {
                "key": key,
                "spellings": counter,
                "total": int(sum(counter.values())),
            }
        )
    clusters.sort(key=lambda r: (-r["total"], -len(r["spellings"]), r["key"]))
    return clusters[:top_n]


def write_clusters(pair: str, clusters: list[dict], dak_sets: list[set[str]]) -> None:
    path = PROC_ROOT / pair / "romanization_clusters_clean.tsv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["key", "spellings", "total", "dakshina_attested"])
        for cl in clusters:
            spell = "|".join(f"{form}:{n}" for form, n in cl["spellings"].most_common())
            att = dakshina_attested(list(cl["spellings"]), dak_sets)
            cl["dakshina_attested"] = "yes" if att else "no"
            w.writerow([cl["key"], spell, cl["total"], cl["dakshina_attested"]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", nargs="+", default=list(PAIRS))
    parser.add_argument("--min-service-terms", type=int, default=1)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    for pair in args.pairs:
        log(f"=== {pair} ===")
        rows = load_utterances(pair)
        log(f"  pool: {len(rows)}")
        cands, n_qual = select_service(pair, rows, args.min_service_terms, args.limit)
        write_candidates(pair, cands)
        log(f"  qualifying: {n_qual}; kept: {len(cands)} (min_service_terms={args.min_service_terms})")
        if cands:
            log("  top 8:")
            for c in cands[:8]:
                log(f"    [{c['score']}] {c['service_terms']} || {c['text'][:140]}")

        lang = LOCAL[pair]
        dak = load_dakshina_sets(lang)
        clusters = build_clusters(pair, rows, top_n=300)
        write_clusters(pair, clusters, dak)
        log(f"  romanization clusters: {len(clusters)}")
        for cl in clusters[:10]:
            spell = ", ".join(f"{f}={n}" for f, n in cl["spellings"].most_common(6))
            log(f"    {cl['key']} n={cl['total']} dak={cl['dakshina_attested']}: {spell}")


if __name__ == "__main__":
    main()
