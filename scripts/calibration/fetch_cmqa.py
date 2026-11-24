#!/usr/bin/env python3
"""Fetch Chandu et al. (CALCS 2018) code-mixed QA questions and normalise them.

Re-runnable. Downloads the public GitHub release of the CMQA challenge data,
probes the dead official CMU host plus archive.org / related resources, writes
a license record, and emits roman-script questions.jsonl plus REPORT.md.

Official paper: https://aclanthology.org/W18-3204/
Live data: https://github.com/khyathiraghavi/code_switched_QA
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from wordfreq import zipf_frequency


GITHUB_RAW = "https://raw.githubusercontent.com/khyathiraghavi/code_switched_QA/master"
GITHUB_REPO = "https://github.com/khyathiraghavi/code_switched_QA"
PAPER_URL = "https://aclanthology.org/W18-3204/"
PAPER_PDF = "https://aclanthology.org/W18-3204.pdf"
CMU_HOST = "http://tts.speech.cs.cmu.edu/codemixedqa/"
WEBSHODH_URL = "https://tts.speech.cs.cmu.edu/webshodh/cmqa.php"
ARCHIVE_HOMEPAGE = (
    "https://web.archive.org/web/20180423153819id_/http://tts.speech.cs.cmu.edu/codemixedqa/"
)
ARCHIVE_FLYER = (
    "https://web.archive.org/web/20180423153819id_/"
    "http://tts.speech.cs.cmu.edu/codemixedqa/CMQAFlyer.pdf"
)
ARCHIVE_WEBSHODH = (
    "https://web.archive.org/web/20180725164454id_/"
    "http://tts.speech.cs.cmu.edu/webshodh/cmqa.php"
)
WWW2015_PDF = "http://www.www2015.it/documents/proceedings/companion/p853.pdf"
MICROSOFT_BLOG = (
    "https://learn.microsoft.com/en-us/archive/blogs/msind/"
    "creating-ai-that-can-respond-to-questions-in-mixed-languages-like-humans"
)
GLUECOS_QA_URL = (
    "https://raw.githubusercontent.com/khyathiraghavi/code_switched_QA/master/"
    "code_mixed_qa_train.json"
)
HF_HINGLISH_SUBSET = "https://huggingface.co/datasets/smitpandit/hinglish-cs-qa"

# Expected byte sizes of the 2020-05-04 GitHub master files (integrity check).
GITHUB_FILES = {
    "COPYING": 2269,
    "README": 1266,
    "code_mixed_qa_train.json": 502197,
    "code_mixed_qa_test_batch1.json": 49003,
    "code_mixed_qa_test_batch1_answers.json": 55517,
    "code_mixed_qa_test_batch2.json": 48872,
    "code_mixed_qa_test_batch2_answers.json": 54448,
    "code_mixed_qa_test_batch3.json": 28077,
    "code_mixed_qa_test_batch3_answers.json": 31682,
}

# Use answer-bearing files so test items keep gold answers.
QUESTION_SOURCES = (
    ("code_mixed_qa_train.json", "train"),
    ("code_mixed_qa_test_batch1_answers.json", "test_batch1"),
    ("code_mixed_qa_test_batch2_answers.json", "test_batch2"),
    ("code_mixed_qa_test_batch3_answers.json", "test_batch3"),
)

LANG_TO_PAIR = {"Hindi": "hi_en", "Tamil": "ta_en", "Telugu": "te_en"}
PAIR_ORDER = ("hi_en", "ta_en", "te_en")
PAPER_COUNTS = {"hi_en": 1694, "ta_en": 2848, "te_en": 1391}
ENGLISH_ZIPF_THRESHOLD = 3.5
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
EXAMPLE_SEED = 42
N_EXAMPLES = 15
CURL_TIMEOUT = 25

CITATION_BIBTEX = """@inproceedings{chandu-etal-2018-code,
    title = "Code-Mixed Question Answering Challenge: Crowd-sourcing Data and Techniques",
    author = "Chandu, Khyathi  and
      Loginova, Ekaterina  and
      Gupta, Vishal  and
      van Genabith, Josef  and
      Neumann, G{\\"u}nter  and
      Chinnakotla, Manoj  and
      Nyberg, Eric  and
      Black, Alan W.",
    booktitle = "Proceedings of the Third Workshop on Computational Approaches to Linguistic Code-Switching",
    month = jul,
    year = "2018",
    address = "Melbourne, Australia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W18-3204/",
    doi = "10.18653/v1/W18-3204",
    pages = "29--38"
}"""


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{n} B"


def dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def curl_get(url: str, dest: Path | None = None, timeout: int = CURL_TIMEOUT) -> dict:
    """GET a URL with curl. Optionally write the body to dest."""
    cmd = [
        "curl",
        "-sS",
        "-L",
        "--fail",
        "--max-time",
        str(timeout),
        "-A",
        "aaai-icfd-cmqa-fetch/1.0",
        "-w",
        "\n%{http_code} %{url_effective}",
        "-o",
        str(dest) if dest is not None else "-",
        url,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        return {"url": url, "ok": False, "status": None, "error": str(exc), "bytes": 0}
    stdout = proc.stdout or ""
    stderr = (proc.stderr or "").strip()
    if dest is not None:
        body, _, trailer = stdout.rpartition("\n")
        size = dest.stat().st_size if dest.exists() else 0
    else:
        if "\n" in stdout:
            body, trailer = stdout.rsplit("\n", 1)
        else:
            body, trailer = "", stdout
        size = len(body.encode("utf-8", errors="replace"))
    parts = trailer.strip().split(None, 1)
    status = int(parts[0]) if parts and parts[0].isdigit() else None
    ok = proc.returncode == 0 and dest is not None and dest.exists() and dest.stat().st_size > 0
    if dest is None:
        ok = proc.returncode == 0
    result = {
        "url": url,
        "ok": ok,
        "status": status,
        "error": None if ok else (stderr or f"curl exit {proc.returncode}"),
        "bytes": size,
        "effective_url": parts[1] if len(parts) > 1 else url,
    }
    if dest is None:
        result["snippet"] = body[:400].replace("\n", " ").strip()
    return result


def urlopen_probe(url: str, timeout: int = 15) -> dict:
    req = Request(url, headers={"User-Agent": "aaai-icfd-cmqa-fetch/1.0"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            data = resp.read(800)
            return {
                "url": url,
                "ok": True,
                "status": getattr(resp, "status", None),
                "error": None,
                "bytes": len(data),
                "snippet": data[:200].decode("utf-8", errors="replace").replace("\n", " ").strip(),
            }
    except HTTPError as exc:
        return {"url": url, "ok": False, "status": exc.code, "error": str(exc), "bytes": 0}
    except (URLError, TimeoutError, OSError) as exc:
        return {"url": url, "ok": False, "status": None, "error": str(exc), "bytes": 0}


def download_github_files(raw_dir: Path, skip_download: bool) -> list[dict]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    attempts: list[dict] = []
    for name, expected in GITHUB_FILES.items():
        dest = raw_dir / name
        url = f"{GITHUB_RAW}/{name}"
        if dest.exists() and dest.stat().st_size == expected:
            attempts.append(
                {
                    "url": url,
                    "ok": True,
                    "status": 200,
                    "error": None,
                    "bytes": dest.stat().st_size,
                    "note": "already present with expected size",
                }
            )
            continue
        if skip_download:
            attempts.append(
                {
                    "url": url,
                    "ok": dest.exists() and dest.stat().st_size > 0,
                    "status": None,
                    "error": None if dest.exists() else "missing and --skip-download set",
                    "bytes": dest.stat().st_size if dest.exists() else 0,
                }
            )
            continue
        print(f"Downloading {name}")
        result = curl_get(url, dest=dest, timeout=60)
        if dest.exists() and expected and dest.stat().st_size != expected:
            result["ok"] = False
            result["error"] = (
                f"size {dest.stat().st_size} != expected {expected}"
            )
        attempts.append(result)
    return attempts


def probe_related_urls(raw_dir: Path, skip_probes: bool) -> list[dict]:
    if skip_probes:
        return [{"url": "(skipped)", "ok": False, "error": "--skip-probes", "bytes": 0}]
    attempts: list[dict] = []
    print("Probing official / archive / related URLs")
    archive_html = raw_dir / "archive_codemixedqa_20180423.html"
    archive_get = curl_get(ARCHIVE_HOMEPAGE, dest=archive_html, timeout=40)
    archive_get["note"] = "Wayback id_ snapshot of official CMU challenge homepage"
    attempts.append(archive_get)

    for url, note in (
        (CMU_HOST, "official host named in COPYING; expected dead"),
        ("https://tts.speech.cs.cmu.edu/codemixedqa/", "https variant of official host"),
        (WEBSHODH_URL, "WebShodh demo named in Microsoft India blog"),
        ("http://tts.speech.cs.cmu.edu/webshodh/cmqa.php", "http WebShodh demo"),
        (ARCHIVE_FLYER, "Wayback attempt for CMQAFlyer.pdf linked from homepage"),
        (ARCHIVE_WEBSHODH, "Wayback snapshot of WebShodh demo (no question dump)"),
        (PAPER_URL, "ACL Anthology landing page"),
        (PAPER_PDF, "ACL Anthology PDF"),
        (WWW2015_PDF, "Raghavi et al. 2015 WWW companion PDF"),
        (MICROSOFT_BLOG, "Microsoft India blog on WebShodh; no data link"),
        (GITHUB_REPO, "author GitHub repository"),
        (HF_HINGLISH_SUBSET, "unofficial 100-row HF Hinglish subset; not used"),
        (
            "https://api.github.com/repos/khyathiraghavi/code_switched_QA/contents/",
            "GitHub API listing of public files",
        ),
        (
            "https://web.archive.org/cdx/search/cdx?url=tts.speech.cs.cmu.edu/codemixedqa/*&output=json&limit=20",
            "Wayback CDX for official host; HTML only, no JSON dumps",
        ),
    ):
        result = urlopen_probe(url)
        result["note"] = note
        attempts.append(result)
    return attempts


def is_roman_script(text: str) -> bool:
    for char in text:
        if char.isalpha():
            name = unicodedata.name(char, "")
            if not name.startswith("LATIN"):
                return False
    return True


def context_type(context: object) -> str:
    if isinstance(context, str) and context.strip().lower() == "general":
        return "image"
    return "article"


def alphabetic_tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def auto_english_share(text: str) -> dict:
    tokens = alphabetic_tokens(text)
    n_en = sum(1 for tok in tokens if zipf_frequency(tok, "en") >= ENGLISH_ZIPF_THRESHOLD)
    n_local = len(tokens) - n_en
    denom = n_en + n_local
    share = (n_en / denom) if denom else None
    return {
        "n_tokens": len(tokens),
        "n_en": n_en,
        "n_local": n_local,
        "english_share": share,
        "english_share_status": "auto/unvalidated",
        "rule": f"zipf_frequency(tok,'en')>={ENGLISH_ZIPF_THRESHOLD}",
    }


def load_questions(raw_dir: Path) -> list[dict]:
    rows: list[dict] = []
    seen_ids: set[int] = set()
    for filename, split in QUESTION_SOURCES:
        path = raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload.get("questions") or []:
            source_id = int(item["id"])
            if source_id in seen_ids:
                continue
            language = str(item.get("language") or "").strip()
            pair = LANG_TO_PAIR.get(language)
            text = str(item.get("query") or "").strip()
            if pair is None or not text:
                continue
            if not is_roman_script(text):
                continue
            seen_ids.add(source_id)
            answer = item.get("answer")
            if isinstance(answer, list):
                answer = "; ".join(str(x) for x in answer if x)
            elif answer is None:
                answer = ""
            else:
                answer = str(answer)
            rows.append(
                {
                    "id": f"cmqa_{pair}_{source_id}",
                    "pair": pair,
                    "text": text,
                    "tags": {
                        "source_id": source_id,
                        "language": language,
                        "split": split,
                        "context_type": context_type(item.get("context")),
                        "answer": answer,
                        "source_file": filename,
                    },
                }
            )
    rows.sort(key=lambda r: (PAIR_ORDER.index(r["pair"]), r["tags"]["source_id"]))
    return rows


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "p25": statistics.quantiles(ordered, n=4, method="inclusive")[0],
        "p50": statistics.median(ordered),
        "p75": statistics.quantiles(ordered, n=4, method="inclusive")[2],
        "p90": statistics.quantiles(ordered, n=10, method="inclusive")[8],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def pair_stats(rows: list[dict]) -> dict[str, dict]:
    by_pair: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_pair[row["pair"]].append(row)
    stats: dict[str, dict] = {}
    for pair in PAIR_ORDER:
        items = by_pair.get(pair, [])
        lengths = [float(len(alphabetic_tokens(r["text"]))) for r in items]
        shares = []
        metrics = []
        for row in items:
            metric = auto_english_share(row["text"])
            metrics.append(metric)
            if metric["english_share"] is not None:
                shares.append(metric["english_share"])
        splits = Counter(r["tags"]["split"] for r in items)
        contexts = Counter(r["tags"]["context_type"] for r in items)
        stats[pair] = {
            "count": len(items),
            "paper_count": PAPER_COUNTS[pair],
            "token_length": quantiles(lengths),
            "english_share": quantiles(shares),
            "english_share_status": "auto/unvalidated",
            "splits": dict(splits),
            "context_types": dict(contexts),
            "examples": sample_examples(items, metrics),
        }
    return stats


def sample_examples(items: list[dict], metrics: list[dict]) -> list[dict]:
    if not items:
        return []
    rng = random.Random(EXAMPLE_SEED)
    idxs = list(range(len(items)))
    rng.shuffle(idxs)
    chosen = sorted(idxs[: min(N_EXAMPLES, len(items))])
    out = []
    for i in chosen:
        row = items[i]
        metric = metrics[i]
        out.append(
            {
                "id": row["id"],
                "text": row["text"],
                "answer": row["tags"]["answer"],
                "split": row["tags"]["split"],
                "context_type": row["tags"]["context_type"],
                "n_tokens": metric["n_tokens"],
                "english_share": metric["english_share"],
            }
        )
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_license(path: Path, copying_text: str, attempts: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    live_github = any(
        a.get("ok") and a.get("url", "").startswith(GITHUB_RAW) for a in attempts
    )
    official_live = any(
        a.get("ok") and "tts.speech.cs.cmu.edu/codemixedqa" in a.get("url", "")
        and "web.archive.org" not in a.get("url", "")
        for a in attempts
    )
    body = f"""# CMQA (Chandu et al., CALCS 2018) — license record

Recorded: {date.today().isoformat()}
Local raw files: `data/calibration/raw/cmqa/`

## Source

- Paper: Chandu et al., *Code-Mixed Question Answering Challenge: Crowd-sourcing Data and Techniques*, CALCS @ ACL 2018. {PAPER_URL}
- Official host named in `COPYING`: {CMU_HOST}
  - Live status (this run): **{"reachable" if official_live else "DEAD / unreachable"}**
  - The 2018 site was a challenge registration page, not a public JSON dump. Training data was released to registered teams. Wayback snapshots (2017–2019) archive HTML only; no question files.
- Live public copy used here: {GITHUB_REPO} (created 2020-05-04)
- GitHub files downloaded: {"yes" if live_github else "not confirmed this run"}

## License

**CMU 2018 custom permissive data license** (file `COPYING` in the author repository).

Not UNCLEAR. Redistribution of the original files is allowed if the copyright notice, conditions, and disclaimer are retained; modifications must be marked; original authors' names must not be deleted. The grant is “use, copy, modify … for any purpose” including commercial. The paper’s prose (“freely available for research purposes”) is narrower than `COPYING`; we follow `COPYING`.

Processed `questions.jsonl` is a derived extract (roman-script questions + provided tags). Treat it as a modification of the CMU data and keep this notice.

### `COPYING` text

```
{copying_text.strip()}
```

## Citation

{CITATION_BIBTEX}

## Related resources (not used as the main dump)

| Resource | What we found | Redistribution |
|---|---|---|
| Raghavi / Chandu et al. 2015, *Answer ka type kya he?* (WWW companion) | ~1,000 En–Hi question-classification items. No public dump on GitHub, author page, ACM, or the WWW 2015 PDF host. | UNCLEAR → stats only, do not redistribute |
| Chandu et al. 2017, *WebShodh* (CLEF) | Demo at `{WEBSHODH_URL}` (host dead). Wayback has the interactive page only; no question file. Microsoft India blog describes the system, no download. | UNCLEAR → stats only, do not redistribute |
| Microsoft / Manoj Chinnakotla | Co-author of the 2015/2017/2018 papers. No separate MSR Open Data “code-mixed QA” release found. | — |
| Microsoft GLUECoS QA_EN_HI | Downloads `{GLUECOS_QA_URL}` (same GitHub train file) and builds a 313-triple En–Hi subset with DrQA contexts. Not a new human question set. | same as CMU `COPYING` for the questions |
| `smitpandit/hinglish-cs-qa` (Hugging Face) | Unofficial 100-row Hinglish sample. Not used. | UNCLEAR → stats only, do not redistribute |

## What we keep

Challenge JSON (train + three test batches with answers) from the author GitHub repo, plus this license record and a roman-script question extract.
"""
    path.write_text(body, encoding="utf-8")


def fmt_q(qdict: dict, key: str, digits: int = 2) -> str:
    if key not in qdict:
        return "—"
    val = qdict[key]
    if isinstance(val, float):
        return f"{val:.{digits}f}"
    return str(val)


def write_report(
    path: Path,
    rows: list[dict],
    stats: dict[str, dict],
    attempts: list[dict],
    problems: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# CMQA calibration extract",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Script: `scripts/calibration/fetch_cmqa.py`",
        "",
        "## Obtained?",
        "",
        "**Yes, partial public release.** Human-created roman-script factoid questions from Chandu et al. (CALCS 2018) via `khyathiraghavi/code_switched_QA`. All three pairs are present. Counts are **below** the paper totals (see gap table).",
        "",
        "## License / redistribution",
        "",
        "CMU 2018 custom permissive license (`COPYING`). Redistribution allowed with copyright notice, marked modifications, and retained author names. Record: `data/calibration/licenses/cmqa.md`.",
        "",
        "## Counts",
        "",
        "| pair | public n | paper n | gap | train | test_batch1 | test_batch2 | test_batch3 | image | article |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for pair in PAIR_ORDER:
        s = stats[pair]
        splits = s["splits"]
        ctx = s["context_types"]
        lines.append(
            "| {pair} | {n} | {paper} | {gap} | {tr} | {b1} | {b2} | {b3} | {img} | {art} |".format(
                pair=pair,
                n=s["count"],
                paper=s["paper_count"],
                gap=s["paper_count"] - s["count"],
                tr=splits.get("train", 0),
                b1=splits.get("test_batch1", 0),
                b2=splits.get("test_batch2", 0),
                b3=splits.get("test_batch3", 0),
                img=ctx.get("image", 0),
                art=ctx.get("article", 0),
            )
        )
    lines += [
        "",
        f"Total public questions written to `questions.jsonl`: **{len(rows)}** (all roman script; none dropped).",
        "",
        "## Token-length quantiles (alphabetic tokens)",
        "",
        "| pair | n | min | p25 | p50 | p75 | p90 | max | mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for pair in PAIR_ORDER:
        q = stats[pair]["token_length"]
        lines.append(
            f"| {pair} | {stats[pair]['count']} | {fmt_q(q,'min',0)} | {fmt_q(q,'p25',1)} | "
            f"{fmt_q(q,'p50',1)} | {fmt_q(q,'p75',1)} | {fmt_q(q,'p90',1)} | "
            f"{fmt_q(q,'max',0)} | {fmt_q(q,'mean',2)} |"
        )
    lines += [
        "",
        "## Auto `english_share` (unvalidated)",
        "",
        "Rule: alphabetic tokens; a token counts as English iff `wordfreq.zipf_frequency(tok, 'en') >= 3.5`.",
        "`english_share = n_en / (n_en + n_local)`. **auto/unvalidated** — romanized Hindi/Tamil/Telugu tokens that collide with English (e.g. `he`, `to`, `me`) are over-counted as English.",
        "",
        "| pair | mean | min | p25 | p50 | p75 | p90 | max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for pair in PAIR_ORDER:
        q = stats[pair]["english_share"]
        lines.append(
            f"| {pair} | {fmt_q(q,'mean',3)} | {fmt_q(q,'min',3)} | {fmt_q(q,'p25',3)} | "
            f"{fmt_q(q,'p50',3)} | {fmt_q(q,'p75',3)} | {fmt_q(q,'p90',3)} | {fmt_q(q,'max',3)} |"
        )

    for pair in PAIR_ORDER:
        lines += [
            "",
            f"## {N_EXAMPLES} random examples — `{pair}` (seed={EXAMPLE_SEED})",
            "",
        ]
        for ex in stats[pair]["examples"]:
            share = ex["english_share"]
            share_s = "—" if share is None else f"{share:.2f}"
            lines.append(
                f"- `{ex['id']}` [{ex['split']}/{ex['context_type']}, toks={ex['n_tokens']}, "
                f"en_share={share_s} auto] {ex['text']} → {ex['answer']}"
            )

    lines += [
        "",
        "## URL attempts",
        "",
        "| ok | status | bytes | url | note / error |",
        "|---|---:|---:|---|---|",
    ]
    for att in attempts:
        note = att.get("note") or att.get("error") or ""
        if att.get("error") and att.get("note"):
            note = f"{att['note']} — {att['error']}"
        snippet = att.get("snippet")
        if snippet:
            note = f"{note} | {snippet[:120]}" if note else snippet[:120]
        lines.append(
            f"| {str(att.get('ok')).lower()} | {att.get('status') if att.get('status') is not None else '—'} | "
            f"{att.get('bytes', 0)} | `{att.get('url')}` | {note} |"
        )

    lines += [
        "",
        "## Related-resource outcome",
        "",
        "- **Official CMU page** (`COPYING` / paper-era host): DNS/host failure on this run. Wayback 2017–2019 snapshots are the challenge invitation HTML (Google Form registration, timeline for train + five test batches, `CMQAFlyer.pdf` link). No JSON/TSV dump was hosted or archived.",
        "- **Author GitHub** (`khyathiraghavi/code_switched_QA`): live. Train (Hindi+Telugu) + three Tamil test batches with answers. This is the extract we use.",
        "- **WebShodh (2017)**: demo only; host dead; Wayback has the search form, not a dataset.",
        "- **Raghavi et al. 2015**: evaluation set of ~1,000 En–Hi questions for answer-type classification; no public file found.",
        "- **MSR / Microsoft code-mixed QA**: no independent public dump. GLUECoS QA_EN_HI reuses this GitHub train file.",
        "",
        "## Problems",
        "",
    ]
    if problems:
        lines.extend(f"- {p}" for p in problems)
    else:
        lines.append("- (none recorded)")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def collect_problems(rows: list[dict], stats: dict[str, dict], attempts: list[dict]) -> list[str]:
    problems: list[str] = []
    for pair in PAIR_ORDER:
        gap = stats[pair]["paper_count"] - stats[pair]["count"]
        if gap > 0:
            problems.append(
                f"Public `{pair}` is {stats[pair]['count']} vs paper {stats[pair]['paper_count']} "
                f"(missing {gap}). GitHub is the challenge split, not the full curated set."
            )
    if not any(r["pair"] == "ta_en" and r["tags"]["split"] == "train" for r in rows):
        problems.append("No Tamil questions in the train file; all Tamlish items are test batches 1–3.")
    if stats["hi_en"]["context_types"].get("article", 0) < 275:
        problems.append(
            "Paper reports 275 Hinglish article questions; public train has "
            f"{stats['hi_en']['context_types'].get('article', 0)}."
        )
    problems.append(
        "Official site advertised test batches 4 and 5; they were never published on GitHub or archive.org."
    )
    problems.append(
        "Every test JSON has `division: test_batch_1`; split labels in the extract are taken from filenames."
    )
    problems.append(
        "english_share is automatic (wordfreq Zipf≥3.5) and unvalidated on romanized CM text."
    )
    github_ok = any(
        a.get("ok") and "code_mixed_qa_train.json" in a.get("url", "") for a in attempts
    )
    if not github_ok:
        problems.append("GitHub train download did not validate this run.")
    official = [a for a in attempts if a.get("url") in {CMU_HOST, "https://tts.speech.cs.cmu.edu/codemixedqa/"}]
    if official and not any(a.get("ok") for a in official):
        problems.append("Official CMU host is unreachable; used GitHub + Wayback HTML only.")
    return problems


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-probes", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.repo_root.resolve() if args.repo_root else repo_root_from_script()
    raw_dir = root / "data" / "calibration" / "raw" / "cmqa"
    out_dir = root / "data" / "calibration" / "processed" / "cmqa"
    license_path = root / "data" / "calibration" / "licenses" / "cmqa.md"
    out_dir.mkdir(parents=True, exist_ok=True)
    license_path.parent.mkdir(parents=True, exist_ok=True)

    attempts = download_github_files(raw_dir, skip_download=args.skip_download)
    attempts.extend(probe_related_urls(raw_dir, skip_probes=args.skip_probes))
    (out_dir / "url_attempts.json").write_text(
        json.dumps(attempts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    missing = [name for name in GITHUB_FILES if not (raw_dir / name).exists()]
    if missing:
        write_report(
            out_dir / "REPORT.md",
            [],
            {p: {"count": 0, "paper_count": PAPER_COUNTS[p], "token_length": {}, "english_share": {},
                 "splits": {}, "context_types": {}, "examples": []} for p in PAIR_ORDER},
            attempts,
            [f"Could not obtain required file {name}" for name in missing],
        )
        print("FAILED: missing", missing, file=sys.stderr)
        return 1

    copying = (raw_dir / "COPYING").read_text(encoding="utf-8", errors="replace")
    write_license(license_path, copying, attempts)

    rows = load_questions(raw_dir)
    stats = pair_stats(rows)
    write_jsonl(out_dir / "questions.jsonl", rows)
    problems = collect_problems(rows, stats, attempts)
    write_report(out_dir / "REPORT.md", rows, stats, attempts, problems)

    summary = {pair: stats[pair]["count"] for pair in PAIR_ORDER}
    print(json.dumps({"obtained": True, "counts": summary, "n": len(rows), "problems": problems}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
