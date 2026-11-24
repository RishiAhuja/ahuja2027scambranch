#!/usr/bin/env python3
"""Extract attested Dakshina romanization variants for Hindi, Tamil, Telugu.

Re-runnable. Downloads Dakshina v1.0 if needed, keeps only hi/ta/te lexicons
and romanized folders, then writes processed variant tables, LID lexicons,
license record, and a short report.

Official source: https://github.com/google-research-datasets/dakshina
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import statistics
import subprocess
import sys
import tarfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from urllib.request import urlopen


LANGS = ("hi", "ta", "te")
DAKSHINA_URL = "https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar"
GITHUB_README_URL = (
    "https://raw.githubusercontent.com/google-research-datasets/dakshina/master/README.md"
)
VERSION = "1.0"
RELEASE_DATE = "2020-05-27"
DEFAULT_TOP_WORDS = 5000
DEFAULT_TOP_LEXICON = 500
DEFAULT_TOP_ALTERNATIONS = 50

KEEP_DIR_MARKERS = (
    "/hi/lexicons",
    "/hi/romanized",
    "/ta/lexicons",
    "/ta/romanized",
    "/te/lexicons",
    "/te/romanized",
)
LICENSE_BASENAMES = {
    "license",
    "license.txt",
    "license.md",
    "copying",
    "copying.txt",
    "notice",
    "notice.txt",
    "readme",
    "readme.md",
    "readme.txt",
}

LATIN_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
SENTINELS = {"</s>", "<s>", "<unk>"}

CITATION_BIBTEX = """@inproceedings{roark-etal-2020-processing,
    title = "Processing {South} {Asian} Languages Written in the {Latin} Script:
    the {Dakshina} Dataset",
    author = "Roark, Brian and
      Wolf-Sonkin, Lawrence and
      Kirov, Christo and
      Mielke, Sabrina J. and
      Johny, Cibu and
      Demir{\\c{s}}ahin, I{\\c{s}}in and
      Hall, Keith",
    booktitle = "Proceedings of The 12th Language Resources and Evaluation Conference (LREC)",
    year = "2020",
    url = "https://www.aclweb.org/anthology/2020.lrec-1.294",
    pages = "2413--2423"
}"""


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{n} B"


def should_extract(name: str) -> bool:
    normalized = name.replace("\\", "/").rstrip("/")
    base = normalized.split("/")[-1].lower()
    if base in LICENSE_BASENAMES:
        return True
    for marker in KEEP_DIR_MARKERS:
        if normalized.endswith(marker) or f"{marker}/" in f"/{normalized}/":
            return True
    return False


def lang_ready(raw_dir: Path, lang: str) -> bool:
    lang_dir = find_lang_dir(raw_dir, lang)
    if lang_dir is None:
        return False
    lex = lang_dir / "lexicons"
    rom = lang_dir / "romanized"
    return lex.is_dir() and any(lex.iterdir()) and rom.is_dir() and any(rom.iterdir())


def find_lang_dir(raw_dir: Path, lang: str) -> Path | None:
    direct = raw_dir / "dakshina_dataset_v1.0" / lang
    if (direct / "lexicons").is_dir() or (direct / "romanized").is_dir():
        return direct
    hits = [
        p.parent
        for p in raw_dir.rglob("lexicons")
        if p.is_dir() and p.parent.name == lang
    ]
    return hits[0] if hits else None


def download_and_extract(raw_dir: Path) -> list[str]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    if all(lang_ready(raw_dir, lang) for lang in LANGS):
        print("Dakshina hi/ta/te lexicons and romanized folders already present.", flush=True)
        return []

    print(f"Streaming {DAKSHINA_URL}", flush=True)
    print("Extracting only hi/ta/te lexicons + romanized, plus license/README files.", flush=True)
    curl = subprocess.Popen(
        [
            "curl",
            "-L",
            "--fail",
            "--retry",
            "5",
            "--retry-delay",
            "8",
            "--retry-all-errors",
            DAKSHINA_URL,
        ],
        stdout=subprocess.PIPE,
    )
    extracted: list[str] = []
    try:
        assert curl.stdout is not None
        with tarfile.open(fileobj=curl.stdout, mode="r|") as tar:
            for member in tar:
                if member.isfile() or member.isdir():
                    if should_extract(member.name):
                        tar.extract(member, path=raw_dir, filter="data")
                        extracted.append(member.name)
    finally:
        returncode = curl.wait()
    if returncode != 0:
        raise RuntimeError(f"curl failed with exit code {returncode}")

    leftover = raw_dir / "dakshina_dataset_v1.0.tar"
    if leftover.exists():
        leftover.unlink()
        print(f"Deleted leftover tarball {leftover}")
    print(f"Extracted {len(extracted)} archive members.")
    return extracted


def fetch_official_readme(raw_dir: Path) -> Path | None:
    dest = raw_dir / "dakshina_github_README.md"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    try:
        with urlopen(GITHUB_README_URL, timeout=60) as resp:
            text = resp.read().decode("utf-8")
        dest.write_text(text, encoding="utf-8")
        return dest
    except OSError as exc:
        print(f"Warning: could not fetch official GitHub README: {exc}")
        return dest if dest.exists() else None


def iter_license_source_files(raw_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in raw_dir.rglob("*"):
        if path.is_file() and path.name.lower() in LICENSE_BASENAMES:
            files.append(path)
    return sorted(files)


def verify_license(raw_dir: Path) -> dict:
    sources = iter_license_source_files(raw_dir)
    snippets: list[str] = []
    verified_from: list[str] = []
    cc_by_sa = False
    for path in sources:
        text = path.read_text(encoding="utf-8", errors="replace")
        snippets.append(f"----- {path.relative_to(raw_dir)} -----\n{text[:4000]}")
        lowered = text.lower()
        if "cc by-sa 4.0" in lowered or (
            "creative commons" in lowered and "attribution-sharealike" in lowered
        ) or ("creativecommons.org/licenses/by-sa/4.0" in lowered):
            cc_by_sa = True
            verified_from.append(str(path.relative_to(raw_dir)))
    return {
        "license": "CC BY-SA 4.0" if cc_by_sa else "UNVERIFIED",
        "verified": cc_by_sa,
        "verified_from": verified_from,
        "source_files": [str(p.relative_to(raw_dir)) for p in sources],
        "snippets": snippets,
    }


def write_license_record(path: Path, license_info: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    status = (
        "Verified from official Dakshina files listed below."
        if license_info["verified"]
        else "NOT verified from local files; do not assume a license."
    )
    source_list = "\n".join(f"- `{name}`" for name in license_info["source_files"]) or "- (none found)"
    verified_list = (
        "\n".join(f"- `{name}`" for name in license_info["verified_from"]) or "- (none)"
    )
    body = f"""# Dakshina dataset — license record

Recorded: {date.today().isoformat()}
Local extract: `data/calibration/raw/dakshina/` (Hindi, Tamil, Telugu lexicons and romanized folders only)

## Source

- Official repository: https://github.com/google-research-datasets/dakshina
- Tarball: {DAKSHINA_URL}
- Version: {VERSION} ({RELEASE_DATE}, initial data release)
- Paper: Roark et al., LREC 2020, https://aclanthology.org/2020.lrec-1.294/

## License

**{license_info["license"]}**

{status}

Files inspected for license/README text:

{source_list}

Files that contained an explicit CC BY-SA 4.0 (or Creative Commons Attribution-ShareAlike 4.0) statement:

{verified_list}

The official dataset README states:

> The dataset is licensed under [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/).
> Any third party content or data is provided "As Is" without any warranty, express or implied.

Verification used the v1.0 tarball `README.md`. There is **no standalone LICENSE file** in the tarball or in the GitHub repository (confirmed: GitHub tree contains only `README.md`). The license statement lives in that README.

## Citation

{CITATION_BIBTEX}

## Redistribution terms (CC BY-SA 4.0)

- You may share and adapt the material.
- **Attribution** is required (credit Roark et al. 2020 / Google Research Dakshina; link the license; note changes).
- **ShareAlike**: adapted material, including derived lexicons and variant tables produced from this dataset, must be licensed under CC BY-SA 4.0 (or a compatible license).
- Native-script Wikipedia strings in Dakshina are derived from Wikipedia and remain subject to Wikipedia/CC BY-SA provenance in addition to the dataset license.
- This repository keeps only `hi`/`ta`/`te` `lexicons/` and `romanized/` folders, not the full 12-language tarball. Do not treat processed TSVs as public-domain spelling lists.

## What we extracted

Only romanization lexicons and sentence-level romanized Wikipedia strings for `hi`, `ta`, and `te`. Native-script Wikipedia dumps for those languages were not kept.
"""
    path.write_text(body, encoding="utf-8")


def parse_lexicon_row(parts: list[str]) -> tuple[str, str, int] | None:
    if len(parts) < 2:
        return None
    native = parts[0].strip()
    roman = parts[1].strip()
    if not native or not roman or native in SENTINELS:
        return None
    if " " in native or "\t" in native:
        return None
    count = 1
    if len(parts) >= 3 and parts[2].strip().isdigit():
        count = int(parts[2].strip())
    return native, roman, count


def collect_lexicon_files(lex_dir: Path) -> list[Path]:
    if not lex_dir.is_dir():
        return []
    rom_named = sorted(lex_dir.glob("*.rom*.tsv"))
    if rom_named:
        return rom_named
    translit = sorted(lex_dir.glob("*.translit*.tsv"))
    if translit:
        return translit
    return sorted(lex_dir.glob("*.tsv"))


def add_pair(
    word_rom: dict[str, Counter[str]],
    native: str,
    roman: str,
    count: int,
) -> None:
    roman_n = roman.strip().lower()
    if not roman_n or roman_n in SENTINELS:
        return
    if not LATIN_TOKEN_RE.fullmatch(roman_n.replace(" ", "")) and " " in roman_n:
        return
    if " " in roman_n:
        return
    if not LATIN_TOKEN_RE.fullmatch(roman_n):
        roman_n = "".join(ch for ch in roman_n if ch.isalpha() or ch == "'")
        if not roman_n or not LATIN_TOKEN_RE.fullmatch(roman_n):
            return
    word_rom[native][roman_n] += count


def load_lexicon_pairs(files: list[Path]) -> dict[str, Counter[str]]:
    word_rom: dict[str, Counter[str]] = defaultdict(Counter)
    for path in files:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                parsed = parse_lexicon_row(parts)
                if parsed is None:
                    continue
                native, roman, count = parsed
                add_pair(word_rom, native, roman, count)
    return word_rom


def collect_aligned_files(rom_dir: Path) -> list[Path]:
    if not rom_dir.is_dir():
        return []
    preferred = sorted(rom_dir.glob("*.romanized.rejoined.aligned.cased_nopunct.tsv"))
    if preferred:
        return preferred
    rejoined = sorted(rom_dir.glob("*.romanized.rejoined.aligned*.tsv"))
    if rejoined:
        return rejoined
    return sorted(rom_dir.glob("*.aligned*.tsv"))


def load_aligned_pairs(files: list[Path]) -> dict[str, Counter[str]]:
    word_rom: dict[str, Counter[str]] = defaultdict(Counter)
    for path in files:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                native, roman = parts[0].strip(), parts[1].strip()
                if not native or native in SENTINELS or " " in native:
                    continue
                add_pair(word_rom, native, roman, 1)
    return word_rom


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for ca in a:
        curr = [prev[0] + 1]
        for j, cb in enumerate(b):
            curr.append(
                min(curr[j] + 1, prev[j + 1] + 1, prev[j] + (ca != cb))
            )
        prev = curr
    return prev[-1]


def close_roman(a: str, b: str) -> bool:
    if a == b:
        return True
    if a.startswith(b) or b.startswith(a):
        return abs(len(a) - len(b)) <= 2
    if levenshtein(a, b) > 2:
        return False
    if not a or not b:
        return False
    if a[0] == b[0]:
        return True
    return {a[0], b[0]} in ({"v", "w"}, {"i", "y"})


def filter_word_variants(
    counter: Counter[str], lex_romans: set[str]
) -> Counter[str]:
    """Drop likely sentence-alignment errors; always keep lexicon spellings."""
    if not counter:
        return counter
    total = sum(counter.values())
    thresh = max(2, int(0.02 * total))
    kept: Counter[str] = Counter()
    for roman, count in counter.items():
        if roman in lex_romans or count >= thresh:
            kept[roman] = count
    refs = lex_romans or {roman for roman, _ in kept.most_common(1)}
    for roman, count in counter.items():
        if roman in kept:
            continue
        if any(close_roman(roman, ref) for ref in refs):
            kept[roman] = count
    if kept:
        return kept
    modal, modal_n = counter.most_common(1)[0]
    return Counter({modal: modal_n})


def merge_pair_maps(
    lex_pairs: dict[str, Counter[str]],
    sent_pairs: dict[str, Counter[str]],
) -> dict[str, Counter[str]]:
    natives = set(lex_pairs) | set(sent_pairs)
    merged: dict[str, Counter[str]] = {}
    for native in natives:
        combined: Counter[str] = Counter()
        combined.update(lex_pairs.get(native, Counter()))
        combined.update(sent_pairs.get(native, Counter()))
        merged[native] = filter_word_variants(
            combined, set(lex_pairs.get(native, ()))
        )
    return merged


def top_words(
    word_rom: dict[str, Counter[str]], limit: int
) -> list[tuple[str, Counter[str], int]]:
    ranked: list[tuple[str, Counter[str], int]] = []
    for native, counter in word_rom.items():
        ranked.append((native, counter, int(sum(counter.values()))))
    ranked.sort(key=lambda item: (-item[2], item[0]))
    return ranked[:limit]


def write_variants_tsv(
    path: Path, rows: list[tuple[str, Counter[str], int]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["native_word", "n_variants", "variants", "total_attestations"])
        for native, counter, total in rows:
            variants = [rom for rom, _ in counter.most_common()]
            writer.writerow([native, len(variants), "|".join(variants), total])


def canonicalize_seg_pair(left: str, right: str) -> str:
    if len(left) != len(right):
        longer, shorter = (left, right) if len(left) > len(right) else (right, left)
        return f"{longer}~{shorter}"
    a, b = sorted((left, right))
    return f"{a}~{b}"


ASPIRATE_BASES = set("bcdgjpkst")


def _length_alternation(seg: str, source: str, i1: int, i2: int) -> str | None:
    if len(seg) != 1:
        return None
    if (i1 > 0 and source[i1 - 1] == seg) or (i2 < len(source) and source[i2] == seg):
        return canonicalize_seg_pair(seg + seg, seg)
    return None


def _aspiration(seg: str, source: str, i1: int) -> str | None:
    if seg == "h" and i1 > 0 and source[i1 - 1] in ASPIRATE_BASES:
        base = source[i1 - 1]
        return canonicalize_seg_pair(base + "h", base)
    return None


def pair_alternations(v1: str, v2: str) -> list[str]:
    a, b = v1.lower(), v2.lower()
    if a == b:
        return []
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    alts: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        left, right = a[i1:i2], b[j1:j2]
        if tag == "replace":
            alts.append(canonicalize_seg_pair(left, right))
        elif tag == "delete" and left:
            asp = _aspiration(left, a, i1)
            if asp:
                alts.append(asp)
            elif i2 == len(a):
                alts.append(f"final_{left}_dropped")
            else:
                alts.append(_length_alternation(left, a, i1, i2) or f"{left}~")
        elif tag == "insert" and right:
            asp = _aspiration(right, b, j1)
            if asp:
                alts.append(asp)
            elif j2 == len(b):
                alts.append(f"final_{right}_dropped")
            else:
                alts.append(_length_alternation(right, b, j1, j2) or f"{right}~")
    return alts


def alternation_counts(rows: list[tuple[str, Counter[str], int]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for _native, counter, _total in rows:
        variants = list(counter)
        for i, left in enumerate(variants):
            for right in variants[i + 1 :]:
                for alt in pair_alternations(left, right):
                    counts[alt] += 1
    return counts


def write_variant_stats(
    path: Path,
    lang: str,
    rows: list[tuple[str, Counter[str], int]],
    sources: dict,
    top_n: int,
) -> dict:
    n_variants = [len(counter) for _native, counter, _total in rows]
    multi = sum(1 for n in n_variants if n > 1)
    alts = alternation_counts(rows)
    top_alts = [{"alternation": k, "count": v} for k, v in alts.most_common(top_n)]
    payload = {
        "lang": lang,
        "n_words": len(rows),
        "mean_variants_per_word": float(statistics.mean(n_variants)) if n_variants else 0.0,
        "median_variants_per_word": float(statistics.median(n_variants)) if n_variants else 0.0,
        "n_words_with_multiple_variants": multi,
        "total_attestations": int(sum(total for _n, _c, total in rows)),
        "sources": sources,
        "top_alternations": top_alts,
        "note": (
            "Alternations are substitution/insert/delete segments from "
            "difflib.SequenceMatcher over unique romanization pairs of the same native word. "
            "Segments are canonicalized as longer~shorter (or alphabetical if same length). "
            "Word-final insertions/deletions are labeled final_<seg>_dropped."
        ),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def collect_roman_sentence_files(rom_dir: Path) -> list[Path]:
    if not rom_dir.is_dir():
        return []
    preferred = sorted(rom_dir.glob("*.romanized.rejoined*.roman.txt"))
    if preferred:
        return preferred
    roman_txt = sorted(rom_dir.glob("*.roman.txt"))
    if roman_txt:
        return roman_txt
    return []


def collect_rejoined_tsv_files(rom_dir: Path) -> list[Path]:
    if not rom_dir.is_dir():
        return []
    files = [
        p
        for p in sorted(rom_dir.glob("*.romanized.rejoined.tsv"))
        if "aligned" not in p.name
    ]
    return files


def roman_tokens_from_txt(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            counts.update(LATIN_TOKEN_RE.findall(line.lower()))
    return counts


def roman_tokens_from_rejoined_tsv(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            roman = parts[1] if len(parts) >= 2 else parts[0]
            counts.update(LATIN_TOKEN_RE.findall(roman.lower()))
    return counts


def write_roman_lexicon(path: Path, counts: Counter[str], limit: int) -> list[tuple[str, int]]:
    top = counts.most_common(limit)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["token", "freq"])
        writer.writerows(top)
    return top


def example_words(
    rows: list[tuple[str, Counter[str], int]], n: int = 5
) -> list[dict]:
    examples: list[dict] = []
    for native, counter, total in rows:
        if len(counter) < 2:
            continue
        variants = [f"{rom} ({cnt})" for rom, cnt in counter.most_common()]
        examples.append(
            {
                "native_word": native,
                "n_variants": len(counter),
                "total_attestations": total,
                "variants": variants,
            }
        )
        if len(examples) >= n:
            break
    return examples


def write_report(
    path: Path,
    per_lang: dict[str, dict],
    license_info: dict,
    disk: dict[str, str],
    problems: list[str],
) -> None:
    lines = [
        "# Dakshina romanization calibration extract",
        "",
        f"Date: {date.today().isoformat()}",
        "Source: Dakshina v1.0 (Roark et al., LREC 2020).",
        "Script: `scripts/calibration/extract_dakshina.py`.",
        "",
        "## Important caveat",
        "",
        "The romanized **sentence** files are human romanizations of **Wikipedia**",
        "strings, not naturally occurring code-mixed chat or social-media text.",
        "Use them to calibrate spelling variation and as a romanized-local LID",
        "lexicon, not as a sample of fraud-conversation language.",
        "",
        "## License",
        "",
        f"Recorded as **{license_info['license']}** "
        f"({'verified' if license_info['verified'] else 'NOT verified'}).",
        "See `data/calibration/licenses/dakshina.md`.",
        "",
        "## Per-language counts",
        "",
    ]
    for lang in LANGS:
        info = per_lang[lang]
        lines.extend(
            [
                f"### {lang}",
                "",
                f"- Native words kept: {info['n_words']}",
                f"- Mean variants / word: {info['mean_variants_per_word']:.3f}",
                f"- Median variants / word: {info['median_variants_per_word']:.1f}",
                f"- Words with ≥2 variants: {info['n_words_with_multiple_variants']}",
                f"- Total attestations (top words): {info['total_attestations']}",
                f"- Lexicon files: {', '.join(info['sources']['lexicon_files']) or '(none)'}",
                f"- Aligned sentence files: {', '.join(info['sources']['aligned_files']) or '(none)'}",
                f"- Roman sentence files: {', '.join(info['sources']['roman_sentence_files']) or '(none)'}",
                f"- Roman-lexicon tokens written: {info['roman_lexicon_n']}",
                "",
                "Top 10 character-level alternations:",
                "",
            ]
        )
        for item in info["top_alternations"][:10]:
            lines.append(f"- `{item['alternation']}`: {item['count']}")
        lines.extend(["", "Example words:", ""])
        for ex in info["examples"]:
            var = ", ".join(ex["variants"])
            lines.append(
                f"- `{ex['native_word']}` ({ex['n_variants']} variants, "
                f"{ex['total_attestations']} attestations): {var}"
            )
        lines.append("")
    lines.extend(
        [
            "## Disk space",
            "",
            f"- Raw extract: {disk['raw']}",
            f"- Processed romanization: {disk['processed']}",
            f"- License record: {disk['license']}",
            "",
            "## Problems / notes",
            "",
        ]
    )
    lines.extend(
        [
            "- Lexicon files are named `*.translit.sampled.*.tsv`, not `*.rom*.tsv`.",
            "- Token alignments in `*.aligned.cased_nopunct.tsv` are not always 1-1; "
            "low-count sentence romanizations far from lexicon spellings were dropped.",
            "- No standalone LICENSE file exists in the tarball or GitHub repo; "
            "CC BY-SA 4.0 is stated in `dakshina_dataset_v1.0/README.md`.",
        ]
    )
    if problems:
        lines.extend(f"- {p}" for p in problems)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def process_language(
    raw_dir: Path,
    out_dir: Path,
    lang: str,
    top_words_n: int,
    top_lexicon_n: int,
    top_alts: int,
    problems: list[str],
) -> dict:
    lang_dir = find_lang_dir(raw_dir, lang)
    if lang_dir is None:
        raise FileNotFoundError(f"No extracted directory for {lang} under {raw_dir}")

    lex_files = collect_lexicon_files(lang_dir / "lexicons")
    aligned_files = collect_aligned_files(lang_dir / "romanized")
    roman_txts = collect_roman_sentence_files(lang_dir / "romanized")
    rejoined_tsvs = collect_rejoined_tsv_files(lang_dir / "romanized")

    if not lex_files:
        problems.append(f"{lang}: no lexicon TSV files matching *.rom*.tsv or *.translit*.tsv")
    if not aligned_files:
        problems.append(f"{lang}: no aligned romanized TSV files")
    if not roman_txts and not rejoined_tsvs:
        problems.append(f"{lang}: no romanized sentence files for the LID lexicon")

    lex_pairs = load_lexicon_pairs(lex_files)
    sent_pairs = load_aligned_pairs(aligned_files)
    merged = merge_pair_maps(lex_pairs, sent_pairs)
    rows = top_words(merged, top_words_n)
    if len(rows) < top_words_n:
        problems.append(
            f"{lang}: only {len(rows)} native words available (requested {top_words_n})"
        )

    rel = lambda p: str(p.relative_to(raw_dir))
    sources = {
        "lexicon_files": [rel(p) for p in lex_files],
        "aligned_files": [rel(p) for p in aligned_files],
        "roman_sentence_files": [rel(p) for p in (roman_txts or rejoined_tsvs)],
        "lexicon_native_words": len(lex_pairs),
        "aligned_native_words": len(sent_pairs),
        "merged_native_words": len(merged),
        "note": (
            "Lexicon files are named *.translit.sampled.*.tsv (not *.rom*.tsv). "
            "Lexicon counts are annotator attestation counts. "
            "Aligned sentence pairs add +1 per 1-1 token occurrence. "
            "Only rejoined aligned files are used to avoid double-counting split fragments. "
            "Singleton/low-count sentence alignments that are far from lexicon spellings "
            "are dropped as token-alignment noise."
        ),
    }

    write_variants_tsv(out_dir / f"{lang}_variants.tsv", rows)
    stats = write_variant_stats(
        out_dir / f"{lang}_variant_stats.json", lang, rows, sources, top_alts
    )

    token_counts: Counter[str] = Counter()
    if roman_txts:
        for path in roman_txts:
            token_counts.update(roman_tokens_from_txt(path))
        sources["roman_lexicon_source"] = "roman.txt"
    else:
        for path in rejoined_tsvs:
            token_counts.update(roman_tokens_from_rejoined_tsv(path))
        sources["roman_lexicon_source"] = "rejoined.tsv roman column (no *.roman.txt found)"
        if rejoined_tsvs:
            problems.append(
                f"{lang}: no *.roman.txt files; LID lexicon built from rejoined.tsv roman side"
            )

    lexicon_rows = write_roman_lexicon(
        out_dir / f"{lang}_roman_lexicon.tsv", token_counts, top_lexicon_n
    )
    stats["examples"] = example_words(rows)
    stats["roman_lexicon_n"] = len(lexicon_rows)
    stats["roman_lexicon_top5"] = lexicon_rows[:5]
    stats["sources"] = sources
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--top-words", type=int, default=DEFAULT_TOP_WORDS)
    parser.add_argument("--top-lexicon", type=int, default=DEFAULT_TOP_LEXICON)
    parser.add_argument("--top-alternations", type=int, default=DEFAULT_TOP_ALTERNATIONS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.repo_root.resolve() if args.repo_root else repo_root_from_script()
    raw_dir = root / "data" / "calibration" / "raw" / "dakshina"
    out_dir = root / "data" / "calibration" / "processed" / "romanization"
    license_path = root / "data" / "calibration" / "licenses" / "dakshina.md"
    out_dir.mkdir(parents=True, exist_ok=True)
    license_path.parent.mkdir(parents=True, exist_ok=True)

    problems: list[str] = []
    if not args.skip_download:
        download_and_extract(raw_dir)
    elif not all(lang_ready(raw_dir, lang) for lang in LANGS):
        raise FileNotFoundError(
            f"Missing hi/ta/te extract under {raw_dir}; rerun without --skip-download"
        )

    readme = fetch_official_readme(raw_dir)
    if readme is None:
        problems.append("Could not fetch official GitHub README for license verification.")

    license_info = verify_license(raw_dir)
    if not license_info["verified"]:
        problems.append("CC BY-SA 4.0 was not confirmed in local README/LICENSE files.")
    write_license_record(license_path, license_info)

    per_lang: dict[str, dict] = {}
    for lang in LANGS:
        print(f"Processing {lang}...", flush=True)
        per_lang[lang] = process_language(
            raw_dir,
            out_dir,
            lang,
            args.top_words,
            args.top_lexicon,
            args.top_alternations,
            problems,
        )

    disk = {
        "raw": human_bytes(dir_size_bytes(raw_dir)),
        "processed": human_bytes(dir_size_bytes(out_dir)),
        "license": human_bytes(dir_size_bytes(license_path)),
    }
    write_report(out_dir / "REPORT.md", per_lang, license_info, disk, problems)
    print(json.dumps({"license": license_info["license"], "disk": disk, "problems": problems}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
