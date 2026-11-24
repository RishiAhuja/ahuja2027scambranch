#!/usr/bin/env python3
"""Validate scenario JSON files against scenario.schema.json and extra invariants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON = REPO_ROOT / "scenarios" / "json"
DEFAULT_SCHEMA = REPO_ROOT / "schemas" / "scenario.schema.json"
CONTINUATION_KEYS = ("cooperative", "clarifying", "resistant", "suspicious")


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def extra_checks(record: dict, path: Path) -> list[str]:
    errors: list[str] = []
    family_id = record.get("family_id", "")
    if family_id != path.stem:
        errors.append(f"family_id {family_id!r} does not match filename {path.name}")

    cases = record.get("cases") or {}
    fraud = cases.get("fraud") or {}
    legit = cases.get("legitimate") or {}
    fraud_branch = fraud.get("branch_after")
    legit_branch = legit.get("branch_after")
    if fraud_branch != legit_branch:
        errors.append(f"branch_after mismatch: fraud={fraud_branch} legitimate={legit_branch}")

    required_ids = {
        item.get("id")
        for item in fraud.get("evidence") or []
        if item.get("role") == "required" and item.get("id")
    }
    if not required_ids:
        errors.append("fraud case has no required evidence ids")
    else:
        branch_after = fraud_branch if isinstance(fraud_branch, int) else -1
        seen = False
        for turn in fraud.get("dialogue_plan") or []:
            turn_no = turn.get("turn")
            if not isinstance(turn_no, int) or turn_no > branch_after:
                continue
            if required_ids.intersection(turn.get("evidence_ids") or []):
                seen = True
                break
        if not seen:
            errors.append(
                "no required evidence id appears in a fraud dialogue_plan turn "
                f"<= branch_after ({branch_after}); required={sorted(required_ids)}"
            )

    continuations = record.get("continuations") or {}
    for key in CONTINUATION_KEYS:
        pair = continuations.get(key) or {}
        if key not in continuations:
            errors.append(f"missing continuation condition {key}")
            continue
        if not str(pair.get("fraud") or "").strip() or not str(pair.get("legitimate") or "").strip():
            errors.append(f"empty continuation string(s) for {key}")
    return errors


def format_schema_error(error) -> str:
    location = ".".join(str(part) for part in error.absolute_path) or "<root>"
    return f"schema {location}: {error.message}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-dir", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    args = parser.parse_args(argv)

    schema = load_json(args.schema)
    validator = Draft202012Validator(schema)
    files = sorted(args.json_dir.glob("*.json"))
    if not files:
        print(f"no JSON files in {args.json_dir}", file=sys.stderr)
        return 1

    rows: list[tuple[str, str, str]] = []
    failures = 0
    for path in files:
        reasons: list[str] = []
        try:
            record = load_json(path)
        except json.JSONDecodeError as exc:
            reasons.append(f"invalid JSON: {exc}")
            record = {}
        if record:
            reasons.extend(format_schema_error(err) for err in validator.iter_errors(record))
            reasons.extend(extra_checks(record, path))
        status = "PASS" if not reasons else "FAIL"
        if reasons:
            failures += 1
        rows.append((path.stem, status, "; ".join(reasons) if reasons else ""))

    family_w = max(len("family"), *(len(r[0]) for r in rows))
    status_w = 6
    print(f"{'family':<{family_w}}  {'result':<{status_w}}  detail")
    print(f"{'-' * family_w}  {'-' * status_w}  {'-' * 6}")
    for family, status, detail in rows:
        print(f"{family:<{family_w}}  {status:<{status_w}}  {detail}")
    print()
    print(f"{len(rows) - failures}/{len(rows)} passed; {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
