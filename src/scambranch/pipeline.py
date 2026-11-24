"""End-to-end ScamBranch generation: plan → prefix → branches → review → repair."""

from __future__ import annotations

import copy
import difflib
import hashlib
import json
import re
import shutil
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

from scambranch.batch import ReviewBatchRunner, make_custom_id
from scambranch.checks import (
    prefix_hash,
    run_automatic_checks,
    schema_errors,
    should_stop_for_repetition,
    to_detector_export,
)
from scambranch.llm import (
    BudgetExceeded,
    CostLedger,
    IncompleteOutputError,
    LLMRequest,
    Provider,
    RefusalError,
    make_provider,
    schema_for,
)
from scambranch.placeholders import keyed_rng, resolve_placeholders
from scambranch.render import (
    CONDITION_DEFINITIONS,
    CONDITION_INSTRUCTIONS,
    FIRST_TURN_ENDING_HINT,
    LATER_TURN_ENDING_HINT,
    behaviour_hint,
    local_lexicon,
    load_profile,
    render_language_block,
    render_module,
    sanitize_allowed_endings,
    split_system_user,
)

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS_DIR = ROOT / "scenarios" / "json"
CONFIG_PATH = ROOT / "configs" / "generation.yaml"
SPLITS_PATH = ROOT / "manifests" / "splits_v1.json"
CONDITION_NAMES = ("cooperative", "clarifying", "resistant", "suspicious")
SEMREV03_VERSION = "semrev-0.3"
CONDITION_NAME_RE = re.compile(
    r"\b(cooperative|clarifying|resistant|suspicious)\b",
    re.IGNORECASE,
)


def load_generation_config(path: Path | None = None) -> dict[str, Any]:
    with (path or CONFIG_PATH).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_split_assignment() -> dict[str, str]:
    if not SPLITS_PATH.exists():
        return {}
    data = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    assignment = data.get("assignment")
    if isinstance(assignment, dict):
        return {str(k): str(v) for k, v in assignment.items()}
    return {}


def families_and_splits(config: dict[str, Any], cli_families: list[str] | None) -> tuple[list[str], dict[str, str]]:
    assignment = load_split_assignment()
    splits: dict[str, str] = dict(assignment)
    if cli_families:
        families = list(cli_families)
    elif config.get("families"):
        families = list(config["families"])
    elif assignment:
        families = sorted(assignment)
    elif SPLITS_PATH.exists():
        data = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
        families = []
        if isinstance(data.get("families"), list):
            for row in data["families"]:
                fid = row.get("family_id") or row.get("id")
                if fid:
                    families.append(fid)
                    splits[fid] = row.get("split") or "unassigned"
        else:
            blob = data.get("splits") if isinstance(data.get("splits"), dict) else data
            for split in ("train", "dev", "test"):
                for fid in blob.get(split) or []:
                    families.append(fid)
                    splits[fid] = split
    else:
        families = sorted(p.stem for p in SCENARIOS_DIR.glob("*.json"))
    for path in SCENARIOS_DIR.glob("*.json"):
        try:
            card = json.loads(path.read_text(encoding="utf-8"))
            splits.setdefault(card.get("family_id") or path.stem, card.get("split") or "unassigned")
        except json.JSONDecodeError:
            continue
    splits.update(assignment)
    return families, splits


def conversation_id(case_id: str, pair: str, history_id: str, condition: str) -> str:
    return f"{case_id}_{pair}_{history_id}_{condition}"


def format_transcript(turns: list[dict[str, Any]]) -> str:
    lines = []
    for t in turns:
        lines.append(f"{t['turn_id']} {t['speaker']}: {t['text']}")
    return "\n".join(lines)


def validate_prefix_against_plan(prefix_turns: list[dict[str, Any]], plan: dict[str, Any]) -> None:
    planned = list(plan.get("turns") or [])
    if len(prefix_turns) != len(planned):
        raise ValueError(
            f"prefix has {len(prefix_turns)} turns, plan has {len(planned)}"
        )
    for got, exp in zip(prefix_turns, planned, strict=True):
        if got.get("speaker") != exp.get("speaker"):
            raise ValueError(
                f"turn {got.get('turn_id')} speaker {got.get('speaker')} != plan {exp.get('speaker')}"
            )
        if int(got.get("turn_id") or 0) != int(exp.get("turn") or 0):
            raise ValueError("prefix turn ids do not match the plan")
        if not (got.get("text") or "").strip():
            raise ValueError(f"empty prefix text at turn {got.get('turn_id')}")


def _turn_diff(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> str:
    a = [f"{t['turn_id']} {t['speaker']}: {t['text']}" for t in before]
    b = [f"{t['turn_id']} {t['speaker']}: {t['text']}" for t in after]
    return "\n".join(difflib.unified_diff(a, b, lineterm=""))


def _caller_instruction_text(system: str, user: str) -> str:
    """Instruction chrome only. Filled data may say 'Cooperative Bank' or 'suspicious'."""
    sys_head = (system or "").split("\n\n", 1)[0]
    user_task = user or ""
    if "## Task" in user_task:
        user_task = user_task.split("## Task", 1)[1]
    else:
        for marker in ("## Conversation so far", "## Transcript", "## Your role (private)"):
            if marker in user_task:
                user_task = user_task.split(marker, 1)[0]
    return f"{sys_head}\n{user_task}"


def _assert_caller_prompt_clean(system: str, user: str) -> None:
    blob = _caller_instruction_text(system, user)
    hit = CONDITION_NAME_RE.search(blob)
    if hit:
        raise RuntimeError(f"caller prompt contains condition name {hit.group(0)!r}")


HARD_LANGUAGE_FLAG_TYPES = frozenset({"script", "wrong_language", "artifact"})
SOFT_LANGUAGE_FLAG_TYPES = frozenset(
    {"grammar", "switch_point", "forced_mixing", "unnatural_register"}
)
LANGUAGE_GATES = frozenset({"hard_flags_only", "hard_flags_only_no_scores", "scores"})
DEFAULT_LANGUAGE_GATE = "hard_flags_only_no_scores"
FALLBACK_MODEL = "claude-haiku-4-5"
FALLBACK_PROVIDER = "anthropic"
GENERATION_FALLBACK_ROLES = frozenset({"plan", "prefix", "caller", "recipient", "repair"})


def as_review(obj: Any) -> dict[str, Any]:
    if not isinstance(obj, dict) or obj.get("review_missing"):
        return {}
    return obj


def looks_like_placeholder_language_review(lang: Any) -> bool:
    """True for the old missing-review default (repair + 3/3 + no flags)."""
    if lang is None:
        return True
    if not isinstance(lang, dict):
        return True
    if lang.get("review_missing"):
        return True
    return (
        lang.get("overall") == "repair"
        and lang.get("switching_naturalness") == 3
        and lang.get("grammatical_acceptability") == 3
        and not (lang.get("turn_flags") or [])
    )


def store_reviews(
    gs: dict[str, Any],
    sem: dict[str, Any] | None,
    lang: dict[str, Any] | None,
    *,
    errors: dict[str, str] | None = None,
) -> None:
    sem_missing = sem is None or bool(sem.get("review_missing"))
    lang_missing = lang is None or bool(lang.get("review_missing"))
    extra = {
        key: val
        for key, val in (gs.get("reviews") or {}).items()
        if key not in {"semantic", "language"}
    }
    gs["reviews"] = {
        **extra,
        "semantic": None if sem_missing else sem,
        "language": None if lang_missing else lang,
    }
    gs["review_missing"] = {"semantic": sem_missing, "language": lang_missing}
    if errors:
        gs["review_errors"] = errors
    elif "review_errors" in gs and not sem_missing and not lang_missing:
        gs.pop("review_errors", None)


def archive_semantic_v02(gs: dict[str, Any]) -> bool:
    reviews = gs.setdefault("reviews", {})
    if "semantic_v02" in reviews:
        return False
    sem = reviews.get("semantic")
    if not isinstance(sem, dict) or sem.get("review_missing"):
        return False
    if gs.get("semantic_review_version") == SEMREV03_VERSION:
        return False
    reviews["semantic_v02"] = copy.deepcopy(sem)
    return True


def semantic_is_v03(gs: dict[str, Any]) -> bool:
    if gs.get("semantic_review_version") != SEMREV03_VERSION:
        return False
    sem = (gs.get("reviews") or {}).get("semantic")
    return isinstance(sem, dict) and not sem.get("review_missing")


def language_driven_repair(rec: dict[str, Any]) -> bool:
    tokens = (
        "script",
        "wrong_language",
        "artifact",
        "grammar",
        "switch_point",
        "forced_mixing",
        "unnatural",
        "switching_naturalness",
        "grammatical_acceptability",
    )
    for entry in rec.get("provenance", {}).get("repair_history") or []:
        blob = str(entry.get("reason") or "").lower()
        if any(token in blob for token in tokens):
            return True
    return False


def prefix_broken_reasons(rec: dict[str, Any]) -> list[str]:
    gs = rec.get("provenance", {}).get("generation_settings", {})
    sem = as_review((gs.get("reviews") or {}).get("semantic"))
    checks = gs.get("checks") or {}
    reasons: list[str] = []
    if sem.get("overall") == "reject":
        failed = failed_semantic_check_names(sem)
        reasons.extend(failed if failed else ["semantic_prefix_reject"])
    if checks.get("prefix_broken"):
        reasons.append("prefix_broken")
    if checks.get("prefix_recipient_reaction"):
        reasons.append("prefix_recipient_reaction")
    out: list[str] = []
    seen: set[str] = set()
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            out.append(reason)
    return out


def review_from_result(result: Any) -> tuple[dict[str, Any] | None, str | None]:
    if result is None:
        return None, None
    if getattr(result, "missing", False):
        return None, getattr(result, "error", None) or "missing"
    parsed = getattr(result, "parsed", None)
    if not parsed or parsed.get("review_missing"):
        return None, getattr(result, "error", None) or "missing"
    return parsed, None


def normalize_language_gate(value: Any) -> str:
    gate = str(value or DEFAULT_LANGUAGE_GATE)
    if gate not in LANGUAGE_GATES:
        raise ValueError(f"unknown language_gate {gate!r}")
    return gate


def language_flag_type(flag: dict[str, Any]) -> str:
    return str(flag.get("type") or "").strip()


def hard_language_flags(lang: dict[str, Any] | None) -> list[dict[str, Any]]:
    lang = as_review(lang)
    return [
        flag
        for flag in (lang.get("turn_flags") or [])
        if language_flag_type(flag) in HARD_LANGUAGE_FLAG_TYPES
    ]


def soft_language_flags(lang: dict[str, Any] | None) -> list[dict[str, Any]]:
    lang = as_review(lang)
    return [
        flag
        for flag in (lang.get("turn_flags") or [])
        if language_flag_type(flag) not in HARD_LANGUAGE_FLAG_TYPES
    ]


def confirmed_hard_language_flags(
    lang: dict[str, Any] | None,
    checks: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Hard flags that may gate. LLM `script` counts only if checks.script_issues agrees."""
    checks = checks or {}
    det_turns = {
        int(issue.get("turn_id") or 0)
        for issue in (checks.get("script_issues") or [])
        if issue.get("turn_id") is not None
    }
    out: list[dict[str, Any]] = []
    for flag in hard_language_flags(lang):
        ftype = language_flag_type(flag)
        if ftype == "script":
            try:
                tid = int(flag.get("turn_id") or 0)
            except (TypeError, ValueError):
                tid = 0
            if tid and tid in det_turns:
                out.append(flag)
            continue
        if ftype in {"wrong_language", "artifact"}:
            out.append(flag)
    return out


def language_score_reject(lang: dict[str, Any] | None) -> bool:
    lang = as_review(lang)
    if lang.get("overall") != "reject":
        return False
    scores = (
        lang.get("switching_naturalness"),
        lang.get("grammatical_acceptability"),
    )
    return any(isinstance(score, (int, float)) and score <= 2 for score in scores)


def language_metrics(lang: dict[str, Any] | None) -> dict[str, Any]:
    lang = as_review(lang)
    missing = not lang
    return {
        "switching_naturalness": lang.get("switching_naturalness"),
        "grammatical_acceptability": lang.get("grammatical_acceptability"),
        "overall": lang.get("overall"),
        "soft_flags": soft_language_flags(lang),
        "hard_flags": hard_language_flags(lang),
        "review_missing": missing,
    }


def failed_semantic_check_names(sem: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for name, val in (sem.get("checks") or {}).items():
        if isinstance(val, dict) and val.get("pass") is False:
            names.append(str(name))
    return names


def failed_auto_check_names(checks: dict[str, Any]) -> list[str]:
    names: list[str] = []
    if checks.get("prefix_broken"):
        names.append("prefix_broken")
    if checks.get("prefix_script"):
        names.append("prefix_script")
    if checks.get("prefix_recipient_reaction"):
        names.append("prefix_recipient_reaction")
    if checks.get("cooperative_refusal"):
        names.append("cooperative_refusal")
    leak = checks.get("leakage") or {}
    if leak and not leak.get("ok"):
        names.append("leakage")
    if checks.get("script_issues") and not checks.get("prefix_script"):
        names.append("script_issues")
    if checks.get("prefix_hash_ok") is False:
        names.append("prefix_hash")
    return names


def auto_checks_ok(checks: dict[str, Any], *, language_gate: str) -> bool:
    if checks.get("prefix_broken") or checks.get("prefix_recipient_reaction"):
        return False
    if checks.get("cooperative_refusal"):
        return False
    leak = checks.get("leakage") or {}
    if leak and not leak.get("ok"):
        return False
    if checks.get("script_issues"):
        return False
    if checks.get("prefix_hash_ok") is False:
        return False
    if language_gate == "scores" and checks.get("english_share_ok") is False:
        return False
    return True


def combine_verdict(
    sem: dict[str, Any],
    lang: dict[str, Any],
    checks: dict[str, Any],
    *,
    honor_auto_flags: bool = True,
    language_gate: str = DEFAULT_LANGUAGE_GATE,
) -> str:
    language_gate = normalize_language_gate(language_gate)
    sem = as_review(sem)
    lang = as_review(lang)
    if checks.get("prefix_broken") or checks.get("prefix_recipient_reaction"):
        return "reject"
    if sem.get("overall") == "reject":
        return "reject"
    if language_gate == "scores":
        if lang.get("overall") == "reject" and checks.get("prefix_script"):
            return "reject"
        auto_repair = (
            sem.get("overall") in {"repair", "reject"}
            or lang.get("overall") in {"repair", "reject"}
            or (
                checks.get("leakage")
                and not checks["leakage"].get("ok")
                and not checks["leakage"].get("prefix_broken")
            )
            or (checks.get("script_issues") and not checks.get("prefix_script"))
        )
        if honor_auto_flags and checks.get("repair_candidate"):
            return "repair"
        if auto_repair:
            return "repair"
        return "accept"
    if language_gate == "hard_flags_only_no_scores":
        lang_hard = bool(confirmed_hard_language_flags(lang, checks))
    else:
        lang_hard = bool(hard_language_flags(lang)) or language_score_reject(lang)
    if sem.get("overall") == "accept" and auto_checks_ok(checks, language_gate=language_gate) and not lang_hard:
        return "accept"
    return "repair"


def collect_rejection_reasons(
    sem: dict[str, Any],
    lang: dict[str, Any],
    checks: dict[str, Any],
    *,
    language_gate: str = DEFAULT_LANGUAGE_GATE,
) -> list[str]:
    language_gate = normalize_language_gate(language_gate)
    sem = as_review(sem)
    lang = as_review(lang)
    reasons: list[str] = []
    if sem.get("overall") != "accept":
        failed = failed_semantic_check_names(sem)
        reasons.extend(failed if failed else ["semantic"])
    reasons.extend(failed_auto_check_names(checks))
    seen_types: set[str] = set()
    gate_flags = (
        confirmed_hard_language_flags(lang, checks)
        if language_gate == "hard_flags_only_no_scores"
        else hard_language_flags(lang)
    )
    for flag in gate_flags:
        flag_type = language_flag_type(flag)
        if flag_type and flag_type not in seen_types:
            seen_types.add(flag_type)
            reasons.append(flag_type)
    if language_gate == "hard_flags_only" and language_score_reject(lang):
        if (lang.get("switching_naturalness") or 5) <= 2:
            reasons.append("switching_naturalness")
        if (lang.get("grammatical_acceptability") or 5) <= 2:
            reasons.append("grammatical_acceptability")
    elif language_gate == "scores" and lang.get("overall") != "accept":
        reasons.append("language")
    out: list[str] = []
    seen: set[str] = set()
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            out.append(reason)
    return out


def collect_repair_targets(
    *,
    sem: dict[str, Any],
    lang: dict[str, Any],
    checks: dict[str, Any],
    turns: list[dict[str, Any]],
    branch_after: int,
    language_gate: str = DEFAULT_LANGUAGE_GATE,
) -> list[dict[str, Any]]:
    language_gate = normalize_language_gate(language_gate)
    sem = as_review(sem)
    lang = as_review(lang)
    targets: list[dict[str, Any]] = list(sem.get("repair_targets") or [])
    flags = list(lang.get("turn_flags") or [])
    if language_gate == "hard_flags_only_no_scores":
        flags = confirmed_hard_language_flags(lang, checks)
    elif language_gate == "hard_flags_only":
        flags = [flag for flag in flags if language_flag_type(flag) in HARD_LANGUAGE_FLAG_TYPES]
    for flag in flags:
        targets.append(
            {
                "turn_id": flag.get("turn_id"),
                "problem": f"{flag.get('type')}: {flag.get('note')}",
            }
        )
    for issue in (checks.get("leakage") or {}).get("issues") or []:
        if int(issue["turn_id"]) > branch_after:
            targets.append({"turn_id": issue["turn_id"], "problem": ",".join(issue["flags"])})
    for issue in checks.get("script_issues") or []:
        if int(issue.get("turn_id") or 0) > branch_after:
            scripts = ",".join(issue.get("scripts") or [])
            targets.append({"turn_id": issue["turn_id"], "problem": f"script: {scripts}"})
    if checks.get("cooperative_refusal"):
        first_post = next(
            (
                turn
                for turn in turns
                if turn.get("speaker") == "recipient"
                and int(turn.get("turn_id") or 0) > branch_after
            ),
            None,
        )
        if first_post:
            targets.append(
                {
                    "turn_id": first_post["turn_id"],
                    "problem": "cooperative_refusal: first post-branch recipient turn refuses",
                }
            )
    if not targets:
        post = [turn for turn in turns if int(turn.get("turn_id") or 0) > branch_after]
        if post:
            targets.append({"turn_id": post[0]["turn_id"], "problem": "post-branch wording"})
    return targets


class GenerationRun:
    def __init__(
        self,
        *,
        run_id: str,
        config: dict[str, Any],
        families: list[str],
        pairs: list[str],
        conditions: list[str],
        dry_run: bool,
        sync_review: bool,
        budget: float | None,
        workers: int | None,
        run_root: Path | None = None,
        history_ids: list[str] | None = None,
    ) -> None:
        self.run_id = run_id
        self.config = config
        self.families = families
        self.pairs = pairs
        self.conditions = conditions
        self.dry_run = dry_run
        self.sync_review = bool(sync_review or dry_run)
        self.seed = int(config.get("seed") or 42)
        self.max_post = int(config.get("max_post_branch_turns") or 6)
        self.max_repair = int(config.get("max_repair_rounds") or 2)
        self.workers = int(workers if workers is not None else config.get("workers") or 8)
        self.audit_sample = float(config.get("audit_sample") or 0.0)
        self.prices = config["prices"]
        self.roles = config["roles"]
        self.history_ids = list(history_ids or config.get("history_ids") or ["h1"])
        self.language_gate = normalize_language_gate(config.get("language_gate"))
        review_cfg = config.get("review") or {}
        self.review_chunk_size = int(review_cfg.get("chunk_size") or 200)
        self.sync_repair_threshold = int(review_cfg.get("sync_repair_threshold") or 20)
        self.run_dir = (run_root or (ROOT / "runs")) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        cap = float(budget if budget is not None else config.get("budget_usd_cap") or 3.0)
        self.ledger = CostLedger(
            self.run_dir / "ledger.jsonl",
            budget_usd_cap=cap,
            prices=self.prices,
        )
        self.prompt_log: list[dict[str, Any]] = []
        self._providers: dict[str, Provider] = {}
        self._prov_lock = threading.Lock()
        self._splits: dict[str, str] = {}
        self.fallback_events: list[dict[str, Any]] = []

    def provider_for(self, role: str) -> Provider:
        cfg = self.roles[role if role in self.roles else role.split("_")[0]]
        if role.startswith("audit"):
            cfg = self.roles["audit"]
        key = "mock" if self.dry_run else cfg["provider"]
        with self._prov_lock:
            if key not in self._providers:
                self._providers[key] = make_provider(
                    key,
                    dry_run=self.dry_run,
                    ledger=self.ledger,
                    prices=self.prices,
                    raw_dir=self.run_dir / "raw",
                    prompt_log=self.prompt_log,
                )
            return self._providers[key]

    def role_cfg(self, role: str) -> dict[str, Any]:
        if role.startswith("audit"):
            return self.roles["audit"]
        return self.roles[role]

    def call_llm(
        self,
        role: str,
        *,
        system: str,
        user: str,
        conversation_id: str = "",
        mock_context: dict[str, Any] | None = None,
        cache_system: bool = True,
    ) -> dict[str, Any]:
        cfg = self.role_cfg(role)
        schema_role = "language_review" if role == "audit_language" else (
            "semantic_review" if role in {"audit", "audit_semantic"} else role
        )
        req = LLMRequest(
            role=role,
            system=system,
            user=user,
            schema=schema_for(schema_role),
            model=cfg["model"],
            provider=cfg["provider"],
            max_output_tokens=int(cfg["max_output_tokens"]),
            reasoning_effort=cfg.get("reasoning_effort"),
            temperature=cfg.get("temperature"),
            conversation_id=conversation_id,
            mock_context=mock_context or {},
            cache_system=cache_system,
        )
        try:
            result = self.provider_for(role).complete(req)
            return result.parsed
        except BudgetExceeded:
            raise
        except RefusalError as exc:
            if role not in GENERATION_FALLBACK_ROLES or self.dry_run:
                raise
            try:
                result = self.provider_for(role).complete(req)
                return result.parsed
            except BudgetExceeded:
                raise
            except Exception as retry_exc:
                result = self._fallback_complete(req, role, f"refusal:{retry_exc}")
                return result.parsed
        except (IncompleteOutputError, json.JSONDecodeError, ValueError) as exc:
            if role not in GENERATION_FALLBACK_ROLES or self.dry_run:
                raise
            result = self._fallback_complete(req, role, f"{type(exc).__name__}:{exc}")
            return result.parsed

    def _fallback_complete(self, req: LLMRequest, role: str, reason: str):
        haiku_req = replace(
            req,
            model=FALLBACK_MODEL,
            provider=FALLBACK_PROVIDER,
            reasoning_effort=None,
            max_output_tokens=max(int(req.max_output_tokens), 800),
        )
        with self._prov_lock:
            if "anthropic" not in self._providers:
                self._providers["anthropic"] = make_provider(
                    "anthropic",
                    dry_run=self.dry_run,
                    ledger=self.ledger,
                    prices=self.prices,
                    raw_dir=self.run_dir / "raw",
                    prompt_log=self.prompt_log,
                )
            provider = self._providers["anthropic"]
        result = provider.complete(haiku_req)
        event = {
            "conversation_id": req.conversation_id,
            "role": role,
            "original_model": req.model,
            "fallback_model": FALLBACK_MODEL,
            "reason": reason[:500],
        }
        with self._prov_lock:
            self.fallback_events.append(event)
        return result

    def attach_fallbacks(self, rec: dict[str, Any]) -> None:
        cid = rec.get("conversation_id")
        with self._prov_lock:
            events = [e for e in self.fallback_events if e.get("conversation_id") == cid]
        if not events:
            return
        gs = rec["provenance"]["generation_settings"]
        gs["fallbacks"] = events
        models = dict(gs.get("models") or {})
        for event in events:
            models[event["role"]] = event["fallback_model"]
        gs["models"] = models

    def render_cached(self, name: str, values: dict[str, Any]) -> tuple[str, str, str]:
        version, filled = render_module(name, values)
        system, user = split_system_user(filled)
        if name == "language_review" and "## Transcript" in user:
            ref, rest = user.split("## Transcript", 1)
            system = (system + "\n\n" + ref.strip()).strip()
            user = "## Transcript" + rest
        return version, system, user


def _evidence_turn_ids(plan: dict[str, Any]) -> list[int]:
    ids: list[int] = []
    for row in plan.get("turns") or []:
        if row.get("evidence_ids"):
            ids.append(int(row["turn"]))
    return ids


def _prefix_evidence_states(plan: dict[str, Any]) -> list[dict[str, Any]]:
    states = []
    for row in plan.get("turns") or []:
        state = "evidence" if row.get("evidence_ids") else "no_evidence"
        states.append({"turn_id": int(row["turn"]), "state": state})
    return states


def _observed_outcome(turns: list[dict[str, Any]], ended_by: str | None) -> str:
    if ended_by == "repetition":
        return "stopped_for_repetition"
    if ended_by == "recipient":
        return "recipient_ended"
    if ended_by == "caller":
        return "caller_ended"
    return "max_turns"


class Pipeline:
    def __init__(self, run: GenerationRun, splits: dict[str, str]) -> None:
        self.run = run
        self.splits = splits
        self._progress_lock = threading.Lock()
        self._last_progress_n = 0
        self._projection_checked = False
        self._projection_stop = False
        self._projected_usd: float | None = None
        self._batch_runner: ReviewBatchRunner | None = None
        self._inflight: list[tuple[list[dict[str, Any]], list[LLMRequest], dict[str, Any]]] = []

    def execute(self) -> dict[str, Any]:
        units: list[tuple[str, str, str, str]] = []
        for history_id in self.run.history_ids:
            for family in self.run.families:
                for case_kind in ("fraud", "legitimate"):
                    for pair in self.run.pairs:
                        units.append((family, case_kind, pair, history_id))
        self._target_n = max(1, len(units) * len(self.run.conditions))
        results: list[dict[str, Any]] = []
        budget_hit: BudgetExceeded | None = None
        try:
            if not self.run.sync_review:
                self._resume_inflight_reviews()
            pending_chunk: list[dict[str, Any]] = []
            if not self.run.sync_review:
                pending_chunk.extend(self._pending_records())
            workers = max(1, min(self.run.workers, len(units) or 1))
            for history_id in self.run.history_ids:
                group = [unit for unit in units if unit[3] == history_id]
                todo = [unit for unit in group if not self._unit_finished(*unit)]
                if not todo:
                    continue
                use_workers = max(1, min(workers, len(todo)))
                if use_workers == 1:
                    for unit in todo:
                        pending_chunk = self._ingest_generated(
                            self._run_unit(*unit), pending_chunk
                        )
                        if self._projection_stop:
                            break
                else:
                    with ThreadPoolExecutor(max_workers=use_workers) as pool:
                        futs = [pool.submit(self._run_unit, *u) for u in todo]
                        for fut in as_completed(futs):
                            try:
                                recs = fut.result()
                            except BudgetExceeded:
                                raise
                            except Exception as exc:
                                print(f"UNIT ERROR: {type(exc).__name__}: {exc}")
                                recs = []
                            pending_chunk = self._ingest_generated(recs, pending_chunk)
                            if self._projection_stop:
                                break
                if self._projection_stop:
                    break
            if not self.run.sync_review:
                if pending_chunk:
                    self._submit_review_chunk(pending_chunk)
                while self._inflight:
                    self._collect_oldest_batch()
            elif pending_chunk:
                results.extend(pending_chunk)
        except BudgetExceeded as exc:
            print(f"BUDGET STOP: {exc}")
            budget_hit = exc
        disk = {rec["conversation_id"]: rec for rec in self._load_disk_records()}
        disk.update({rec["conversation_id"]: rec for rec in results})
        results = list(disk.values())
        leftover = [
            rec
            for rec in results
            if rec["provenance"]["generation_settings"].get("disposition") == "pending"
        ]
        if leftover and not self.run.sync_review and budget_hit is None:
            try:
                results = self._batch_review_and_repair(results)
            except BudgetExceeded as exc:
                print(f"BUDGET STOP: {exc}")
                budget_hit = exc
                results = self._load_disk_records()
        self._write_exports(results)
        self._write_readable_sample(results)
        self._log_progress(force=True)
        if budget_hit is not None:
            raise budget_hit
        return {
            "conversations": results,
            "run_dir": str(self.run.run_dir),
            "projection_stop": self._projection_stop,
            "projected_usd": self._projected_usd,
        }

    def _load_disk_records(self) -> list[dict[str, Any]]:
        rows = []
        for path in sorted((self.run.run_dir / "conversations").glob("*.json")):
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        return rows

    def _pending_records(self) -> list[dict[str, Any]]:
        inflight_ids: set[str] = set()
        for _recs, _jobs, entry in self._inflight:
            inflight_ids.update(entry.get("conversation_ids") or [])
        return [
            rec
            for rec in self._load_disk_records()
            if rec["provenance"]["generation_settings"].get("disposition") == "pending"
            and rec["conversation_id"] not in inflight_ids
        ]

    def _unit_finished(self, family: str, case_kind: str, pair: str, history_id: str) -> bool:
        card_path = SCENARIOS_DIR / f"{family}.json"
        card = json.loads(card_path.read_text(encoding="utf-8"))
        case_id = card["cases"][case_kind]["case_id"]
        for condition in self.run.conditions:
            path = self.run.run_dir / "conversations" / f"{conversation_id(case_id, pair, history_id, condition)}.json"
            if not path.exists():
                return False
            rec = json.loads(path.read_text(encoding="utf-8"))
            disp = (rec.get("provenance") or {}).get("generation_settings", {}).get("disposition")
            if disp not in {"accepted", "repaired", "rejected"}:
                return False
        return True

    def _review_runner(self, purpose: str | None = None) -> ReviewBatchRunner:
        if self._batch_runner is None or (
            purpose and getattr(self._batch_runner, "purpose", None) != purpose
        ):
            provider = (self.run.config.get("review") or {}).get("batch_provider") or "anthropic"
            self._batch_runner = ReviewBatchRunner(
                run_dir=self.run.run_dir,
                ledger=self.run.ledger,
                prices=self.run.prices,
                provider=provider,
                dry_run=self.run.dry_run,
                purpose=purpose,
            )
        return self._batch_runner

    def _lang_vars_for(self, rec: dict[str, Any]) -> dict[str, Any]:
        stored = rec["provenance"]["generation_settings"].get("lang_vars") or {}
        defaults = {
            "pair_name": rec["language_pair"],
            "region": "",
            "english_share_p50_pct": "36",
            "english_share_p25_pct": "16",
            "english_share_p75_pct": "75",
            "monolingual_share_pct": "18",
            "light_verb_examples": "",
            "mixed_morphology_examples": "",
            "romanization_variants": "",
            "style_anchors": "",
            "unnatural_examples_with_reasons": "",
            "local_language": "",
            "english_share_p10": 0.0,
            "english_share_p90": 1.0,
        }
        defaults.update(stored)
        return defaults

    def _jobs_for_records(self, records: list[dict[str, Any]]) -> list[LLMRequest]:
        jobs: list[LLMRequest] = []
        for rec in records:
            gs = rec["provenance"]["generation_settings"]
            jobs.extend(
                self._review_jobs_for(
                    rec, gs.get("language_block") or "", self._lang_vars_for(rec)
                )
            )
        return jobs

    def _ingest_generated(
        self, recs: list[dict[str, Any]], pending_chunk: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        for rec in recs:
            if rec["provenance"]["generation_settings"].get("disposition") == "pending":
                pending_chunk.append(rec)
        self._maybe_log_progress()
        self._maybe_stop_for_projection()
        if (
            not self.run.sync_review
            and not self._projection_stop
            and len(pending_chunk) >= self.run.review_chunk_size
        ):
            self._submit_review_chunk(pending_chunk)
            pending_chunk = []
            if len(self._inflight) >= 2:
                self._collect_oldest_batch()
        return pending_chunk

    def _submit_review_chunk(self, records: list[dict[str, Any]]) -> None:
        pending = [
            rec
            for rec in records
            if rec["provenance"]["generation_settings"].get("disposition") == "pending"
        ]
        if not pending:
            return
        jobs = self._jobs_for_records(pending)
        entry = self._review_runner().submit(jobs)
        self._inflight.append((pending, jobs, entry))

    def _collect_oldest_batch(self) -> None:
        if not self._inflight:
            return
        records, jobs, entry = self._inflight.pop(0)
        try:
            results = self._review_runner().collect(entry, jobs)
        except BudgetExceeded:
            self._inflight.insert(0, (records, jobs, entry))
            raise
        except Exception as exc:
            print(f"BATCH COLLECT ERROR: {type(exc).__name__}: {exc}")
            return
        self._process_review_results(records, results)

    def _resume_inflight_reviews(self) -> None:
        runner = self._review_runner()
        disk = {rec["conversation_id"]: rec for rec in self._load_disk_records()}
        for entry in list(runner.state.get("batches") or []):
            if entry.get("status") in {"ended", "completed"}:
                continue
            cids = entry.get("conversation_ids") or []
            recs = [
                disk[cid]
                for cid in cids
                if cid in disk
                and disk[cid]["provenance"]["generation_settings"].get("disposition") == "pending"
            ]
            if not recs:
                continue
            jobs = self._jobs_for_records(recs)
            print(f"resuming pending batch {entry.get('batch_id')}")
            try:
                results = runner.collect(entry, jobs)
            except Exception as exc:
                print(f"RESUME BATCH ERROR: {type(exc).__name__}: {exc}")
                continue
            self._process_review_results(recs, results)

    def _process_review_results(
        self, records: list[dict[str, Any]], results: dict[str, Any]
    ) -> None:
        need_repair: list[dict[str, Any]] = []
        for rec in records:
            gs = rec["provenance"]["generation_settings"]
            if gs.get("disposition") != "pending":
                continue
            cid = rec["conversation_id"]
            sem_r = results.get(make_custom_id(cid, "semantic_review"))
            lang_r = results.get(make_custom_id(cid, "language_review"))
            sem, sem_err = review_from_result(sem_r)
            lang, lang_err = review_from_result(lang_r)
            errors = {}
            if sem_err:
                errors["semantic"] = sem_err
            if lang_err:
                errors["language"] = lang_err
            store_reviews(gs, sem, lang, errors=errors or None)
            gs["language_metrics"] = language_metrics(lang)
            checks = gs.get("checks") or {}
            verdict = combine_verdict(sem, lang, checks, language_gate=self.run.language_gate)
            if verdict == "accept":
                gs["disposition"] = "accepted"
                gs["rejection_reason"] = []
                rec["provenance"]["review_status"] = "accepted"
                self._persist_conversation(rec)
            elif verdict == "reject":
                gs["disposition"] = "rejected"
                gs["prefix_reject"] = True
                gs["rejection_reason"] = collect_rejection_reasons(
                    sem, lang, checks, language_gate=self.run.language_gate
                )
                rec["provenance"]["review_status"] = "rejected"
                self._persist_conversation(rec)
            else:
                need_repair.append(rec)
        if need_repair:
            self._repair_and_rereview(need_repair)

    def _repair_and_rereview(self, records: list[dict[str, Any]]) -> None:
        for rec in records:
            gs = rec["provenance"]["generation_settings"]
            reviews = gs.get("reviews") or {}
            try:
                self._apply_repair(
                    rec,
                    reviews.get("semantic") or {},
                    reviews.get("language") or {},
                    gs.get("checks") or {},
                    gs.get("language_block") or "",
                    {},
                    {},
                )
            except Exception as exc:
                print(f"REPAIR ERROR {rec['conversation_id']}: {type(exc).__name__}: {exc}")
            rec["provenance"]["generation_settings"]["disposition"] = "pending"
            rec["provenance"]["review_status"] = "pending"
            self._persist_conversation(rec)
        if len(records) < self.run.sync_repair_threshold:
            for rec in records:
                gs = rec["provenance"]["generation_settings"]
                self._review_and_repair(
                    rec,
                    gs.get("language_block") or "",
                    self._lang_vars_for(rec),
                    {},
                    {},
                )
            return
        jobs = self._jobs_for_records(records)
        results = self._review_runner().submit_and_collect(jobs)
        for rec in records:
            gs = rec["provenance"]["generation_settings"]
            cid = rec["conversation_id"]
            sem_r = results.get(make_custom_id(cid, "semantic_review"))
            lang_r = results.get(make_custom_id(cid, "language_review"))
            sem, sem_err = review_from_result(sem_r)
            lang, lang_err = review_from_result(lang_r)
            errors = {}
            if sem_err:
                errors["semantic"] = sem_err
            if lang_err:
                errors["language"] = lang_err
            store_reviews(gs, sem, lang, errors=errors or None)
            gs["language_metrics"] = language_metrics(lang)
            checks = run_automatic_checks(
                record=rec,
                turns=rec["turns"],
                condition=rec["private_annotations"]["recipient_condition"],
                branch_after=rec["private_annotations"]["branch_after_turn"],
                sibling_hashes=None,
                expected_prefix_hash=rec["private_annotations"]["prefix_hash"],
                lexicon=local_lexicon(load_profile(rec["language_pair"])),
                p10=float(self._lang_vars_for(rec).get("english_share_p10") or 0.0),
                p90=float(self._lang_vars_for(rec).get("english_share_p90") or 1.0),
                pair=rec["language_pair"],
            )
            gs["checks"] = checks
            verdict = combine_verdict(sem, lang, checks, language_gate=self.run.language_gate)
            if verdict == "accept":
                gs["disposition"] = "repaired"
                gs["rejection_reason"] = []
                rec["provenance"]["review_status"] = "accepted"
            else:
                gs["disposition"] = "rejected"
                gs["rejection_reason"] = collect_rejection_reasons(
                    sem, lang, checks, language_gate=self.run.language_gate
                )
                rec["provenance"]["review_status"] = "rejected"
            self._persist_conversation(rec)

    def _maybe_log_progress(self, force: bool = False) -> None:
        recs = self._load_disk_records()
        n = len(recs)
        if not force and n - self._last_progress_n < 50 and n < self._target_n:
            return
        self._log_progress(force=True)

    def _log_progress(self, force: bool = False) -> None:
        recs = self._load_disk_records()
        n = len(recs)
        if not force and n - self._last_progress_n < 50:
            return
        self._last_progress_n = n
        disp = {"accepted": 0, "repaired": 0, "rejected": 0, "pending": 0}
        for rec in recs:
            key = rec["provenance"]["generation_settings"].get("disposition") or "pending"
            disp[key] = disp.get(key, 0) + 1
        target_n = getattr(self, "_target_n", None) or max(n, 1)
        units_done = n // max(1, len(self.run.conditions))
        units_total = target_n // max(1, len(self.run.conditions))
        line = (
            f"{_now_iso()} units_done={units_done}/{units_total} convos={n}/{target_n} "
            f"accepted={disp.get('accepted', 0)} repaired={disp.get('repaired', 0)} "
            f"rejected={disp.get('rejected', 0)} pending={disp.get('pending', 0)} "
            f"spend=${self.run.ledger.spent:.4f}"
        )
        path = self.run.run_dir / "progress.log"
        with self._progress_lock:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        print(line)

    def _maybe_stop_for_projection(self) -> None:
        if self._projection_checked or self._projection_stop:
            return
        recs = self._load_disk_records()
        n = len(recs)
        if n < max(1, int(0.25 * self._target_n)):
            return
        self._projection_checked = True
        projected = self.run.ledger.spent * (self._target_n / max(n, 1))
        self._projected_usd = projected
        line = (
            f"{_now_iso()} 25% check n={n}/{self._target_n} spent=${self.run.ledger.spent:.4f} "
            f"projected=${projected:.2f} cap=${self.run.ledger.budget_usd_cap:.2f}"
        )
        path = self.run.run_dir / "progress.log"
        with self._progress_lock:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        print(line)
        if projected > self.run.ledger.budget_usd_cap:
            self._projection_stop = True
            print(f"PROJECTION STOP: ${projected:.2f} exceeds ${self.run.ledger.budget_usd_cap:.2f}")

    def _write_readable_sample(self, records: list[dict[str, Any]]) -> None:
        usable = [
            rec
            for rec in records
            if rec["provenance"]["generation_settings"].get("disposition") in {"accepted", "repaired"}
        ]
        if not usable:
            return
        rng = keyed_rng("readable_sample", self.run.run_id, seed=self.run.seed)
        strata: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for rec in usable:
            key = (rec["language_pair"], rec["private_annotations"]["scenario_label"])
            strata.setdefault(key, []).append(rec)
        chosen: list[dict[str, Any]] = []
        per = max(1, 30 // max(1, len(strata)))
        for key in sorted(strata):
            pool = list(strata[key])
            rng.shuffle(pool)
            chosen.extend(pool[:per])
        if len(chosen) < 30:
            leftover = [rec for rec in usable if rec not in chosen]
            rng.shuffle(leftover)
            chosen.extend(leftover[: 30 - len(chosen)])
        chosen = chosen[:30]
        write_readable_dump(self.run.run_dir, chosen, filename="READABLE_sample.md")

    def _run_unit(
        self,
        family: str,
        case_kind: str,
        pair: str,
        history_id: str,
    ) -> list[dict[str, Any]]:
        card_path = SCENARIOS_DIR / f"{family}.json"
        card = json.loads(card_path.read_text(encoding="utf-8"))
        case = card["cases"][case_kind]
        case_id = case["case_id"]
        finished: list[dict[str, Any]] = []
        pending_existing: list[dict[str, Any]] = []
        missing: list[str] = []
        for condition in self.run.conditions:
            cid = conversation_id(case_id, pair, history_id, condition)
            path = self.run.run_dir / "conversations" / f"{cid}.json"
            if path.exists():
                rec = json.loads(path.read_text(encoding="utf-8"))
                disp = (rec.get("provenance") or {}).get("generation_settings", {}).get("disposition")
                if disp in {"accepted", "repaired", "rejected"}:
                    finished.append(rec)
                else:
                    pending_existing.append(rec)
            else:
                missing.append(condition)
        if not missing and not pending_existing:
            return finished
        if not missing and pending_existing:
            if self.run.sync_review:
                reviewed = []
                for rec in pending_existing:
                    gs = rec["provenance"]["generation_settings"]
                    done = self._review_and_repair(
                        rec,
                        gs.get("language_block") or "",
                        gs.get("lang_vars") or {},
                        card,
                        case,
                    )
                    self._persist_conversation(done)
                    reviewed.append(done)
                return finished + reviewed
            return finished + pending_existing

        resolved, mapping, persona = resolve_placeholders(
            card,
            family=family,
            case=case_kind,
            pair=pair,
            history_id=history_id,
            seed=self.run.seed,
        )
        case = resolved["cases"][case_kind]
        profile = load_profile(pair)
        lang_rng = keyed_rng(family, case_kind, pair, history_id, "language", seed=self.run.seed)
        language_block, lang_vars = render_language_block(pair, profile, lang_rng)
        lexicon = local_lexicon(profile)
        prompt_versions: dict[str, str] = {
            "language_block": lang_vars.get("language_block_version") or "lang-0.1"
        }

        sibling = (finished or pending_existing or [None])[0]
        if sibling:
            gs = sibling["provenance"]["generation_settings"]
            plan = gs.get("plan") or {}
            prompt_versions = dict(gs.get("prompt_versions") or prompt_versions)
            language_block = gs.get("language_block") or language_block
            lang_vars = {**lang_vars, **(gs.get("lang_vars") or {})}
            mapping = gs.get("resolved_placeholders") or mapping
            persona = gs.get("persona") or persona
            branch_after = sibling["private_annotations"]["branch_after_turn"]
            prefix_turns = [
                dict(t) for t in sibling["turns"] if int(t["turn_id"]) <= int(branch_after)
            ]
            p_hash = sibling["private_annotations"]["prefix_hash"]
        else:
            plan, plan_ver = self._make_plan(
                resolved, case, pair, language_block, mapping, persona
            )
            prompt_versions["plan"] = plan_ver
            prefix_turns, prefix_ver, p_hash = self._make_prefix(
                plan, language_block, mapping, case, pair, history_id
            )
            prompt_versions["prefix"] = prefix_ver

        records: list[dict[str, Any]] = []
        prefix_reject = False
        regenerated = False
        while True:
            records = list(finished)
            for condition in self.run.conditions:
                if condition not in missing:
                    continue
                cid = conversation_id(case["case_id"], pair, history_id, condition)
                existing_path = self.run.run_dir / "conversations" / f"{cid}.json"
                if existing_path.exists():
                    existing = json.loads(existing_path.read_text(encoding="utf-8"))
                    disp = (existing.get("provenance") or {}).get("generation_settings", {}).get(
                        "disposition"
                    )
                    if disp in {"accepted", "repaired", "rejected"}:
                        records.append(existing)
                        continue
                try:
                    rec = self._generate_branch(
                        resolved=resolved,
                        case=case,
                        case_kind=case_kind,
                        pair=pair,
                        history_id=history_id,
                        condition=condition,
                        plan=plan,
                        prefix_turns=prefix_turns,
                        prefix_hash=p_hash,
                        language_block=language_block,
                        lang_vars=lang_vars,
                        mapping=mapping,
                        persona=persona,
                        profile=profile,
                        lexicon=lexicon,
                        prompt_versions=prompt_versions,
                    )
                    self._persist_conversation(rec)
                    records.append(rec)
                except BudgetExceeded:
                    raise
                except Exception as exc:
                    print(f"BRANCH ERROR {cid}: {type(exc).__name__}: {exc}")
                    continue
            records.extend(pending_existing)
            hashes = [r["private_annotations"]["prefix_hash"] for r in records]
            if any(h != p_hash for h in hashes):
                prefix_reject = True
            if self.run.sync_review:
                reviewed = []
                need_prefix_regen = False
                for rec in records:
                    if rec["provenance"]["generation_settings"].get("disposition") in {
                        "accepted",
                        "repaired",
                        "rejected",
                    }:
                        reviewed.append(rec)
                        continue
                    rec = self._review_and_repair(rec, language_block, lang_vars, resolved, case)
                    if rec["provenance"]["generation_settings"].get("prefix_reject"):
                        need_prefix_regen = True
                    reviewed.append(rec)
                records = reviewed
                if need_prefix_regen and not regenerated:
                    regenerated = True
                    plan, plan_ver = self._make_plan(
                        resolved, case, pair, language_block, mapping, persona
                    )
                    prompt_versions["plan"] = plan_ver
                    prefix_turns, prefix_ver, p_hash = self._make_prefix(
                        plan, language_block, mapping, case, pair, history_id
                    )
                    prompt_versions["prefix"] = prefix_ver
                    continue
            break

        sibling_hashes = [r["private_annotations"]["prefix_hash"] for r in records]
        for rec in records:
            rec["private_annotations"]["invariance_eligible"] = (
                all(h == sibling_hashes[0] for h in sibling_hashes)
                and rec["provenance"]["generation_settings"].get("disposition")
                in {"accepted", "repaired"}
            )
            rec["provenance"]["generation_settings"]["sibling_prefix_hashes_match"] = (
                len(set(sibling_hashes)) == 1
            )
            self._persist_conversation(rec)
        _ = prefix_reject
        return records

    def _make_plan(
        self,
        card: dict[str, Any],
        case: dict[str, Any],
        pair: str,
        language_block: str,
        mapping: dict[str, str],
        persona: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        gc = card.get("generation_constraints") or {}
        values = {
            "case_id": case["case_id"],
            "scenario_label": case["scenario_label"],
            "card_case_json": dumps(case),
            "shared_setting_json": dumps(card.get("shared_setting") or {}),
            "resolved_placeholders_json": dumps(mapping),
            "recipient_persona_json": dumps(persona),
            "pair_name": pair,
            "branch_after": str(case.get("branch_after") or 7),
            "must_preserve": dumps(gc.get("must_preserve") or []),
            "may_vary": dumps(gc.get("may_vary") or []),
            "must_not_emit": dumps(gc.get("must_not_emit") or []),
            "language_block": language_block,
        }
        version, system, user = self.run.render_cached("plan", values)
        parsed = self.run.call_llm(
            "plan",
            system=system,
            user=user,
            conversation_id=case["case_id"],
            mock_context={
                "dialogue_plan": case.get("dialogue_plan") or [],
                "case_id": case["case_id"],
                "branch_after": case.get("branch_after"),
                "stable_facts": case.get("stable_facts") or [],
            },
        )
        return parsed, version

    def _make_prefix(
        self,
        plan: dict[str, Any],
        language_block: str,
        mapping: dict[str, str],
        case: dict[str, Any],
        pair: str,
        history_id: str,
    ) -> tuple[list[dict[str, Any]], str, str]:
        values = {
            "language_block": language_block,
            "plan_json": dumps(plan),
            "resolved_placeholders_json": dumps(mapping),
            "branch_after": str(plan.get("branch_after") or case.get("branch_after") or 7),
        }
        version, system, user = self.run.render_cached("prefix", values)
        last_err: Exception | None = None
        for _ in range(2):
            parsed = self.run.call_llm(
                "prefix",
                system=system,
                user=user,
                conversation_id=f"{case['case_id']}_{pair}_{history_id}",
                mock_context={"plan": plan},
            )
            turns = parsed.get("turns") or []
            try:
                validate_prefix_against_plan(turns, plan)
                p_hash = prefix_hash(turns, int(plan.get("branch_after") or turns[-1]["turn_id"]))
                return turns, version, p_hash
            except ValueError as exc:
                last_err = exc
        raise last_err or ValueError("prefix validation failed")

    def _generate_branch(
        self,
        *,
        resolved: dict[str, Any],
        case: dict[str, Any],
        case_kind: str,
        pair: str,
        history_id: str,
        condition: str,
        plan: dict[str, Any],
        prefix_turns: list[dict[str, Any]],
        prefix_hash: str,
        language_block: str,
        lang_vars: dict[str, Any],
        mapping: dict[str, str],
        persona: dict[str, Any],
        profile: dict[str, Any],
        lexicon: set[str],
        prompt_versions: dict[str, str],
    ) -> dict[str, Any]:
        cid = conversation_id(case["case_id"], pair, history_id, condition)
        turns = [dict(t) for t in prefix_turns]
        branch_after = int(plan.get("branch_after") or case.get("branch_after") or len(turns))
        ended_by: str | None = None
        stopped_for_repetition = False
        next_speaker = "recipient"
        versions = dict(prompt_versions)
        for step in range(1, self.run.max_post + 1):
            if next_speaker == "recipient":
                hint = FIRST_TURN_ENDING_HINT if step == 1 else LATER_TURN_ENDING_HINT
                values = {
                    "language_block": language_block,
                    "recipient_persona_json": dumps(persona),
                    "condition_instruction": CONDITION_INSTRUCTIONS[condition],
                    "behaviour_hint": behaviour_hint(resolved, condition, case_kind),
                    "visible_transcript": format_transcript(turns),
                    "ending_hint": hint,
                }
                ver, system, user = self.run.render_cached("recipient", values)
                versions["recipient"] = ver
                parsed = self.run.call_llm(
                    "recipient",
                    system=system,
                    user=user,
                    conversation_id=cid,
                    mock_context={
                        "condition": condition,
                        "post_branch_turn": step,
                        "end_after": 2,
                    },
                )
                tid = turns[-1]["turn_id"] + 1
                candidate = {"turn_id": tid, "speaker": "recipient", "text": parsed["text"]}
                if should_stop_for_repetition(turns, candidate):
                    stopped_for_repetition = True
                    ended_by = "repetition"
                    break
                turns.append(candidate)
                if parsed.get("end_call"):
                    ended_by = "recipient"
                    break
                next_speaker = "caller"
            else:
                brief = {
                    "scenario_label": case["scenario_label"],
                    "caller_objective": case.get("caller_objective"),
                    "stable_facts": case.get("stable_facts") or [],
                    "facts_committed": plan.get("facts_committed") or [],
                    "notes_for_continuation": plan.get("notes_for_continuation") or "",
                }
                allowed = sanitize_allowed_endings(
                    (resolved.get("allowed_endings") or {}).get(case_kind) or []
                )
                values = {
                    "language_block": language_block,
                    "caller_brief_json": dumps(brief),
                    "caller_continuation_limits": resolved.get("caller_continuation_limits") or "",
                    "allowed_endings": dumps(allowed),
                    "visible_transcript": format_transcript(turns),
                    "post_branch_turn": str(step),
                    "max_post_branch_turns": str(self.run.max_post),
                }
                ver, system, user = self.run.render_cached("caller_continuation", values)
                versions["caller"] = ver
                _assert_caller_prompt_clean(system, user)
                parsed = self.run.call_llm(
                    "caller",
                    system=system,
                    user=user,
                    conversation_id=cid,
                    mock_context={"post_branch_turn": step, "end_after": 2},
                )
                tid = turns[-1]["turn_id"] + 1
                candidate = {"turn_id": tid, "speaker": "caller", "text": parsed["text"]}
                if should_stop_for_repetition(turns, candidate):
                    stopped_for_repetition = True
                    ended_by = "repetition"
                    break
                turns.append(candidate)
                if parsed.get("end_call") or step >= self.run.max_post:
                    ended_by = "caller"
                    break
                next_speaker = "recipient"

        checks = run_automatic_checks(
            record=None,
            turns=turns,
            condition=condition,
            branch_after=branch_after,
            sibling_hashes=None,
            expected_prefix_hash=prefix_hash,
            lexicon=lexicon,
            p10=float(lang_vars.get("english_share_p10") or 0.0),
            p90=float(lang_vars.get("english_share_p90") or 1.0),
            pair=pair,
        )
        ev_or_markers = case.get("evidence") or case.get("markers") or []
        record = {
            "schema_version": "1.0",
            "conversation_id": cid,
            "scenario_family_id": resolved["family_id"],
            "case_id": case["case_id"],
            "history_id": history_id,
            "branch_id": condition,
            "language_pair": pair,
            "script_profile": "roman",
            "split": self.splits.get(resolved["family_id"], resolved.get("split") or "unassigned"),
            "turns": turns,
            "private_annotations": {
                "scenario_label": case["scenario_label"],
                "recipient_condition": condition,
                "prefix_hash": prefix_hash,
                "branch_after_turn": branch_after,
                "evidence_turn_ids": _evidence_turn_ids(plan),
                "prefix_evidence_states": _prefix_evidence_states(plan),
                "observed_outcome": _observed_outcome(turns, ended_by),
                "invariance_eligible": False,
            },
            "provenance": {
                "generator_model_id": self.run.roles["plan"]["model"],
                "prompt_version": ";".join(f"{k}={v}" for k, v in sorted(versions.items())),
                "scenario_version": resolved.get("version") or "0.1",
                "language_profile_version": profile.get("version") or "0.1",
                "generation_settings": {
                    "seed": self.run.seed,
                    "dry_run": self.run.dry_run,
                    "sync_review": self.run.sync_review,
                    "models": {k: v.get("model") for k, v in self.run.roles.items()},
                    "reasoning_effort": {
                        k: v.get("reasoning_effort") for k, v in self.run.roles.items()
                    },
                    "max_post_branch_turns": self.run.max_post,
                    "language_target_english_share": lang_vars.get("target_english_share"),
                    "resolved_placeholders": mapping,
                    "prompt_versions": versions,
                    "checks": checks,
                    "reviews": {},
                    "audit": None,
                    "disposition": "pending",
                    "stopped_for_repetition": stopped_for_repetition,
                    "rejection_reason": [],
                    "language_gate": self.run.language_gate,
                    "language_metrics": {},
                    "prefix_reject": False,
                    "case_kind": case_kind,
                    "evidence_or_markers": ev_or_markers,
                    "label_rationale": case.get("label_rationale") or "",
                    "must_preserve": (resolved.get("generation_constraints") or {}).get("must_preserve") or [],
                    "must_not_emit": (resolved.get("generation_constraints") or {}).get("must_not_emit") or [],
                    "caller_continuation_limits": resolved.get("caller_continuation_limits") or "",
                    "plan": plan,
                    "persona": persona,
                    "language_block": language_block,
                    "lang_vars": {k: v for k, v in lang_vars.items() if isinstance(v, (str, int, float))},
                },
                "repair_history": [],
                "review_status": "pending",
            },
        }
        errs = schema_errors(record, "conversation")
        record["provenance"]["generation_settings"]["schema_errors"] = errs
        return record

    def _review_jobs_for(self, rec: dict[str, Any], language_block: str, lang_vars: dict[str, Any]) -> list[LLMRequest]:
        jobs = []
        for role, name in (
            ("semantic_review", "semantic_review"),
            ("language_review", "language_review"),
        ):
            system, user, schema_role = self._review_prompt(rec, role, language_block, lang_vars)
            cfg = self.run.role_cfg(role)
            jobs.append(
                LLMRequest(
                    role=role,
                    system=system,
                    user=user,
                    schema=schema_for(schema_role),
                    model=cfg["model"],
                    provider=cfg["provider"],
                    max_output_tokens=int(cfg["max_output_tokens"]),
                    reasoning_effort=cfg.get("reasoning_effort"),
                    temperature=cfg.get("temperature"),
                    conversation_id=rec["conversation_id"],
                    mock_context={
                        "branch_after": rec["private_annotations"]["branch_after_turn"]
                    },
                    cache_system=True,
                )
            )
            _ = name
        return jobs

    def _review_prompt(
        self,
        rec: dict[str, Any],
        role: str,
        language_block: str,
        lang_vars: dict[str, Any],
    ) -> tuple[str, str, str]:
        gs = rec["provenance"]["generation_settings"]
        condition = rec["private_annotations"]["recipient_condition"]
        branch_after = rec["private_annotations"]["branch_after_turn"]
        if role in {"semantic_review", "audit_semantic"}:
            values = {
                "case_id": rec["case_id"],
                "scenario_label": rec["private_annotations"]["scenario_label"],
                "label_rationale": gs.get("label_rationale") or "",
                "evidence_or_markers_json": dumps(gs.get("evidence_or_markers") or []),
                "branch_after": str(branch_after),
                "recipient_condition": condition,
                "condition_definition": CONDITION_DEFINITIONS[condition],
                "stable_facts_json": dumps((gs.get("plan") or {}).get("facts_committed") or []),
                "resolved_placeholders_json": dumps(gs.get("resolved_placeholders") or {}),
                "must_not_emit": dumps(gs.get("must_not_emit") or []),
                "caller_continuation_limits": gs.get("caller_continuation_limits") or "",
                "transcript_numbered": format_transcript(rec["turns"]),
                "language_block": language_block,
            }
            _ver, system, user = self.run.render_cached("semantic_review", values)
            return system, user, "semantic_review"
        values = {
            "pair_name": lang_vars.get("pair_name") or rec["language_pair"],
            "region": lang_vars.get("region") or "",
            "english_share_p50_pct": lang_vars.get("english_share_p50_pct") or "36",
            "english_share_p25_pct": lang_vars.get("english_share_p25_pct") or "16",
            "english_share_p75_pct": lang_vars.get("english_share_p75_pct") or "75",
            "monolingual_share_pct": lang_vars.get("monolingual_share_pct") or "18",
            "light_verb_examples": lang_vars.get("light_verb_examples") or "",
            "mixed_morphology_examples": lang_vars.get("mixed_morphology_examples") or "",
            "romanization_variants": lang_vars.get("romanization_variants") or "",
            "style_anchors": lang_vars.get("style_anchors") or "",
            "unnatural_examples_with_reasons": lang_vars.get("unnatural_examples_with_reasons") or "",
            "local_language": lang_vars.get("local_language") or "",
            "transcript_numbered": format_transcript(rec["turns"]),
            "language_block": language_block,
        }
        _ver, system, user = self.run.render_cached("language_review", values)
        return system, user, "language_review"

    def _review_and_repair(
        self,
        rec: dict[str, Any],
        language_block: str,
        lang_vars: dict[str, Any],
        resolved: dict[str, Any],
        case: dict[str, Any],
    ) -> dict[str, Any]:
        gs = rec["provenance"]["generation_settings"]
        condition = rec["private_annotations"]["recipient_condition"]
        branch_after = rec["private_annotations"]["branch_after_turn"]
        used_repair = False
        for rnd in range(0, self.run.max_repair + 1):
            checks = run_automatic_checks(
                record=rec,
                turns=rec["turns"],
                condition=condition,
                branch_after=branch_after,
                sibling_hashes=None,
                expected_prefix_hash=rec["private_annotations"]["prefix_hash"],
                lexicon=local_lexicon(load_profile(rec["language_pair"])),
                p10=float(lang_vars.get("english_share_p10") or 0.0),
                p90=float(lang_vars.get("english_share_p90") or 1.0),
                pair=rec["language_pair"],
            )
            gs["checks"] = checks
            review_errors: dict[str, str] = {}
            try:
                sem = self._sync_review(rec, "semantic_review", language_block, lang_vars)
            except Exception as exc:
                sem = None
                review_errors["semantic"] = f"{type(exc).__name__}:{exc}"
            try:
                lang = self._sync_review(rec, "language_review", language_block, lang_vars)
            except Exception as exc:
                lang = None
                review_errors["language"] = f"{type(exc).__name__}:{exc}"
            store_reviews(gs, sem, lang, errors=review_errors or None)
            gs["language_metrics"] = language_metrics(lang)
            last_round = rnd >= self.run.max_repair
            verdict = combine_verdict(
                sem,
                lang,
                checks,
                honor_auto_flags=not last_round if self.run.language_gate == "scores" else True,
                language_gate=self.run.language_gate,
            )
            if verdict == "accept":
                gs["disposition"] = "repaired" if used_repair else "accepted"
                gs["rejection_reason"] = []
                rec["provenance"]["review_status"] = "accepted"
                self._maybe_audit(rec, language_block, lang_vars)
                self._persist_conversation(rec)
                return rec
            if verdict == "reject":
                gs["prefix_reject"] = True
                gs["disposition"] = "rejected"
                gs["rejection_reason"] = collect_rejection_reasons(
                    sem, lang, checks, language_gate=self.run.language_gate
                )
                rec["provenance"]["review_status"] = "rejected"
                self._maybe_audit(rec, language_block, lang_vars)
                self._persist_conversation(rec)
                return rec
            if rnd >= self.run.max_repair:
                break
            used_repair = True
            self._apply_repair(rec, sem, lang, checks, language_block, resolved, case)
        gs["disposition"] = "rejected"
        gs["rejection_reason"] = collect_rejection_reasons(
            sem, lang, checks, language_gate=self.run.language_gate
        )
        rec["provenance"]["review_status"] = "rejected"
        self._maybe_audit(rec, language_block, lang_vars)
        self._persist_conversation(rec)
        return rec

    def _sync_review(
        self,
        rec: dict[str, Any],
        role: str,
        language_block: str,
        lang_vars: dict[str, Any],
    ) -> dict[str, Any]:
        system, user, schema_role = self._review_prompt(rec, role, language_block, lang_vars)
        return self.run.call_llm(
            role,
            system=system,
            user=user,
            conversation_id=rec["conversation_id"],
            mock_context={
                "branch_after": rec["private_annotations"]["branch_after_turn"]
            },
        )

    def _apply_repair(
        self,
        rec: dict[str, Any],
        sem: dict[str, Any],
        lang: dict[str, Any],
        checks: dict[str, Any],
        language_block: str,
        resolved: dict[str, Any],
        case: dict[str, Any],
    ) -> None:
        gs = rec["provenance"]["generation_settings"]
        condition = rec["private_annotations"]["recipient_condition"]
        branch_after = rec["private_annotations"]["branch_after_turn"]
        targets = collect_repair_targets(
            sem=sem,
            lang=lang,
            checks=checks,
            turns=rec["turns"],
            branch_after=branch_after,
            language_gate=self.run.language_gate,
        )
        before = [dict(t) for t in rec["turns"]]
        values = {
            "language_block": language_block,
            "frozen_until": str(branch_after),
            "case_id": rec["case_id"],
            "scenario_label": rec["private_annotations"]["scenario_label"],
            "recipient_condition": condition,
            "condition_definition": CONDITION_DEFINITIONS[condition],
            "must_preserve": dumps(gs.get("must_preserve") or []),
            "must_not_emit": dumps(gs.get("must_not_emit") or []),
            "caller_continuation_limits": gs.get("caller_continuation_limits") or "",
            "transcript_numbered": format_transcript(rec["turns"]),
            "repair_targets_json": dumps(targets),
        }
        _ver, system, user = self.run.render_cached("repair", values)
        try:
            parsed = self.run.call_llm(
                "repair",
                system=system,
                user=user,
                conversation_id=rec["conversation_id"],
                mock_context={"repair_targets": targets},
            )
        except Exception:
            rec["provenance"]["repair_history"].append(
                {
                    "round": len(rec["provenance"]["repair_history"]) + 1,
                    "reason": dumps(targets),
                    "diff": "repair call failed (empty or invalid JSON)",
                }
            )
            return
        by_id = {t["turn_id"]: t for t in rec["turns"]}
        for rev in parsed.get("revised_turns") or []:
            tid = int(rev["turn_id"])
            if tid <= branch_after:
                continue
            if tid in by_id:
                by_id[tid]["text"] = rev["text"]
        rec["turns"] = [by_id[k] for k in sorted(by_id)]
        rec["provenance"]["repair_history"].append(
            {
                "round": len(rec["provenance"]["repair_history"]) + 1,
                "reason": dumps(targets),
                "diff": _turn_diff(before, rec["turns"]),
            }
        )

    def _maybe_audit(
        self,
        rec: dict[str, Any],
        language_block: str,
        lang_vars: dict[str, Any],
    ) -> None:
        if self.run.audit_sample <= 0:
            return
        rng = keyed_rng(rec["conversation_id"], "audit", seed=self.run.seed)
        if rng.random() >= self.run.audit_sample:
            return
        audit = {}
        for role in ("audit_semantic", "audit_language"):
            system, user, _schema = self._review_prompt(
                rec,
                "semantic_review" if role == "audit_semantic" else "language_review",
                language_block,
                lang_vars,
            )
            audit[role] = self.run.call_llm(
                role,
                system=system,
                user=user,
                conversation_id=rec["conversation_id"],
                mock_context={
                    "branch_after": rec["private_annotations"]["branch_after_turn"]
                },
            )
        rec["provenance"]["generation_settings"]["audit"] = audit

    def _batch_review_and_repair(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        pending = [
            r
            for r in records
            if r["provenance"]["generation_settings"].get("disposition") == "pending"
        ]
        if not pending:
            return records
        provider = (self.run.config.get("review") or {}).get("batch_provider") or "anthropic"
        runner = ReviewBatchRunner(
            run_dir=self.run.run_dir,
            ledger=self.run.ledger,
            prices=self.run.prices,
            provider=provider,
            dry_run=self.run.dry_run,
        )
        jobs: list[LLMRequest] = []
        for rec in pending:
            gs = rec["provenance"]["generation_settings"]
            lang_vars = {
                "pair_name": rec["language_pair"],
                "region": "",
                "english_share_p50_pct": "36",
                "english_share_p25_pct": "16",
                "english_share_p75_pct": "75",
                "monolingual_share_pct": "18",
                "light_verb_examples": "",
                "mixed_morphology_examples": "",
                "romanization_variants": "",
                "style_anchors": "",
                "unnatural_examples_with_reasons": "",
                "local_language": "",
                "english_share_p10": 0.0,
                "english_share_p90": 1.0,
            }
            jobs.extend(self._review_jobs_for(rec, "", lang_vars))
            gs["_lang_vars_stub"] = True
        results = runner.submit_and_collect(jobs)
        out = []
        for rec in records:
            if rec["provenance"]["generation_settings"].get("disposition") != "pending":
                out.append(rec)
                continue
            cid = rec["conversation_id"]
            sem_r = results.get(make_custom_id(cid, "semantic_review"))
            lang_r = results.get(make_custom_id(cid, "language_review"))
            sem, sem_err = review_from_result(sem_r)
            lang, lang_err = review_from_result(lang_r)
            errors = {}
            if sem_err:
                errors["semantic"] = sem_err
            if lang_err:
                errors["language"] = lang_err
            gs = rec["provenance"]["generation_settings"]
            store_reviews(gs, sem, lang, errors=errors or None)
            gs["language_metrics"] = language_metrics(lang)
            checks = gs.get("checks") or {}
            verdict = combine_verdict(
                sem, lang, checks, language_gate=self.run.language_gate
            )
            if verdict == "accept":
                gs["disposition"] = "accepted"
                gs["rejection_reason"] = []
                rec["provenance"]["review_status"] = "accepted"
            elif verdict == "reject":
                gs["disposition"] = "rejected"
                gs["prefix_reject"] = True
                gs["rejection_reason"] = collect_rejection_reasons(
                    sem, lang, checks, language_gate=self.run.language_gate
                )
                rec["provenance"]["review_status"] = "rejected"
            else:
                # Sync repair + re-review for the flagged subset (small).
                rec = self._review_and_repair(rec, "", {}, {}, rec)
            out.append(rec)
            self._persist_conversation(rec)
        return out

    def _language_jobs_for(self, records: list[dict[str, Any]]) -> list[LLMRequest]:
        jobs: list[LLMRequest] = []
        for rec in records:
            gs = rec["provenance"]["generation_settings"]
            system, user, schema_role = self._review_prompt(
                rec,
                "language_review",
                gs.get("language_block") or "",
                self._lang_vars_for(rec),
            )
            cfg = self.run.role_cfg("language_review")
            jobs.append(
                LLMRequest(
                    role="language_review",
                    system=system,
                    user=user,
                    schema=schema_for(schema_role),
                    model=cfg["model"],
                    provider=cfg["provider"],
                    max_output_tokens=int(cfg["max_output_tokens"]),
                    reasoning_effort=cfg.get("reasoning_effort"),
                    temperature=cfg.get("temperature"),
                    conversation_id=rec["conversation_id"],
                    mock_context={
                        "branch_after": rec["private_annotations"]["branch_after_turn"]
                    },
                    cache_system=True,
                )
            )
        return jobs

    def _language_review_is_missing(self, rec: dict[str, Any], ledger_cids: set[str]) -> bool:
        gs = rec["provenance"]["generation_settings"]
        missing_flag = (gs.get("review_missing") or {}).get("language")
        lang = (gs.get("reviews") or {}).get("language")
        if missing_flag or lang is None:
            return True
        if looks_like_placeholder_language_review(lang) and rec["conversation_id"] not in ledger_cids:
            return True
        return False

    def _reapply_disposition(
        self, rec: dict[str, Any], *, allow_repair: bool = True
    ) -> dict[str, Any] | None:
        gs = rec["provenance"]["generation_settings"]
        sem = (gs.get("reviews") or {}).get("semantic")
        lang = (gs.get("reviews") or {}).get("language")
        checks = gs.get("checks") or {}
        old = gs.get("disposition")
        old_reasons = list(gs.get("rejection_reason") or [])
        verdict = combine_verdict(sem, lang, checks, language_gate=self.run.language_gate)
        new = old
        if verdict == "accept":
            new = "repaired" if old == "repaired" else "accepted"
            gs["rejection_reason"] = []
            rec["provenance"]["review_status"] = "accepted"
        elif verdict == "reject":
            new = "rejected"
            gs["rejection_reason"] = collect_rejection_reasons(
                sem, lang, checks, language_gate=self.run.language_gate
            )
            rec["provenance"]["review_status"] = "rejected"
        elif allow_repair and old in {"accepted", "repaired"}:
            try:
                self._apply_repair(
                    rec,
                    sem or {},
                    lang or {},
                    checks,
                    gs.get("language_block") or "",
                    {},
                    {},
                )
                rec = self._review_and_repair(
                    rec,
                    gs.get("language_block") or "",
                    self._lang_vars_for(rec),
                    {},
                    {},
                )
                gs = rec["provenance"]["generation_settings"]
                new = gs.get("disposition")
            except BudgetExceeded:
                raise
            except Exception as exc:
                new = "rejected"
                gs["disposition"] = "rejected"
                gs["rejection_reason"] = collect_rejection_reasons(
                    sem, lang, checks, language_gate=self.run.language_gate
                ) or ["language_repair_failed"]
                rec["provenance"]["review_status"] = "rejected"
                gs.setdefault("review_errors", {})["disposition_repair"] = (
                    f"{type(exc).__name__}:{exc}"
                )
        else:
            new = old
            if new == "rejected":
                gs["rejection_reason"] = collect_rejection_reasons(
                    sem, lang, checks, language_gate=self.run.language_gate
                )
        gs["disposition"] = new
        skipped_repair = (
            verdict == "repair" and old in {"accepted", "repaired"} and not allow_repair
        )
        if (
            new == old
            and list(gs.get("rejection_reason") or []) == old_reasons
            and not skipped_repair
        ):
            return None
        change = {
            "conversation_id": rec["conversation_id"],
            "old": old,
            "new": new,
            "verdict": verdict,
            "old_reasons": old_reasons,
            "new_reasons": list(gs.get("rejection_reason") or []),
        }
        if skipped_repair:
            change["applied"] = False
            change["note"] = "language verdict is repair; repair skipped to keep review budget"
            gs["language_would_repair"] = True
        return change

    def rerun_language_reviews(self) -> dict[str, Any]:
        records = self._load_disk_records()
        ledger_cids = language_review_ledger_cids(self.run.run_dir)
        need: list[dict[str, Any]] = []
        for rec in records:
            if self._language_review_is_missing(rec, ledger_cids):
                gs = rec["provenance"]["generation_settings"]
                sem = (gs.get("reviews") or {}).get("semantic")
                store_reviews(gs, sem, None)
                gs["language_metrics"] = language_metrics(None)
                self._persist_conversation(rec)
                need.append(rec)
        changes: list[dict[str, Any]] = []
        reviewed = 0
        if need:
            runner = self._review_runner()
            chunk = max(1, self.run.review_chunk_size)
            for i in range(0, len(need), chunk):
                batch = need[i : i + chunk]
                jobs = self._language_jobs_for(batch)
                try:
                    results = runner.submit_and_collect(jobs)
                except BudgetExceeded:
                    print(f"BUDGET STOP during language re-review at {i}/{len(need)}")
                    break
                for rec in batch:
                    cid = rec["conversation_id"]
                    lang, lang_err = review_from_result(
                        results.get(make_custom_id(cid, "language_review"))
                    )
                    gs = rec["provenance"]["generation_settings"]
                    sem = (gs.get("reviews") or {}).get("semantic")
                    errors = dict(gs.get("review_errors") or {})
                    if lang_err:
                        errors["language"] = lang_err
                    elif "language" in errors:
                        errors.pop("language", None)
                    store_reviews(gs, sem, lang, errors=errors or None)
                    gs["language_metrics"] = language_metrics(lang)
                    if lang is not None:
                        reviewed += 1
                    change = self._reapply_disposition(rec, allow_repair=False)
                    self._persist_conversation(rec)
                    if change:
                        changes.append(change)
        else:
            for rec in records:
                change = self._reapply_disposition(rec, allow_repair=False)
                if change:
                    changes.append(change)
                    self._persist_conversation(rec)
        records = self._load_disk_records()
        self._write_exports(records)
        self._write_readable_sample(records)
        self._log_progress(force=True)
        change_path = self.run.run_dir / "disposition_changes.jsonl"
        atomic_write_jsonl(change_path, changes)
        return {
            "run_dir": str(self.run.run_dir),
            "language_reviews_needed": len(need),
            "language_reviews_recovered": reviewed,
            "disposition_changes": len(changes),
            "changes": changes,
        }

    def _semantic_jobs_for(self, records: list[dict[str, Any]]) -> list[LLMRequest]:
        jobs: list[LLMRequest] = []
        for rec in records:
            gs = rec["provenance"]["generation_settings"]
            system, user, schema_role = self._review_prompt(
                rec,
                "semantic_review",
                gs.get("language_block") or "",
                self._lang_vars_for(rec),
            )
            cfg = self.run.role_cfg("semantic_review")
            jobs.append(
                LLMRequest(
                    role="semantic_review",
                    system=system,
                    user=user,
                    schema=schema_for(schema_role),
                    model=cfg["model"],
                    provider=cfg["provider"],
                    max_output_tokens=int(cfg["max_output_tokens"]),
                    reasoning_effort=cfg.get("reasoning_effort"),
                    temperature=cfg.get("temperature"),
                    conversation_id=rec["conversation_id"],
                    mock_context={
                        "branch_after": rec["private_annotations"]["branch_after_turn"]
                    },
                    cache_system=True,
                )
            )
        return jobs

    def _collect_review_chunks(
        self, jobs: list[LLMRequest], label: str
    ) -> tuple[dict[str, Any], bool]:
        if not jobs:
            return {}, False
        runner = self._review_runner()
        chunk = max(1, self.run.review_chunk_size)
        out: dict[str, Any] = {}
        for i in range(0, len(jobs), chunk):
            part = jobs[i : i + chunk]
            print(f"{label}: submitting {i + 1}-{i + len(part)}/{len(jobs)}")
            try:
                out.update(runner.submit_and_collect(part))
            except BudgetExceeded:
                print(f"BUDGET STOP during {label} at {i}/{len(jobs)}")
                return out, True
        return out, False

    def _run_semantic_chunks(
        self, records: list[dict[str, Any]], label: str
    ) -> tuple[int, bool]:
        runner = self._review_runner()
        chunk = max(1, self.run.review_chunk_size)
        stored = 0
        for i in range(0, len(records), chunk):
            part = [
                rec
                for rec in records[i : i + chunk]
                if not semantic_is_v03(rec["provenance"]["generation_settings"])
            ]
            if not part:
                continue
            print(f"{label}: submitting {i + 1}-{i + len(records[i : i + chunk])}/{len(records)}")
            try:
                results = runner.submit_and_collect(self._semantic_jobs_for(part))
            except BudgetExceeded:
                print(f"BUDGET STOP during {label} at {i}/{len(records)}")
                return stored, True
            for rec in part:
                result = results.get(make_custom_id(rec["conversation_id"], "semantic_review"))
                if result is not None and self._store_semrev03_result(rec, result):
                    stored += 1
                self._persist_conversation(rec)
            print(f"{label}: stored {stored} so far")
        return stored, False

    def _run_language_chunks(
        self, records: list[dict[str, Any]], label: str
    ) -> tuple[int, bool]:
        runner = self._review_runner()
        chunk = max(1, self.run.review_chunk_size)
        stored = 0
        for i in range(0, len(records), chunk):
            part = records[i : i + chunk]
            print(f"{label}: submitting {i + 1}-{i + len(part)}/{len(records)}")
            try:
                results = runner.submit_and_collect(self._language_jobs_for(part))
            except BudgetExceeded:
                print(f"BUDGET STOP during {label} at {i}/{len(records)}")
                return stored, True
            for rec in part:
                gs = rec["provenance"]["generation_settings"]
                lang, lang_err = review_from_result(
                    results.get(make_custom_id(rec["conversation_id"], "language_review"))
                )
                sem = (gs.get("reviews") or {}).get("semantic")
                errors = dict(gs.get("review_errors") or {})
                if lang_err:
                    errors["language"] = lang_err
                else:
                    errors.pop("language", None)
                store_reviews(gs, sem, lang, errors=errors or None)
                gs["language_metrics"] = language_metrics(lang)
                if lang is not None:
                    stored += 1
                self._persist_conversation(rec)
            print(f"{label}: stored {stored} so far")
        return stored, False

    def _refresh_auto_checks(self, rec: dict[str, Any]) -> None:
        gs = rec["provenance"]["generation_settings"]
        lang_vars = self._lang_vars_for(rec)
        gs["checks"] = run_automatic_checks(
            record=rec,
            turns=rec["turns"],
            condition=rec["private_annotations"]["recipient_condition"],
            branch_after=rec["private_annotations"]["branch_after_turn"],
            sibling_hashes=None,
            expected_prefix_hash=rec["private_annotations"]["prefix_hash"],
            lexicon=local_lexicon(load_profile(rec["language_pair"])),
            p10=float(lang_vars.get("english_share_p10") or 0.0),
            p90=float(lang_vars.get("english_share_p90") or 1.0),
            pair=rec["language_pair"],
        )

    def _store_semrev03_result(self, rec: dict[str, Any], result: Any) -> bool:
        gs = rec["provenance"]["generation_settings"]
        archive_semantic_v02(gs)
        sem, sem_err = review_from_result(result)
        lang = (gs.get("reviews") or {}).get("language")
        errors = dict(gs.get("review_errors") or {})
        if sem_err:
            errors["semantic"] = sem_err
        else:
            errors.pop("semantic", None)
        store_reviews(gs, sem, lang, errors=errors or None)
        if sem is not None:
            gs["semantic_review_version"] = SEMREV03_VERSION
            return True
        return False

    def _store_semrev03_parsed(self, rec: dict[str, Any], sem: dict[str, Any] | None, err: str | None) -> bool:
        gs = rec["provenance"]["generation_settings"]
        archive_semantic_v02(gs)
        lang = (gs.get("reviews") or {}).get("language")
        errors = dict(gs.get("review_errors") or {})
        if err:
            errors["semantic"] = err
        else:
            errors.pop("semantic", None)
        store_reviews(gs, sem, lang, errors=errors or None)
        if sem is not None:
            gs["semantic_review_version"] = SEMREV03_VERSION
            return True
        return False

    def _apply_semrev03_disposition(self, rec: dict[str, Any]) -> str:
        """Apply current policy. Returns verdict: accept|reject|repair|missing."""
        gs = rec["provenance"]["generation_settings"]
        if not semantic_is_v03(gs):
            return "missing"
        sem = (gs.get("reviews") or {}).get("semantic")
        lang = (gs.get("reviews") or {}).get("language")
        checks = gs.get("checks") or {}
        verdict = combine_verdict(sem, lang, checks, language_gate=self.run.language_gate)
        if verdict == "accept":
            gs["disposition"] = (
                "repaired" if rec["provenance"].get("repair_history") else "accepted"
            )
            gs["rejection_reason"] = []
            gs["prefix_reject"] = False
            gs.pop("semrev03_needs_repair", None)
            rec["provenance"]["review_status"] = "accepted"
            return verdict
        if verdict == "reject":
            gs["disposition"] = "rejected"
            gs["prefix_reject"] = True
            gs["rejection_reason"] = collect_rejection_reasons(
                sem, lang, checks, language_gate=self.run.language_gate
            )
            gs.pop("semrev03_needs_repair", None)
            rec["provenance"]["review_status"] = "rejected"
            return verdict
        gs["semrev03_needs_repair"] = True
        return "repair"

    def _semrev03_apply_repairs(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        def work(rec: dict[str, Any]) -> dict[str, Any]:
            gs = rec["provenance"]["generation_settings"]
            try:
                self._apply_repair(
                    rec,
                    (gs.get("reviews") or {}).get("semantic") or {},
                    (gs.get("reviews") or {}).get("language") or {},
                    gs.get("checks") or {},
                    gs.get("language_block") or "",
                    {},
                    {},
                )
                self._refresh_auto_checks(rec)
                gs["semrev03_repaired"] = True
            except BudgetExceeded:
                raise
            except Exception as exc:
                print(f"REPAIR ERROR {rec['conversation_id']}: {exc}")
                gs.setdefault("review_errors", {})["semrev03_repair"] = (
                    f"{type(exc).__name__}:{exc}"
                )
            self._persist_conversation(rec)
            return rec

        if len(records) <= 1 or self.run.workers <= 1:
            return [work(rec) for rec in records]
        out: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.run.workers) as pool:
            futs = [pool.submit(work, rec) for rec in records]
            for fut in as_completed(futs):
                out.append(fut.result())
        by_id = {rec["conversation_id"]: rec for rec in out}
        return [by_id[rec["conversation_id"]] for rec in records if rec["conversation_id"] in by_id]

    def _semrev03_rereview(
        self, records: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], bool]:
        if not records:
            return [], False
        stored = 0
        if len(records) < self.run.sync_repair_threshold:
            print(f"SEMREV03 re-review sync n={len(records)}")
            for rec in records:
                gs = rec["provenance"]["generation_settings"]
                try:
                    sem = self._sync_review(
                        rec,
                        "semantic_review",
                        gs.get("language_block") or "",
                        self._lang_vars_for(rec),
                    )
                    self._store_semrev03_parsed(rec, sem, None)
                    stored += 1
                except BudgetExceeded:
                    print("BUDGET STOP during SEMREV03 sync re-review")
                    self._persist_conversation(rec)
                    return records, True
                except Exception as exc:
                    self._store_semrev03_parsed(
                        rec, None, f"{type(exc).__name__}:{exc}"
                    )
                self._persist_conversation(rec)
            return records, False
        jobs = self._semantic_jobs_for(records)
        results, stopped = self._collect_review_chunks(jobs, "SEMREV03 re-review")
        for rec in records:
            cid = rec["conversation_id"]
            result = results.get(make_custom_id(cid, "semantic_review"))
            if result is not None and self._store_semrev03_result(rec, result):
                stored += 1
            self._persist_conversation(rec)
        print(f"SEMREV03 re-review stored {stored}/{len(records)}")
        return records, stopped

    def _semrev03_repair_loop(self, need: list[dict[str, Any]]) -> dict[str, int]:
        remaining = list(need)
        stats = {"repair_rounds": 0, "repaired_ok": 0, "repair_exhausted": 0}
        for rnd in range(self.run.max_repair):
            if not remaining:
                break
            stats["repair_rounds"] = rnd + 1
            print(f"SEMREV03 repair round {rnd + 1}: {len(remaining)}")
            remaining = self._semrev03_apply_repairs(remaining)
            remaining, stopped = self._semrev03_rereview(remaining)
            still: list[dict[str, Any]] = []
            for rec in remaining:
                verdict = self._apply_semrev03_disposition(rec)
                self._persist_conversation(rec)
                if verdict == "accept":
                    stats["repaired_ok"] += 1
                elif verdict == "repair":
                    still.append(rec)
            remaining = still
            if stopped:
                break
        for rec in remaining:
            gs = rec["provenance"]["generation_settings"]
            sem = (gs.get("reviews") or {}).get("semantic")
            lang = (gs.get("reviews") or {}).get("language")
            checks = gs.get("checks") or {}
            gs["disposition"] = "rejected"
            gs["prefix_reject"] = False
            gs["rejection_reason"] = collect_rejection_reasons(
                sem, lang, checks, language_gate=self.run.language_gate
            ) or ["repair_exhausted"]
            gs.pop("semrev03_needs_repair", None)
            rec["provenance"]["review_status"] = "rejected"
            self._persist_conversation(rec)
            stats["repair_exhausted"] += 1
        return stats

    def _write_semrev03_artifacts(
        self,
        records: list[dict[str, Any]],
        *,
        stats: dict[str, Any],
    ) -> None:
        run_dir = self.run.run_dir
        v02_path = run_dir / "dispositions_semrev02.json"
        v02 = json.loads(v02_path.read_text(encoding="utf-8")) if v02_path.exists() else {}
        prefix_rows = []
        for rec in records:
            gs = rec["provenance"]["generation_settings"]
            if gs.get("disposition") != "rejected":
                continue
            reasons = prefix_broken_reasons(rec)
            if not reasons and not gs.get("prefix_reject"):
                continue
            if reasons or gs.get("prefix_reject"):
                prefix_rows.append(
                    {
                        "conversation_id": rec["conversation_id"],
                        "case": _case_kind_of(rec),
                        "condition": rec["private_annotations"]["recipient_condition"],
                        "pair": rec["language_pair"],
                        "split": rec.get("split"),
                        "reasons": reasons or list(gs.get("rejection_reason") or []),
                    }
                )
        prefix_rows.sort(key=lambda row: row["conversation_id"])
        atomic_write_json(
            run_dir / "prefix_broken_v03.json",
            {"n": len(prefix_rows), "items": prefix_rows},
        )
        transitions: dict[str, int] = {}
        for rec in records:
            cid = rec["conversation_id"]
            old = (v02.get(cid) or {}).get("status") or "missing"
            new = rec["provenance"]["generation_settings"].get("disposition") or "pending"
            key = f"{old}->{new}"
            transitions[key] = transitions.get(key, 0) + 1

        def snap_counts(items: list[dict[str, Any]], dim: str) -> dict[str, dict[str, int]]:
            out: dict[str, dict[str, int]] = {}
            for item in items:
                val = str(item.get(dim) or "")
                bucket = out.setdefault(
                    val, {"n": 0, "accepted": 0, "repaired": 0, "rejected": 0, "pending": 0}
                )
                bucket["n"] += 1
                st = item.get("status") or "pending"
                if st in bucket:
                    bucket[st] += 1
                else:
                    bucket["pending"] += 1
            return out

        v02_items = list(v02.values())
        v03_items = []
        for rec in records:
            gs = rec["provenance"]["generation_settings"]
            v03_items.append(
                {
                    "status": gs.get("disposition") or "pending",
                    "case": _case_kind_of(rec),
                    "condition": rec["private_annotations"]["recipient_condition"],
                    "pair": rec["language_pair"],
                    "split": rec.get("split"),
                }
            )
        v02_by = {dim: snap_counts(v02_items, dim) for dim in ("case", "condition", "pair", "split")}
        v03_by = {dim: snap_counts(v03_items, dim) for dim in ("case", "condition", "pair", "split")}
        complete = write_sibling_completeness(run_dir, records)
        lang_null = sum(
            1
            for rec in records
            if (rec["provenance"]["generation_settings"].get("reviews") or {}).get("language")
            is None
            or rec["provenance"]["generation_settings"].get("review_missing", {}).get("language")
        )
        v03_n = sum(
            1
            for rec in records
            if semantic_is_v03(rec["provenance"]["generation_settings"])
        )
        lines = [
            "# Semantic review v0.3 vs v0.2",
            "",
            "Policy: `semrev-0.3` accept + automatic checks ok + no hard language flags. "
            "Post-branch `repair` ran Luna (max 2 rounds). Prefix-broken rejects were kept.",
            "",
            f"- conversations: **{len(records)}**",
            f"- v0.3 semantic stored: **{v03_n}**",
            f"- language reviews still null: **{lang_null}**",
            f"- prefix-broken rejects (listed, not regenerated): **{len(prefix_rows)}**",
            f"- spend after this pass: **${self.run.ledger.spent:.4f}**",
            "",
            "## Disposition totals",
            "",
            "| version | accepted | repaired | rejected | pending |",
            "|---|---:|---:|---:|---:|",
        ]

        def totals(items: list[dict[str, Any]]) -> dict[str, int]:
            out = {"accepted": 0, "repaired": 0, "rejected": 0, "pending": 0}
            for item in items:
                st = item.get("status") or "pending"
                out[st] = out.get(st, 0) + 1
            return out

        t02, t03 = totals(v02_items), totals(v03_items)
        lines.append(
            f"| v0.2 | {t02['accepted']} | {t02['repaired']} | {t02['rejected']} | {t02['pending']} |"
        )
        lines.append(
            f"| v0.3 | {t03['accepted']} | {t03['repaired']} | {t03['rejected']} | {t03['pending']} |"
        )
        lines += ["", "## Transitions (v0.2 → v0.3)", "", "| transition | n |", "|---|---:|"]
        for key in sorted(transitions):
            lines.append(f"| `{key}` | {transitions[key]} |")
        for dim in ("case", "condition", "pair", "split"):
            keys = sorted(set(v02_by[dim]) | set(v03_by[dim]))
            lines += [
                "",
                f"## By {dim}",
                "",
                f"| {dim} | v02 acc | v02 rep | v02 rej | v03 acc | v03 rep | v03 rej |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
            for key in keys:
                a = v02_by[dim].get(key, {})
                b = v03_by[dim].get(key, {})
                lines.append(
                    f"| {key} | {a.get('accepted', 0)} | {a.get('repaired', 0)} | "
                    f"{a.get('rejected', 0)} | {b.get('accepted', 0)} | "
                    f"{b.get('repaired', 0)} | {b.get('rejected', 0)} |"
                )
        sc = complete.get("split_case") or {}
        lines += [
            "",
            "## Complete 4-sibling sets (v0.3)",
            "",
            f"- complete sets: **{complete.get('n_complete', 0)}** / {complete.get('n_groups', 0)} groups",
            "",
            "| split | fraud | legitimate | total |",
            "|---|---:|---:|---:|",
        ]
        for split in sorted(sc):
            row = sc[split]
            lines.append(
                f"| {split} | {row.get('fraud', 0)} | {row.get('legitimate', 0)} | "
                f"{row.get('fraud', 0) + row.get('legitimate', 0)} |"
            )
        v02_complete = {}
        old_complete = run_dir / "complete_sets_semrev02.json"
        if old_complete.exists():
            v02_complete = json.loads(old_complete.read_text(encoding="utf-8"))
        lines += [
            "",
            f"v0.2 complete sets: **{v02_complete.get('n_complete', 'n/a')}** → "
            f"v0.3: **{complete.get('n_complete', 0)}**",
            "",
            "## Repair loop",
            "",
            f"- repair rounds run: {stats.get('repair_rounds', 0)}",
            f"- accepted after repair: {stats.get('repaired_ok', 0)}",
            f"- repair exhausted → rejected: {stats.get('repair_exhausted', 0)}",
            f"- semantic jobs submitted: {stats.get('semantic_submitted', 0)}",
            f"- semantic stored: {stats.get('semantic_stored', 0)}",
            f"- language jobs submitted: {stats.get('language_submitted', 0)}",
            f"- language stored: {stats.get('language_stored', 0)}",
            "",
            "## Rejection audit",
            "",
            "See `rejection_audit_semrev03.md` (filled after this pass).",
            "",
        ]
        path = run_dir / "SEMREV03_REPORT.md"
        tmp = path.with_suffix(".tmp")
        tmp.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        tmp.replace(path)
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            summary = {}
        summary.update(
            {
                "semantic_review_version": SEMREV03_VERSION,
                "language_reviews_null": lang_null,
                "prefix_broken_v03": len(prefix_rows),
                "complete_sets": complete.get("n_complete"),
                "semrev03": {
                    "transitions": transitions,
                    **{k: stats.get(k) for k in stats},
                },
            }
        )
        atomic_write_json(summary_path, summary)

    def rerun_semrev03(self) -> dict[str, Any]:
        self._target_n = 2880
        records = self._load_disk_records()
        archived = 0
        for rec in records:
            gs = rec["provenance"]["generation_settings"]
            if archive_semantic_v02(gs):
                archived += 1
                self._persist_conversation(rec)
        print(f"SEMREV03 archived semantic_v02 on {archived} records")
        self._review_runner(purpose="semrev03")
        records = self._load_disk_records()
        need_sem = [
            rec
            for rec in records
            if not semantic_is_v03(rec["provenance"]["generation_settings"])
        ]
        semantic_stored = 0
        budget_stop = False
        if need_sem:
            semantic_stored, budget_stop = self._run_semantic_chunks(
                need_sem, "SEMREV03 semantic"
            )
            print(f"SEMREV03 semantic stored {semantic_stored}/{len(need_sem)}")
        records = self._load_disk_records()
        ledger_cids = language_review_ledger_cids(self.run.run_dir)
        need_lang = [
            rec for rec in records if self._language_review_is_missing(rec, ledger_cids)
        ]
        language_stored = 0
        if need_lang and not budget_stop:
            language_stored, budget_stop = self._run_language_chunks(
                need_lang, "SEMREV03 language"
            )
            print(f"SEMREV03 language stored {language_stored}/{len(need_lang)}")
        records = self._load_disk_records()
        need_repair: list[dict[str, Any]] = []
        for rec in records:
            self._refresh_auto_checks(rec)
            verdict = self._apply_semrev03_disposition(rec)
            self._persist_conversation(rec)
            if verdict == "repair":
                need_repair.append(rec)
        print(f"SEMREV03 disposition pass: {len(need_repair)} need post-branch repair")
        repair_stats = {"repair_rounds": 0, "repaired_ok": 0, "repair_exhausted": 0}
        if need_repair and not budget_stop:
            try:
                repair_stats = self._semrev03_repair_loop(need_repair)
            except BudgetExceeded:
                print("BUDGET STOP during SEMREV03 repair loop")
        records = self._load_disk_records()
        self._write_exports(records)
        self._write_readable_sample(records)
        stats = {
            "archived_v02": archived,
            "semantic_submitted": len(need_sem),
            "semantic_stored": semantic_stored,
            "language_submitted": len(need_lang),
            "language_stored": language_stored,
            **repair_stats,
        }
        self._write_semrev03_artifacts(records, stats=stats)
        self._log_progress(force=True)
        return {
            "run_dir": str(self.run.run_dir),
            **stats,
            "n": len(records),
        }

    def snapshot_dispositions(self, filename: str) -> dict[str, Any]:
        records = self._load_disk_records()
        snap: dict[str, Any] = {}
        for rec in records:
            gs = rec["provenance"]["generation_settings"]
            snap[rec["conversation_id"]] = {
                "status": gs.get("disposition"),
                "reasons": list(gs.get("rejection_reason") or []),
                "case": _case_kind_of(rec),
                "condition": rec["private_annotations"]["recipient_condition"],
                "pair": rec["language_pair"],
                "split": rec.get("split"),
            }
        atomic_write_json(self.run.run_dir / filename, snap)
        return snap

    def recompute_dispositions(self) -> dict[str, Any]:
        """Offline gate change: no API. Keep language-repaired text."""
        self._target_n = 2880
        snap = self.snapshot_dispositions("dispositions_semrev03_langscore.json")
        records = self._load_disk_records()
        transitions: dict[str, int] = {}
        lang_kept = 0
        for rec in records:
            gs = rec["provenance"]["generation_settings"]
            old = gs.get("disposition") or "pending"
            if language_driven_repair(rec):
                gs["language_driven_repair_kept"] = True
                lang_kept += 1
            self._refresh_auto_checks(rec)
            sem = (gs.get("reviews") or {}).get("semantic")
            lang = (gs.get("reviews") or {}).get("language")
            checks = gs.get("checks") or {}
            if not semantic_is_v03(gs) and (sem is None or sem.get("review_missing")):
                new = old
            else:
                verdict = combine_verdict(
                    sem, lang, checks, language_gate=self.run.language_gate
                )
                if verdict == "accept":
                    new = "repaired" if rec["provenance"].get("repair_history") else "accepted"
                    gs["rejection_reason"] = []
                    gs["prefix_reject"] = False
                    rec["provenance"]["review_status"] = "accepted"
                elif verdict == "reject":
                    new = "rejected"
                    gs["prefix_reject"] = True
                    gs["rejection_reason"] = collect_rejection_reasons(
                        sem, lang, checks, language_gate=self.run.language_gate
                    )
                    rec["provenance"]["review_status"] = "rejected"
                else:
                    if old in {"accepted", "repaired"}:
                        new = old
                    else:
                        new = "rejected"
                        gs["rejection_reason"] = collect_rejection_reasons(
                            sem, lang, checks, language_gate=self.run.language_gate
                        ) or list(gs.get("rejection_reason") or [])
                        rec["provenance"]["review_status"] = "rejected"
            gs["disposition"] = new
            gs["language_gate"] = self.run.language_gate
            key = f"{old}->{new}"
            transitions[key] = transitions.get(key, 0) + 1
            self._persist_conversation(rec)
        records = self._load_disk_records()
        self._write_exports(records)
        remaining = [
            rec
            for rec in records
            if rec["provenance"]["generation_settings"].get("disposition") == "rejected"
        ]
        label_balance = {"fraud": 0, "legitimate": 0}
        for rec in remaining:
            label_balance[_case_kind_of(rec)] = label_balance.get(_case_kind_of(rec), 0) + 1
        payload = {
            "gate": self.run.language_gate,
            "transitions": transitions,
            "language_driven_repair_kept": lang_kept,
            "remaining_rejected": len(remaining),
            "remaining_label_balance": label_balance,
        }
        atomic_write_json(self.run.run_dir / "langscore_gate_transitions.json", payload)
        print(
            f"recompute gate={self.run.language_gate} transitions={transitions} "
            f"remaining_rej={len(remaining)} balance={label_balance} lang_kept={lang_kept}"
        )
        return {"run_dir": str(self.run.run_dir), **payload, "n": len(records)}

    def _prefix_broken_units(self) -> list[tuple[str, str, str, str]]:
        path = self.run.run_dir / "prefix_broken_v03.json"
        if not path.exists():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        units: set[tuple[str, str, str, str]] = set()
        cid_re = re.compile(
            r"^(.+)_(F|L)_(hi_en|ta_en|te_en)_(h\d+)_(cooperative|clarifying|resistant|suspicious)$"
        )
        for item in payload.get("items") or []:
            cid = str(item.get("conversation_id") or "")
            match = cid_re.match(cid)
            if not match:
                continue
            family, letter, pair, history, _cond = match.groups()
            case_kind = "fraud" if letter == "F" else "legitimate"
            units.add((family, case_kind, pair, history))
        return sorted(units)

    def _move_to_superseded(self, rec: dict[str, Any]) -> Path:
        dest_dir = self.run.run_dir / "superseded"
        dest_dir.mkdir(parents=True, exist_ok=True)
        src = self.run.run_dir / "conversations" / f"{rec['conversation_id']}.json"
        dest = dest_dir / f"{rec['conversation_id']}.json"
        if dest.exists():
            dest = dest_dir / f"{rec['conversation_id']}.{int(time.time())}.json"
        if src.exists():
            shutil.move(str(src), str(dest))
        return dest

    def regenerate_prefix_broken(self) -> dict[str, Any]:
        units = self._prefix_broken_units()
        n_new = len(units) * len(self.run.conditions)
        print(f"prefix-broken units={len(units)} conversations_to_regen={n_new}")
        self._target_n = max(n_new, 1)
        self.run.sync_review = n_new < 100
        moved = 0
        generated = 0
        by_id: dict[str, dict[str, Any]] = {
            rec["conversation_id"]: rec for rec in self._load_disk_records()
        }
        cid_re = re.compile(
            r"^(.+)_(F|L)_(hi_en|ta_en|te_en)_(h\d+)_(cooperative|clarifying|resistant|suspicious)$"
        )
        budget_hit = False
        conv_dir = self.run.run_dir / "conversations"
        for idx, (family, case_kind, pair, history_id) in enumerate(units, start=1):
            moved_map: dict[str, Path] = {}
            for rec in list(by_id.values()):
                match = cid_re.match(rec["conversation_id"])
                if not match:
                    continue
                fam, letter, rec_pair, rec_hist, _ = match.groups()
                rec_kind = "fraud" if letter == "F" else "legitimate"
                if (fam, rec_kind, rec_pair, rec_hist) != (family, case_kind, pair, history_id):
                    continue
                dest = self._move_to_superseded(rec)
                moved_map[rec["conversation_id"]] = dest
                moved += 1
            regen = 1
            saved_seed = self.run.seed
            self.run.seed = saved_seed + regen
            print(
                f"regen unit {idx}/{len(units)} {family} {case_kind} {pair} {history_id} "
                f"spent=${self.run.ledger.spent:.4f}",
                flush=True,
            )
            try:
                recs = self._run_unit(family, case_kind, pair, history_id)
            except BudgetExceeded:
                for cid, dest in moved_map.items():
                    src = conv_dir / f"{cid}.json"
                    if not src.exists() and dest.exists():
                        shutil.copy2(dest, src)
                budget_hit = True
                print(f"BUDGET during regen after {generated} new conversations", flush=True)
                break
            finally:
                self.run.seed = saved_seed
            for rec in recs:
                gs = rec["provenance"]["generation_settings"]
                gs["regen"] = regen
                gs["seed"] = saved_seed + regen
                self._persist_conversation(rec)
                generated += 1
            if budget_hit:
                break
        if not self.run.sync_review:
            try:
                records = self._load_disk_records()
                self._batch_review_and_repair(records)
            except BudgetExceeded:
                budget_hit = True
                print("BUDGET during prefix-broken review/repair", flush=True)
        records = self._load_disk_records()
        self._write_exports(records)
        self._write_final_data_report(records, prefix_units=len(units), moved=moved, generated=generated)
        return {
            "run_dir": str(self.run.run_dir),
            "units": len(units),
            "moved": moved,
            "generated": generated,
            "sync_review": self.run.sync_review,
            "budget_hit": budget_hit,
        }

    def _write_final_data_report(
        self,
        records: list[dict[str, Any]],
        *,
        prefix_units: int = 0,
        moved: int = 0,
        generated: int = 0,
    ) -> None:
        old_complete = self.run.run_dir / "complete_sets.json"
        if old_complete.exists() and not (self.run.run_dir / "complete_sets_semrev03.json").exists():
            shutil.copy2(old_complete, self.run.run_dir / "complete_sets_semrev03.json")
        complete = write_sibling_completeness(self.run.run_dir, records)
        breakdowns = disposition_breakdowns(records)
        reason_tally: dict[str, int] = {}
        for rec in records:
            for reason in rec["provenance"]["generation_settings"].get("rejection_reason") or []:
                reason_tally[reason] = reason_tally.get(reason, 0) + 1
        cost_by: dict[str, dict[str, float]] = {}
        ledger_path = self.run.run_dir / "ledger.jsonl"
        total = 0.0
        if ledger_path.exists():
            for line in ledger_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                usd = float(row.get("usd") or 0)
                total += usd
                provider = str(row.get("provider") or "unknown")
                role = str(row.get("role") or "unknown")
                cost_by.setdefault(provider, {})
                cost_by[provider][role] = cost_by[provider].get(role, 0.0) + usd
        snapshots = [
            p.name
            for p in sorted(self.run.run_dir.glob("dispositions*.json"))
            + sorted(self.run.run_dir.glob("complete_sets*.json"))
        ]
        lines = [
            "# Final data report",
            "",
            f"- n: **{len(records)}**",
            f"- language_gate: `{self.run.language_gate}`",
            f"- spend to date: **${total:.4f}**",
            f"- prefix-broken units regenerated: {prefix_units} (moved {moved}, wrote {generated})",
            "",
            "## Dispositions",
            "",
            f"| status | n |",
            "|---|---:|",
            f"| accepted | {sum(1 for r in records if r['provenance']['generation_settings'].get('disposition')=='accepted')} |",
            f"| repaired | {sum(1 for r in records if r['provenance']['generation_settings'].get('disposition')=='repaired')} |",
            f"| rejected | {sum(1 for r in records if r['provenance']['generation_settings'].get('disposition')=='rejected')} |",
            "",
        ]
        for dim in ("case", "condition", "pair", "split"):
            lines += [f"## By {dim}", "", f"| {dim} | n | accepted | repaired | rejected |", "|---|---:|---:|---:|---:|"]
            for key, row in sorted((breakdowns.get(dim) or {}).items()):
                lines.append(
                    f"| {key} | {row['n']} | {row['accepted']} | {row['repaired']} | {row['rejected']} |"
                )
            lines.append("")
        sc = complete.get("split_case") or {}
        lines += [
            "## Complete 4-sibling sets",
            "",
            f"- **{complete.get('n_complete', 0)}** / {complete.get('n_groups', 0)} groups",
            "",
            "| split | fraud | legitimate | total |",
            "|---|---:|---:|---:|",
        ]
        for split in sorted(sc):
            row = sc[split]
            lines.append(
                f"| {split} | {row.get('fraud', 0)} | {row.get('legitimate', 0)} | "
                f"{row.get('fraud', 0) + row.get('legitimate', 0)} |"
            )
        pair_case: dict[str, dict[str, int]] = {}
        for row in complete.get("sets") or []:
            # sets may not have split; recount from records
            pass
        # split x case x pair from complete rows if we recompute
        complete_rows = compute_complete_sets(records)
        sxcxp: dict[tuple[str, str, str], int] = {}
        for row in complete_rows:
            key = (row.get("split") or "", row["case"], row["pair"])
            sxcxp[key] = sxcxp.get(key, 0) + 1
        lines += [
            "",
            "## Complete sets split × case × pair",
            "",
            "| split | case | pair | n |",
            "|---|---|---|---:|",
        ]
        for key in sorted(sxcxp):
            lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {sxcxp[key]} |")
        lines += ["", "## Rejection reasons", "", "| reason | n |", "|---|---:|"]
        for reason, n in sorted(reason_tally.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"| {reason} | {n} |")
        lines += ["", "## Cost by provider / role", "", "| provider | role | usd |", "|---|---|---:|"]
        for provider in sorted(cost_by):
            for role, usd in sorted(cost_by[provider].items()):
                lines.append(f"| {provider} | {role} | {usd:.4f} |")
        lines += ["", "## Ablation snapshots", ""]
        for name in snapshots:
            lines.append(f"- `{name}`")
        path = self.run.run_dir / "FINAL_DATA_REPORT.md"
        tmp = path.with_suffix(".tmp")
        tmp.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        tmp.replace(path)

    def freeze_run(self) -> dict[str, Any]:
        records = self._load_disk_records()
        self._write_exports(records)
        usable = []
        hashes: dict[str, str] = {}
        conv_dir = self.run.run_dir / "conversations"
        for rec in records:
            disp = rec["provenance"]["generation_settings"].get("disposition")
            if disp not in {"accepted", "repaired"}:
                continue
            usable.append(rec["conversation_id"])
            path = conv_dir / f"{rec['conversation_id']}.json"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hashes[rec["conversation_id"]] = digest
        export_path = self.run.run_dir / "detector_export.jsonl"
        export_sha = hashlib.sha256(export_path.read_bytes()).hexdigest() if export_path.exists() else None
        from scambranch.render import PROFILES_DIR, PROMPTS_DIR, prompt_version

        prompts = {}
        for path in sorted(PROMPTS_DIR.glob("*.md")):
            if path.name == "README.md":
                continue
            prompts[path.stem] = prompt_version(path.read_text(encoding="utf-8"))
        profiles = {}
        for path in sorted(PROFILES_DIR.glob("*.yaml")):
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            profiles[path.stem] = str((data or {}).get("version") or "unknown")
        scenarios = {}
        for path in sorted(SCENARIOS_DIR.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            scenarios[path.stem] = str(data.get("version") or "unknown")
        payload = {
            "frozen_at": _now_iso(),
            "n_usable": len(usable),
            "n_conversations": len(records),
            "language_gate": self.run.language_gate,
            "detector_export_sha256": export_sha,
            "conversations": hashes,
            "versions": {
                "prompts": prompts,
                "language_profiles": profiles,
                "scenarios": scenarios,
            },
        }
        atomic_write_json(self.run.run_dir / "FROZEN.json", payload)
        print(f"FROZEN n_usable={len(usable)} export={export_sha}")
        return {"run_dir": str(self.run.run_dir), "n_usable": len(usable)}

    def _persist_conversation(self, rec: dict[str, Any]) -> None:
        self.run.attach_fallbacks(rec)
        errs = schema_errors(rec, "conversation")
        rec["provenance"]["generation_settings"]["schema_errors"] = errs
        path = self.run.run_dir / "conversations" / f"{rec['conversation_id']}.json"
        atomic_write_json(path, rec)

    def _write_exports(self, records: list[dict[str, Any]]) -> None:
        detector_rows = []
        private_rows = []
        for rec in records:
            disp = rec["provenance"]["generation_settings"].get("disposition")
            private_rows.append(
                {
                    "conversation_id": rec["conversation_id"],
                    "scenario_family_id": rec["scenario_family_id"],
                    "case_id": rec["case_id"],
                    "history_id": rec["history_id"],
                    "branch_id": rec["branch_id"],
                    "language_pair": rec["language_pair"],
                    "split": rec["split"],
                    "private_annotations": rec["private_annotations"],
                    "review_status": rec["provenance"]["review_status"],
                    "disposition": disp,
                }
            )
            if disp in {"accepted", "repaired"}:
                detector_rows.append(to_detector_export(rec))
        atomic_write_jsonl(self.run.run_dir / "detector_export.jsonl", detector_rows)
        atomic_write_jsonl(self.run.run_dir / "private_manifest.jsonl", private_rows)
        reason_tally: dict[str, int] = {}
        for rec in records:
            for reason in rec["provenance"]["generation_settings"].get("rejection_reason") or []:
                reason_tally[reason] = reason_tally.get(reason, 0) + 1
        summary = {
            "n": len(records),
            "accepted": sum(
                1
                for r in records
                if r["provenance"]["generation_settings"].get("disposition") == "accepted"
            ),
            "repaired": sum(
                1
                for r in records
                if r["provenance"]["generation_settings"].get("disposition") == "repaired"
            ),
            "rejected": sum(
                1
                for r in records
                if r["provenance"]["generation_settings"].get("disposition") == "rejected"
            ),
            "pending": sum(
                1
                for r in records
                if r["provenance"]["generation_settings"].get("disposition") == "pending"
            ),
            "spent_usd": self.run.ledger.spent,
            "language_gate": self.run.language_gate,
            "rejection_reason": reason_tally,
            "disposition": disposition_breakdowns(records),
        }
        atomic_write_json(self.run.run_dir / "summary.json", summary)
        write_disposition_table(self.run.run_dir, records)
        if len(records) <= 200:
            write_readable_dump(self.run.run_dir, records)


def _case_kind_of(rec: dict[str, Any]) -> str:
    gs = rec.get("provenance", {}).get("generation_settings", {})
    if gs.get("case_kind"):
        return str(gs["case_kind"])
    case_id = str(rec.get("case_id") or "")
    if case_id.endswith("_L"):
        return "legitimate"
    return "fraud"


def disposition_breakdowns(records: list[dict[str, Any]]) -> dict[str, Any]:
    dims = ("pair", "case", "split", "condition")

    def dim_value(rec: dict[str, Any], dim: str) -> str:
        if dim == "pair":
            return str(rec.get("language_pair") or "")
        if dim == "case":
            return _case_kind_of(rec)
        if dim == "split":
            return str(rec.get("split") or "")
        return str(rec.get("private_annotations", {}).get("recipient_condition") or rec.get("branch_id") or "")

    out: dict[str, Any] = {}
    for dim in dims:
        bucket: dict[str, dict[str, int]] = {}
        for rec in records:
            key = dim_value(rec, dim)
            row = bucket.setdefault(
                key, {"n": 0, "accepted": 0, "repaired": 0, "rejected": 0, "pending": 0}
            )
            disp = rec["provenance"]["generation_settings"].get("disposition") or "pending"
            row["n"] += 1
            if disp in row:
                row[disp] += 1
        out[dim] = bucket
    return out


def write_disposition_table(run_dir: Path, records: list[dict[str, Any]]) -> None:
    breakdowns = disposition_breakdowns(records)
    lines = [
        "# Disposition table",
        "",
        f"n={len(records)}",
        "",
    ]
    for dim in ("pair", "case", "split", "condition"):
        lines.append(f"## By {dim}")
        lines.append("")
        lines.append("| value | n | accepted | repaired | rejected | pending |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for key in sorted(breakdowns[dim]):
            row = breakdowns[dim][key]
            lines.append(
                f"| {key} | {row['n']} | {row['accepted']} | {row['repaired']} | "
                f"{row['rejected']} | {row['pending']} |"
            )
        lines.append("")
    path = run_dir / "disposition_table.md"
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    tmp.replace(path)


def compute_complete_sets(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for rec in records:
        key = (
            rec["scenario_family_id"],
            _case_kind_of(rec),
            rec["language_pair"],
            rec["history_id"],
        )
        groups.setdefault(key, []).append(rec)
    complete: list[dict[str, Any]] = []
    for (family, case, pair, history), sibs in groups.items():
        by_cond = {
            s["private_annotations"]["recipient_condition"]: s for s in sibs
        }
        if set(CONDITION_NAMES) - set(by_cond):
            continue
        if not all(
            by_cond[cond]["provenance"]["generation_settings"].get("disposition")
            in {"accepted", "repaired"}
            for cond in CONDITION_NAMES
        ):
            continue
        complete.append(
            {
                "family": family,
                "case": case,
                "pair": pair,
                "history": history,
                "split": by_cond["cooperative"].get("split"),
            }
        )
    complete.sort(key=lambda row: (row["family"], row["case"], row["pair"], row["history"]))
    return complete


def write_sibling_completeness(
    run_dir: Path,
    records: list[dict[str, Any]],
    *,
    json_name: str = "complete_sets.json",
    md_name: str = "sibling_completeness.md",
) -> dict[str, Any]:
    complete = compute_complete_sets(records)
    n_groups = len(
        {
            (
                rec["scenario_family_id"],
                _case_kind_of(rec),
                rec["language_pair"],
                rec["history_id"],
            )
            for rec in records
        }
    )
    by_split: dict[str, int] = {}
    by_case: dict[str, int] = {}
    by_pair: dict[str, int] = {}
    by_family: dict[str, int] = {}
    split_case: dict[str, dict[str, int]] = {}
    for row in complete:
        by_split[row["split"] or ""] = by_split.get(row["split"] or "", 0) + 1
        by_case[row["case"]] = by_case.get(row["case"], 0) + 1
        by_pair[row["pair"]] = by_pair.get(row["pair"], 0) + 1
        by_family[row["family"]] = by_family.get(row["family"], 0) + 1
        sc = split_case.setdefault(row["split"] or "", {"fraud": 0, "legitimate": 0})
        sc[row["case"]] = sc.get(row["case"], 0) + 1
    payload = {
        "n_complete": len(complete),
        "n_groups": n_groups,
        "by_split": by_split,
        "by_case": by_case,
        "by_pair": by_pair,
        "by_family": by_family,
        "split_case": split_case,
        "sets": [
            {k: row[k] for k in ("family", "case", "pair", "history")} for row in complete
        ],
    }
    atomic_write_json(run_dir / json_name, payload)
    lines = [
        "# Sibling completeness",
        "",
        "A complete set is a (family, case, pair, history) with all 4 conditions present and usable (`accepted` or `repaired`).",
        "",
        f"- groups (family × case × pair × history): **{n_groups}**",
        f"- complete 4-sibling sets: **{len(complete)}**",
        "",
        "## By split",
        "",
        "| split | complete sets |",
        "|---|---:|",
    ]
    for key in sorted(by_split):
        lines.append(f"| {key} | {by_split[key]} |")
    lines += ["", "## By case", "", "| case | complete sets |", "|---|---:|"]
    for key in sorted(by_case):
        lines.append(f"| {key} | {by_case[key]} |")
    lines += ["", "## By pair", "", "| pair | complete sets |", "|---|---:|"]
    for key in sorted(by_pair):
        lines.append(f"| {key} | {by_pair[key]} |")
    lines += [
        "",
        "## Split × case",
        "",
        "| split | fraud | legitimate | total |",
        "|---|---:|---:|---:|",
    ]
    for split in sorted(split_case):
        row = split_case[split]
        lines.append(
            f"| {split} | {row.get('fraud', 0)} | {row.get('legitimate', 0)} | "
            f"{row.get('fraud', 0) + row.get('legitimate', 0)} |"
        )
    lines += ["", "## By family", "", "| family | complete sets |", "|---|---:|"]
    for key in sorted(by_family):
        lines.append(f"| {key} | {by_family[key]} |")
    path = run_dir / md_name
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    tmp.replace(path)
    return payload


def language_review_ledger_cids(run_dir: Path) -> set[str]:
    path = run_dir / "ledger.jsonl"
    cids: set[str] = set()
    if not path.exists():
        return cids
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("role") == "language_review" and row.get("conversation_id"):
            cids.add(str(row["conversation_id"]))
    return cids


def write_readable_dump(
    run_dir: Path, records: list[dict[str, Any]], filename: str = "READABLE.md"
) -> None:
    lines = ["# Readable conversations", ""]
    for rec in sorted(records, key=lambda r: r["conversation_id"]):
        gs = rec["provenance"]["generation_settings"]
        lines.append(f"## {rec['conversation_id']}")
        lines.append(f"- label: {rec['private_annotations']['scenario_label']}")
        lines.append(f"- condition: {rec['private_annotations']['recipient_condition']}")
        lines.append(f"- disposition: {gs.get('disposition')}")
        lines.append(f"- review_status: {rec['provenance']['review_status']}")
        if gs.get("rejection_reason"):
            lines.append(f"- rejection_reason: {', '.join(gs['rejection_reason'])}")
        lines.append("")
        for turn in rec["turns"]:
            lines.append(f"{turn['turn_id']} {turn['speaker']}: {turn['text']}")
        lines.append("")
    path = run_dir / filename
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    tmp.replace(path)


def run_generation(
    *,
    run_id: str,
    families: list[str] | None = None,
    pairs: list[str] | None = None,
    conditions: list[str] | None = None,
    dry_run: bool = False,
    sync_review: bool = False,
    budget: float | None = None,
    workers: int | None = None,
    run_root: Path | None = None,
    config_path: Path | None = None,
    history_ids: list[str] | None = None,
    language_review_only: bool = False,
    semrev03: bool = False,
    recompute_dispositions: bool = False,
    regen_prefix_broken: bool = False,
    freeze: bool = False,
) -> dict[str, Any]:
    config = load_generation_config(config_path)
    fams, splits = families_and_splits(config, families)
    use_pairs = list(pairs or config.get("pairs") or ["hi_en"])
    use_conds = list(conditions or config.get("conditions") or list(CONDITION_NAMES))
    run = GenerationRun(
        run_id=run_id,
        config=config,
        families=fams,
        pairs=use_pairs,
        conditions=use_conds,
        dry_run=dry_run,
        sync_review=sync_review,
        budget=budget,
        workers=workers,
        run_root=run_root,
        history_ids=history_ids,
    )
    pipe = Pipeline(run, splits)
    if recompute_dispositions:
        return pipe.recompute_dispositions()
    if regen_prefix_broken:
        return pipe.regenerate_prefix_broken()
    if freeze:
        return pipe.freeze_run()
    if semrev03:
        return pipe.rerun_semrev03()
    if language_review_only:
        return pipe.rerun_language_reviews()
    return pipe.execute()
