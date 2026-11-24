"""LLM providers, usage accounting, and a hard budget cap."""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from jsonschema import Draft202012Validator

from scambranch.output_schemas import ROLE_SCHEMAS

JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class BudgetExceeded(RuntimeError):
    """Raised before a call or batch that would exceed budget_usd_cap."""


class RefusalError(ValueError):
    """Model refused to generate the requested content."""


class IncompleteOutputError(ValueError):
    """Empty or truncated model output (often max_output_tokens eaten by reasoning)."""


ANTHROPIC_SCHEMA_DROP = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
    }
)


def sanitize_schema_for_anthropic(schema: Any) -> Any:
    """Drop JSON-Schema keywords Anthropic structured output rejects (e.g. integer min/max)."""
    if isinstance(schema, dict):
        return {
            key: sanitize_schema_for_anthropic(val)
            for key, val in schema.items()
            if key not in ANTHROPIC_SCHEMA_DROP
        }
    if isinstance(schema, list):
        return [sanitize_schema_for_anthropic(val) for val in schema]
    return schema


@dataclass
class Usage:
    input_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def input_uncached(self) -> int:
        return max(int(self.input_tokens) - int(self.cached_tokens), 0)


@dataclass
class LLMResult:
    parsed: dict[str, Any]
    raw_text: str
    usage: Usage
    request_id: str
    latency_s: float
    model: str
    provider: str
    batch: bool = False
    missing: bool = False
    error: str | None = None


@dataclass
class LLMRequest:
    role: str
    system: str
    user: str
    schema: dict[str, Any]
    model: str
    provider: str
    max_output_tokens: int
    reasoning_effort: str | None = None
    temperature: float | None = None
    conversation_id: str = ""
    mock_context: dict[str, Any] = field(default_factory=dict)
    cache_system: bool = True


def estimate_input_tokens(text: str) -> int:
    return max(1, int(len(text) / 3.5)) if text else 1


def price_table(model: str, prices: dict[str, Any], *, batch: bool) -> dict[str, float]:
    row = prices[model]
    if "batch" in row and isinstance(row["batch"], dict) and batch:
        return {
            "input": float(row["batch"]["input"]),
            "cached": float(row["batch"].get("cache_read", row.get("cache_read", 0.0))),
            "cache_write": float(row["batch"].get("cache_write", row.get("cache_write", 0.0))),
            "output": float(row["batch"]["output"]),
        }
    mult = float(row.get("batch_multiplier", 1.0)) if batch else 1.0
    return {
        "input": float(row["input"]) * mult,
        "cached": float(row.get("cached", row.get("cache_read", 0.0))) * mult,
        "cache_write": float(row.get("cache_write", 0.0)) * mult,
        "output": float(row["output"]) * mult,
    }


def usage_usd(model: str, prices: dict[str, Any], usage: Usage, *, batch: bool) -> float:
    p = price_table(model, prices, batch=batch)
    # Reasoning tokens are billed as output; output_tokens already includes them
    # when the provider reports inclusive totals. We never add them twice.
    return (
        usage.input_uncached * p["input"]
        + usage.cached_tokens * p["cached"]
        + usage.cache_write_tokens * p["cache_write"]
        + usage.output_tokens * p["output"]
    ) / 1_000_000.0


def estimate_call_usd(
    *,
    model: str,
    prices: dict[str, Any],
    system: str,
    user: str,
    max_output_tokens: int,
    batch: bool = False,
) -> float:
    usage = Usage(
        input_tokens=estimate_input_tokens((system or "") + (user or "")),
        output_tokens=int(max_output_tokens),
    )
    return usage_usd(model, prices, usage, batch=batch)


class CostLedger:
    def __init__(
        self,
        path: Path,
        *,
        budget_usd_cap: float,
        prices: dict[str, Any],
    ) -> None:
        self.path = path
        self.budget_usd_cap = float(budget_usd_cap)
        self.prices = prices
        self._lock = threading.Lock()
        self.spent = 0.0
        self.reserved = 0.0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                self.spent += float(row.get("usd") or 0.0)

    def _would_exceed(self, extra: float) -> bool:
        return (self.spent + self.reserved + extra) > self.budget_usd_cap + 1e-12

    def reserve(self, estimate: float, *, what: str) -> float:
        with self._lock:
            if self._would_exceed(estimate):
                raise BudgetExceeded(
                    f"budget cap ${self.budget_usd_cap:.4f} would be exceeded by "
                    f"{what} (spent=${self.spent:.4f} reserved=${self.reserved:.4f} "
                    f"estimate=${estimate:.4f})"
                )
            self.reserved += estimate
            return estimate

    def release(self, estimate: float) -> None:
        with self._lock:
            self.reserved = max(0.0, self.reserved - estimate)

    def record(
        self,
        *,
        role: str,
        provider: str,
        model: str,
        usage: Usage,
        usd: float,
        latency_s: float,
        request_id: str,
        batch: bool,
        conversation_id: str = "",
        reserved: float = 0.0,
    ) -> dict[str, Any]:
        row = {
            "role": role,
            "provider": provider,
            "model": model,
            "input_tokens": usage.input_tokens,
            "cached_tokens": usage.cached_tokens,
            "cache_write_tokens": usage.cache_write_tokens,
            "output_tokens": usage.output_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
            "usd": round(usd, 8),
            "latency_s": round(latency_s, 4),
            "request_id": request_id,
            "batch": batch,
            "conversation_id": conversation_id,
        }
        line = json.dumps(row, ensure_ascii=False)
        with self._lock:
            self.reserved = max(0.0, self.reserved - reserved)
            self.spent += usd
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        return row


class Provider(Protocol):
    def complete(self, request: LLMRequest) -> LLMResult: ...


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = JSON_FENCE.sub("", cleaned).strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        obj = json.loads(cleaned[start : end + 1])
        if isinstance(obj, dict):
            return obj
    raise json.JSONDecodeError("no JSON object", cleaned, 0)


def validate_against(schema: dict[str, Any], obj: dict[str, Any]) -> None:
    errors = list(Draft202012Validator(schema).iter_errors(obj))
    if errors:
        raise ValueError("; ".join(e.message for e in errors[:4]))


def _is_retryable(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status in {429, 500, 502, 503, 504}:
        return True
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return "rate" in name or "timeout" in name or "429" in text or "5" in text[:8]


def _backoff_sleep(attempt: int) -> None:
    time.sleep(min(2 ** attempt, 16))


class _BaseProvider:
    def __init__(
        self,
        *,
        ledger: CostLedger,
        prices: dict[str, Any],
        raw_dir: Path,
        prompt_log: list[dict[str, Any]] | None = None,
    ) -> None:
        self.ledger = ledger
        self.prices = prices
        self.raw_dir = raw_dir
        self.prompt_log = prompt_log if prompt_log is not None else []
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def _log_prompt(self, request: LLMRequest) -> None:
        entry = {
            "role": request.role,
            "conversation_id": request.conversation_id,
            "system": request.system,
            "user": request.user,
            "model": request.model,
            "provider": request.provider,
        }
        self.prompt_log.append(entry)
        dest = self.raw_dir.parent / "prompts"
        dest.mkdir(parents=True, exist_ok=True)
        name = f"{len(self.prompt_log):05d}_{request.role}_{request.conversation_id or 'na'}.json"
        path = dest / name.replace(":", "_")
        path.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_raw(self, request_id: str, payload: dict[str, Any]) -> None:
        path = self.raw_dir / f"{request_id}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _reserve(self, request: LLMRequest, *, batch: bool = False) -> float:
        est = estimate_call_usd(
            model=request.model,
            prices=self.prices,
            system=request.system,
            user=request.user,
            max_output_tokens=request.max_output_tokens,
            batch=batch,
        )
        return self.ledger.reserve(est, what=f"{request.role}/{request.model}")


class OpenAIProvider(_BaseProvider):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        from openai import OpenAI

        self.client = OpenAI()

    def _text_format(self, request: LLMRequest) -> dict[str, Any]:
        return {
            "format": {
                "type": "json_schema",
                "name": re.sub(r"[^A-Za-z0-9_]", "_", request.role)[:64] or "out",
                "strict": True,
                "schema": request.schema,
            }
        }

    def _extract_text(self, resp: Any) -> str:
        for item in getattr(resp, "output", None) or []:
            if getattr(item, "type", None) == "refusal":
                raise RefusalError(str(getattr(item, "refusal", None) or "refusal"))
            for content in getattr(item, "content", None) or []:
                ctype = getattr(content, "type", None)
                refusal = getattr(content, "refusal", None)
                if ctype == "refusal" or refusal:
                    raise RefusalError(str(refusal or "refusal"))
        text = getattr(resp, "output_text", None)
        if text:
            return str(text)
        parts: list[str] = []
        for item in getattr(resp, "output", None) or []:
            for content in getattr(item, "content", None) or []:
                if getattr(content, "text", None):
                    parts.append(str(content.text))
        return "".join(parts)

    def _resp_meta(self, resp: Any) -> dict[str, Any]:
        details = getattr(resp, "incomplete_details", None)
        if details is not None and hasattr(details, "model_dump"):
            details = details.model_dump()
        elif details is not None:
            details = {"reason": getattr(details, "reason", None), "repr": str(details)}
        return {
            "status": getattr(resp, "status", None),
            "incomplete_details": details,
        }

    def _usage(self, resp: Any) -> Usage:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return Usage()
        details_in = getattr(usage, "input_tokens_details", None)
        details_out = getattr(usage, "output_tokens_details", None)
        cached = 0
        reasoning = 0
        if details_in is not None:
            cached = int(getattr(details_in, "cached_tokens", 0) or 0)
        if details_out is not None:
            reasoning = int(getattr(details_out, "reasoning_tokens", 0) or 0)
        return Usage(
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            cached_tokens=cached,
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            reasoning_tokens=reasoning,
        )

    def _create(self, request: LLMRequest) -> Any:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "input": [
                {"role": "system", "content": request.system or "Follow the user."},
                {"role": "user", "content": request.user},
            ],
            "max_output_tokens": request.max_output_tokens,
            "text": self._text_format(request),
        }
        if request.reasoning_effort:
            kwargs["reasoning"] = {"effort": request.reasoning_effort}
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        try:
            return self.client.responses.create(**kwargs)
        except Exception as exc:
            # Fall back to JSON mode if structured schema is rejected.
            if "json_schema" in str(exc).lower() or "text.format" in str(exc).lower():
                kwargs["text"] = {"format": {"type": "json_object"}}
                return self.client.responses.create(**kwargs)
            if request.temperature is not None and "temperature" in str(exc).lower():
                kwargs.pop("temperature", None)
                return self.client.responses.create(**kwargs)
            raise

    def complete(self, request: LLMRequest) -> LLMResult:
        self._log_prompt(request)
        reserved = self._reserve(request)
        last_err: Exception | None = None
        parse_tries = 0
        started = time.time()
        try:
            while parse_tries < 3:
                api_attempt = 0
                resp = None
                while True:
                    try:
                        resp = self._create(request)
                        break
                    except RefusalError:
                        raise
                    except Exception as exc:
                        last_err = exc
                        if _is_retryable(exc) and api_attempt < 5:
                            _backoff_sleep(api_attempt)
                            api_attempt += 1
                            continue
                        raise
                assert resp is not None
                request_id = str(getattr(resp, "id", None) or uuid.uuid4())
                usage = self._usage(resp)
                usd = usage_usd(request.model, self.prices, usage, batch=False)
                meta = self._resp_meta(resp)
                try:
                    raw = self._extract_text(resp)
                except RefusalError as exc:
                    self._write_raw(
                        request_id,
                        {
                            "role": request.role,
                            "raw": "",
                            "parsed": None,
                            "error": str(exc),
                            **meta,
                        },
                    )
                    self.ledger.record(
                        role=request.role,
                        provider="openai",
                        model=request.model,
                        usage=usage,
                        usd=usd,
                        latency_s=time.time() - started,
                        request_id=request_id,
                        batch=False,
                        conversation_id=request.conversation_id,
                        reserved=reserved,
                    )
                    reserved = 0.0
                    raise
                status = meta.get("status")
                incomplete = status == "incomplete" or bool(meta.get("incomplete_details"))
                try:
                    if not (raw or "").strip():
                        raise IncompleteOutputError(
                            f"empty output status={status} incomplete={meta.get('incomplete_details')}"
                        )
                    parsed = parse_json_object(raw)
                    validate_against(request.schema, parsed)
                    latency = time.time() - started
                    self._write_raw(
                        request_id,
                        {"role": request.role, "raw": raw, "parsed": parsed, **meta},
                    )
                    self.ledger.record(
                        role=request.role,
                        provider="openai",
                        model=request.model,
                        usage=usage,
                        usd=usd,
                        latency_s=latency,
                        request_id=request_id,
                        batch=False,
                        conversation_id=request.conversation_id,
                        reserved=reserved,
                    )
                    return LLMResult(
                        parsed=parsed,
                        raw_text=raw,
                        usage=usage,
                        request_id=request_id,
                        latency_s=latency,
                        model=request.model,
                        provider="openai",
                    )
                except (json.JSONDecodeError, ValueError) as exc:
                    self._write_raw(
                        request_id,
                        {
                            "role": request.role,
                            "raw": raw,
                            "parsed": None,
                            "error": str(exc),
                            **meta,
                        },
                    )
                    self.ledger.record(
                        role=request.role,
                        provider="openai",
                        model=request.model,
                        usage=usage,
                        usd=usd,
                        latency_s=time.time() - started,
                        request_id=request_id,
                        batch=False,
                        conversation_id=request.conversation_id,
                        reserved=0.0,
                    )
                    if incomplete or not (raw or "").strip():
                        last_err = IncompleteOutputError(
                            f"{exc} status={status} incomplete={meta.get('incomplete_details')}"
                        )
                    else:
                        last_err = exc
                    parse_tries += 1
            raise last_err or ValueError("JSON parse failed")
        except Exception:
            self.ledger.release(reserved)
            raise


class AnthropicProvider(_BaseProvider):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        from anthropic import Anthropic

        self.client = Anthropic()

    def _system_blocks(self, request: LLMRequest) -> list[dict[str, Any]] | str:
        text = request.system or ""
        if not text:
            return ""
        block: dict[str, Any] = {"type": "text", "text": text}
        if request.cache_system:
            block["cache_control"] = {"type": "ephemeral"}
        return [block]

    def _usage(self, resp: Any) -> Usage:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return Usage()
        return Usage(
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            cached_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            cache_write_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        )

    def _extract_text(self, resp: Any) -> str:
        parts: list[str] = []
        for block in getattr(resp, "content", None) or []:
            if getattr(block, "text", None):
                parts.append(str(block.text))
        return "".join(parts)

    def _create(self, request: LLMRequest) -> Any:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_output_tokens,
            "messages": [{"role": "user", "content": request.user}],
        }
        system = self._system_blocks(request)
        if system:
            kwargs["system"] = system
        kwargs["output_config"] = {
            "format": {
                "type": "json_schema",
                "schema": sanitize_schema_for_anthropic(request.schema),
            },
        }
        try:
            return self.client.messages.create(**kwargs)
        except TypeError:
            kwargs.pop("output_config", None)
            return self.client.messages.create(**kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            if "output_config" in msg or "json_schema" in msg:
                kwargs.pop("output_config", None)
                return self.client.messages.create(**kwargs)
            raise

    def complete(self, request: LLMRequest) -> LLMResult:
        self._log_prompt(request)
        reserved = self._reserve(request)
        last_err: Exception | None = None
        parse_tries = 0
        started = time.time()
        try:
            while parse_tries < 3:
                api_attempt = 0
                resp = None
                while True:
                    try:
                        resp = self._create(request)
                        break
                    except Exception as exc:
                        last_err = exc
                        if _is_retryable(exc) and api_attempt < 5:
                            _backoff_sleep(api_attempt)
                            api_attempt += 1
                            continue
                        raise
                assert resp is not None
                raw = self._extract_text(resp)
                request_id = str(getattr(resp, "id", None) or uuid.uuid4())
                usage = self._usage(resp)
                try:
                    parsed = parse_json_object(raw)
                    validate_against(request.schema, parsed)
                    usd = usage_usd(request.model, self.prices, usage, batch=False)
                    latency = time.time() - started
                    self._write_raw(
                        request_id,
                        {"role": request.role, "raw": raw, "parsed": parsed},
                    )
                    self.ledger.record(
                        role=request.role,
                        provider="anthropic",
                        model=request.model,
                        usage=usage,
                        usd=usd,
                        latency_s=latency,
                        request_id=request_id,
                        batch=False,
                        conversation_id=request.conversation_id,
                        reserved=reserved,
                    )
                    return LLMResult(
                        parsed=parsed,
                        raw_text=raw,
                        usage=usage,
                        request_id=request_id,
                        latency_s=latency,
                        model=request.model,
                        provider="anthropic",
                    )
                except (json.JSONDecodeError, ValueError) as exc:
                    last_err = exc
                    parse_tries += 1
            raise last_err or ValueError("JSON parse failed")
        except Exception:
            self.ledger.release(reserved)
            raise


class MockProvider(_BaseProvider):
    """Schema-valid fake JSON for --dry-run and offline tests."""

    def complete(self, request: LLMRequest) -> LLMResult:
        self._log_prompt(request)
        reserved = self._reserve(request)
        started = time.time()
        parsed = self._fake(request)
        validate_against(request.schema, parsed)
        raw = json.dumps(parsed, ensure_ascii=False)
        usage = Usage(
            input_tokens=estimate_input_tokens(request.system + request.user),
            output_tokens=max(8, len(raw) // 4),
        )
        usd = 0.0
        request_id = f"mock-{uuid.uuid4().hex[:12]}"
        latency = time.time() - started
        self._write_raw(request_id, {"role": request.role, "raw": raw, "parsed": parsed})
        self.ledger.record(
            role=request.role,
            provider="mock",
            model=request.model,
            usage=usage,
            usd=usd,
            latency_s=latency,
            request_id=request_id,
            batch=False,
            conversation_id=request.conversation_id,
            reserved=reserved,
        )
        return LLMResult(
            parsed=parsed,
            raw_text=raw,
            usage=usage,
            request_id=request_id,
            latency_s=latency,
            model=request.model,
            provider="mock",
        )

    def _fake(self, request: LLMRequest) -> dict[str, Any]:
        ctx = request.mock_context or {}
        role = request.role
        if role == "plan":
            plan_turns = ctx.get("dialogue_plan") or []
            turns = []
            for row in plan_turns:
                turns.append(
                    {
                        "turn": int(row["turn"]),
                        "speaker": row["speaker"],
                        "act": row.get("act") or "speak",
                        "facts_used": [],
                        "evidence_ids": list(row.get("evidence_ids") or []),
                    }
                )
            if not turns:
                turns = [
                    {
                        "turn": 1,
                        "speaker": "caller",
                        "act": "greet",
                        "facts_used": [],
                        "evidence_ids": [],
                    }
                ]
            return {
                "case_id": ctx.get("case_id") or "bp_01_F",
                "branch_after": int(ctx.get("branch_after") or turns[-1]["turn"]),
                "turns": turns,
                "facts_committed": list(ctx.get("stable_facts") or ["name stated"]),
                "notes_for_continuation": "Keep the same request; do not change mechanism.",
            }
        if role == "prefix":
            plan = ctx.get("plan") or {}
            turns = []
            for row in plan.get("turns") or []:
                speaker = row["speaker"]
                tid = int(row["turn"])
                if speaker == "caller":
                    text = "Namaste, main bank team se call kar raha hoon, ek chhoti si verification ke liye."
                else:
                    text = "Haan, boliye na, main sun raha hoon."
                turns.append({"turn_id": tid, "speaker": speaker, "text": text})
            if not turns:
                turns = [{"turn_id": 1, "speaker": "caller", "text": "Namaste."}]
            return {"turns": turns}
        if role in {"caller", "recipient"}:
            n = int(ctx.get("post_branch_turn") or 1)
            condition = ctx.get("condition") or ""
            end = n >= int(ctx.get("end_after") or 2)
            if role == "recipient":
                if condition == "suspicious" and n == 1:
                    text = "Yeh call genuine nahi lag raha, maine aise scam calls ke baare mein suna hai."
                elif condition == "resistant":
                    text = "Main branch jaake khud handle karunga, abhi kuch share nahi karunga."
                elif condition == "clarifying":
                    text = "Ek minute, yeh step kyun chahiye, thoda explain kar dijiye."
                else:
                    text = "Haan theek hai, aap batao kaise karna hai, main kar deta hoon."
            else:
                text = "Samajh gaya, main aapki help ke liye yahin hoon, aap decide kar lijiye."
                if end:
                    text = "Theek hai, phir aap official number par confirm kar lijiye. Dhanyavaad."
            return {"text": text, "end_call": end}
        if role == "repair":
            targets = ctx.get("repair_targets") or []
            revised = []
            for t in targets:
                revised.append(
                    {
                        "turn_id": int(t.get("turn_id") or 0) or 1,
                        "speaker": t.get("speaker") or "recipient",
                        "text": "Theek hai, main official channel se hi aage badhunga.",
                    }
                )
            return {"revised_turns": revised, "unfixable": []}
        if role in {"language_review", "audit_language"}:
            return {
                "turn_flags": [],
                "switching_naturalness": 4,
                "grammatical_acceptability": 4,
                "overall": "accept",
            }
        # semantic_review / audit / default
        return {
            "checks": {
                "evidence_present": {"pass": True, "turns": [int(ctx.get("branch_after") or 7)], "note": ""},
                "evidence_not_invalidated": {"pass": True, "note": ""},
                "facts_consistent": {"pass": True, "note": ""},
                "condition_followed": {"pass": True, "note": ""},
                "caller_within_limits": {"pass": True, "note": ""},
                "no_leakage": {"pass": True, "note": ""},
                "plausible_ending": {"pass": True, "note": ""},
            },
            "earliest_evidence_turn": int(ctx.get("branch_after") or 7),
            "overall": "accept",
            "repair_targets": [],
        }


def make_provider(
    provider_name: str,
    *,
    dry_run: bool,
    ledger: CostLedger,
    prices: dict[str, Any],
    raw_dir: Path,
    prompt_log: list[dict[str, Any]] | None = None,
) -> Provider:
    kwargs = dict(ledger=ledger, prices=prices, raw_dir=raw_dir, prompt_log=prompt_log)
    if dry_run or provider_name == "mock":
        return MockProvider(**kwargs)
    if provider_name == "openai":
        return OpenAIProvider(**kwargs)
    if provider_name == "anthropic":
        return AnthropicProvider(**kwargs)
    raise ValueError(f"unknown provider {provider_name}")


def schema_for(role: str) -> dict[str, Any]:
    return ROLE_SCHEMAS[role]
