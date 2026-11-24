"""Resumable review batches: Anthropic Message Batches (default) and OpenAI Batch API."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from scambranch.llm import (
    CostLedger,
    LLMRequest,
    LLMResult,
    Usage,
    estimate_call_usd,
    parse_json_object,
    sanitize_schema_for_anthropic,
    usage_usd,
    validate_against,
)

# Anthropic custom_id: ^[a-zA-Z0-9_-]{1,64}$
REVIEW_SUFFIX = {
    "semantic_review": "sem",
    "language_review": "lang",
    "audit_semantic": "auds",
    "audit_language": "audl",
}


def make_custom_id(conversation_id: str, review_type: str) -> str:
    suffix = REVIEW_SUFFIX.get(review_type, review_type[:4])
    cid = f"{conversation_id}_{suffix}"
    cid = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in cid)
    if len(cid) > 64:
        cid = cid[:64]
    return cid


def parse_custom_id(custom_id: str) -> tuple[str, str]:
    for review_type, suffix in REVIEW_SUFFIX.items():
        token = f"_{suffix}"
        if custom_id.endswith(token):
            return custom_id[: -len(token)], review_type
    return custom_id, "semantic_review"


def _load_state(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"batches": []}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _request_to_anthropic_params(req: LLMRequest, *, batch_cache: bool) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": req.model,
        "max_tokens": req.max_output_tokens,
        "messages": [{"role": "user", "content": req.user}],
    }
    if req.system:
        block: dict[str, Any] = {"type": "text", "text": req.system}
        if req.cache_system:
            # Batches often exceed 5 minutes; 1h cache is recommended.
            block["cache_control"] = {"type": "ephemeral", "ttl": "1h"} if batch_cache else {
                "type": "ephemeral"
            }
        params["system"] = [block]
    if req.temperature is not None:
        params["temperature"] = req.temperature
    if req.schema:
        params["output_config"] = {
            "format": {
                "type": "json_schema",
                "schema": sanitize_schema_for_anthropic(req.schema),
            },
        }
    return params


def _request_to_openai_body(req: LLMRequest) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": req.model,
        "input": [
            {"role": "system", "content": req.system or "Follow the user."},
            {"role": "user", "content": req.user},
        ],
        "max_output_tokens": req.max_output_tokens,
        "text": {
            "format": {
                "type": "json_schema",
                "name": req.role.replace("-", "_")[:40] or "out",
                "strict": True,
                "schema": req.schema,
            }
        },
    }
    if req.reasoning_effort:
        body["reasoning"] = {"effort": req.reasoning_effort}
    return body


def _usage_from_anthropic_message(msg: Any) -> Usage:
    usage = msg.get("usage") if isinstance(msg, dict) else getattr(msg, "usage", None)
    if not usage:
        return Usage()
    if isinstance(usage, dict):
        return Usage(
            input_tokens=int(usage.get("input_tokens") or 0),
            cached_tokens=int(usage.get("cache_read_input_tokens") or 0),
            cache_write_tokens=int(usage.get("cache_creation_input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
        )
    return Usage(
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        cached_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
        cache_write_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
    )


def _text_from_anthropic_message(msg: Any) -> str:
    content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
    parts: list[str] = []
    for block in content or []:
        if isinstance(block, dict):
            if block.get("text"):
                parts.append(str(block["text"]))
        elif getattr(block, "text", None):
            parts.append(str(block.text))
    return "".join(parts)


def _usage_from_openai(resp: dict[str, Any]) -> Usage:
    usage = resp.get("usage") or {}
    details_in = usage.get("input_tokens_details") or {}
    details_out = usage.get("output_tokens_details") or {}
    return Usage(
        input_tokens=int(usage.get("input_tokens") or 0),
        cached_tokens=int(details_in.get("cached_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        reasoning_tokens=int(details_out.get("reasoning_tokens") or 0),
    )


def _text_from_openai(resp: dict[str, Any]) -> str:
    if resp.get("output_text"):
        return str(resp["output_text"])
    parts: list[str] = []
    for item in resp.get("output") or []:
        for content in item.get("content") or []:
            if content.get("text"):
                parts.append(str(content["text"]))
    return "".join(parts)


class ReviewBatchRunner:
    def __init__(
        self,
        *,
        run_dir: Path,
        ledger: CostLedger,
        prices: dict[str, Any],
        provider: str = "anthropic",
        dry_run: bool = False,
        purpose: str | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.ledger = ledger
        self.prices = prices
        self.provider = provider
        self.dry_run = dry_run
        self.purpose = purpose
        self.state_path = run_dir / "batches.json"
        self.state = _load_state(self.state_path)

    def submit_and_collect(self, jobs: list[LLMRequest]) -> dict[str, LLMResult]:
        if not jobs:
            return {}
        entry = self.submit(jobs)
        return self.collect(entry, jobs)

    def submit(self, jobs: list[LLMRequest]) -> dict[str, Any]:
        if not jobs:
            return {"batch_id": None, "status": "ended", "custom_ids": [], "conversation_ids": []}
        if self.dry_run:
            entry = {
                "batch_id": f"mock-{uuid.uuid4().hex[:12]}",
                "provider": "mock",
                "status": "in_progress",
                "custom_ids": [make_custom_id(j.conversation_id, j.role) for j in jobs],
                "conversation_ids": sorted({j.conversation_id for j in jobs}),
            }
            self.state.setdefault("batches", []).append(entry)
            _save_state(self.state_path, self.state)
            return entry
        if self.provider == "openai":
            return self._openai_submit(jobs)
        return self._anthropic_submit(jobs)

    def collect(self, entry: dict[str, Any], jobs: list[LLMRequest]) -> dict[str, LLMResult]:
        if not jobs:
            return {}
        if self.dry_run or entry.get("provider") == "mock":
            results = self._mock_results(jobs)
            entry["status"] = "ended"
            _save_state(self.state_path, self.state)
            return results
        if self.provider == "openai":
            return self._openai_collect(entry, jobs)
        return self._anthropic_collect(entry, jobs)

    def _estimate_and_reserve(self, jobs: list[LLMRequest]) -> float:
        total = 0.0
        for req in jobs:
            total += estimate_call_usd(
                model=req.model,
                prices=self.prices,
                system=req.system,
                user=req.user,
                max_output_tokens=req.max_output_tokens,
                batch=True,
            )
        return self.ledger.reserve(total, what=f"batch:{self.provider}:{len(jobs)}")

    def _mock_results(self, jobs: list[LLMRequest]) -> dict[str, LLMResult]:
        from scambranch.llm import MockProvider

        mock = MockProvider(
            ledger=self.ledger,
            prices=self.prices,
            raw_dir=self.run_dir / "raw",
        )
        out: dict[str, LLMResult] = {}
        for req in jobs:
            out[make_custom_id(req.conversation_id, req.role)] = mock.complete(req)
        return out

    def _resume_entry(self, custom_ids: list[str]) -> dict[str, Any] | None:
        wanted = set(custom_ids)
        found = None
        for entry in self.state.get("batches") or []:
            have = set(entry.get("custom_ids") or [])
            if have != wanted or entry.get("provider") != self.provider:
                continue
            if self.purpose and entry.get("purpose") != self.purpose:
                continue
            found = entry
        return found

    def _anthropic_submit(self, jobs: list[LLMRequest]) -> dict[str, Any]:
        from anthropic import Anthropic

        client = Anthropic()
        custom_ids = [make_custom_id(j.conversation_id, j.role) for j in jobs]
        existing = self._resume_entry(custom_ids)
        if existing and existing.get("batch_id"):
            if existing.get("status") != "ended" or self.purpose:
                print(
                    f"resuming Anthropic batch {existing['batch_id']} "
                    f"status={existing.get('status')} purpose={existing.get('purpose')}"
                )
                return existing
        reserved = self._estimate_and_reserve(jobs)
        requests = []
        for req, cid in zip(jobs, custom_ids, strict=True):
            requests.append(
                {
                    "custom_id": cid,
                    "params": _request_to_anthropic_params(req, batch_cache=True),
                }
            )
        try:
            created = client.messages.batches.create(requests=requests)
        except Exception:
            for row in requests:
                row["params"].pop("output_config", None)
            created = client.messages.batches.create(requests=requests)
        entry = {
            "batch_id": created.id,
            "provider": "anthropic",
            "status": getattr(created, "processing_status", None) or "in_progress",
            "custom_ids": custom_ids,
            "conversation_ids": sorted({j.conversation_id for j in jobs}),
            "reserved": reserved,
            "purpose": self.purpose,
        }
        self.state.setdefault("batches", []).append(entry)
        _save_state(self.state_path, self.state)
        print(f"created Anthropic batch {created.id} ({len(jobs)} requests)")
        return entry

    def _anthropic_collect(self, entry: dict[str, Any], jobs: list[LLMRequest]) -> dict[str, LLMResult]:
        from anthropic import Anthropic

        client = Anthropic()
        batch_id = entry["batch_id"]
        job_by_cid = {make_custom_id(j.conversation_id, j.role): j for j in jobs}
        reserved = float(entry.get("reserved") or 0.0)
        already_billed = entry.get("status") == "ended" and reserved <= 0
        batch = self._poll_anthropic(client, batch_id)
        results: dict[str, LLMResult] = {}
        try:
            for item in client.messages.batches.results(batch_id):
                cid = item.custom_id
                result = item.result
                rtype = getattr(result, "type", None) or (result.get("type") if isinstance(result, dict) else None)
                if rtype != "succeeded":
                    err_obj = getattr(result, "error", None)
                    if err_obj is not None and hasattr(err_obj, "model_dump"):
                        err_payload = err_obj.model_dump()
                    elif isinstance(err_obj, dict):
                        err_payload = err_obj
                    else:
                        err_payload = {"repr": str(err_obj or result)[:500]}
                    err_msg = ""
                    if isinstance(err_payload, dict):
                        inner = err_payload.get("error") or err_payload
                        if isinstance(inner, dict):
                            err_msg = str(inner.get("message") or inner.get("type") or err_payload)
                        else:
                            err_msg = str(inner)
                    print(f"batch item {cid} status={rtype} error={err_msg[:180]}")
                    raw_path = self.run_dir / "raw" / f"{batch_id}_{cid}.json"
                    raw_path.write_text(
                        json.dumps(
                            {
                                "custom_id": cid,
                                "batch_id": batch_id,
                                "status": rtype,
                                "error": err_payload,
                            },
                            ensure_ascii=False,
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    results[cid] = LLMResult(
                        parsed={},
                        raw_text="",
                        usage=Usage(),
                        request_id=f"{batch_id}:{cid}",
                        latency_s=0.0,
                        model=(job_by_cid.get(cid).model if job_by_cid.get(cid) else ""),
                        provider="anthropic",
                        batch=True,
                        missing=True,
                        error=err_msg or str(rtype),
                    )
                    continue
                message = getattr(result, "message", None) or (
                    result.get("message") if isinstance(result, dict) else None
                )
                if message is None:
                    continue
                msg = message if isinstance(message, dict) else message.model_dump()
                raw = _text_from_anthropic_message(msg)
                usage = _usage_from_anthropic_message(msg)
                req = job_by_cid.get(cid)
                if req is None:
                    continue
                try:
                    parsed = parse_json_object(raw)
                    validate_against(req.schema, parsed)
                except (json.JSONDecodeError, ValueError) as exc:
                    print(f"batch item {cid} parse error: {exc}")
                    continue
                usd = usage_usd(req.model, self.prices, usage, batch=True)
                rid = str(msg.get("id") or uuid.uuid4())
                if not already_billed:
                    self.ledger.record(
                        role=req.role,
                        provider="anthropic",
                        model=req.model,
                        usage=usage,
                        usd=usd,
                        latency_s=0.0,
                        request_id=rid,
                        batch=True,
                        conversation_id=req.conversation_id,
                        reserved=0.0,
                    )
                results[cid] = LLMResult(
                    parsed=parsed,
                    raw_text=raw,
                    usage=usage,
                    request_id=rid,
                    latency_s=0.0,
                    model=req.model,
                    provider="anthropic",
                    batch=True,
                )
        finally:
            if reserved:
                self.ledger.release(reserved)
            for row in self.state.get("batches") or []:
                if row.get("batch_id") == batch_id:
                    row["status"] = "ended"
                    row["reserved"] = 0.0
            _save_state(self.state_path, self.state)
        _ = batch
        return results

    def _poll_anthropic(self, client: Any, batch_id: str) -> Any:
        delay = 2.0
        while True:
            try:
                batch = client.messages.batches.retrieve(batch_id)
            except Exception as exc:
                name = type(exc).__name__
                transient = any(
                    token in name
                    for token in ("Timeout", "Connection", "APIConnection", "APITimeout")
                )
                if not transient:
                    raise
                print(f"poll retry {batch_id}: {name}: {exc}")
                time.sleep(delay)
                delay = min(delay * 1.5, 30.0)
                continue
            status = getattr(batch, "processing_status", None) or getattr(batch, "status", None)
            counts = getattr(batch, "request_counts", None)
            print(f"Anthropic batch {batch_id} status={status} counts={counts}")
            if status in {"ended", "completed", "failed", "expired", "canceled", "cancelled"}:
                return batch
            time.sleep(delay)
            delay = min(delay * 1.5, 30.0)

    def _openai_submit(self, jobs: list[LLMRequest]) -> dict[str, Any]:
        from openai import OpenAI

        client = OpenAI()
        custom_ids = [make_custom_id(j.conversation_id, j.role) for j in jobs]
        existing = self._resume_entry(custom_ids)
        if existing and existing.get("batch_id") and existing.get("status") not in {"completed", "ended"}:
            print(f"resuming OpenAI batch {existing['batch_id']}")
            return existing
        reserved = self._estimate_and_reserve(jobs)
        jsonl_path = self.run_dir / "raw" / f"openai_batch_{uuid.uuid4().hex[:8]}.jsonl"
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for req, cid in zip(jobs, custom_ids, strict=True):
            lines.append(
                json.dumps(
                    {
                        "custom_id": cid,
                        "method": "POST",
                        "url": "/v1/responses",
                        "body": _request_to_openai_body(req),
                    },
                    ensure_ascii=False,
                )
            )
        jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        uploaded = client.files.create(file=jsonl_path.open("rb"), purpose="batch")
        created = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/responses",
            completion_window="24h",
        )
        entry = {
            "batch_id": created.id,
            "provider": "openai",
            "status": created.status,
            "custom_ids": custom_ids,
            "conversation_ids": sorted({j.conversation_id for j in jobs}),
            "output_file_id": getattr(created, "output_file_id", None),
            "reserved": reserved,
        }
        self.state.setdefault("batches", []).append(entry)
        _save_state(self.state_path, self.state)
        print(f"created OpenAI batch {created.id} ({len(jobs)} requests)")
        return entry

    def _openai_collect(self, entry: dict[str, Any], jobs: list[LLMRequest]) -> dict[str, LLMResult]:
        from openai import OpenAI

        client = OpenAI()
        batch_id = entry["batch_id"]
        job_by_cid = {make_custom_id(j.conversation_id, j.role): j for j in jobs}
        reserved = float(entry.get("reserved") or 0.0)
        batch = self._poll_openai(client, batch_id)
        results: dict[str, LLMResult] = {}
        try:
            output_file_id = getattr(batch, "output_file_id", None)
            if not output_file_id:
                raise RuntimeError(f"OpenAI batch {batch_id} has no output_file_id")
            content = client.files.content(output_file_id)
            text = content.text if hasattr(content, "text") else content.read().decode("utf-8")
            for line in text.splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                cid = row.get("custom_id")
                body = (row.get("response") or {}).get("body") or {}
                raw = _text_from_openai(body)
                usage = _usage_from_openai(body)
                req = job_by_cid[cid]
                parsed = parse_json_object(raw)
                validate_against(req.schema, parsed)
                usd = usage_usd(req.model, self.prices, usage, batch=True)
                rid = str(body.get("id") or uuid.uuid4())
                self.ledger.record(
                    role=req.role,
                    provider="openai",
                    model=req.model,
                    usage=usage,
                    usd=usd,
                    latency_s=0.0,
                    request_id=rid,
                    batch=True,
                    conversation_id=req.conversation_id,
                    reserved=0.0,
                )
                results[cid] = LLMResult(
                    parsed=parsed,
                    raw_text=raw,
                    usage=usage,
                    request_id=rid,
                    latency_s=0.0,
                    model=req.model,
                    provider="openai",
                    batch=True,
                )
        finally:
            if reserved:
                self.ledger.release(reserved)
            for entry in self.state.get("batches") or []:
                if entry.get("batch_id") == batch_id:
                    entry["status"] = getattr(batch, "status", None) or "completed"
            _save_state(self.state_path, self.state)
        return results

    def _poll_openai(self, client: Any, batch_id: str) -> Any:
        delay = 2.0
        while True:
            batch = client.batches.retrieve(batch_id)
            print(f"OpenAI batch {batch_id} status={batch.status}")
            if batch.status in {"completed", "failed", "expired", "cancelled", "canceled"}:
                return batch
            time.sleep(delay)
            delay = min(delay * 1.5, 30.0)
