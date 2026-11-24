"""Zero-shot generative detectors scored from next-token log-probs."""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scambranch.eval.constants import (
    DETECTOR_INSTRUCTION,
    LABEL_FRAUD,
    LABEL_LEGITIMATE,
    MODEL_REGISTRY,
    VERBALIZERS,
)
from scambranch.eval.data import assert_no_private_leak, build_detector_input, render_transcript
from scambranch.eval.resources import MemoryTracker, pick_device, pick_dtype


PARSE_RE = re.compile(r"^\s*([A-Za-z]+)")


def parse_model_ref(name: str) -> tuple[str, str | None]:
    """Parse ``short_id`` or ``short_id@/path/to/adapter`` (or hf_id@adapter)."""
    if "@" in name:
        base, adapter = name.split("@", 1)
        adapter = adapter.strip()
        if not adapter:
            raise ValueError(f"empty adapter path in model ref {name!r}")
        return base.strip(), adapter
    return name, None


def resolve_model_spec(name: str) -> tuple[str, dict[str, Any]]:
    """Return (short_id, spec). Accepts short names or Hugging Face ids.

    Adapter refs (``name@path``) resolve the base model; callers should pass
    the adapter path separately via ``load_detector(..., adapter_path=...)``.
    """
    base, _adapter = parse_model_ref(name)
    if base in MODEL_REGISTRY:
        return base, dict(MODEL_REGISTRY[base])
    for short, spec in MODEL_REGISTRY.items():
        if spec["hf_id"] == base:
            return short, dict(spec)
    raise KeyError(f"unknown model {base!r}; known: {sorted(MODEL_REGISTRY)}")


def adapter_short_id(base_short: str, adapter_path: str | None) -> str:
    """Stable short id that includes arm+seed when path is ``.../<arm>/seed<k>``."""
    if not adapter_path:
        return base_short
    path = Path(adapter_path)
    parts = list(path.parts)
    # Prefer .../<arm>/seed<k> or .../<arm>/seed<k>/adapter
    arm = None
    seed = None
    for i, part in enumerate(parts):
        if part.startswith("seed") and part[4:].isdigit():
            seed = part
            if i > 0:
                arm = parts[i - 1]
            break
    if arm and seed:
        return f"{base_short}+{arm}-{seed}"
    return f"{base_short}+{path.name}"


def continuation_token_ids(tokenizer: Any, prompt: str, word: str) -> list[int]:
    """Tokens added when ``word`` is appended to ``prompt`` (no specials)."""
    pref = tokenizer.encode(prompt, add_special_tokens=False)
    full = tokenizer.encode(prompt + word, add_special_tokens=False)
    if len(full) > len(pref) and full[: len(pref)] == pref:
        return list(full[len(pref) :])
    standalone = tokenizer.encode(word, add_special_tokens=False)
    if standalone:
        return list(standalone)
    raise ValueError(f"tokenizer produced no ids for {word!r}")


def document_verbalizers(tokenizer: Any, prompt: str) -> dict[str, Any]:
    """Pick verbalizer variants whose first continuation tokens differ."""
    candidates = [
        {LABEL_FRAUD: "fraud", LABEL_LEGITIMATE: "legitimate"},
        {LABEL_FRAUD: "Fraud", LABEL_LEGITIMATE: "Legitimate"},
        {LABEL_FRAUD: " fraud", LABEL_LEGITIMATE: " legitimate"},
    ]
    chosen: dict[str, Any] | None = None
    attempts: list[dict[str, Any]] = []
    for variant in candidates:
        entry: dict[str, Any] = {"variant": variant, "verbalizers": {}}
        first_ids: list[int] = []
        ok = True
        for label, word in variant.items():
            ids = continuation_token_ids(tokenizer, prompt, word)
            tokens = tokenizer.convert_ids_to_tokens(ids)
            entry["verbalizers"][label] = {
                "surface": word,
                "token_ids": ids,
                "tokens": tokens,
                "first_token_id": ids[0],
                "first_token": tokens[0] if tokens else None,
            }
            first_ids.append(ids[0])
        entry["first_tokens_distinct"] = len(set(first_ids)) == len(first_ids)
        attempts.append(entry)
        if entry["first_tokens_distinct"] and chosen is None:
            chosen = entry
    if chosen is None:
        raise RuntimeError(
            "verbalizer first tokens collide for every candidate pair; "
            f"attempts={attempts}"
        )
    return {
        "method": "prompt_continuation",
        "chosen": chosen,
        "attempts": attempts,
    }


def parse_verbalizer(text: str) -> str | None:
    if not text:
        return None
    match = PARSE_RE.match(text.strip())
    if not match:
        return None
    word = match.group(1).lower()
    if word == LABEL_FRAUD:
        return LABEL_FRAUD
    if word == LABEL_LEGITIMATE:
        return LABEL_LEGITIMATE
    return None


def pairwise_fraud_score(logit_fraud: float, logit_legit: float) -> float:
    m = max(logit_fraud, logit_legit)
    p_f = math.exp(logit_fraud - m)
    p_l = math.exp(logit_legit - m)
    denom = p_f + p_l
    if denom <= 0.0 or not math.isfinite(denom):
        return float("nan")
    return float(p_f / denom)


@dataclass
class DetectorScore:
    score: float
    parsed: str | None
    generated: str
    invalid: bool
    latency_s: float
    used_logprobs: bool
    fallback: str | None = None


@dataclass
class ZeroShotDetector:
    short_id: str
    hf_id: str
    enable_thinking: bool | None
    gated: bool
    device: str
    dtype_name: str
    model: Any
    tokenizer: Any
    verbalizer_doc: dict[str, Any]
    fraud_token_id: int
    legit_token_id: int
    load_time_s: float
    n_params: int
    tracker: MemoryTracker = field(default_factory=MemoryTracker)
    latencies_s: list[float] = field(default_factory=list)

    def score_turns(self, conversation_id: str, turns: list[dict[str, Any]]) -> DetectorScore:
        detector_input = build_detector_input(conversation_id, turns)
        transcript = render_transcript(detector_input["turns"])
        user_content = f"{DETECTOR_INSTRUCTION}\n\nConversation:\n{transcript}"
        assert_no_private_leak(user_content, detector_input)
        prompt = self._apply_chat(user_content)
        return self._score_prompt(prompt)

    def _apply_chat(self, user_content: str) -> str:
        messages = [{"role": "user", "content": user_content}]
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self.enable_thinking is False:
            kwargs["enable_thinking"] = False
        try:
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            try:
                return self.tokenizer.apply_chat_template(messages, **kwargs)
            except Exception:
                return f"{user_content}\n\nAnswer:"

    def _score_prompt(self, prompt: str) -> DetectorScore:
        import torch

        self.tracker.sample()
        t0 = time.perf_counter()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        generated_text = ""
        logits = None
        fallback = None
        used_logprobs = False
        try:
            with torch.inference_mode():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=8,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=pad_id,
                )
            if out.scores:
                logits = out.scores[0][0].float()
                used_logprobs = True
            seq = out.sequences[0]
            prompt_len = inputs["input_ids"].shape[1]
            gen_ids = seq[prompt_len:]
            generated_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        except Exception as exc:
            fallback = f"generate_failed:{type(exc).__name__}"
            with torch.inference_mode():
                fwd = self.model(**inputs)
            logits = fwd.logits[0, -1].float()
            used_logprobs = True
            top_id = int(torch.argmax(logits).item())
            generated_text = self.tokenizer.decode([top_id], skip_special_tokens=True)

        latency = time.perf_counter() - t0
        self.latencies_s.append(latency)
        self.tracker.sample()

        score = float("nan")
        if logits is not None:
            logit_f = float(logits[self.fraud_token_id].item())
            logit_l = float(logits[self.legit_token_id].item())
            score = pairwise_fraud_score(logit_f, logit_l)

        parsed = parse_verbalizer(generated_text)
        invalid = parsed is None
        if (not math.isfinite(score)) and parsed is not None:
            score = 1.0 if parsed == LABEL_FRAUD else 0.0
            fallback = (fallback + ";parse_score") if fallback else "parse_score"
        return DetectorScore(
            score=score,
            parsed=parsed,
            generated=generated_text,
            invalid=invalid,
            latency_s=latency,
            used_logprobs=used_logprobs,
            fallback=fallback,
        )


def _set_determinism(device: str) -> None:
    import torch

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(False)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed") and device == "mps":
        torch.mps.manual_seed(0)


def load_detector(
    name: str,
    *,
    device: str | None = None,
    dtype_name: str | None = None,
    adapter_path: str | None = None,
) -> ZeroShotDetector:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_name, inline_adapter = parse_model_ref(name)
    adapter_path = adapter_path or inline_adapter
    short_id, spec = resolve_model_spec(base_name)
    short_id = adapter_short_id(short_id, adapter_path)
    device = pick_device(device)
    dtype = pick_dtype(device, dtype_name)
    _set_determinism(device)

    tracker = MemoryTracker()
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(spec["hf_id"], trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    def _load_weights(chosen_dtype):
        kwargs = {"low_cpu_mem_usage": True, "trust_remote_code": True}
        try:
            return AutoModelForCausalLM.from_pretrained(spec["hf_id"], dtype=chosen_dtype, **kwargs)
        except TypeError:
            return AutoModelForCausalLM.from_pretrained(spec["hf_id"], torch_dtype=chosen_dtype, **kwargs)

    try:
        model = _load_weights(dtype)
        model.to(device)
        model.eval()
        used_dtype = dtype
    except Exception:
        if dtype != torch.float32:
            model = _load_weights(torch.float32)
            model.to(device)
            model.eval()
            used_dtype = torch.float32
        else:
            raise

    if adapter_path:
        from peft import PeftModel

        adapter = Path(adapter_path)
        # Accept either the adapter dir itself or a parent containing adapter/.
        if (adapter / "adapter_config.json").is_file():
            load_from = adapter
        elif (adapter / "adapter" / "adapter_config.json").is_file():
            load_from = adapter / "adapter"
        else:
            load_from = adapter
        model = PeftModel.from_pretrained(model, str(load_from))
        model.to(device)
        model.eval()

    n_params = int(sum(p.numel() for p in model.parameters()))
    load_time = time.perf_counter() - t0
    tracker.sample()

    probe = ZeroShotDetector(
        short_id=short_id,
        hf_id=spec["hf_id"],
        enable_thinking=spec.get("enable_thinking"),
        gated=bool(spec.get("gated")),
        device=device,
        dtype_name=str(used_dtype).replace("torch.", ""),
        model=model,
        tokenizer=tokenizer,
        verbalizer_doc={},
        fraud_token_id=0,
        legit_token_id=0,
        load_time_s=load_time,
        n_params=n_params,
        tracker=tracker,
    )
    dummy_user = f"{DETECTOR_INSTRUCTION}\n\nConversation:\nCaller: hello"
    dummy_prompt = probe._apply_chat(dummy_user)
    vdoc = document_verbalizers(tokenizer, dummy_prompt)
    probe.verbalizer_doc = vdoc
    probe.fraud_token_id = int(vdoc["chosen"]["verbalizers"][LABEL_FRAUD]["first_token_id"])
    probe.legit_token_id = int(vdoc["chosen"]["verbalizers"][LABEL_LEGITIMATE]["first_token_id"])
    if probe.fraud_token_id == probe.legit_token_id:
        raise RuntimeError("verbalizer first tokens are not distinct")

    # Warm-up + fp16 stability check on MPS.
    warm = probe._score_prompt(dummy_prompt)
    if not math.isfinite(warm.score) and used_dtype != torch.float32:
        model = _load_weights(torch.float32)
        model.to(device)
        model.eval()
        if adapter_path:
            from peft import PeftModel

            adapter = Path(adapter_path)
            if (adapter / "adapter_config.json").is_file():
                load_from = adapter
            elif (adapter / "adapter" / "adapter_config.json").is_file():
                load_from = adapter / "adapter"
            else:
                load_from = adapter
            model = PeftModel.from_pretrained(model, str(load_from))
            model.to(device)
            model.eval()
        probe.model = model
        probe.dtype_name = "float32"
        probe._score_prompt(dummy_prompt)
    # Drop warm-up from recorded latencies.
    probe.latencies_s.clear()
    probe.tracker.sample()
    return probe


def try_load_detector(
    name: str,
    *,
    device: str | None = None,
    dtype_name: str | None = None,
    adapter_path: str | None = None,
) -> tuple[ZeroShotDetector | None, dict[str, Any]]:
    """Load a model; on download/auth failure return a skip record instead of raising."""
    base_name, inline_adapter = parse_model_ref(name)
    adapter_path = adapter_path or inline_adapter
    short_id, spec = resolve_model_spec(base_name)
    display_id = adapter_short_id(short_id, adapter_path)
    try:
        det = load_detector(
            name, device=device, dtype_name=dtype_name, adapter_path=adapter_path
        )
        return det, {
            "short_id": det.short_id,
            "hf_id": spec["hf_id"],
            "adapter_path": adapter_path,
            "skipped": False,
            "skip_reason": None,
            "gated": bool(spec.get("gated")),
        }
    except Exception as exc:
        reason = str(exc)
        lowered = reason.lower()
        kind = "load_failed"
        if spec.get("gated") or any(s in lowered for s in ("401", "403", "gated", "unauthorized", "restricted")):
            kind = "gated_or_auth"
        elif any(s in lowered for s in ("connection", "timeout", "maxretry", "download", "huggingface")):
            kind = "download_failed"
        return None, {
            "short_id": display_id,
            "hf_id": spec["hf_id"],
            "adapter_path": adapter_path,
            "skipped": True,
            "skip_reason": f"{kind}: {type(exc).__name__}: {reason[:400]}",
            "gated": bool(spec.get("gated")),
        }
