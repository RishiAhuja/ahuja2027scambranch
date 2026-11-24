"""Shared constants for detector evaluation."""

from __future__ import annotations

VIEW_FULL = "FULL"
VIEW_PREFIX_ONLY = "PREFIX_ONLY"
VIEW_PREFIX_PLUS_REACTION = "PREFIX_PLUS_REACTION"
VIEW_STREAMING = "STREAMING"
VIEW_CALLER_ONLY = "CALLER_ONLY"

ALL_VIEWS = (
    VIEW_FULL,
    VIEW_PREFIX_ONLY,
    VIEW_PREFIX_PLUS_REACTION,
    VIEW_STREAMING,
)

# CALLER_ONLY is opt-in via --views so existing CLI defaults stay unchanged.
KNOWN_VIEWS = ALL_VIEWS + (VIEW_CALLER_ONLY,)

CONDITIONS = ("cooperative", "clarifying", "resistant", "suspicious")

USABLE_STATUSES = frozenset({"accepted", "repaired"})
REJECTED_STATUSES = frozenset({"rejected"})

SPEAKER_LABELS = {"caller": "Caller:", "recipient": "Recipient:"}

# Field names that must never appear in a detector prompt.
PRIVATE_PROMPT_MARKERS = (
    "private_annotations",
    "generation_settings",
    "scenario_label",
    "label_rationale",
    "evidence_turn_ids",
    "prefix_evidence_states",
    "invariance_eligible",
    "branch_after_turn",
    "prefix_hash",
    "repair_history",
    "review_status",
    "recipient_condition",
    "must_preserve",
    "must_not_emit",
    "caller_continuation_limits",
)

FRAUD_LABELS = frozenset({"fraud_attempt", "fraud"})
LABEL_FRAUD = "fraud"
LABEL_LEGITIMATE = "legitimate"
VERBALIZERS = (LABEL_FRAUD, LABEL_LEGITIMATE)

# Short fixed instruction. The model must answer with one of the verbalizers.
DETECTOR_INSTRUCTION = (
    "Does the conversation contain a fraudulent attempt by the caller? "
    "Reply with exactly one word: fraud or legitimate."
)

BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 42

MODEL_REGISTRY = {
    "qwen3-0.6b": {
        "hf_id": "Qwen/Qwen3-0.6B",
        "enable_thinking": False,
        "gated": False,
    },
    "qwen3-1.7b": {
        "hf_id": "Qwen/Qwen3-1.7B",
        "enable_thinking": False,
        "gated": False,
    },
    "qwen2.5-1.5b": {
        "hf_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "enable_thinking": None,
        "gated": False,
    },
    "llama-3.2-1b": {
        "hf_id": "meta-llama/Llama-3.2-1B-Instruct",
        "enable_thinking": None,
        "gated": True,
    },
    "smollm2-1.7b": {
        "hf_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "enable_thinking": None,
        "gated": False,
    },
    "qwen2.5-0.5b": {
        "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "enable_thinking": None,
        "gated": False,
    },
    "qwen2-0.5b": {
        "hf_id": "Qwen/Qwen2-0.5B-Instruct",
        "enable_thinking": None,
        "gated": False,
    },
    "qwen2-1.5b": {
        "hf_id": "Qwen/Qwen2-1.5B-Instruct",
        "enable_thinking": None,
        "gated": False,
    },
    "smollm2-135m": {
        "hf_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "enable_thinking": None,
        "gated": False,
    },
    "smollm2-360m": {
        "hf_id": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "enable_thinking": None,
        "gated": False,
    },
    "gemma3-270m": {
        "hf_id": "google/gemma-3-270m-it",
        "enable_thinking": None,
        "gated": False,
    },
    "gemma3-1b": {
        "hf_id": "google/gemma-3-1b-it",
        "enable_thinking": None,
        "gated": False,
    },
    "tinyllama-1.1b": {
        "hf_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "enable_thinking": None,
        "gated": False,
    },
    "stablelm2-1.6b": {
        "hf_id": "stabilityai/stablelm-2-zephyr-1_6b",
        "enable_thinking": None,
        "gated": False,
    },
    "internlm2-1.8b": {
        "hf_id": "internlm/internlm2-chat-1_8b",
        "enable_thinking": None,
        "gated": False,
    },
    "danube2-1.8b": {
        "hf_id": "h2oai/h2o-danube2-1.8b-chat",
        "enable_thinking": None,
        "gated": False,
    },
    "granite-1b": {
        "hf_id": "ibm-granite/granite-3.1-1b-a400m-instruct",
        "enable_thinking": None,
        "gated": False,
    },
    "bloomz-1b7": {
        "hf_id": "bigscience/bloomz-1b7",
        "enable_thinking": None,
        "gated": False,
    },
    "olmo-1b": {
        "hf_id": "allenai/OLMo-1B-0724-hf",
        "enable_thinking": None,
        "gated": False,
    },
}

DEFAULT_MODELS = tuple(MODEL_REGISTRY.keys())
STREAMING_MODELS = frozenset({"qwen3-1.7b", "qwen2.5-1.5b"})
