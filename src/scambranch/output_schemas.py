"""JSON schemas for structured model outputs."""

PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "case_id",
        "branch_after",
        "turns",
        "facts_committed",
        "notes_for_continuation",
    ],
    "properties": {
        "case_id": {"type": "string"},
        "branch_after": {"type": "integer"},
        "turns": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["turn", "speaker", "act", "facts_used", "evidence_ids"],
                "properties": {
                    "turn": {"type": "integer"},
                    "speaker": {"type": "string", "enum": ["caller", "recipient"]},
                    "act": {"type": "string"},
                    "facts_used": {"type": "array", "items": {"type": "string"}},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "facts_committed": {"type": "array", "items": {"type": "string"}},
        "notes_for_continuation": {"type": "string"},
    },
}

PREFIX_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["turns"],
    "properties": {
        "turns": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["turn_id", "speaker", "text"],
                "properties": {
                    "turn_id": {"type": "integer"},
                    "speaker": {"type": "string", "enum": ["caller", "recipient"]},
                    "text": {"type": "string"},
                },
            },
        }
    },
}

TURN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["text", "end_call"],
    "properties": {
        "text": {"type": "string"},
        "end_call": {"type": "boolean"},
    },
}

SEMANTIC_REVIEW_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["checks", "earliest_evidence_turn", "overall", "repair_targets"],
    "properties": {
        "checks": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "evidence_present",
                "evidence_not_invalidated",
                "facts_consistent",
                "condition_followed",
                "caller_within_limits",
                "no_leakage",
                "plausible_ending",
            ],
            "properties": {
                "evidence_present": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["pass", "turns", "note"],
                    "properties": {
                        "pass": {"type": "boolean"},
                        "turns": {"type": "array", "items": {"type": "integer"}},
                        "note": {"type": "string"},
                    },
                },
                "evidence_not_invalidated": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["pass", "note"],
                    "properties": {
                        "pass": {"type": "boolean"},
                        "note": {"type": "string"},
                    },
                },
                "facts_consistent": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["pass", "note"],
                    "properties": {
                        "pass": {"type": "boolean"},
                        "note": {"type": "string"},
                    },
                },
                "condition_followed": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["pass", "note"],
                    "properties": {
                        "pass": {"type": "boolean"},
                        "note": {"type": "string"},
                    },
                },
                "caller_within_limits": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["pass", "note"],
                    "properties": {
                        "pass": {"type": "boolean"},
                        "note": {"type": "string"},
                    },
                },
                "no_leakage": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["pass", "note"],
                    "properties": {
                        "pass": {"type": "boolean"},
                        "note": {"type": "string"},
                    },
                },
                "plausible_ending": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["pass", "note"],
                    "properties": {
                        "pass": {"type": "boolean"},
                        "note": {"type": "string"},
                    },
                },
            },
        },
        "earliest_evidence_turn": {"type": ["integer", "null"]},
        "overall": {"type": "string", "enum": ["accept", "repair", "reject"]},
        "repair_targets": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["turn_id", "problem"],
                "properties": {
                    "turn_id": {"type": "integer"},
                    "problem": {"type": "string"},
                },
            },
        },
    },
}

LANGUAGE_REVIEW_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "turn_flags",
        "switching_naturalness",
        "grammatical_acceptability",
        "overall",
    ],
    "properties": {
        "turn_flags": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["turn_id", "type", "span", "note"],
                "properties": {
                    "turn_id": {"type": "integer"},
                    "type": {"type": "string"},
                    "span": {"type": "string"},
                    "note": {"type": "string"},
                },
            },
        },
        "switching_naturalness": {"type": "integer", "minimum": 1, "maximum": 5},
        "grammatical_acceptability": {"type": "integer", "minimum": 1, "maximum": 5},
        "overall": {"type": "string", "enum": ["accept", "repair", "reject"]},
    },
}

REPAIR_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["revised_turns", "unfixable"],
    "properties": {
        "revised_turns": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["turn_id", "speaker", "text"],
                "properties": {
                    "turn_id": {"type": "integer"},
                    "speaker": {"type": "string", "enum": ["caller", "recipient"]},
                    "text": {"type": "string"},
                },
            },
        },
        "unfixable": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["turn_id", "reason"],
                "properties": {
                    "turn_id": {"type": "integer"},
                    "reason": {"type": "string"},
                },
            },
        },
    },
}

ROLE_SCHEMAS = {
    "plan": PLAN_SCHEMA,
    "prefix": PREFIX_SCHEMA,
    "caller": TURN_SCHEMA,
    "recipient": TURN_SCHEMA,
    "repair": REPAIR_SCHEMA,
    "semantic_review": SEMANTIC_REVIEW_SCHEMA,
    "language_review": LANGUAGE_REVIEW_SCHEMA,
    "audit": SEMANTIC_REVIEW_SCHEMA,
    "audit_semantic": SEMANTIC_REVIEW_SCHEMA,
    "audit_language": LANGUAGE_REVIEW_SCHEMA,
}
