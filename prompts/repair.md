<!-- prompt_version: repair-0.1 -->
# System

You make minimal, targeted fixes to fictional conversation transcripts in a fraud-detection research dataset. You change only the flagged turns and only as much as needed. Output JSON only.

{{language_block}}

# User

## Frozen turns (do not change)

Turns 1..{{frozen_until}} are frozen and must not change.

## Specification (private)

- Case `{{case_id}}`, label `{{scenario_label}}`, recipient condition `{{recipient_condition}}` — {{condition_definition}}
- must_preserve: {{must_preserve}}
- must_not_emit: {{must_not_emit}}
- Caller continuation limits: {{caller_continuation_limits}}

## Transcript

{{transcript_numbered}}

## Problems to fix

{{repair_targets_json}}

## Task

Rewrite only the turns listed in the problems (and, if strictly necessary for coherence, the single turn immediately after each). Keep speakers, turn ids, meaning, facts, and the recipient's attitude unchanged. Fix the stated problem and nothing else. If a problem cannot be fixed without changing a frozen turn, say so.

Return JSON only:

```json
{
  "revised_turns": [{"turn_id": 9, "speaker": "caller|recipient", "text": "..."}],
  "unfixable": [{"turn_id": 0, "reason": ""}]
}
```
