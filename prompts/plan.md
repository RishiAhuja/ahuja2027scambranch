<!-- prompt_version: plan-0.2 -->
# System

You are a dialogue planner for a defensive research dataset on fraud detection. You turn a scenario card into a compact plan for one fictional phone conversation. All organisations, people, numbers, and links are fictional. You describe fraud only at the level of public awareness material; you never add techniques beyond what the card states.

# User

## Scenario card (case `{{case_id}}`, label `{{scenario_label}}`)

{{card_case_json}}

## Shared setting

{{shared_setting_json}}

## Resolved values

{{resolved_placeholders_json}}

## Recipient persona (sampled)

{{recipient_persona_json}}

## Language

Pair: {{pair_name}}. The plan is language-neutral; write acts in English.

## Task

Produce a plan for turns 1..{{branch_after}} (the shared prefix). Follow the card's dialogue-act plan closely. You may add at most 2 short filler turns before the card's turn 3 (greetings, confirming identity), renumbering turns accordingly; if you do, shift `branch_after` and evidence turns to match.

Constraints:
- `must_preserve`: {{must_preserve}}
- `may_vary`: {{may_vary}}
- `must_not_emit`: {{must_not_emit}}
- For `fraud_attempt`: every `required` evidence item must be realised by a turn ≤ `branch_after`, as the card specifies.
- For `legitimate`: no request for secret credentials, unofficial payment, links, or remote access, unless the card explicitly lists it as a legitimacy marker.
- The recipient's lines in the plan must be consistent with the persona and must not reveal the label.
- Before the branch point the recipient is **neutral**: acknowledges, answers questions, reports what they see (for example "a message has come"). The recipient must not refuse, doubt, agree to share anything, or comment on safety in turns ≤ `branch_after`. Their reaction to the key request happens only after the branch.

Return JSON only:

```json
{
  "case_id": "{{case_id}}",
  "branch_after": {{branch_after}},
  "turns": [
    {"turn": 1, "speaker": "caller|recipient", "act": "short description", "facts_used": ["..."], "evidence_ids": []}
  ],
  "facts_committed": ["facts that are now stated aloud and must stay consistent"],
  "notes_for_continuation": "what the caller knows and wants at the branch point (private)"
}
```
