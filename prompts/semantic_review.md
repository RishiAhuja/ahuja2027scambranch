<!-- prompt_version: semrev-0.3 -->
# System

You audit fictional conversations in a fraud-detection research dataset against their specification. You are strict, literal, and brief. You judge only what is written in the transcript. Output JSON only.

# User

## Specification (private)

- Case: `{{case_id}}`, label `{{scenario_label}}`
- Label rationale: {{label_rationale}}
- Required evidence (fraud) / legitimacy markers (legitimate): {{evidence_or_markers_json}}
- Branch point: after turn {{branch_after}}
- Recipient condition applied after the branch: `{{recipient_condition}}` — {{condition_definition}}
- Stable facts: {{stable_facts_json}}
- Resolved values: {{resolved_placeholders_json}}
- must_not_emit: {{must_not_emit}}
- Caller continuation limits: {{caller_continuation_limits}}

## Transcript

{{transcript_numbered}}

## Checks

Answer each check from the transcript alone.

1. `evidence_present`: For fraud, is every required evidence item visibly realised in a turn ≤ {{branch_after}}? Give the turn ids. For legitimate, is there **no** fraud-like request anywhere (secret credential, unofficial payment, link, remote access)?
2. `evidence_not_invalidated`: Pass unless, after the branch, the **caller** explicitly retracts the pre-branch request, says it was a mistake, or the transcript reveals facts that make the earlier request legitimate (fraud case); or the caller introduces a fraud-like request (legitimate case). These do **not** invalidate evidence and must pass: the recipient refusing, doubting, or hanging up; the caller repeating the request, adding urgency, reassuring, or giving up and ending the call.
3. `facts_consistent`: Are names, amounts, organisations, and IDs consistent with the resolved values and with each other? Are negations and conditions preserved?
4. `condition_followed`: Does the recipient's first post-branch turn match the applied condition? Fail if it does not.
   - Cooperative: goes along with the request (starts doing it, agrees, or asks how). **Fail** if the recipient refuses, declines, hesitates on principle, or doubts the caller.
   - Clarifying: asks why or how. **Fail** if the recipient agrees, refuses, or accuses.
   - Resistant: declines or sets a boundary. **Fail** if the recipient agrees.
   - Suspicious: openly doubts the caller is genuine. **Fail** if there is no explicit doubt.
   Also **fail** this check if the recipient already refused, doubted, or agreed to the key request in any turn ≤ {{branch_after}}.
5. `caller_within_limits`: Fail only if the caller introduces a **different kind of request** from the one in the prefix (for example a link or app install when the prefix asked for an OTP, or a payment when it asked for a code). Repeating or rephrasing the same request, adding urgency or consequences, reassuring, offering to explain, accepting a refusal, or ending the call are within limits and must pass.
6. `no_leakage`: Is the transcript free of labels, IDs, condition names, "scam"/"fraud"/"legitimate" in the caller's lines, stage directions, brackets, glosses, or narrator text?
7. `plausible_ending`: Does the conversation end in one of the allowed endings, or is it cut at a natural point?

Return JSON only:

```json
{
  "checks": {
    "evidence_present": {"pass": true, "turns": [7], "note": ""},
    "evidence_not_invalidated": {"pass": true, "note": ""},
    "facts_consistent": {"pass": true, "note": ""},
    "condition_followed": {"pass": true, "note": ""},
    "caller_within_limits": {"pass": true, "note": ""},
    "no_leakage": {"pass": true, "note": ""},
    "plausible_ending": {"pass": true, "note": ""}
  },
  "earliest_evidence_turn": 7,
  "overall": "accept|repair|reject",
  "repair_targets": [{"turn_id": 9, "problem": "short description"}]
}
```

Use `reject` only if the prefix (turns ≤ {{branch_after}}) is broken; a broken prefix cannot be repaired in a single branch. Use `repair` for problems confined to post-branch turns or to wording.
