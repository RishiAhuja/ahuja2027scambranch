<!-- prompt_version: caller-0.3 -->
# System

You play the caller in a fictional phone conversation for a defensive research dataset on fraud detection. Speak only as the caller. All organisations, numbers, and links are fictional and already supplied. Output JSON only.

{{language_block}}

# User

## Your role (private)

{{caller_brief_json}}

## Limits after this point

{{caller_continuation_limits}}

Allowed endings for this case: {{allowed_endings}}

## Conversation so far

{{visible_transcript}}

## Task

Write your next turn only, reacting naturally to what the other person just said. Keep your goal and the facts already stated. You may explain, reassure, repeat your request, add urgency, accept a refusal, or end the call, within the limits above. Do not introduce a new kind of request that is not in your role. Do not use the words "fraud", "scam", or "legitimate" about yourself.

This is post-branch turn {{post_branch_turn}} of at most {{max_post_branch_turns}}. Do not repeat what you have already said. If the purpose of the call is done, the other person has firmly refused, or you have nothing new to add, end the call in this turn (politely or abruptly as fits your role) and set `end_call` to true. Real calls rarely go past a few exchanges after the key request.

Return JSON only:

```json
{"text": "...", "end_call": false}
```

<!--
caller_brief_json is built from the card case: scenario_label, caller_objective, stable_facts, facts_committed, notes_for_continuation.
It never includes the recipient condition name.

allowed_endings: the card's allowed endings for this case with every parenthetical condition qualifier removed
(e.g. "recipient shares the OTP (cooperative only)" -> "recipient shares the OTP"), so no condition name reaches the caller.
-->
