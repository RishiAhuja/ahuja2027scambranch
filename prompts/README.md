# Prompt modules

Versioned templates for the generation pipeline (plan Section 9.5). Placeholders use `{{name}}` and are filled by the pipeline; nothing is filled by hand. Bump `prompt_version` in a template's header whenever its text changes; the version is stored in every conversation's provenance.

## Modules and call order

```text
plan ─► prefix ─► [freeze prefix, hash] ─► for each of 4 conditions:
                                             recipient ⇄ caller_continuation (turn by turn)
                                           ─► semantic_review + language_review
                                           ─► repair (max 2 rounds) ─► re-review
```

| Module | File | Output |
|---|---|---|
| Semantic plan | `plan.md` | JSON plan: dialogue acts, facts, evidence placement |
| Prefix | `prefix.md` | JSON turns 1..`branch_after` |
| Recipient turn | `recipient.md` | One recipient turn (JSON) |
| Caller turn | `caller_continuation.md` | One caller turn or end signal (JSON) |
| Semantic review | `semantic_review.md` | JSON verdict on facts, evidence, leakage |
| Language review | `language_review.md` | JSON verdict on code-switching quality |
| Repair | `repair.md` | Revised turns for the flagged span only |

Every generation module includes `_language_block.md`, rendered from `configs/language_profiles/<pair>.yaml`.

## Information flow

The generator knows more than the detector. The recipient knows less than the generator.

| Information | plan | prefix | caller | recipient | reviewers | detector |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| Scenario label, rationale, caller objective | yes | yes | yes | **no** | semantic only | **no** |
| Evidence IDs and `branch_after` | yes | yes | yes | **no** | semantic only | **no** |
| Recipient condition name | no | no | **no** | yes (as behaviour, not label) | semantic only | **no** |
| Stable facts | yes | yes | yes | only what was said aloud | yes | no |
| Visible transcript | — | — | yes | yes | yes | yes |
| Language profile | yes | yes | yes | yes | language only | no |

The caller never learns which recipient condition was applied; it only sees the recipient's words. This keeps the caller's reaction natural and stops it from "playing to" a condition.

## Conventions

- All model outputs are JSON only. The pipeline rejects any non-JSON output and retries at most twice.
- Transcript `text` fields contain spoken words only: no stage directions, brackets, narrator text, glosses, or translations.
- Placeholders from `scenarios/fictional_entities.md` (for example `[AMOUNT_1]`) are resolved to concrete fictional values by the pipeline **before** prompting, so the model never has to invent numbers, IDs, or links.
