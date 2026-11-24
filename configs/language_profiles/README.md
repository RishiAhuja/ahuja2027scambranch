# Language profiles (v0.1-auto)

Automatically derived from human-written calibration corpora. **Not yet reviewed by a fluent speaker.** Generation targets `mixing_bands` (human, mostly gold-tagged). `service_register_bands` are selection-biased upward (candidates were required to have CMI > 0 and English service terms), so use them only as an upper reference, not a target.

## Pair comparison

| Field | hi_en | ta_en | te_en |
|---|---|---|---|
| Sources | lince, lince_pos, phinc, hinge, sentimix, code_mixed_dialog, hinglish_top, dakshina | tamilmixsentiment, offenseval_dravidian, tanglish_corpus_185k, coli_dravidian_ta, cmqa_tamlish, code_mixed_dialog, dakshina | cmtet_sentiment, cmtet_lid, icon_2016, chandu_cmqa, dakshina |
| Mixing-band origin | gold (n=26969) | auto (n=210532) | gold (n=52239) |
| Social english_share p50 | 0.360 (p10=0.056, p90=1.000) | 0.375 (p10=0.100, p90=0.692) | 0.400 (p10=0.000, p90=1.000) |
| Social CMI p50 | 18.182 (p10=0.000, p90=42.105) | 32.258 (p10=7.692, p90=46.667) | 25.000 (p10=0.000, p90=45.455) |
| Service english_share p50 | 0.500 (p10=0.200, p90=0.539) | 0.500 (p10=0.321, p90=0.600) | 0.425 (p10=0.256, p90=0.647) |
| Service CMI p50 | 47.017 (p10=20.000, p90=50.000) | 46.154 (p10=29.647, p90=50.000) | 36.667 (p10=25.577, p90=48.846) |
| # reference utterances | 200 | 200 | 200 |
| # by register | | | |
| `service` | 55 | 120 | 14 |
| `service_dialog_user` | 50 | 50 | 0 |
| `service_dialog_system_template` | 15 | 15 | 0 |
| `assistant_command` | 60 | 0 | 0 |
| `question` | 0 | 0 | 86 |
| `conversational` | 20 | 15 | 100 |
| Known gaps | Social-media calibration; no open phone-call corpus; LinCE not redistributable | Gold LID <2000 (auto bands); YouTube-heavy; no call-centre CS | No public Roman TE-EN service dialogue; 14 strict candidates only |

Full `known_gaps` lists live in each YAML.

## Fluent-reviewer instructions

1. Open `configs/language_profiles/{hi_en,ta_en,te_en}.yaml`.
2. Read `status_note` and `known_gaps` before judging items.
3. For each `reference_utterances` row, set `accepted` to `yes` or `no`. Leave `pending` only if you skip the item.
4. Check that brand names were replaced with fictional placeholders (Kaveri Bank, ShopKart, SwiftShip Couriers, PaySetu, `[TELECOM]`, `[BRAND]`). Do not restore real brand strings.
5. Skim `light_verb_constructions` and `mixed_morphology`; film/sports leftovers belong in `excluded_domain_specific`, not the main lists.
6. When you finish a pair, append a `reviewer_signoff` object: `role`, `name`, `date` (YYYY-MM-DD), `notes`.
7. Add a `change_history` entry and bump `version` (for example `0.1-reviewed`).
8. Two bilingual reviewers per pair is the plan-§10.3 target; this file is the first-pass auto draft.

Neither band set is phone-call truth. `mixing_bands` come from social media; `service_register_bands` are inflated by the CMI > 0 selection filter. Sample mixing from `mixing_bands`, and treat values above the service p75 as review flags. Telugu–English has no public Roman service dialogue.
