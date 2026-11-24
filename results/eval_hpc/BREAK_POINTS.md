# Break points (`eval_hpc`, TEST)

Models: danube2-1.8b, granite-1b, qwen2-0.5b, qwen2-1.5b, qwen2.5-0.5b, qwen2.5-1.5b, qwen3-0.6b, qwen3-1.7b, smollm2-1.7b, smollm2-360m, stablelm2-1.6b. Excluded (100% invalid outputs): smollm2-135m, tinyllama-1.1b. Strong = qwen2.5-1.5b, qwen3-1.7b, qwen2-1.5b.
Verdicts use the DEV-fit threshold for each (model, view).

## 1. Per-family error rate (FULL view)

F = share of fraud chats missed; L = share of legit chats flagged as fraud.

| family | case | all models | strong models | qwen2.5-1.5b | qwen3-1.7b | qwen2-1.5b |
|---|---|---:|---:|---:|---:|---:|
| ec_08 *(OTP mentioned in a legit delivery call)* | L | 0.54 | 0.57 | 0.67 | 0.54 | 0.50 |
| ec_10 | L | 0.45 | 0.53 | 0.65 | 0.70 | 0.25 |
| bp_08 | L | 0.59 | 0.49 | 0.50 | 0.54 | 0.42 |
| jr_03 | F | 0.50 | 0.48 | 0.65 | 0.26 | 0.52 |
| jr_08 | F | 0.52 | 0.46 | 0.54 | 0.29 | 0.54 |
| bp_15 | F | 0.49 | 0.39 | 0.41 | 0.35 | 0.41 |
| ec_10 | F | 0.45 | 0.36 | 0.36 | 0.32 | 0.41 |
| bp_14 | L | 0.45 | 0.35 | 0.20 | 0.25 | 0.60 |
| jr_14 | F | 0.48 | 0.33 | 0.42 | 0.21 | 0.38 |
| bp_14 | F | 0.47 | 0.29 | 0.08 | 0.29 | 0.50 |
| bp_05 | F | 0.41 | 0.29 | 0.26 | 0.17 | 0.43 |
| jr_17 | F | 0.52 | 0.29 | 0.43 | 0.10 | 0.33 |
| ec_19 | L | 0.47 | 0.28 | 0.17 | 0.17 | 0.50 |
| jr_03 | L | 0.26 | 0.28 | 0.08 | 0.46 | 0.29 |
| ec_18 | F | 0.31 | 0.22 | 0.21 | 0.33 | 0.12 |
| ec_18 | L | 0.38 | 0.22 | 0.42 | 0.21 | 0.04 |
| jr_08 *(account number / IFSC requested legitimately)* | L | 0.22 | 0.22 | 0.08 | 0.46 | 0.12 |
| bp_05 | L | 0.43 | 0.21 | 0.04 | 0.21 | 0.38 |
| ec_08 | F | 0.33 | 0.20 | 0.05 | 0.50 | 0.05 |
| ec_19 | F | 0.32 | 0.19 | 0.09 | 0.43 | 0.04 |
| bp_15 | L | 0.36 | 0.15 | 0.04 | 0.29 | 0.12 |
| jr_14 *(fee disclosed legitimately (gig onboarding))* | L | 0.27 | 0.15 | 0.00 | 0.29 | 0.17 |
| bp_08 | F | 0.26 | 0.12 | 0.09 | 0.04 | 0.22 |
| jr_17 | L | 0.19 | 0.10 | 0.14 | 0.14 | 0.00 |

## 2. Keyword-confound legit families vs other legit families (FULL, false-alarm rate)

| model | confound legit (ec_08, jr_14, jr_08) | other legit | gap |
|---|---:|---:|---:|
| danube2-1.8b | 0.29 (21/72) | 0.52 (102/198) | -0.22 |
| granite-1b | 0.51 (37/72) | 0.40 (79/198) | +0.11 |
| qwen2-0.5b | 0.31 (22/72) | 0.54 (107/198) | -0.23 |
| qwen2-1.5b | 0.26 (19/72) | 0.30 (59/198) | -0.03 |
| qwen2.5-0.5b | 0.42 (30/72) | 0.42 (84/198) | -0.01 |
| qwen2.5-1.5b | 0.25 (18/72) | 0.25 (49/198) | +0.00 |
| qwen3-0.6b | 0.36 (26/72) | 0.31 (61/198) | +0.05 |
| qwen3-1.7b | 0.43 (31/72) | 0.33 (66/198) | +0.10 |
| smollm2-1.7b | 0.43 (31/72) | 0.51 (101/198) | -0.08 |
| smollm2-360m | 0.35 (25/72) | 0.54 (107/198) | -0.19 |
| stablelm2-1.6b | 0.18 (13/72) | 0.36 (71/198) | -0.18 |

## 3. Surface-keyword shortcut on LEGIT chats (FULL, false-alarm rate)

Keyword searched in the caller's lines only. Positive gap = the word alone pushes a genuine call toward 'fraud'.

| keyword | legit chats with it | qwen2.5-1.5b with / without | qwen3-1.7b with / without | qwen2-1.5b with / without |
|---|---:|---|---|---|
| otp | 81 | 0.30 / 0.23 | 0.37 / 0.35 | 0.33 / 0.27 |
| pin/cvv | 76 | 0.21 / 0.26 | 0.28 / 0.39 | 0.21 / 0.32 |
| link | 29 | 0.10 / 0.27 | 0.24 / 0.37 | 0.03 / 0.32 |
| upi | 44 | 0.32 / 0.23 | 0.43 / 0.35 | 0.32 / 0.28 |
| fee/payment | 199 | 0.23 / 0.30 | 0.40 / 0.25 | 0.25 / 0.39 |
| account/ifsc | 115 | 0.20 / 0.28 | 0.41 / 0.32 | 0.30 / 0.28 |
| app/install | 175 | 0.20 / 0.34 | 0.31 / 0.45 | 0.34 / 0.20 |
| kyc | 0 | — / 0.25 | — / 0.36 | — / 0.29 |

## 4. Legit false alarms by language (FULL, strong models pooled)

| family | hi_en | ta_en | te_en |
|---|---:|---:|---:|
| bp_05 | 0.12 | 0.42 | 0.08 |
| bp_08 | 0.21 | 0.67 | 0.58 |
| bp_14 | 0.17 | 0.71 | 0.00 |
| bp_15 | 0.12 | 0.25 | 0.08 |
| ec_08 | 0.21 | 0.79 | 0.71 |
| ec_10 | 0.38 | 0.78 | 0.48 |
| ec_18 | 0.38 | 0.04 | 0.25 |
| ec_19 | 0.33 | 0.25 | 0.25 |
| jr_03 | 0.12 | 0.42 | 0.29 |
| jr_08 | 0.00 | 0.50 | 0.17 |
| jr_14 | 0.04 | 0.29 | 0.12 |
| jr_17 | 0.12 | 0.11 | 0.00 |
| **all** | 0.18 | 0.45 | 0.28 |

## 5. Verdict flips when the recipient's first line is added

Share of chats whose verdict changes from PREFIX_ONLY to PREFIX_PLUS_REACTION.
Fraud chats: flip = fraud -> legitimate (victim reaction clears the caller).
Legit chats: flip = legitimate -> fraud (victim reaction creates a false alarm).

| model | label | cooperative | clarifying | resistant | suspicious |
|---|---|---:|---:|---:|---:|
| danube2-1.8b | fraud | 0.39 (18/46) | 0.29 (13/45) | 0.30 (14/47) | 0.13 (6/47) |
| danube2-1.8b | legit | 0.17 (4/23) | 0.26 (6/23) | 0.14 (3/22) | 0.61 (14/23) |
| granite-1b | fraud | 0.58 (31/53) | 0.32 (17/53) | 0.77 (44/57) | 0.62 (35/56) |
| granite-1b | legit | 0.00 (0/20) | 0.58 (11/19) | 0.22 (4/18) | 0.26 (5/19) |
| qwen2-0.5b | fraud | 0.06 (2/32) | 0.15 (5/33) | 0.29 (10/35) | 0.38 (13/34) |
| qwen2-0.5b | legit | 0.14 (4/29) | 0.21 (6/29) | 0.18 (5/28) | 0.14 (4/29) |
| qwen2-1.5b | fraud | 0.24 (12/49) | 0.12 (6/52) | 0.33 (18/54) | 0.34 (18/53) |
| qwen2-1.5b | legit | 0.06 (3/52) | 0.12 (6/51) | 0.06 (3/49) | 0.22 (11/51) |
| qwen2.5-0.5b | fraud | 0.18 (7/38) | 0.29 (11/38) | 0.22 (9/41) | 0.32 (13/41) |
| qwen2.5-0.5b | legit | 0.00 (0/35) | 0.00 (0/35) | 0.06 (2/33) | 0.11 (4/35) |
| qwen2.5-1.5b | fraud | 0.29 (15/52) | 0.06 (3/53) | 0.23 (13/56) | 0.25 (14/56) |
| qwen2.5-1.5b | legit | 0.05 (2/39) | 0.10 (4/39) | 0.06 (2/36) | 0.54 (21/39) |
| qwen3-0.6b | fraud | 0.31 (9/29) | 0.46 (12/26) | 0.32 (9/28) | 0.59 (17/29) |
| qwen3-0.6b | legit | 0.30 (16/53) | 0.13 (7/54) | 0.32 (17/53) | 0.17 (9/54) |
| qwen3-1.7b | fraud | 0.12 (6/49) | 0.21 (10/48) | 0.25 (13/52) | 0.13 (7/52) |
| qwen3-1.7b | legit | 0.00 (0/38) | 0.05 (2/38) | 0.03 (1/36) | 0.32 (12/38) |
| smollm2-1.7b | fraud | 0.29 (15/52) | 0.21 (11/52) | 0.18 (10/56) | 0.20 (11/55) |
| smollm2-1.7b | legit | 0.08 (1/12) | 0.00 (0/12) | 0.27 (3/11) | 0.08 (1/12) |
| smollm2-360m | fraud | 0.24 (7/29) | 0.22 (7/32) | 0.14 (5/35) | 0.18 (6/34) |
| smollm2-360m | legit | 0.26 (9/35) | 0.29 (10/34) | 0.44 (15/34) | 0.31 (11/35) |
| stablelm2-1.6b | fraud | 0.53 (25/47) | 0.35 (16/46) | 0.53 (26/49) | 0.46 (22/48) |
| stablelm2-1.6b | legit | 0.00 (0/23) | 0.21 (5/24) | 0.00 (0/21) | 0.00 (0/24) |

