# Example four-branch sets (TEST split)

Selected for the dataset-documentation supplement. Scores are P(fraud).
Decision rule: `score >= thresholds[model][view]` from the corresponding `metrics.json`.

- Zero-shot model: `qwen2.5-1.5b` / Qwen2.5-1.5B-Instruct
  - thresholds: PREFIX_PLUS_REACTION=0.20181322226037882, FULL=0.20181322226037882
- Fine-tuned: `qwen2.5-1.5b+branched-seed1` from `results/eval_hpc_ft_rest/`
  - FULL threshold: 0.34864513533394575

## (a) `ec_08_L_hi_en_h1` — Genuine courier call (Hindi--English)

Chosen because it is the paper's running genuine-courier example: zero-shot Qwen2.5-1.5B flags only the suspicious branch after the first reply (PREFIX_PLUS_REACTION), while cooperative/clarifying/resistant stay legitimate.

| branch | zs open+reply | zs full | ft full |
|---|---:|---:|---:|
| cooperative | 0.0110 (ok) | 0.0076 (ok) | 0.0010 (ok) |
| clarifying | 0.0180 (ok) | 0.0534 (ok) | 0.0028 (ok) |
| resistant | 0.0019 (ok) | 0.0012 (ok) | 0.0004 (ok) |
| suspicious | 0.9797 (WRONG) | 0.0097 (ok) | 0.0003 (ok) |

LaTeX: `examples/ec_08_L_hi_en_h1.tex`

## (b) `bp_05_F_te_en_h1` — Fraud call where cooperation hides the scam (zero-shot)

Chosen as a TEST fraud set in Tamil-- or Telugu--English where zero-shot Qwen2.5-1.5B misses the cooperative branch on the full call but catches the suspicious branch (largest score gap among matching candidates).

| branch | zs open+reply | zs full | ft full |
|---|---:|---:|---:|
| cooperative | 0.0851 (WRONG) | 0.0046 (WRONG) | 0.7982 (ok) |
| clarifying | 0.9740 (ok) | 0.9841 (ok) | 0.9841 (ok) |
| resistant | 0.1480 (WRONG) | 0.5312 (ok) | 0.9985 (ok) |
| suspicious | 0.0953 (WRONG) | 0.9903 (ok) | 0.9990 (ok) |

LaTeX: `examples/bp_05_F_te_en_h1.tex`

## (c) `bp_08_L_te_en_h1` — Genuine call with a zero-shot false alarm

Chosen as a TEST genuine set in Tamil-- or Telugu--English with at least one zero-shot false alarm on PREFIX_PLUS_REACTION or FULL.

| branch | zs open+reply | zs full | ft full |
|---|---:|---:|---:|
| cooperative | 0.6225 (WRONG) | 0.0953 (ok) | 0.0046 (ok) |
| clarifying | 0.5927 (WRONG) | 0.2227 (WRONG) | 0.0046 (ok) |
| resistant | 0.0953 (ok) | 0.3208 (WRONG) | 0.0025 (ok) |
| suspicious | 0.9997 (WRONG) | 0.7058 (WRONG) | 0.0110 (ok) |

LaTeX: `examples/bp_08_L_te_en_h1.tex`

