# Supplement dataset statistics report

Usable conversations: **2643** (from `runs/full_v1/FROZEN.json`).
Complete four-branch sets: **597** (train 356 / dev 119 / test 122).

## Tagging method (shared with generator checks)

Word tokens via `scambranch.checks.WORD_RE` (`[A-Za-z]+(?:'[A-Za-z]+)?`). Tag **en** if `wordfreq.zipf_frequency(w, "en") >= 3.5`, else **local** if in `render.local_lexicon(profile)`, else **untagged**. English share = `n_en / (n_en + n_local)`. CMI (Gambäck & Das 2016) = `100 * (1 - max(n_lang) / (n - n_u))` with `n_u` = untagged. Switch points = adjacent tagged-label changes, normalized by `n_tagged`. Recomputed call-level english_share matches stored generator checks exactly (max abs diff = 0).

**Reference corpora.** Full mined pools: `data/calibration/processed/{hi_en,ta_en,te_en}/utterances.jsonl` (n = 45,465 / 235,986 / 52,684). The **200 reference utterances** are `reference_utterances` in each `configs/language_profiles/{pair}.yaml` (service-oriented calibration subset used in prompts).

Re-run: `scripts/analysis/supp_dataset.py` → `results/supplement/dataset/`.

## 1. Size and structure

- Overall turns/call: mean 9.89, median 9, IQR 2.0.
- Words/call mean 187.4; words/turn mean 18.95.
- Opening turns mean 6.99; post-branch turns mean 2.90.
- Per-language usable n: Hindi–English 889, Tamil–English 878, Telugu–English 876.
- Tables: `tables/size_counts.tex`, `tables/lengths.tex`, `tables/complete_sets.tex`.

## 2. Code-mixing

- **Hindi–English**: synth EN-share median 0.69, CMI median 30.0, mono share 0.06; vs full ref KS(EN)=0.39, W1=0.12.
- **Tamil–English**: synth EN-share median 0.74, CMI median 25.0, mono share 0.11; vs full ref KS(EN)=0.17, W1=0.07.
- **Telugu–English**: synth EN-share median 0.75, CMI median 25.0, mono share 0.09; vs full ref KS(EN)=0.28, W1=0.10.
- **Target vs realized EN share is essentially uncorrelated**: Pearson r=0.005, Spearman ρ=0.011 (n=2643). Band checks only enforce p10–p90, not the sampled target.
- Figures: `figures/cmi_en_share_dist.pdf`, `figures/target_vs_realized.pdf`. Tables: `tables/code_mixing.tex`, `tables/target_realized.tex`.

### Caveats
- Auto-tagging leaves many Romanized content words as `unk` (excluded from share/CMI denominators).
- Full reference pools differ in genre (tweets/QA/dialogs vs phone calls); the 200 profile refs are a service-biased subset.

## 3. Lexical diversity

TTR on contiguous 10k-token windows (mean of 20 random starts). MATTR window 50. Zipf: OLS of log freq on log rank for ranks 1..min(1000, V). Heaps: OLS of log V on log N.

- **Hindi–English**: synth V=2573, TTR10k=0.067, MATTR=0.814, Zipf slope=−1.090; ref V=60,705, TTR10k=0.357, Zipf=−0.934.
- **Tamil–English**: synth V=3245, TTR10k=0.080, MATTR=0.817, Zipf=−1.001; ref V=234,767, TTR10k=0.399, Zipf=−0.820.
- **Telugu–English**: synth V=3329, TTR10k=0.082, MATTR=0.831, Zipf=−1.001; ref V=70,855, TTR10k=0.433, Zipf=−0.798.
- Synthetic vocabulary is much smaller and less diverse (lower TTR10k) than the large reference pools — expected for a closed scenario set.
- Figure: `figures/zipf.pdf`. Table: `tables/lexical_diversity.tex`.

## 4. Distributional similarity

Reference sources are tweets, QA, and service dialogues — not phone-call transcripts. JSD / OOV / bits-per-char are **upper-bound-of-mismatch** indicators, not naturalness scores. Char LM: interpolated 3–5-gram, add-k=0.01, trained on 90% of reference utterances; bits/char on 10% holdout vs synthetic.

- **Hindi–English**: JSD uni 0.58, bi 0.91, OOV 0.04, BPC holdout 2.84, BPC synth 3.28.
- **Tamil–English**: JSD uni 0.65, bi 0.95, OOV 0.05, BPC holdout 2.42, BPC synth 3.73.
- **Telugu–English**: JSD uni 0.65, bi 0.96, OOV 0.09, BPC holdout 2.66, BPC synth 3.25.
- Low OOV with high JSD: synthetic reuses common types but with very different n-gram distributions (genre + closed scenarios).
- Table: `tables/distributional.tex`.

## 5. Romanization variation

Clusters from profile `romanization_variants` with `kind=corpus_cluster` and ≥2 variants. Share of attested clusters with >1 variant; mean Shannon entropy over variant counts.

- **Hindi–English**: synth multi-variant 0.09, H=0.03; ref multi 1.00, H=0.44.
- **Tamil–English**: synth multi-variant 0.17, H=0.09; ref multi 1.00, H=0.82.
- **Telugu–English**: synth multi-variant 0.28, H=0.16; ref multi 1.00, H=0.91.
- Synthetic spelling is much more canonical than the large human pools (reference multi≈1.0 is expected at that scale).
- Table: `tables/romanization.tex`.

## 6. Branch analysis

- Complete sets used: 597.
- First-reply length (words): Cooperative mean 16.8; Clarifying 24.7; Resistant 19.1; Suspicious 36.6.
- Pairwise post-branch Jaccard: unigram mean 0.193, bigram mean 0.042 (continuations differ substantially).
- Weighted log-odds (Monroe et al. 2008) for first replies by condition and fraud vs genuine openings: `tables/logodds_condition.tex`, `tables/logodds_label.tex`.
- Figure: `figures/first_reply_length.pdf`. Also `tables/first_reply_length.tex`, `tables/branch_jaccard.tex`.

## 7. Lexical shortcut probes

TF–IDF word 1–2grams + logistic regression (liblinear for binary; lbfgs for 4-way condition; `class_weight=balanced`). C chosen on dev by macro-F1. Test macro-F1 with 95% bootstrap CI over 2000 resamples of the 12 test scenario families with replacement (seed 42).

| Probe | Test macro-F1 | 95% CI | Dev F1 | C |
|---|---:|---|---:|---:|
| Opening only | 0.80 | [.70, .89] | 0.88 | 0.1 |
| First reply | 0.88 | [.84, .93] | 0.92 | 2.0 |
| Recipient turns | 0.89 | [.84, .94] | 0.94 | 8.0 |
| Caller turns | 0.90 | [.85, .95] | 0.96 | 8.0 |
| Full call | 0.91 | [.85, .96] | 0.97 | 0.5 |
| Condition from first reply | 0.96 | [.93, .98] | 0.96 | 1.0 |

### Interpretation (what the numbers say)
- **Condition probe is very high (0.96):** recipient conditions are strongly lexically realized in first replies.
- **Opening-only label probe is strong (0.80):** fraud vs genuine openings are lexically separable.
- **SURPRISING: first-reply / recipient-only label probes are not near chance (0.88 / 0.89).** Victim wording alone still predicts the label on this family-held-out test split. Do **not** claim that victim words are label-safe; this is a lexical shortcut risk.
- Full call / caller-only are highest (0.91 / 0.90).
- Table: `tables/lexical_probes.tex`.

## Anomalies / flags

1. Target vs realized English share ≈ uncorrelated (r≈0).
2. First-reply and recipient-only **label** probes are strong, contrary to the hoped-for near-chance pattern.
3. Large EN-share distribution shift vs reference for Hindi–English (KS=0.39) and Telugu–English (KS=0.28).
4. Synthetic romanization much less variant-diverse than reference pools.
5. High JSD with low OOV: type overlap without distributional match (genre/scenario mismatch).
