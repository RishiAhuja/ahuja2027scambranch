# Detector eval `eval_hpc_gated`

- run_dir: `runs/full_v1`
- n_dev_scored: 540
- n_test_scored: 540
- views: FULL, PREFIX_ONLY, PREFIX_PLUS_REACTION, STREAMING
- models: llama-3.2-1b, gemma3-270m, gemma3-1b

## Thresholds (fit on DEV, max macro-F1)

| model | view | threshold |
|---|---|---|
| llama-3.2-1b | FULL | 0.7400 |
| llama-3.2-1b | PREFIX_ONLY | 0.7100 |
| llama-3.2-1b | PREFIX_PLUS_REACTION | 0.7800 |
| gemma3-270m | FULL | 1.0000 |
| gemma3-270m | PREFIX_ONLY | 1.0000 |
| gemma3-270m | PREFIX_PLUS_REACTION | 1.0000 |
| gemma3-1b | FULL | 0.9995 |
| gemma3-1b | PREFIX_ONLY | 0.9978 |
| gemma3-1b | PREFIX_PLUS_REACTION | 1.0000 |

## Test quality (pooled)

| model | view | n | macro-F1 | fraud recall | legit FPR | ROC-AUC | invalid | worst-cond F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gemma3-1b | FULL | 540 | 0.621 [0.568, 0.665] | 0.681 [0.624, 0.737] | 0.437 [0.364, 0.531] | 0.653 | 0.033 | 0.546 |
| gemma3-1b | PREFIX_ONLY | 540 | 0.536 [0.440, 0.641] | 0.659 [0.435, 0.848] | 0.574 [0.410, 0.708] | 0.588 | 0.015 | 0.534 |
| gemma3-1b | PREFIX_PLUS_REACTION | 540 | 0.596 [0.528, 0.662] | 0.719 [0.585, 0.848] | 0.515 [0.426, 0.594] | 0.652 | 0.009 | 0.465 |
| gemma3-1b | STREAMING | 0 | — | — | — | — | — | — |
| gemma3-270m | FULL | 540 | 0.480 [0.396, 0.558] | 0.481 [0.383, 0.585] | 0.522 [0.423, 0.632] | 0.489 | 0.044 | 0.423 |
| gemma3-270m | PREFIX_ONLY | 540 | 0.516 [0.444, 0.590] | 0.396 [0.260, 0.548] | 0.348 [0.229, 0.463] | 0.540 | 0.000 | 0.505 |
| gemma3-270m | PREFIX_PLUS_REACTION | 540 | 0.531 [0.450, 0.618] | 0.700 [0.580, 0.804] | 0.615 [0.488, 0.744] | 0.549 | 0.000 | 0.485 |
| gemma3-270m | STREAMING | 0 | — | — | — | — | — | — |
| llama-3.2-1b | FULL | 540 | 0.592 [0.501, 0.671] | 0.615 [0.463, 0.751] | 0.430 [0.234, 0.612] | 0.634 | 0.000 | 0.533 |
| llama-3.2-1b | PREFIX_ONLY | 540 | 0.487 [0.380, 0.587] | 0.422 [0.249, 0.582] | 0.444 [0.169, 0.692] | 0.533 | 0.007 | 0.478 |
| llama-3.2-1b | PREFIX_PLUS_REACTION | 540 | 0.507 [0.436, 0.560] | 0.415 [0.252, 0.580] | 0.393 [0.180, 0.591] | 0.549 | 0.000 | 0.476 |
| llama-3.2-1b | STREAMING | 0 | — | — | — | — | — | — |

## Per language (test, pooled views listed separately)

| model | view | language | n | macro-F1 | fraud recall | legit FPR |
|---|---|---|---:|---:|---:|---:|
| gemma3-1b | FULL | hi_en | 188 | 0.723 | 0.699 | 0.253 |
| gemma3-1b | FULL | ta_en | 177 | 0.572 | 0.784 | 0.607 |
| gemma3-1b | FULL | te_en | 175 | 0.548 | 0.562 | 0.465 |
| gemma3-1b | PREFIX_ONLY | hi_en | 188 | 0.647 | 0.516 | 0.211 |
| gemma3-1b | PREFIX_ONLY | ta_en | 177 | 0.408 | 0.727 | 0.820 |
| gemma3-1b | PREFIX_ONLY | te_en | 175 | 0.485 | 0.742 | 0.721 |
| gemma3-1b | PREFIX_PLUS_REACTION | hi_en | 188 | 0.702 | 0.667 | 0.263 |
| gemma3-1b | PREFIX_PLUS_REACTION | ta_en | 177 | 0.570 | 0.830 | 0.640 |
| gemma3-1b | PREFIX_PLUS_REACTION | te_en | 175 | 0.488 | 0.663 | 0.663 |
| gemma3-270m | FULL | hi_en | 188 | 0.476 | 0.720 | 0.716 |
| gemma3-270m | FULL | ta_en | 177 | 0.439 | 0.386 | 0.506 |
| gemma3-270m | FULL | te_en | 175 | 0.483 | 0.326 | 0.326 |
| gemma3-270m | PREFIX_ONLY | hi_en | 188 | 0.518 | 0.806 | 0.705 |
| gemma3-270m | PREFIX_ONLY | ta_en | 177 | 0.492 | 0.216 | 0.124 |
| gemma3-270m | PREFIX_ONLY | te_en | 175 | 0.412 | 0.146 | 0.186 |
| gemma3-270m | PREFIX_PLUS_REACTION | hi_en | 188 | 0.443 | 0.925 | 0.863 |
| gemma3-270m | PREFIX_PLUS_REACTION | ta_en | 177 | 0.522 | 0.670 | 0.607 |
| gemma3-270m | PREFIX_PLUS_REACTION | te_en | 175 | 0.569 | 0.494 | 0.349 |
| llama-3.2-1b | FULL | hi_en | 188 | 0.640 | 0.860 | 0.547 |
| llama-3.2-1b | FULL | ta_en | 177 | 0.546 | 0.477 | 0.382 |
| llama-3.2-1b | FULL | te_en | 175 | 0.569 | 0.494 | 0.349 |
| llama-3.2-1b | PREFIX_ONLY | hi_en | 188 | 0.587 | 0.688 | 0.505 |
| llama-3.2-1b | PREFIX_ONLY | ta_en | 177 | 0.392 | 0.250 | 0.438 |
| llama-3.2-1b | PREFIX_ONLY | te_en | 175 | 0.452 | 0.315 | 0.384 |
| llama-3.2-1b | PREFIX_PLUS_REACTION | hi_en | 188 | 0.619 | 0.720 | 0.474 |
| llama-3.2-1b | PREFIX_PLUS_REACTION | ta_en | 177 | 0.417 | 0.239 | 0.360 |
| llama-3.2-1b | PREFIX_PLUS_REACTION | te_en | 175 | 0.443 | 0.270 | 0.337 |

## Per recipient condition (test)

| model | view | condition | n | fraud recall | legit FPR | macro-F1 |
|---|---|---|---:|---:|---:|---:|
| gemma3-1b | FULL | cooperative | 133 | 0.415 | 0.309 | 0.546 |
| gemma3-1b | FULL | clarifying | 134 | 0.576 | 0.397 | 0.589 |
| gemma3-1b | FULL | resistant | 136 | 0.800 | 0.364 | 0.718 |
| gemma3-1b | FULL | suspicious | 137 | 0.913 | 0.676 | 0.583 |
| gemma3-1b | PREFIX_ONLY | cooperative | 133 | 0.662 | 0.574 | 0.536 |
| gemma3-1b | PREFIX_ONLY | clarifying | 134 | 0.667 | 0.574 | 0.539 |
| gemma3-1b | PREFIX_ONLY | resistant | 136 | 0.657 | 0.576 | 0.536 |
| gemma3-1b | PREFIX_ONLY | suspicious | 137 | 0.652 | 0.574 | 0.534 |
| gemma3-1b | PREFIX_PLUS_REACTION | cooperative | 133 | 0.523 | 0.353 | 0.584 |
| gemma3-1b | PREFIX_PLUS_REACTION | clarifying | 134 | 0.742 | 0.426 | 0.655 |
| gemma3-1b | PREFIX_PLUS_REACTION | resistant | 136 | 0.643 | 0.424 | 0.609 |
| gemma3-1b | PREFIX_PLUS_REACTION | suspicious | 137 | 0.957 | 0.853 | 0.465 |
| gemma3-270m | FULL | cooperative | 133 | 0.754 | 0.603 | 0.559 |
| gemma3-270m | FULL | clarifying | 134 | 0.227 | 0.309 | 0.430 |
| gemma3-270m | FULL | resistant | 136 | 0.500 | 0.576 | 0.462 |
| gemma3-270m | FULL | suspicious | 137 | 0.449 | 0.603 | 0.423 |
| gemma3-270m | PREFIX_ONLY | cooperative | 133 | 0.415 | 0.338 | 0.533 |
| gemma3-270m | PREFIX_ONLY | clarifying | 134 | 0.379 | 0.353 | 0.505 |
| gemma3-270m | PREFIX_ONLY | resistant | 136 | 0.400 | 0.348 | 0.516 |
| gemma3-270m | PREFIX_ONLY | suspicious | 137 | 0.391 | 0.353 | 0.511 |
| gemma3-270m | PREFIX_PLUS_REACTION | cooperative | 133 | 0.800 | 0.544 | 0.614 |
| gemma3-270m | PREFIX_PLUS_REACTION | clarifying | 134 | 0.621 | 0.632 | 0.485 |
| gemma3-270m | PREFIX_PLUS_REACTION | resistant | 136 | 0.729 | 0.636 | 0.533 |
| gemma3-270m | PREFIX_PLUS_REACTION | suspicious | 137 | 0.652 | 0.647 | 0.492 |
| llama-3.2-1b | FULL | cooperative | 133 | 0.615 | 0.353 | 0.631 |
| llama-3.2-1b | FULL | clarifying | 134 | 0.742 | 0.500 | 0.615 |
| llama-3.2-1b | FULL | resistant | 136 | 0.571 | 0.394 | 0.588 |
| llama-3.2-1b | FULL | suspicious | 137 | 0.536 | 0.471 | 0.533 |
| llama-3.2-1b | PREFIX_ONLY | cooperative | 133 | 0.415 | 0.456 | 0.478 |
| llama-3.2-1b | PREFIX_ONLY | clarifying | 134 | 0.424 | 0.441 | 0.490 |
| llama-3.2-1b | PREFIX_ONLY | resistant | 136 | 0.429 | 0.439 | 0.491 |
| llama-3.2-1b | PREFIX_ONLY | suspicious | 137 | 0.420 | 0.441 | 0.487 |
| llama-3.2-1b | PREFIX_PLUS_REACTION | cooperative | 133 | 0.338 | 0.368 | 0.476 |
| llama-3.2-1b | PREFIX_PLUS_REACTION | clarifying | 134 | 0.455 | 0.441 | 0.506 |
| llama-3.2-1b | PREFIX_PLUS_REACTION | resistant | 136 | 0.471 | 0.348 | 0.556 |
| llama-3.2-1b | PREFIX_PLUS_REACTION | suspicious | 137 | 0.391 | 0.412 | 0.484 |

## Branch metrics (complete sibling sets, test)

| model | view | n histories | disagreement | score spread (mean) | all-correct |
|---|---|---:|---:|---:|---:|
| gemma3-1b | FULL | 122 | 0.770 | 0.594 | 0.172 |
| gemma3-1b | PREFIX_ONLY | 122 | 0.000 | 0.000 | 0.541 |
| gemma3-1b | PREFIX_PLUS_REACTION | 122 | 0.607 | 0.277 | 0.262 |
| gemma3-1b | STREAMING | 0 | — | — | — |
| gemma3-270m | FULL | 122 | 0.664 | 0.000 | 0.172 |
| gemma3-270m | PREFIX_ONLY | 122 | 0.000 | 0.000 | 0.533 |
| gemma3-270m | PREFIX_PLUS_REACTION | 122 | 0.336 | 0.000 | 0.377 |
| gemma3-270m | STREAMING | 0 | — | — | — |
| llama-3.2-1b | FULL | 122 | 0.492 | 0.190 | 0.344 |
| llama-3.2-1b | PREFIX_ONLY | 122 | 0.000 | 0.001 | 0.484 |
| llama-3.2-1b | PREFIX_PLUS_REACTION | 122 | 0.295 | 0.137 | 0.352 |
| llama-3.2-1b | STREAMING | 0 | — | — | — |

## PREFIX_PLUS_REACTION − PREFIX_ONLY (mean Δ P(fraud), test)

| model | condition / label | mean Δ | 95% CI |
|---|---|---:|---|
| gemma3-1b | all | 0.115 | |
| gemma3-1b | cooperative | 0.013 | |
| gemma3-1b | clarifying | 0.066 | |
| gemma3-1b | resistant | 0.155 | |
| gemma3-1b | suspicious | 0.222 | |
| gemma3-1b | label=fraud | 0.120 | |
| gemma3-1b | label=legitimate | 0.110 | |
| gemma3-1b | cooperative|fraud | 0.047 |  [-0.013, 0.141] |
| gemma3-1b | cooperative|legitimate | -0.020 |  [-0.081, 0.054] |
| gemma3-1b | clarifying|fraud | 0.065 |  [0.014, 0.123] |
| gemma3-1b | clarifying|legitimate | 0.066 |  [-0.067, 0.207] |
| gemma3-1b | resistant|fraud | 0.167 |  [0.040, 0.320] |
| gemma3-1b | resistant|legitimate | 0.142 |  [0.019, 0.288] |
| gemma3-1b | suspicious|fraud | 0.193 |  [0.058, 0.341] |
| gemma3-1b | suspicious|legitimate | 0.253 |  [0.144, 0.381] |
| gemma3-270m | all | -0.000 | |
| gemma3-270m | cooperative | -0.000 | |
| gemma3-270m | clarifying | -0.000 | |
| gemma3-270m | resistant | 0.000 | |
| gemma3-270m | suspicious | -0.000 | |
| gemma3-270m | label=fraud | -0.000 | |
| gemma3-270m | label=legitimate | -0.000 | |
| gemma3-270m | cooperative|fraud | 0.000 |  [-0.000, 0.000] |
| gemma3-270m | cooperative|legitimate | -0.000 |  [-0.000, 0.000] |
| gemma3-270m | clarifying|fraud | -0.000 |  [-0.000, 0.000] |
| gemma3-270m | clarifying|legitimate | -0.000 |  [-0.000, 0.000] |
| gemma3-270m | resistant|fraud | -0.000 |  [-0.000, 0.000] |
| gemma3-270m | resistant|legitimate | 0.000 |  [-0.000, 0.000] |
| gemma3-270m | suspicious|fraud | -0.000 |  [-0.000, 0.000] |
| gemma3-270m | suspicious|legitimate | -0.000 |  [-0.000, 0.000] |
| llama-3.2-1b | all | 0.052 | |
| llama-3.2-1b | cooperative | 0.018 | |
| llama-3.2-1b | clarifying | 0.088 | |
| llama-3.2-1b | resistant | 0.044 | |
| llama-3.2-1b | suspicious | 0.060 | |
| llama-3.2-1b | label=fraud | 0.054 | |
| llama-3.2-1b | label=legitimate | 0.051 | |
| llama-3.2-1b | cooperative|fraud | 0.024 |  [0.000, 0.046] |
| llama-3.2-1b | cooperative|legitimate | 0.012 |  [-0.017, 0.048] |
| llama-3.2-1b | clarifying|fraud | 0.096 |  [0.072, 0.119] |
| llama-3.2-1b | clarifying|legitimate | 0.080 |  [0.038, 0.126] |
| llama-3.2-1b | resistant|fraud | 0.047 |  [0.023, 0.070] |
| llama-3.2-1b | resistant|legitimate | 0.040 |  [0.018, 0.065] |
| llama-3.2-1b | suspicious|fraud | 0.050 |  [0.018, 0.080] |
| llama-3.2-1b | suspicious|legitimate | 0.070 |  [0.031, 0.114] |

## Complete-set slice (test)

| model | view | n | macro-F1 | disagreement | all-correct |
|---|---|---:|---:|---:|---:|
| gemma3-1b | FULL | 488 | 0.629 | 0.770 | 0.172 |
| gemma3-1b | PREFIX_ONLY | 488 | 0.538 | 0.000 | 0.541 |
| gemma3-1b | PREFIX_PLUS_REACTION | 488 | 0.602 | 0.607 | 0.262 |
| gemma3-1b | STREAMING | 0 | — | — | — |
| gemma3-270m | FULL | 488 | 0.484 | 0.664 | 0.172 |
| gemma3-270m | PREFIX_ONLY | 488 | 0.524 | 0.000 | 0.533 |
| gemma3-270m | PREFIX_PLUS_REACTION | 488 | 0.526 | 0.336 | 0.377 |
| gemma3-270m | STREAMING | 0 | — | — | — |
| llama-3.2-1b | FULL | 488 | 0.584 | 0.492 | 0.344 |
| llama-3.2-1b | PREFIX_ONLY | 488 | 0.481 | 0.000 | 0.484 |
| llama-3.2-1b | PREFIX_PLUS_REACTION | 488 | 0.504 | 0.295 | 0.352 |
| llama-3.2-1b | STREAMING | 0 | — | — | — |

## Streaming delay (test; misses counted)

| model | eligible | detected | miss rate | mean delay (hits) | pre-evidence alert rate |
|---|---:|---:|---:|---:|---:|
| gemma3-1b | 0 | 0 | — | — | — |
| gemma3-270m | 0 | 0 | — | — | — |
| llama-3.2-1b | 0 | 0 | — | — | — |

## Runtime projection (full panel)

- median per-inference: 0.028 s
- mean turns: 9.94
- assumed test+dev conversations: 540+540
- one model, 4 views: 0.11 h
- 5 models × 4 views: 2.09 h
- note: STREAMING costs one inference per turn; the other three views cost one each. Naive 4×views undercounts STREAMING.
