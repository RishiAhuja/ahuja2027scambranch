# Detector eval `eval_hpc_caller_only`

- run_dir: `runs/full_v1`
- n_dev_scored: 540
- n_test_scored: 540
- views: FULL, CALLER_ONLY
- models: qwen3-0.6b, qwen3-1.7b, qwen2.5-1.5b, smollm2-1.7b, qwen2.5-0.5b, qwen2-0.5b, qwen2-1.5b, smollm2-135m, smollm2-360m, tinyllama-1.1b, stablelm2-1.6b, danube2-1.8b, granite-1b, llama-3.2-1b, gemma3-270m, gemma3-1b

## Thresholds (fit on DEV, max macro-F1)

| model | view | threshold |
|---|---|---|
| qwen3-0.6b | FULL | 0.9195 |
| qwen3-0.6b | CALLER_ONLY | 0.9648 |
| qwen3-1.7b | FULL | 1.0000 |
| qwen3-1.7b | CALLER_ONLY | 1.0000 |
| qwen2.5-1.5b | FULL | 0.2018 |
| qwen2.5-1.5b | CALLER_ONLY | 0.1645 |
| smollm2-1.7b | FULL | 0.9481 |
| smollm2-1.7b | CALLER_ONLY | 0.9591 |
| qwen2.5-0.5b | FULL | 0.7100 |
| qwen2.5-0.5b | CALLER_ONLY | 0.5700 |
| qwen2-0.5b | FULL | 0.9790 |
| qwen2-0.5b | CALLER_ONLY | 0.9790 |
| qwen2-1.5b | FULL | 0.7400 |
| qwen2-1.5b | CALLER_ONLY | 0.7800 |
| smollm2-135m | FULL | 0.9496 |
| smollm2-135m | CALLER_ONLY | 0.9496 |
| smollm2-360m | FULL | 0.8376 |
| smollm2-360m | CALLER_ONLY | 0.5700 |
| tinyllama-1.1b | FULL | 0.9907 |
| tinyllama-1.1b | CALLER_ONLY | 0.9888 |
| stablelm2-1.6b | FULL | 0.8200 |
| stablelm2-1.6b | CALLER_ONLY | 0.8595 |
| danube2-1.8b | FULL | 0.0159 |
| danube2-1.8b | CALLER_ONLY | 0.0216 |
| granite-1b | FULL | 0.5400 |
| granite-1b | CALLER_ONLY | 0.5700 |
| llama-3.2-1b | FULL | 0.7800 |
| llama-3.2-1b | CALLER_ONLY | 0.7100 |
| gemma3-270m | FULL | 1.0000 |
| gemma3-270m | CALLER_ONLY | 1.0000 |
| gemma3-1b | FULL | 0.9995 |
| gemma3-1b | CALLER_ONLY | 0.9576 |

## Test quality (pooled)

| model | view | n | macro-F1 | fraud recall | legit FPR | ROC-AUC | invalid | worst-cond F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| danube2-1.8b | FULL | 540 | 0.590 [0.530, 0.641] | 0.637 [0.541, 0.739] | 0.456 [0.337, 0.571] | 0.614 | 0.017 | 0.557 |
| danube2-1.8b | CALLER_ONLY | 540 | 0.541 [0.492, 0.594] | 0.441 [0.342, 0.552] | 0.348 [0.254, 0.440] | 0.549 | 0.015 | 0.487 |
| gemma3-1b | FULL | 540 | 0.621 [0.568, 0.665] | 0.681 [0.624, 0.737] | 0.437 [0.364, 0.531] | 0.653 | 0.033 | 0.546 |
| gemma3-1b | CALLER_ONLY | 540 | 0.568 [0.509, 0.624] | 0.715 [0.612, 0.817] | 0.563 [0.475, 0.648] | 0.614 | 0.030 | 0.492 |
| gemma3-270m | FULL | 540 | 0.480 [0.396, 0.558] | 0.481 [0.383, 0.585] | 0.522 [0.423, 0.632] | 0.489 | 0.044 | 0.423 |
| gemma3-270m | CALLER_ONLY | 540 | 0.487 [0.418, 0.544] | 0.467 [0.358, 0.585] | 0.493 [0.369, 0.612] | 0.497 | 0.002 | 0.416 |
| granite-1b | FULL | 540 | 0.539 [0.495, 0.586] | 0.459 [0.403, 0.516] | 0.374 [0.309, 0.449] | 0.542 | 0.000 | 0.510 |
| granite-1b | CALLER_ONLY | 540 | 0.478 [0.430, 0.525] | 0.581 [0.481, 0.664] | 0.615 [0.556, 0.689] | 0.501 | 0.000 | 0.432 |
| llama-3.2-1b | FULL | 540 | 0.594 [0.512, 0.660] | 0.578 [0.425, 0.721] | 0.389 [0.196, 0.564] | 0.629 | 0.000 | 0.547 |
| llama-3.2-1b | CALLER_ONLY | 540 | 0.511 [0.422, 0.595] | 0.381 [0.234, 0.528] | 0.341 [0.165, 0.512] | 0.554 | 0.000 | 0.458 |
| qwen2-0.5b | FULL | 540 | 0.560 [0.455, 0.658] | 0.600 [0.466, 0.735] | 0.478 [0.307, 0.649] | 0.602 | 0.041 | 0.511 |
| qwen2-0.5b | CALLER_ONLY | 540 | 0.559 [0.418, 0.679] | 0.585 [0.422, 0.739] | 0.467 [0.259, 0.679] | 0.554 | 0.019 | 0.525 |
| qwen2-1.5b | FULL | 540 | 0.691 [0.625, 0.754] | 0.670 [0.573, 0.771] | 0.289 [0.185, 0.385] | 0.758 | 0.000 | 0.660 |
| qwen2-1.5b | CALLER_ONLY | 540 | 0.690 [0.618, 0.750] | 0.789 [0.697, 0.872] | 0.404 [0.281, 0.529] | 0.783 | 0.000 | 0.654 |
| qwen2.5-0.5b | FULL | 540 | 0.574 [0.511, 0.625] | 0.570 [0.453, 0.694] | 0.422 [0.254, 0.606] | 0.599 | 0.000 | 0.492 |
| qwen2.5-0.5b | CALLER_ONLY | 540 | 0.557 [0.466, 0.634] | 0.700 [0.586, 0.798] | 0.570 [0.394, 0.740] | 0.562 | 0.000 | 0.510 |
| qwen2.5-1.5b | FULL | 540 | 0.728 [0.666, 0.783] | 0.704 [0.598, 0.817] | 0.248 [0.129, 0.393] | 0.784 | 0.000 | 0.628 |
| qwen2.5-1.5b | CALLER_ONLY | 540 | 0.656 [0.594, 0.720] | 0.563 [0.430, 0.699] | 0.244 [0.096, 0.416] | 0.743 | 0.000 | 0.577 |
| qwen3-0.6b | FULL | 540 | 0.540 [0.476, 0.601] | 0.419 [0.328, 0.509] | 0.322 [0.194, 0.453] | 0.563 | 0.000 | 0.448 |
| qwen3-0.6b | CALLER_ONLY | 540 | 0.518 [0.441, 0.594] | 0.281 [0.176, 0.392] | 0.174 [0.084, 0.285] | 0.617 | 0.000 | 0.489 |
| qwen3-1.7b | FULL | 540 | 0.683 [0.626, 0.739] | 0.726 [0.652, 0.800] | 0.359 [0.269, 0.447] | 0.759 | 0.000 | 0.649 |
| qwen3-1.7b | CALLER_ONLY | 540 | 0.637 [0.572, 0.707] | 0.626 [0.545, 0.704] | 0.352 [0.244, 0.453] | 0.689 | 0.000 | 0.582 |
| smollm2-1.7b | FULL | 540 | 0.564 [0.519, 0.603] | 0.619 [0.532, 0.707] | 0.489 [0.398, 0.581] | 0.590 | 0.000 | 0.528 |
| smollm2-1.7b | CALLER_ONLY | 540 | 0.552 [0.515, 0.596] | 0.700 [0.629, 0.768] | 0.578 [0.531, 0.622] | 0.612 | 0.000 | 0.499 |
| smollm2-135m | FULL | 540 | 0.502 [0.459, 0.540] | 0.774 [0.669, 0.863] | 0.711 [0.629, 0.793] | 0.542 | 1.000 | 0.478 |
| smollm2-135m | CALLER_ONLY | 540 | 0.563 [0.521, 0.601] | 0.748 [0.630, 0.860] | 0.596 [0.515, 0.680] | 0.606 | 1.000 | 0.511 |
| smollm2-360m | FULL | 540 | 0.540 [0.482, 0.598] | 0.570 [0.484, 0.660] | 0.489 [0.368, 0.605] | 0.553 | 0.048 | 0.474 |
| smollm2-360m | CALLER_ONLY | 540 | 0.504 [0.452, 0.561] | 0.337 [0.266, 0.414] | 0.296 [0.211, 0.376] | 0.526 | 0.011 | 0.436 |
| stablelm2-1.6b | FULL | 540 | 0.524 [0.489, 0.553] | 0.381 [0.307, 0.459] | 0.311 [0.219, 0.407] | 0.558 | 0.261 | 0.492 |
| stablelm2-1.6b | CALLER_ONLY | 540 | 0.573 [0.518, 0.625] | 0.467 [0.386, 0.551] | 0.311 [0.222, 0.416] | 0.598 | 0.554 | 0.485 |
| tinyllama-1.1b | FULL | 540 | 0.498 [0.453, 0.541] | 0.559 [0.480, 0.637] | 0.559 [0.492, 0.625] | 0.505 | 1.000 | 0.465 |
| tinyllama-1.1b | CALLER_ONLY | 540 | 0.412 [0.354, 0.457] | 0.270 [0.159, 0.393] | 0.419 [0.307, 0.526] | 0.392 | 1.000 | 0.307 |

## Per language (test, pooled views listed separately)

| model | view | language | n | macro-F1 | fraud recall | legit FPR |
|---|---|---|---:|---:|---:|---:|
| danube2-1.8b | FULL | hi_en | 188 | 0.648 | 0.699 | 0.400 |
| danube2-1.8b | FULL | ta_en | 177 | 0.501 | 0.716 | 0.674 |
| danube2-1.8b | FULL | te_en | 175 | 0.596 | 0.494 | 0.291 |
| danube2-1.8b | CALLER_ONLY | hi_en | 188 | 0.563 | 0.419 | 0.274 |
| danube2-1.8b | CALLER_ONLY | ta_en | 177 | 0.462 | 0.420 | 0.494 |
| danube2-1.8b | CALLER_ONLY | te_en | 175 | 0.595 | 0.483 | 0.279 |
| gemma3-1b | FULL | hi_en | 188 | 0.723 | 0.699 | 0.253 |
| gemma3-1b | FULL | ta_en | 177 | 0.572 | 0.784 | 0.607 |
| gemma3-1b | FULL | te_en | 175 | 0.548 | 0.562 | 0.465 |
| gemma3-1b | CALLER_ONLY | hi_en | 188 | 0.618 | 0.473 | 0.221 |
| gemma3-1b | CALLER_ONLY | ta_en | 177 | 0.525 | 0.909 | 0.753 |
| gemma3-1b | CALLER_ONLY | te_en | 175 | 0.483 | 0.775 | 0.744 |
| gemma3-270m | FULL | hi_en | 188 | 0.476 | 0.720 | 0.716 |
| gemma3-270m | FULL | ta_en | 177 | 0.439 | 0.386 | 0.506 |
| gemma3-270m | FULL | te_en | 175 | 0.483 | 0.326 | 0.326 |
| gemma3-270m | CALLER_ONLY | hi_en | 188 | 0.435 | 0.828 | 0.832 |
| gemma3-270m | CALLER_ONLY | ta_en | 177 | 0.490 | 0.318 | 0.303 |
| gemma3-270m | CALLER_ONLY | te_en | 175 | 0.430 | 0.236 | 0.314 |
| granite-1b | FULL | hi_en | 188 | 0.642 | 0.473 | 0.168 |
| granite-1b | FULL | ta_en | 177 | 0.444 | 0.511 | 0.618 |
| granite-1b | FULL | te_en | 175 | 0.513 | 0.393 | 0.349 |
| granite-1b | CALLER_ONLY | hi_en | 188 | 0.509 | 0.688 | 0.642 |
| granite-1b | CALLER_ONLY | ta_en | 177 | 0.422 | 0.625 | 0.742 |
| granite-1b | CALLER_ONLY | te_en | 175 | 0.484 | 0.427 | 0.453 |
| llama-3.2-1b | FULL | hi_en | 188 | 0.665 | 0.860 | 0.505 |
| llama-3.2-1b | FULL | ta_en | 177 | 0.536 | 0.432 | 0.348 |
| llama-3.2-1b | FULL | te_en | 175 | 0.553 | 0.427 | 0.302 |
| llama-3.2-1b | CALLER_ONLY | hi_en | 188 | 0.524 | 0.613 | 0.558 |
| llama-3.2-1b | CALLER_ONLY | ta_en | 177 | 0.510 | 0.307 | 0.236 |
| llama-3.2-1b | CALLER_ONLY | te_en | 175 | 0.454 | 0.213 | 0.209 |
| qwen2-0.5b | FULL | hi_en | 188 | 0.595 | 0.731 | 0.526 |
| qwen2-0.5b | FULL | ta_en | 177 | 0.548 | 0.534 | 0.438 |
| qwen2-0.5b | FULL | te_en | 175 | 0.531 | 0.528 | 0.465 |
| qwen2-0.5b | CALLER_ONLY | hi_en | 188 | 0.563 | 0.613 | 0.484 |
| qwen2-0.5b | CALLER_ONLY | ta_en | 177 | 0.565 | 0.591 | 0.461 |
| qwen2-0.5b | CALLER_ONLY | te_en | 175 | 0.549 | 0.551 | 0.453 |
| qwen2-1.5b | FULL | hi_en | 188 | 0.715 | 0.559 | 0.116 |
| qwen2-1.5b | FULL | ta_en | 177 | 0.591 | 0.830 | 0.607 |
| qwen2-1.5b | FULL | te_en | 175 | 0.735 | 0.629 | 0.151 |
| qwen2-1.5b | CALLER_ONLY | hi_en | 188 | 0.744 | 0.796 | 0.305 |
| qwen2-1.5b | CALLER_ONLY | ta_en | 177 | 0.603 | 0.841 | 0.596 |
| qwen2-1.5b | CALLER_ONLY | te_en | 175 | 0.708 | 0.730 | 0.314 |
| qwen2.5-0.5b | FULL | hi_en | 188 | 0.579 | 0.710 | 0.537 |
| qwen2.5-0.5b | FULL | ta_en | 177 | 0.599 | 0.580 | 0.382 |
| qwen2.5-0.5b | FULL | te_en | 175 | 0.531 | 0.416 | 0.337 |
| qwen2.5-0.5b | CALLER_ONLY | hi_en | 188 | 0.475 | 0.753 | 0.737 |
| qwen2.5-0.5b | CALLER_ONLY | ta_en | 177 | 0.600 | 0.716 | 0.506 |
| qwen2.5-0.5b | CALLER_ONLY | te_en | 175 | 0.587 | 0.629 | 0.453 |
| qwen2.5-1.5b | FULL | hi_en | 188 | 0.830 | 0.806 | 0.147 |
| qwen2.5-1.5b | FULL | ta_en | 177 | 0.655 | 0.648 | 0.337 |
| qwen2.5-1.5b | FULL | te_en | 175 | 0.691 | 0.652 | 0.267 |
| qwen2.5-1.5b | CALLER_ONLY | hi_en | 188 | 0.736 | 0.634 | 0.158 |
| qwen2.5-1.5b | CALLER_ONLY | ta_en | 177 | 0.573 | 0.443 | 0.281 |
| qwen2.5-1.5b | CALLER_ONLY | te_en | 175 | 0.651 | 0.607 | 0.302 |
| qwen3-0.6b | FULL | hi_en | 188 | 0.574 | 0.538 | 0.389 |
| qwen3-0.6b | FULL | ta_en | 177 | 0.522 | 0.364 | 0.292 |
| qwen3-0.6b | FULL | te_en | 175 | 0.516 | 0.348 | 0.279 |
| qwen3-0.6b | CALLER_ONLY | hi_en | 188 | 0.557 | 0.355 | 0.200 |
| qwen3-0.6b | CALLER_ONLY | ta_en | 177 | 0.490 | 0.250 | 0.191 |
| qwen3-0.6b | CALLER_ONLY | te_en | 175 | 0.501 | 0.236 | 0.128 |
| qwen3-1.7b | FULL | hi_en | 188 | 0.792 | 0.871 | 0.284 |
| qwen3-1.7b | FULL | ta_en | 177 | 0.605 | 0.602 | 0.393 |
| qwen3-1.7b | FULL | te_en | 175 | 0.644 | 0.697 | 0.407 |
| qwen3-1.7b | CALLER_ONLY | hi_en | 188 | 0.692 | 0.828 | 0.432 |
| qwen3-1.7b | CALLER_ONLY | ta_en | 177 | 0.691 | 0.591 | 0.202 |
| qwen3-1.7b | CALLER_ONLY | te_en | 175 | 0.513 | 0.449 | 0.419 |
| smollm2-1.7b | FULL | hi_en | 188 | 0.546 | 0.280 | 0.105 |
| smollm2-1.7b | FULL | ta_en | 177 | 0.526 | 0.830 | 0.708 |
| smollm2-1.7b | FULL | te_en | 175 | 0.516 | 0.764 | 0.686 |
| smollm2-1.7b | CALLER_ONLY | hi_en | 188 | 0.627 | 0.387 | 0.084 |
| smollm2-1.7b | CALLER_ONLY | ta_en | 177 | 0.396 | 0.977 | 0.933 |
| smollm2-1.7b | CALLER_ONLY | te_en | 175 | 0.466 | 0.753 | 0.756 |
| smollm2-135m | FULL | hi_en | 188 | 0.447 | 0.796 | 0.800 |
| smollm2-135m | FULL | ta_en | 177 | 0.351 | 0.932 | 0.966 |
| smollm2-135m | FULL | te_en | 175 | 0.623 | 0.596 | 0.349 |
| smollm2-135m | CALLER_ONLY | hi_en | 188 | 0.485 | 0.796 | 0.747 |
| smollm2-135m | CALLER_ONLY | ta_en | 177 | 0.435 | 0.875 | 0.854 |
| smollm2-135m | CALLER_ONLY | te_en | 175 | 0.699 | 0.573 | 0.163 |
| smollm2-360m | FULL | hi_en | 188 | 0.467 | 0.634 | 0.674 |
| smollm2-360m | FULL | ta_en | 177 | 0.515 | 0.625 | 0.584 |
| smollm2-360m | FULL | te_en | 175 | 0.617 | 0.449 | 0.186 |
| smollm2-360m | CALLER_ONLY | hi_en | 188 | 0.494 | 0.398 | 0.400 |
| smollm2-360m | CALLER_ONLY | ta_en | 177 | 0.535 | 0.420 | 0.337 |
| smollm2-360m | CALLER_ONLY | te_en | 175 | 0.463 | 0.191 | 0.140 |
| stablelm2-1.6b | FULL | hi_en | 188 | 0.459 | 0.194 | 0.168 |
| stablelm2-1.6b | FULL | ta_en | 177 | 0.583 | 0.489 | 0.315 |
| stablelm2-1.6b | FULL | te_en | 175 | 0.503 | 0.472 | 0.465 |
| stablelm2-1.6b | CALLER_ONLY | hi_en | 188 | 0.475 | 0.269 | 0.263 |
| stablelm2-1.6b | CALLER_ONLY | ta_en | 177 | 0.610 | 0.602 | 0.382 |
| stablelm2-1.6b | CALLER_ONLY | te_en | 175 | 0.621 | 0.539 | 0.291 |
| tinyllama-1.1b | FULL | hi_en | 188 | 0.400 | 0.118 | 0.168 |
| tinyllama-1.1b | FULL | ta_en | 177 | 0.497 | 0.705 | 0.674 |
| tinyllama-1.1b | FULL | te_en | 175 | 0.424 | 0.876 | 0.872 |
| tinyllama-1.1b | CALLER_ONLY | hi_en | 188 | 0.376 | 0.118 | 0.242 |
| tinyllama-1.1b | CALLER_ONLY | ta_en | 177 | 0.440 | 0.341 | 0.449 |
| tinyllama-1.1b | CALLER_ONLY | te_en | 175 | 0.388 | 0.360 | 0.581 |

## Per recipient condition (test)

| model | view | condition | n | fraud recall | legit FPR | macro-F1 |
|---|---|---|---:|---:|---:|---:|
| danube2-1.8b | FULL | cooperative | 133 | 0.554 | 0.397 | 0.578 |
| danube2-1.8b | FULL | clarifying | 134 | 0.652 | 0.529 | 0.557 |
| danube2-1.8b | FULL | resistant | 136 | 0.671 | 0.394 | 0.639 |
| danube2-1.8b | FULL | suspicious | 137 | 0.667 | 0.500 | 0.581 |
| danube2-1.8b | CALLER_ONLY | cooperative | 133 | 0.277 | 0.250 | 0.487 |
| danube2-1.8b | CALLER_ONLY | clarifying | 134 | 0.500 | 0.397 | 0.551 |
| danube2-1.8b | CALLER_ONLY | resistant | 136 | 0.529 | 0.364 | 0.580 |
| danube2-1.8b | CALLER_ONLY | suspicious | 137 | 0.449 | 0.382 | 0.530 |
| gemma3-1b | FULL | cooperative | 133 | 0.415 | 0.309 | 0.546 |
| gemma3-1b | FULL | clarifying | 134 | 0.576 | 0.397 | 0.589 |
| gemma3-1b | FULL | resistant | 136 | 0.800 | 0.364 | 0.718 |
| gemma3-1b | FULL | suspicious | 137 | 0.913 | 0.676 | 0.583 |
| gemma3-1b | CALLER_ONLY | cooperative | 133 | 0.631 | 0.485 | 0.571 |
| gemma3-1b | CALLER_ONLY | clarifying | 134 | 0.803 | 0.471 | 0.659 |
| gemma3-1b | CALLER_ONLY | resistant | 136 | 0.686 | 0.591 | 0.541 |
| gemma3-1b | CALLER_ONLY | suspicious | 137 | 0.739 | 0.706 | 0.492 |
| gemma3-270m | FULL | cooperative | 133 | 0.754 | 0.603 | 0.559 |
| gemma3-270m | FULL | clarifying | 134 | 0.227 | 0.309 | 0.430 |
| gemma3-270m | FULL | resistant | 136 | 0.500 | 0.576 | 0.462 |
| gemma3-270m | FULL | suspicious | 137 | 0.449 | 0.603 | 0.423 |
| gemma3-270m | CALLER_ONLY | cooperative | 133 | 0.585 | 0.412 | 0.586 |
| gemma3-270m | CALLER_ONLY | clarifying | 134 | 0.591 | 0.647 | 0.463 |
| gemma3-270m | CALLER_ONLY | resistant | 136 | 0.314 | 0.348 | 0.465 |
| gemma3-270m | CALLER_ONLY | suspicious | 137 | 0.391 | 0.559 | 0.416 |
| granite-1b | FULL | cooperative | 133 | 0.477 | 0.456 | 0.510 |
| granite-1b | FULL | clarifying | 134 | 0.333 | 0.279 | 0.510 |
| granite-1b | FULL | resistant | 136 | 0.486 | 0.409 | 0.536 |
| granite-1b | FULL | suspicious | 137 | 0.536 | 0.353 | 0.590 |
| granite-1b | CALLER_ONLY | cooperative | 133 | 0.523 | 0.603 | 0.457 |
| granite-1b | CALLER_ONLY | clarifying | 134 | 0.576 | 0.632 | 0.465 |
| granite-1b | CALLER_ONLY | resistant | 136 | 0.600 | 0.712 | 0.432 |
| granite-1b | CALLER_ONLY | suspicious | 137 | 0.623 | 0.515 | 0.552 |
| llama-3.2-1b | FULL | cooperative | 133 | 0.554 | 0.324 | 0.614 |
| llama-3.2-1b | FULL | clarifying | 134 | 0.682 | 0.485 | 0.595 |
| llama-3.2-1b | FULL | resistant | 136 | 0.571 | 0.333 | 0.617 |
| llama-3.2-1b | FULL | suspicious | 137 | 0.507 | 0.412 | 0.547 |
| llama-3.2-1b | CALLER_ONLY | cooperative | 133 | 0.369 | 0.309 | 0.519 |
| llama-3.2-1b | CALLER_ONLY | clarifying | 134 | 0.576 | 0.426 | 0.575 |
| llama-3.2-1b | CALLER_ONLY | resistant | 136 | 0.257 | 0.227 | 0.477 |
| llama-3.2-1b | CALLER_ONLY | suspicious | 137 | 0.333 | 0.397 | 0.458 |
| qwen2-0.5b | FULL | cooperative | 133 | 0.538 | 0.471 | 0.534 |
| qwen2-0.5b | FULL | clarifying | 134 | 0.758 | 0.529 | 0.605 |
| qwen2-0.5b | FULL | resistant | 136 | 0.586 | 0.561 | 0.511 |
| qwen2-0.5b | FULL | suspicious | 137 | 0.522 | 0.353 | 0.583 |
| qwen2-0.5b | CALLER_ONLY | cooperative | 133 | 0.585 | 0.485 | 0.549 |
| qwen2-0.5b | CALLER_ONLY | clarifying | 134 | 0.667 | 0.441 | 0.611 |
| qwen2-0.5b | CALLER_ONLY | resistant | 136 | 0.529 | 0.424 | 0.551 |
| qwen2-0.5b | CALLER_ONLY | suspicious | 137 | 0.565 | 0.515 | 0.525 |
| qwen2-1.5b | FULL | cooperative | 133 | 0.600 | 0.279 | 0.660 |
| qwen2-1.5b | FULL | clarifying | 134 | 0.697 | 0.309 | 0.694 |
| qwen2-1.5b | FULL | resistant | 136 | 0.729 | 0.364 | 0.682 |
| qwen2-1.5b | FULL | suspicious | 137 | 0.652 | 0.206 | 0.721 |
| qwen2-1.5b | CALLER_ONLY | cooperative | 133 | 0.662 | 0.353 | 0.654 |
| qwen2-1.5b | CALLER_ONLY | clarifying | 134 | 0.742 | 0.368 | 0.686 |
| qwen2-1.5b | CALLER_ONLY | resistant | 136 | 0.886 | 0.530 | 0.666 |
| qwen2-1.5b | CALLER_ONLY | suspicious | 137 | 0.855 | 0.368 | 0.741 |
| qwen2.5-0.5b | FULL | cooperative | 133 | 0.323 | 0.309 | 0.492 |
| qwen2.5-0.5b | FULL | clarifying | 134 | 0.530 | 0.426 | 0.552 |
| qwen2.5-0.5b | FULL | resistant | 136 | 0.714 | 0.439 | 0.636 |
| qwen2.5-0.5b | FULL | suspicious | 137 | 0.696 | 0.515 | 0.586 |
| qwen2.5-0.5b | CALLER_ONLY | cooperative | 133 | 0.585 | 0.559 | 0.510 |
| qwen2.5-0.5b | CALLER_ONLY | clarifying | 134 | 0.545 | 0.471 | 0.537 |
| qwen2.5-0.5b | CALLER_ONLY | resistant | 136 | 0.857 | 0.636 | 0.589 |
| qwen2.5-0.5b | CALLER_ONLY | suspicious | 137 | 0.797 | 0.618 | 0.572 |
| qwen2.5-1.5b | FULL | cooperative | 133 | 0.431 | 0.147 | 0.628 |
| qwen2.5-1.5b | FULL | clarifying | 134 | 0.833 | 0.265 | 0.783 |
| qwen2.5-1.5b | FULL | resistant | 136 | 0.743 | 0.273 | 0.735 |
| qwen2.5-1.5b | FULL | suspicious | 137 | 0.797 | 0.309 | 0.744 |
| qwen2.5-1.5b | CALLER_ONLY | cooperative | 133 | 0.400 | 0.221 | 0.577 |
| qwen2.5-1.5b | CALLER_ONLY | clarifying | 134 | 0.636 | 0.235 | 0.700 |
| qwen2.5-1.5b | CALLER_ONLY | resistant | 136 | 0.600 | 0.288 | 0.654 |
| qwen2.5-1.5b | CALLER_ONLY | suspicious | 137 | 0.609 | 0.235 | 0.684 |
| qwen3-0.6b | FULL | cooperative | 133 | 0.431 | 0.324 | 0.548 |
| qwen3-0.6b | FULL | clarifying | 134 | 0.515 | 0.265 | 0.621 |
| qwen3-0.6b | FULL | resistant | 136 | 0.500 | 0.439 | 0.529 |
| qwen3-0.6b | FULL | suspicious | 137 | 0.232 | 0.265 | 0.448 |
| qwen3-0.6b | CALLER_ONLY | cooperative | 133 | 0.308 | 0.176 | 0.538 |
| qwen3-0.6b | CALLER_ONLY | clarifying | 134 | 0.242 | 0.132 | 0.509 |
| qwen3-0.6b | CALLER_ONLY | resistant | 136 | 0.300 | 0.152 | 0.535 |
| qwen3-0.6b | CALLER_ONLY | suspicious | 137 | 0.275 | 0.235 | 0.489 |
| qwen3-1.7b | FULL | cooperative | 133 | 0.815 | 0.353 | 0.728 |
| qwen3-1.7b | FULL | clarifying | 134 | 0.742 | 0.368 | 0.686 |
| qwen3-1.7b | FULL | resistant | 136 | 0.557 | 0.227 | 0.659 |
| qwen3-1.7b | FULL | suspicious | 137 | 0.797 | 0.485 | 0.649 |
| qwen3-1.7b | CALLER_ONLY | cooperative | 133 | 0.615 | 0.324 | 0.646 |
| qwen3-1.7b | CALLER_ONLY | clarifying | 134 | 0.803 | 0.397 | 0.699 |
| qwen3-1.7b | CALLER_ONLY | resistant | 136 | 0.457 | 0.273 | 0.582 |
| qwen3-1.7b | CALLER_ONLY | suspicious | 137 | 0.638 | 0.412 | 0.613 |
| smollm2-1.7b | FULL | cooperative | 133 | 0.646 | 0.574 | 0.529 |
| smollm2-1.7b | FULL | clarifying | 134 | 0.455 | 0.338 | 0.554 |
| smollm2-1.7b | FULL | resistant | 136 | 0.657 | 0.591 | 0.528 |
| smollm2-1.7b | FULL | suspicious | 137 | 0.710 | 0.456 | 0.625 |
| smollm2-1.7b | CALLER_ONLY | cooperative | 133 | 0.600 | 0.559 | 0.517 |
| smollm2-1.7b | CALLER_ONLY | clarifying | 134 | 0.788 | 0.544 | 0.610 |
| smollm2-1.7b | CALLER_ONLY | resistant | 136 | 0.714 | 0.682 | 0.499 |
| smollm2-1.7b | CALLER_ONLY | suspicious | 137 | 0.696 | 0.529 | 0.578 |
| smollm2-135m | FULL | cooperative | 133 | 0.723 | 0.647 | 0.519 |
| smollm2-135m | FULL | clarifying | 134 | 0.864 | 0.735 | 0.519 |
| smollm2-135m | FULL | resistant | 136 | 0.757 | 0.742 | 0.478 |
| smollm2-135m | FULL | suspicious | 137 | 0.754 | 0.721 | 0.489 |
| smollm2-135m | CALLER_ONLY | cooperative | 133 | 0.754 | 0.588 | 0.568 |
| smollm2-135m | CALLER_ONLY | clarifying | 134 | 0.758 | 0.574 | 0.579 |
| smollm2-135m | CALLER_ONLY | resistant | 136 | 0.743 | 0.682 | 0.511 |
| smollm2-135m | CALLER_ONLY | suspicious | 137 | 0.739 | 0.544 | 0.590 |
| smollm2-360m | FULL | cooperative | 133 | 0.631 | 0.485 | 0.571 |
| smollm2-360m | FULL | clarifying | 134 | 0.727 | 0.529 | 0.591 |
| smollm2-360m | FULL | resistant | 136 | 0.457 | 0.409 | 0.521 |
| smollm2-360m | FULL | suspicious | 137 | 0.478 | 0.529 | 0.474 |
| smollm2-360m | CALLER_ONLY | cooperative | 133 | 0.462 | 0.353 | 0.551 |
| smollm2-360m | CALLER_ONLY | clarifying | 134 | 0.470 | 0.338 | 0.562 |
| smollm2-360m | CALLER_ONLY | resistant | 136 | 0.200 | 0.197 | 0.447 |
| smollm2-360m | CALLER_ONLY | suspicious | 137 | 0.232 | 0.294 | 0.436 |
| stablelm2-1.6b | FULL | cooperative | 133 | 0.369 | 0.353 | 0.500 |
| stablelm2-1.6b | FULL | clarifying | 134 | 0.424 | 0.235 | 0.584 |
| stablelm2-1.6b | FULL | resistant | 136 | 0.386 | 0.318 | 0.521 |
| stablelm2-1.6b | FULL | suspicious | 137 | 0.348 | 0.338 | 0.492 |
| stablelm2-1.6b | CALLER_ONLY | cooperative | 133 | 0.323 | 0.279 | 0.504 |
| stablelm2-1.6b | CALLER_ONLY | clarifying | 134 | 0.606 | 0.191 | 0.705 |
| stablelm2-1.6b | CALLER_ONLY | resistant | 136 | 0.486 | 0.515 | 0.485 |
| stablelm2-1.6b | CALLER_ONLY | suspicious | 137 | 0.449 | 0.265 | 0.583 |
| tinyllama-1.1b | FULL | cooperative | 133 | 0.615 | 0.662 | 0.465 |
| tinyllama-1.1b | FULL | clarifying | 134 | 0.500 | 0.500 | 0.500 |
| tinyllama-1.1b | FULL | resistant | 136 | 0.586 | 0.606 | 0.486 |
| tinyllama-1.1b | FULL | suspicious | 137 | 0.536 | 0.471 | 0.533 |
| tinyllama-1.1b | CALLER_ONLY | cooperative | 133 | 0.415 | 0.426 | 0.492 |
| tinyllama-1.1b | CALLER_ONLY | clarifying | 134 | 0.424 | 0.574 | 0.425 |
| tinyllama-1.1b | CALLER_ONLY | resistant | 136 | 0.129 | 0.212 | 0.387 |
| tinyllama-1.1b | CALLER_ONLY | suspicious | 137 | 0.130 | 0.456 | 0.307 |

## Branch metrics (complete sibling sets, test)

| model | view | n histories | disagreement | score spread (mean) | all-correct |
|---|---|---:|---:|---:|---:|
| danube2-1.8b | FULL | 122 | 0.680 | 0.025 | 0.230 |
| danube2-1.8b | CALLER_ONLY | 122 | 0.705 | 0.025 | 0.197 |
| gemma3-1b | FULL | 122 | 0.770 | 0.594 | 0.172 |
| gemma3-1b | CALLER_ONLY | 122 | 0.451 | 0.403 | 0.344 |
| gemma3-270m | FULL | 122 | 0.664 | 0.000 | 0.172 |
| gemma3-270m | CALLER_ONLY | 122 | 0.492 | 0.000 | 0.254 |
| granite-1b | FULL | 122 | 0.770 | 0.375 | 0.156 |
| granite-1b | CALLER_ONLY | 122 | 0.738 | 0.320 | 0.131 |
| llama-3.2-1b | FULL | 122 | 0.434 | 0.183 | 0.377 |
| llama-3.2-1b | CALLER_ONLY | 122 | 0.385 | 0.218 | 0.352 |
| qwen2-0.5b | FULL | 122 | 0.582 | 0.028 | 0.262 |
| qwen2-0.5b | CALLER_ONLY | 122 | 0.451 | 0.024 | 0.336 |
| qwen2-1.5b | FULL | 122 | 0.475 | 0.236 | 0.443 |
| qwen2-1.5b | CALLER_ONLY | 122 | 0.500 | 0.172 | 0.402 |
| qwen2.5-0.5b | FULL | 122 | 0.623 | 0.377 | 0.230 |
| qwen2.5-0.5b | CALLER_ONLY | 122 | 0.484 | 0.353 | 0.295 |
| qwen2.5-1.5b | FULL | 122 | 0.467 | 0.450 | 0.475 |
| qwen2.5-1.5b | CALLER_ONLY | 122 | 0.443 | 0.331 | 0.410 |
| qwen3-0.6b | FULL | 122 | 0.664 | 0.171 | 0.189 |
| qwen3-0.6b | CALLER_ONLY | 122 | 0.426 | 0.080 | 0.352 |
| qwen3-1.7b | FULL | 122 | 0.607 | 0.000 | 0.336 |
| qwen3-1.7b | CALLER_ONLY | 122 | 0.541 | 0.000 | 0.361 |
| smollm2-1.7b | FULL | 122 | 0.525 | 0.063 | 0.295 |
| smollm2-1.7b | CALLER_ONLY | 122 | 0.377 | 0.035 | 0.369 |
| smollm2-135m | FULL | 122 | 0.328 | 0.023 | 0.361 |
| smollm2-135m | CALLER_ONLY | 122 | 0.393 | 0.023 | 0.385 |
| smollm2-360m | FULL | 122 | 0.631 | 0.252 | 0.221 |
| smollm2-360m | CALLER_ONLY | 122 | 0.631 | 0.358 | 0.205 |
| stablelm2-1.6b | FULL | 122 | 0.557 | 0.247 | 0.246 |
| stablelm2-1.6b | CALLER_ONLY | 122 | 0.639 | 0.210 | 0.246 |
| tinyllama-1.1b | FULL | 122 | 0.500 | 0.015 | 0.246 |
| tinyllama-1.1b | CALLER_ONLY | 122 | 0.623 | 0.058 | 0.123 |

## PREFIX_PLUS_REACTION − PREFIX_ONLY (mean Δ P(fraud), test)

| model | condition / label | mean Δ | 95% CI |
|---|---|---:|---|

## Complete-set slice (test)

| model | view | n | macro-F1 | disagreement | all-correct |
|---|---|---:|---:|---:|---:|
| danube2-1.8b | FULL | 488 | 0.596 | 0.680 | 0.230 |
| danube2-1.8b | CALLER_ONLY | 488 | 0.555 | 0.705 | 0.197 |
| gemma3-1b | FULL | 488 | 0.629 | 0.770 | 0.172 |
| gemma3-1b | CALLER_ONLY | 488 | 0.571 | 0.451 | 0.344 |
| gemma3-270m | FULL | 488 | 0.484 | 0.664 | 0.172 |
| gemma3-270m | CALLER_ONLY | 488 | 0.495 | 0.492 | 0.254 |
| granite-1b | FULL | 488 | 0.543 | 0.770 | 0.156 |
| granite-1b | CALLER_ONLY | 488 | 0.488 | 0.738 | 0.131 |
| llama-3.2-1b | FULL | 488 | 0.590 | 0.434 | 0.377 |
| llama-3.2-1b | CALLER_ONLY | 488 | 0.522 | 0.385 | 0.352 |
| qwen2-0.5b | FULL | 488 | 0.555 | 0.582 | 0.262 |
| qwen2-0.5b | CALLER_ONLY | 488 | 0.555 | 0.451 | 0.336 |
| qwen2-1.5b | FULL | 488 | 0.688 | 0.475 | 0.443 |
| qwen2-1.5b | CALLER_ONLY | 488 | 0.696 | 0.500 | 0.402 |
| qwen2.5-0.5b | FULL | 488 | 0.572 | 0.623 | 0.230 |
| qwen2.5-0.5b | CALLER_ONLY | 488 | 0.545 | 0.484 | 0.295 |
| qwen2.5-1.5b | FULL | 488 | 0.739 | 0.467 | 0.475 |
| qwen2.5-1.5b | CALLER_ONLY | 488 | 0.660 | 0.443 | 0.410 |
| qwen3-0.6b | FULL | 488 | 0.541 | 0.664 | 0.189 |
| qwen3-0.6b | CALLER_ONLY | 488 | 0.524 | 0.426 | 0.352 |
| qwen3-1.7b | FULL | 488 | 0.703 | 0.607 | 0.336 |
| qwen3-1.7b | CALLER_ONLY | 488 | 0.635 | 0.541 | 0.361 |
| smollm2-1.7b | FULL | 488 | 0.557 | 0.525 | 0.295 |
| smollm2-1.7b | CALLER_ONLY | 488 | 0.546 | 0.377 | 0.369 |
| smollm2-135m | FULL | 488 | 0.498 | 0.328 | 0.361 |
| smollm2-135m | CALLER_ONLY | 488 | 0.571 | 0.393 | 0.385 |
| smollm2-360m | FULL | 488 | 0.543 | 0.631 | 0.221 |
| smollm2-360m | CALLER_ONLY | 488 | 0.505 | 0.631 | 0.205 |
| stablelm2-1.6b | FULL | 488 | 0.527 | 0.557 | 0.246 |
| stablelm2-1.6b | CALLER_ONLY | 488 | 0.574 | 0.639 | 0.246 |
| tinyllama-1.1b | FULL | 488 | 0.475 | 0.500 | 0.246 |
| tinyllama-1.1b | CALLER_ONLY | 488 | 0.399 | 0.623 | 0.123 |

## Streaming delay (test; misses counted)

| model | eligible | detected | miss rate | mean delay (hits) | pre-evidence alert rate |
|---|---:|---:|---:|---:|---:|

## Runtime projection (full panel)

- median per-inference: 0.054 s
- mean turns: 9.94
- assumed test+dev conversations: 540+540
- one model, 4 views: 0.21 h
- 5 models × 4 views: 3.96 h
- note: STREAMING costs one inference per turn; the other three views cost one each. Naive 4×views undercounts STREAMING.
