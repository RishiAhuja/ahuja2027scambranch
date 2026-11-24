# Detector eval `eval_hpc`

- run_dir: `runs/full_v1`
- n_dev_scored: 540
- n_test_scored: 540
- views: FULL, PREFIX_ONLY, PREFIX_PLUS_REACTION, STREAMING
- models: qwen3-0.6b, qwen3-1.7b, qwen2.5-1.5b, smollm2-1.7b, qwen2.5-0.5b, qwen2-0.5b, qwen2-1.5b, smollm2-135m, smollm2-360m, tinyllama-1.1b, stablelm2-1.6b, danube2-1.8b, granite-1b

## Skipped models

- `llama-3.2-1b` (meta-llama/Llama-3.2-1B-Instruct): gated_or_auth: OSError: You are trying to access a gated repo.
Make sure to have access to it at https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct.
401 Client Error. (Request ID: Root=1-6ab68508-0788b5b61affeb7d24717010;e95b8003-e840-485c-a8d3-42ed0b948d73)

Cannot access gated repo for url https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/resolve/main/config.json.
Access to model meta-llama/Llama-3.2-1B-Inst
- `gemma3-270m` (google/gemma-3-270m-it): gated_or_auth: OSError: You are trying to access a gated repo.
Make sure to have access to it at https://huggingface.co/google/gemma-3-270m-it.
401 Client Error. (Request ID: Root=1-6ab68ff5-0cf5cc8b34a375d92c77dde1;f61df53e-29e7-4316-9eec-7ffcc1d882f3)

Cannot access gated repo for url https://huggingface.co/google/gemma-3-270m-it/resolve/main/config.json.
Access to model google/gemma-3-270m-it is restricted. You must h
- `gemma3-1b` (google/gemma-3-1b-it): gated_or_auth: OSError: You are trying to access a gated repo.
Make sure to have access to it at https://huggingface.co/google/gemma-3-1b-it.
401 Client Error. (Request ID: Root=1-6ab69126-402ce0ff4a575e721ab80bfa;95f9ba02-a266-405d-8f1a-a2f0bf0759e3)

Cannot access gated repo for url https://huggingface.co/google/gemma-3-1b-it/resolve/main/config.json.
Access to model google/gemma-3-1b-it is restricted. You must have ac
- `internlm2-1.8b` (internlm/internlm2-chat-1_8b): load_failed: ImportError: This modeling file requires the following packages that were not found in your environment: sentencepiece. Run `pip install sentencepiece`
- `bloomz-1b7` (bigscience/bloomz-1b7): download_failed: ValueError: Cannot use chat template functions because tokenizer.chat_template is not set and no template argument was passed! For information about writing templates and setting the tokenizer.chat_template attribute, please see the documentation at https://huggingface.co/docs/transformers/main/en/chat_templating
- `olmo-1b` (allenai/OLMo-1B-0724-hf): download_failed: ValueError: Cannot use chat template functions because tokenizer.chat_template is not set and no template argument was passed! For information about writing templates and setting the tokenizer.chat_template attribute, please see the documentation at https://huggingface.co/docs/transformers/main/en/chat_templating

## Thresholds (fit on DEV, max macro-F1)

| model | view | threshold |
|---|---|---|
| qwen3-0.6b | FULL | 0.9195 |
| qwen3-0.6b | PREFIX_ONLY | 0.9362 |
| qwen3-0.6b | PREFIX_PLUS_REACTION | 0.9283 |
| qwen3-1.7b | FULL | 1.0000 |
| qwen3-1.7b | PREFIX_ONLY | 1.0000 |
| qwen3-1.7b | PREFIX_PLUS_REACTION | 1.0000 |
| qwen3-1.7b | STREAMING | 1.0000 |
| qwen2.5-1.5b | FULL | 0.2018 |
| qwen2.5-1.5b | PREFIX_ONLY | 0.2689 |
| qwen2.5-1.5b | PREFIX_PLUS_REACTION | 0.2018 |
| qwen2.5-1.5b | STREAMING | 0.2018 |
| smollm2-1.7b | FULL | 0.9481 |
| smollm2-1.7b | PREFIX_ONLY | 0.9450 |
| smollm2-1.7b | PREFIX_PLUS_REACTION | 0.9381 |
| qwen2.5-0.5b | FULL | 0.7100 |
| qwen2.5-0.5b | PREFIX_ONLY | 0.7800 |
| qwen2.5-0.5b | PREFIX_PLUS_REACTION | 0.8700 |
| qwen2-0.5b | FULL | 0.9790 |
| qwen2-0.5b | PREFIX_ONLY | 0.9872 |
| qwen2-0.5b | PREFIX_PLUS_REACTION | 0.9846 |
| qwen2-1.5b | FULL | 0.7400 |
| qwen2-1.5b | PREFIX_ONLY | 0.8595 |
| qwen2-1.5b | PREFIX_PLUS_REACTION | 0.8400 |
| smollm2-135m | FULL | 0.9496 |
| smollm2-135m | PREFIX_ONLY | 0.9600 |
| smollm2-135m | PREFIX_PLUS_REACTION | 0.9496 |
| smollm2-360m | FULL | 0.8376 |
| smollm2-360m | PREFIX_ONLY | 0.6893 |
| smollm2-360m | PREFIX_PLUS_REACTION | 0.7341 |
| tinyllama-1.1b | FULL | 0.9907 |
| tinyllama-1.1b | PREFIX_ONLY | 0.9829 |
| tinyllama-1.1b | PREFIX_PLUS_REACTION | 0.9890 |
| stablelm2-1.6b | FULL | 0.8200 |
| stablelm2-1.6b | PREFIX_ONLY | 0.7800 |
| stablelm2-1.6b | PREFIX_PLUS_REACTION | 0.8400 |
| danube2-1.8b | FULL | 0.0159 |
| danube2-1.8b | PREFIX_ONLY | 0.0302 |
| danube2-1.8b | PREFIX_PLUS_REACTION | 0.0191 |
| granite-1b | FULL | 0.5100 |
| granite-1b | PREFIX_ONLY | 0.6300 |
| granite-1b | PREFIX_PLUS_REACTION | 0.6000 |

## Test quality (pooled)

| model | view | n | macro-F1 | fraud recall | legit FPR | ROC-AUC | invalid | worst-cond F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| danube2-1.8b | FULL | 540 | 0.590 [0.530, 0.641] | 0.637 [0.541, 0.739] | 0.456 [0.337, 0.571] | 0.614 | 0.017 | 0.557 |
| danube2-1.8b | PREFIX_ONLY | 540 | 0.496 [0.366, 0.604] | 0.685 [0.521, 0.842] | 0.663 [0.461, 0.868] | 0.491 | 0.007 | 0.490 |
| danube2-1.8b | PREFIX_PLUS_REACTION | 540 | 0.602 [0.503, 0.684] | 0.663 [0.521, 0.806] | 0.456 [0.322, 0.602] | 0.637 | 0.015 | 0.541 |
| danube2-1.8b | STREAMING | 0 | — | — | — | — | — | — |
| granite-1b | FULL | 540 | 0.531 [0.477, 0.576] | 0.493 [0.426, 0.551] | 0.430 [0.354, 0.515] | 0.541 | 0.000 | 0.478 |
| granite-1b | PREFIX_ONLY | 540 | 0.512 [0.405, 0.604] | 0.811 [0.689, 0.913] | 0.719 [0.538, 0.891] | 0.590 | 0.000 | 0.505 |
| granite-1b | PREFIX_PLUS_REACTION | 540 | 0.485 [0.443, 0.522] | 0.385 [0.321, 0.442] | 0.404 [0.305, 0.500] | 0.485 | 0.000 | 0.399 |
| granite-1b | STREAMING | 0 | — | — | — | — | — | — |
| qwen2-0.5b | FULL | 540 | 0.560 [0.455, 0.658] | 0.600 [0.466, 0.735] | 0.478 [0.307, 0.649] | 0.602 | 0.041 | 0.511 |
| qwen2-0.5b | PREFIX_ONLY | 540 | 0.460 [0.297, 0.597] | 0.496 [0.314, 0.682] | 0.574 [0.356, 0.799] | 0.498 | 0.009 | 0.458 |
| qwen2-0.5b | PREFIX_PLUS_REACTION | 540 | 0.561 [0.444, 0.656] | 0.559 [0.407, 0.704] | 0.437 [0.241, 0.641] | 0.572 | 0.024 | 0.507 |
| qwen2-0.5b | STREAMING | 0 | — | — | — | — | — | — |
| qwen2-1.5b | FULL | 540 | 0.691 [0.625, 0.754] | 0.670 [0.573, 0.771] | 0.289 [0.185, 0.385] | 0.758 | 0.000 | 0.660 |
| qwen2-1.5b | PREFIX_ONLY | 540 | 0.761 [0.692, 0.830] | 0.770 [0.629, 0.896] | 0.248 [0.123, 0.357] | 0.814 | 0.000 | 0.757 |
| qwen2-1.5b | PREFIX_PLUS_REACTION | 540 | 0.703 [0.635, 0.758] | 0.648 [0.522, 0.755] | 0.241 [0.153, 0.332] | 0.783 | 0.000 | 0.642 |
| qwen2-1.5b | STREAMING | 0 | — | — | — | — | — | — |
| qwen2.5-0.5b | FULL | 540 | 0.574 [0.511, 0.625] | 0.570 [0.453, 0.694] | 0.422 [0.254, 0.606] | 0.599 | 0.000 | 0.492 |
| qwen2.5-0.5b | PREFIX_ONLY | 540 | 0.548 [0.462, 0.629] | 0.585 [0.349, 0.801] | 0.489 [0.244, 0.735] | 0.585 | 0.000 | 0.543 |
| qwen2.5-0.5b | PREFIX_PLUS_REACTION | 540 | 0.625 [0.545, 0.707] | 0.548 [0.378, 0.717] | 0.293 [0.117, 0.496] | 0.648 | 0.000 | 0.602 |
| qwen2.5-0.5b | STREAMING | 0 | — | — | — | — | — | — |
| qwen2.5-1.5b | FULL | 540 | 0.728 [0.666, 0.783] | 0.704 [0.598, 0.817] | 0.248 [0.129, 0.393] | 0.784 | 0.000 | 0.628 |
| qwen2.5-1.5b | PREFIX_ONLY | 540 | 0.681 [0.553, 0.800] | 0.804 [0.635, 0.946] | 0.433 [0.204, 0.675] | 0.729 | 0.000 | 0.669 |
| qwen2.5-1.5b | PREFIX_PLUS_REACTION | 540 | 0.671 [0.612, 0.725] | 0.744 [0.636, 0.854] | 0.400 [0.270, 0.540] | 0.734 | 0.000 | 0.499 |
| qwen2.5-1.5b | STREAMING | 540 | 0.728 [0.666, 0.783] | 0.704 [0.598, 0.817] | 0.248 [0.129, 0.393] | 0.784 | 0.000 | 0.628 |
| qwen3-0.6b | FULL | 540 | 0.540 [0.476, 0.601] | 0.419 [0.328, 0.509] | 0.322 [0.194, 0.453] | 0.563 | 0.000 | 0.448 |
| qwen3-0.6b | PREFIX_ONLY | 540 | 0.589 [0.457, 0.710] | 0.415 [0.227, 0.611] | 0.207 [0.059, 0.400] | 0.640 | 0.000 | 0.579 |
| qwen3-0.6b | PREFIX_PLUS_REACTION | 540 | 0.522 [0.451, 0.593] | 0.370 [0.235, 0.533] | 0.300 [0.158, 0.474] | 0.545 | 0.000 | 0.508 |
| qwen3-0.6b | STREAMING | 0 | — | — | — | — | — | — |
| qwen3-1.7b | FULL | 540 | 0.683 [0.626, 0.739] | 0.726 [0.652, 0.800] | 0.359 [0.269, 0.447] | 0.759 | 0.000 | 0.649 |
| qwen3-1.7b | PREFIX_ONLY | 540 | 0.647 [0.543, 0.736] | 0.744 [0.652, 0.840] | 0.444 [0.302, 0.603] | 0.783 | 0.000 | 0.640 |
| qwen3-1.7b | PREFIX_PLUS_REACTION | 540 | 0.694 [0.624, 0.760] | 0.696 [0.589, 0.792] | 0.307 [0.219, 0.405] | 0.777 | 0.000 | 0.620 |
| qwen3-1.7b | STREAMING | 540 | 0.683 [0.626, 0.739] | 0.726 [0.652, 0.800] | 0.359 [0.269, 0.447] | 0.759 | 0.000 | 0.649 |
| smollm2-1.7b | FULL | 540 | 0.564 [0.519, 0.603] | 0.619 [0.532, 0.707] | 0.489 [0.398, 0.581] | 0.590 | 0.000 | 0.528 |
| smollm2-1.7b | PREFIX_ONLY | 540 | 0.430 [0.374, 0.487] | 0.796 [0.663, 0.912] | 0.826 [0.755, 0.893] | 0.569 | 0.000 | 0.427 |
| smollm2-1.7b | PREFIX_PLUS_REACTION | 540 | 0.524 [0.494, 0.553] | 0.700 [0.633, 0.768] | 0.626 [0.593, 0.658] | 0.627 | 0.000 | 0.480 |
| smollm2-1.7b | STREAMING | 0 | — | — | — | — | — | — |
| smollm2-135m | FULL | 540 | 0.502 [0.459, 0.540] | 0.774 [0.669, 0.863] | 0.711 [0.629, 0.793] | 0.542 | 1.000 | 0.478 |
| smollm2-135m | PREFIX_ONLY | 540 | 0.536 [0.448, 0.635] | 0.611 [0.444, 0.775] | 0.533 [0.412, 0.656] | 0.557 | 1.000 | 0.526 |
| smollm2-135m | PREFIX_PLUS_REACTION | 540 | 0.468 [0.418, 0.508] | 0.763 [0.640, 0.869] | 0.756 [0.654, 0.849] | 0.523 | 1.000 | 0.443 |
| smollm2-135m | STREAMING | 0 | — | — | — | — | — | — |
| smollm2-360m | FULL | 540 | 0.540 [0.482, 0.598] | 0.570 [0.484, 0.660] | 0.489 [0.368, 0.605] | 0.553 | 0.048 | 0.474 |
| smollm2-360m | PREFIX_ONLY | 540 | 0.496 [0.360, 0.608] | 0.481 [0.332, 0.640] | 0.489 [0.313, 0.685] | 0.502 | 0.000 | 0.480 |
| smollm2-360m | PREFIX_PLUS_REACTION | 540 | 0.548 [0.460, 0.636] | 0.619 [0.539, 0.699] | 0.519 [0.360, 0.664] | 0.547 | 0.004 | 0.504 |
| smollm2-360m | STREAMING | 0 | — | — | — | — | — | — |
| stablelm2-1.6b | FULL | 540 | 0.524 [0.489, 0.553] | 0.381 [0.307, 0.459] | 0.311 [0.219, 0.407] | 0.558 | 0.261 | 0.492 |
| stablelm2-1.6b | PREFIX_ONLY | 540 | 0.506 [0.428, 0.572] | 0.704 [0.589, 0.804] | 0.659 [0.530, 0.786] | 0.526 | 0.730 | 0.493 |
| stablelm2-1.6b | PREFIX_PLUS_REACTION | 540 | 0.510 [0.453, 0.563] | 0.411 [0.298, 0.517] | 0.381 [0.310, 0.441] | 0.508 | 0.463 | 0.492 |
| stablelm2-1.6b | STREAMING | 0 | — | — | — | — | — | — |
| tinyllama-1.1b | FULL | 540 | 0.498 [0.453, 0.541] | 0.559 [0.480, 0.637] | 0.559 [0.492, 0.625] | 0.505 | 1.000 | 0.465 |
| tinyllama-1.1b | PREFIX_ONLY | 540 | 0.455 [0.337, 0.569] | 0.474 [0.280, 0.662] | 0.563 [0.376, 0.737] | 0.417 | 1.000 | 0.451 |
| tinyllama-1.1b | PREFIX_PLUS_REACTION | 540 | 0.462 [0.403, 0.511] | 0.556 [0.420, 0.684] | 0.622 [0.547, 0.693] | 0.445 | 1.000 | 0.398 |
| tinyllama-1.1b | STREAMING | 0 | — | — | — | — | — | — |

## Per language (test, pooled views listed separately)

| model | view | language | n | macro-F1 | fraud recall | legit FPR |
|---|---|---|---:|---:|---:|---:|
| danube2-1.8b | FULL | hi_en | 188 | 0.648 | 0.699 | 0.400 |
| danube2-1.8b | FULL | ta_en | 177 | 0.501 | 0.716 | 0.674 |
| danube2-1.8b | FULL | te_en | 175 | 0.596 | 0.494 | 0.291 |
| danube2-1.8b | PREFIX_ONLY | hi_en | 188 | 0.552 | 0.892 | 0.705 |
| danube2-1.8b | PREFIX_ONLY | ta_en | 177 | 0.378 | 0.398 | 0.640 |
| danube2-1.8b | PREFIX_ONLY | te_en | 175 | 0.541 | 0.753 | 0.640 |
| danube2-1.8b | PREFIX_PLUS_REACTION | hi_en | 188 | 0.699 | 0.817 | 0.411 |
| danube2-1.8b | PREFIX_PLUS_REACTION | ta_en | 177 | 0.546 | 0.614 | 0.517 |
| danube2-1.8b | PREFIX_PLUS_REACTION | te_en | 175 | 0.554 | 0.551 | 0.442 |
| granite-1b | FULL | hi_en | 188 | 0.608 | 0.473 | 0.242 |
| granite-1b | FULL | ta_en | 177 | 0.458 | 0.568 | 0.640 |
| granite-1b | FULL | te_en | 175 | 0.507 | 0.438 | 0.419 |
| granite-1b | PREFIX_ONLY | hi_en | 188 | 0.519 | 0.828 | 0.716 |
| granite-1b | PREFIX_ONLY | ta_en | 177 | 0.560 | 0.955 | 0.730 |
| granite-1b | PREFIX_ONLY | te_en | 175 | 0.455 | 0.652 | 0.709 |
| granite-1b | PREFIX_PLUS_REACTION | hi_en | 188 | 0.542 | 0.398 | 0.295 |
| granite-1b | PREFIX_PLUS_REACTION | ta_en | 177 | 0.418 | 0.443 | 0.607 |
| granite-1b | PREFIX_PLUS_REACTION | te_en | 175 | 0.481 | 0.315 | 0.314 |
| qwen2-0.5b | FULL | hi_en | 188 | 0.595 | 0.731 | 0.526 |
| qwen2-0.5b | FULL | ta_en | 177 | 0.548 | 0.534 | 0.438 |
| qwen2-0.5b | FULL | te_en | 175 | 0.531 | 0.528 | 0.465 |
| qwen2-0.5b | PREFIX_ONLY | hi_en | 188 | 0.462 | 0.516 | 0.589 |
| qwen2-0.5b | PREFIX_ONLY | ta_en | 177 | 0.486 | 0.500 | 0.528 |
| qwen2-0.5b | PREFIX_ONLY | te_en | 175 | 0.433 | 0.472 | 0.605 |
| qwen2-0.5b | PREFIX_PLUS_REACTION | hi_en | 188 | 0.610 | 0.677 | 0.453 |
| qwen2-0.5b | PREFIX_PLUS_REACTION | ta_en | 177 | 0.560 | 0.466 | 0.337 |
| qwen2-0.5b | PREFIX_PLUS_REACTION | te_en | 175 | 0.502 | 0.528 | 0.523 |
| qwen2-1.5b | FULL | hi_en | 188 | 0.715 | 0.559 | 0.116 |
| qwen2-1.5b | FULL | ta_en | 177 | 0.591 | 0.830 | 0.607 |
| qwen2-1.5b | FULL | te_en | 175 | 0.735 | 0.629 | 0.151 |
| qwen2-1.5b | PREFIX_ONLY | hi_en | 188 | 0.851 | 0.828 | 0.126 |
| qwen2-1.5b | PREFIX_ONLY | ta_en | 177 | 0.602 | 0.693 | 0.483 |
| qwen2-1.5b | PREFIX_ONLY | te_en | 175 | 0.823 | 0.787 | 0.140 |
| qwen2-1.5b | PREFIX_PLUS_REACTION | hi_en | 188 | 0.738 | 0.591 | 0.105 |
| qwen2-1.5b | PREFIX_PLUS_REACTION | ta_en | 177 | 0.630 | 0.727 | 0.461 |
| qwen2-1.5b | PREFIX_PLUS_REACTION | te_en | 175 | 0.729 | 0.629 | 0.163 |
| qwen2.5-0.5b | FULL | hi_en | 188 | 0.579 | 0.710 | 0.537 |
| qwen2.5-0.5b | FULL | ta_en | 177 | 0.599 | 0.580 | 0.382 |
| qwen2.5-0.5b | FULL | te_en | 175 | 0.531 | 0.416 | 0.337 |
| qwen2.5-0.5b | PREFIX_ONLY | hi_en | 188 | 0.562 | 0.634 | 0.505 |
| qwen2.5-0.5b | PREFIX_ONLY | ta_en | 177 | 0.491 | 0.477 | 0.494 |
| qwen2.5-0.5b | PREFIX_ONLY | te_en | 175 | 0.587 | 0.640 | 0.465 |
| qwen2.5-0.5b | PREFIX_PLUS_REACTION | hi_en | 188 | 0.643 | 0.699 | 0.411 |
| qwen2.5-0.5b | PREFIX_PLUS_REACTION | ta_en | 177 | 0.625 | 0.455 | 0.180 |
| qwen2.5-0.5b | PREFIX_PLUS_REACTION | te_en | 175 | 0.595 | 0.483 | 0.279 |
| qwen2.5-1.5b | FULL | hi_en | 188 | 0.830 | 0.806 | 0.147 |
| qwen2.5-1.5b | FULL | ta_en | 177 | 0.655 | 0.648 | 0.337 |
| qwen2.5-1.5b | FULL | te_en | 175 | 0.691 | 0.652 | 0.267 |
| qwen2.5-1.5b | PREFIX_ONLY | hi_en | 188 | 0.723 | 0.785 | 0.337 |
| qwen2.5-1.5b | PREFIX_ONLY | ta_en | 177 | 0.625 | 0.784 | 0.517 |
| qwen2.5-1.5b | PREFIX_ONLY | te_en | 175 | 0.689 | 0.843 | 0.453 |
| qwen2.5-1.5b | PREFIX_PLUS_REACTION | hi_en | 188 | 0.763 | 0.882 | 0.347 |
| qwen2.5-1.5b | PREFIX_PLUS_REACTION | ta_en | 177 | 0.604 | 0.648 | 0.438 |
| qwen2.5-1.5b | PREFIX_PLUS_REACTION | te_en | 175 | 0.638 | 0.697 | 0.419 |
| qwen2.5-1.5b | STREAMING | hi_en | 188 | 0.830 | 0.806 | 0.147 |
| qwen2.5-1.5b | STREAMING | ta_en | 177 | 0.655 | 0.648 | 0.337 |
| qwen2.5-1.5b | STREAMING | te_en | 175 | 0.691 | 0.652 | 0.267 |
| qwen3-0.6b | FULL | hi_en | 188 | 0.574 | 0.538 | 0.389 |
| qwen3-0.6b | FULL | ta_en | 177 | 0.522 | 0.364 | 0.292 |
| qwen3-0.6b | FULL | te_en | 175 | 0.516 | 0.348 | 0.279 |
| qwen3-0.6b | PREFIX_ONLY | hi_en | 188 | 0.591 | 0.419 | 0.211 |
| qwen3-0.6b | PREFIX_ONLY | ta_en | 177 | 0.588 | 0.489 | 0.303 |
| qwen3-0.6b | PREFIX_ONLY | te_en | 175 | 0.581 | 0.337 | 0.105 |
| qwen3-0.6b | PREFIX_PLUS_REACTION | hi_en | 188 | 0.544 | 0.387 | 0.274 |
| qwen3-0.6b | PREFIX_PLUS_REACTION | ta_en | 177 | 0.562 | 0.432 | 0.292 |
| qwen3-0.6b | PREFIX_PLUS_REACTION | te_en | 175 | 0.457 | 0.292 | 0.337 |
| qwen3-1.7b | FULL | hi_en | 188 | 0.792 | 0.871 | 0.284 |
| qwen3-1.7b | FULL | ta_en | 177 | 0.605 | 0.602 | 0.393 |
| qwen3-1.7b | FULL | te_en | 175 | 0.644 | 0.697 | 0.407 |
| qwen3-1.7b | PREFIX_ONLY | hi_en | 188 | 0.771 | 0.828 | 0.284 |
| qwen3-1.7b | PREFIX_ONLY | ta_en | 177 | 0.613 | 0.705 | 0.472 |
| qwen3-1.7b | PREFIX_ONLY | te_en | 175 | 0.543 | 0.697 | 0.593 |
| qwen3-1.7b | PREFIX_PLUS_REACTION | hi_en | 188 | 0.750 | 0.763 | 0.263 |
| qwen3-1.7b | PREFIX_PLUS_REACTION | ta_en | 177 | 0.666 | 0.648 | 0.315 |
| qwen3-1.7b | PREFIX_PLUS_REACTION | te_en | 175 | 0.663 | 0.674 | 0.349 |
| qwen3-1.7b | STREAMING | hi_en | 188 | 0.792 | 0.871 | 0.284 |
| qwen3-1.7b | STREAMING | ta_en | 177 | 0.605 | 0.602 | 0.393 |
| qwen3-1.7b | STREAMING | te_en | 175 | 0.644 | 0.697 | 0.407 |
| smollm2-1.7b | FULL | hi_en | 188 | 0.546 | 0.280 | 0.105 |
| smollm2-1.7b | FULL | ta_en | 177 | 0.526 | 0.830 | 0.708 |
| smollm2-1.7b | FULL | te_en | 175 | 0.516 | 0.764 | 0.686 |
| smollm2-1.7b | PREFIX_ONLY | hi_en | 188 | 0.531 | 0.570 | 0.505 |
| smollm2-1.7b | PREFIX_ONLY | ta_en | 177 | 0.332 | 1.000 | 1.000 |
| smollm2-1.7b | PREFIX_ONLY | te_en | 175 | 0.297 | 0.831 | 1.000 |
| smollm2-1.7b | PREFIX_PLUS_REACTION | hi_en | 188 | 0.565 | 0.301 | 0.095 |
| smollm2-1.7b | PREFIX_PLUS_REACTION | ta_en | 177 | 0.357 | 0.955 | 0.966 |
| smollm2-1.7b | PREFIX_PLUS_REACTION | te_en | 175 | 0.430 | 0.865 | 0.860 |
| smollm2-135m | FULL | hi_en | 188 | 0.447 | 0.796 | 0.800 |
| smollm2-135m | FULL | ta_en | 177 | 0.351 | 0.932 | 0.966 |
| smollm2-135m | FULL | te_en | 175 | 0.623 | 0.596 | 0.349 |
| smollm2-135m | PREFIX_ONLY | hi_en | 188 | 0.520 | 0.581 | 0.537 |
| smollm2-135m | PREFIX_ONLY | ta_en | 177 | 0.363 | 0.830 | 0.921 |
| smollm2-135m | PREFIX_ONLY | te_en | 175 | 0.629 | 0.427 | 0.128 |
| smollm2-135m | PREFIX_PLUS_REACTION | hi_en | 188 | 0.430 | 0.860 | 0.853 |
| smollm2-135m | PREFIX_PLUS_REACTION | ta_en | 177 | 0.348 | 0.920 | 0.966 |
| smollm2-135m | PREFIX_PLUS_REACTION | te_en | 175 | 0.537 | 0.506 | 0.430 |
| smollm2-360m | FULL | hi_en | 188 | 0.467 | 0.634 | 0.674 |
| smollm2-360m | FULL | ta_en | 177 | 0.515 | 0.625 | 0.584 |
| smollm2-360m | FULL | te_en | 175 | 0.617 | 0.449 | 0.186 |
| smollm2-360m | PREFIX_ONLY | hi_en | 188 | 0.536 | 0.667 | 0.579 |
| smollm2-360m | PREFIX_ONLY | ta_en | 177 | 0.432 | 0.511 | 0.640 |
| smollm2-360m | PREFIX_ONLY | te_en | 175 | 0.477 | 0.258 | 0.233 |
| smollm2-360m | PREFIX_PLUS_REACTION | hi_en | 188 | 0.528 | 0.763 | 0.663 |
| smollm2-360m | PREFIX_PLUS_REACTION | ta_en | 177 | 0.605 | 0.727 | 0.506 |
| smollm2-360m | PREFIX_PLUS_REACTION | te_en | 175 | 0.483 | 0.360 | 0.372 |
| stablelm2-1.6b | FULL | hi_en | 188 | 0.459 | 0.194 | 0.168 |
| stablelm2-1.6b | FULL | ta_en | 177 | 0.583 | 0.489 | 0.315 |
| stablelm2-1.6b | FULL | te_en | 175 | 0.503 | 0.472 | 0.465 |
| stablelm2-1.6b | PREFIX_ONLY | hi_en | 188 | 0.602 | 0.505 | 0.295 |
| stablelm2-1.6b | PREFIX_ONLY | ta_en | 177 | 0.453 | 0.875 | 0.831 |
| stablelm2-1.6b | PREFIX_ONLY | te_en | 175 | 0.370 | 0.742 | 0.884 |
| stablelm2-1.6b | PREFIX_PLUS_REACTION | hi_en | 188 | 0.455 | 0.151 | 0.084 |
| stablelm2-1.6b | PREFIX_PLUS_REACTION | ta_en | 177 | 0.535 | 0.602 | 0.528 |
| stablelm2-1.6b | PREFIX_PLUS_REACTION | te_en | 175 | 0.468 | 0.494 | 0.558 |
| tinyllama-1.1b | FULL | hi_en | 188 | 0.400 | 0.118 | 0.168 |
| tinyllama-1.1b | FULL | ta_en | 177 | 0.497 | 0.705 | 0.674 |
| tinyllama-1.1b | FULL | te_en | 175 | 0.424 | 0.876 | 0.872 |
| tinyllama-1.1b | PREFIX_ONLY | hi_en | 188 | 0.411 | 0.215 | 0.337 |
| tinyllama-1.1b | PREFIX_ONLY | ta_en | 177 | 0.506 | 0.693 | 0.652 |
| tinyllama-1.1b | PREFIX_ONLY | te_en | 175 | 0.395 | 0.528 | 0.721 |
| tinyllama-1.1b | PREFIX_PLUS_REACTION | hi_en | 188 | 0.447 | 0.204 | 0.221 |
| tinyllama-1.1b | PREFIX_PLUS_REACTION | ta_en | 177 | 0.461 | 0.705 | 0.730 |
| tinyllama-1.1b | PREFIX_PLUS_REACTION | te_en | 175 | 0.324 | 0.775 | 0.953 |

## Per recipient condition (test)

| model | view | condition | n | fraud recall | legit FPR | macro-F1 |
|---|---|---|---:|---:|---:|---:|
| danube2-1.8b | FULL | cooperative | 133 | 0.554 | 0.397 | 0.578 |
| danube2-1.8b | FULL | clarifying | 134 | 0.652 | 0.529 | 0.557 |
| danube2-1.8b | FULL | resistant | 136 | 0.671 | 0.394 | 0.639 |
| danube2-1.8b | FULL | suspicious | 137 | 0.667 | 0.500 | 0.581 |
| danube2-1.8b | PREFIX_ONLY | cooperative | 133 | 0.708 | 0.662 | 0.504 |
| danube2-1.8b | PREFIX_ONLY | clarifying | 134 | 0.682 | 0.662 | 0.494 |
| danube2-1.8b | PREFIX_ONLY | resistant | 136 | 0.671 | 0.667 | 0.490 |
| danube2-1.8b | PREFIX_ONLY | suspicious | 137 | 0.681 | 0.662 | 0.495 |
| danube2-1.8b | PREFIX_PLUS_REACTION | cooperative | 133 | 0.554 | 0.412 | 0.571 |
| danube2-1.8b | PREFIX_PLUS_REACTION | clarifying | 134 | 0.697 | 0.412 | 0.641 |
| danube2-1.8b | PREFIX_PLUS_REACTION | resistant | 136 | 0.629 | 0.348 | 0.640 |
| danube2-1.8b | PREFIX_PLUS_REACTION | suspicious | 137 | 0.768 | 0.647 | 0.541 |
| granite-1b | FULL | cooperative | 133 | 0.554 | 0.500 | 0.526 |
| granite-1b | FULL | clarifying | 134 | 0.348 | 0.294 | 0.513 |
| granite-1b | FULL | resistant | 136 | 0.471 | 0.515 | 0.478 |
| granite-1b | FULL | suspicious | 137 | 0.594 | 0.412 | 0.591 |
| granite-1b | PREFIX_ONLY | cooperative | 133 | 0.815 | 0.706 | 0.519 |
| granite-1b | PREFIX_ONLY | clarifying | 134 | 0.803 | 0.721 | 0.505 |
| granite-1b | PREFIX_ONLY | resistant | 136 | 0.814 | 0.727 | 0.511 |
| granite-1b | PREFIX_ONLY | suspicious | 137 | 0.812 | 0.721 | 0.512 |
| granite-1b | PREFIX_PLUS_REACTION | cooperative | 133 | 0.385 | 0.147 | 0.599 |
| granite-1b | PREFIX_PLUS_REACTION | clarifying | 134 | 0.621 | 0.618 | 0.494 |
| granite-1b | PREFIX_PLUS_REACTION | resistant | 136 | 0.214 | 0.318 | 0.413 |
| granite-1b | PREFIX_PLUS_REACTION | suspicious | 137 | 0.333 | 0.529 | 0.399 |
| qwen2-0.5b | FULL | cooperative | 133 | 0.538 | 0.471 | 0.534 |
| qwen2-0.5b | FULL | clarifying | 134 | 0.758 | 0.529 | 0.605 |
| qwen2-0.5b | FULL | resistant | 136 | 0.586 | 0.561 | 0.511 |
| qwen2-0.5b | FULL | suspicious | 137 | 0.522 | 0.353 | 0.583 |
| qwen2-0.5b | PREFIX_ONLY | cooperative | 133 | 0.492 | 0.574 | 0.458 |
| qwen2-0.5b | PREFIX_ONLY | clarifying | 134 | 0.500 | 0.574 | 0.462 |
| qwen2-0.5b | PREFIX_ONLY | resistant | 136 | 0.500 | 0.576 | 0.462 |
| qwen2-0.5b | PREFIX_ONLY | suspicious | 137 | 0.493 | 0.574 | 0.459 |
| qwen2-0.5b | PREFIX_PLUS_REACTION | cooperative | 133 | 0.723 | 0.441 | 0.637 |
| qwen2-0.5b | PREFIX_PLUS_REACTION | clarifying | 134 | 0.576 | 0.456 | 0.560 |
| qwen2-0.5b | PREFIX_PLUS_REACTION | resistant | 136 | 0.500 | 0.485 | 0.507 |
| qwen2-0.5b | PREFIX_PLUS_REACTION | suspicious | 137 | 0.449 | 0.368 | 0.537 |
| qwen2-1.5b | FULL | cooperative | 133 | 0.600 | 0.279 | 0.660 |
| qwen2-1.5b | FULL | clarifying | 134 | 0.697 | 0.309 | 0.694 |
| qwen2-1.5b | FULL | resistant | 136 | 0.729 | 0.364 | 0.682 |
| qwen2-1.5b | FULL | suspicious | 137 | 0.652 | 0.206 | 0.721 |
| qwen2-1.5b | PREFIX_ONLY | cooperative | 133 | 0.754 | 0.235 | 0.759 |
| qwen2-1.5b | PREFIX_ONLY | clarifying | 134 | 0.788 | 0.250 | 0.769 |
| qwen2-1.5b | PREFIX_ONLY | resistant | 136 | 0.771 | 0.258 | 0.757 |
| qwen2-1.5b | PREFIX_ONLY | suspicious | 137 | 0.768 | 0.250 | 0.759 |
| qwen2-1.5b | PREFIX_PLUS_REACTION | cooperative | 133 | 0.631 | 0.147 | 0.740 |
| qwen2-1.5b | PREFIX_PLUS_REACTION | clarifying | 134 | 0.788 | 0.294 | 0.746 |
| qwen2-1.5b | PREFIX_PLUS_REACTION | resistant | 136 | 0.571 | 0.197 | 0.681 |
| qwen2-1.5b | PREFIX_PLUS_REACTION | suspicious | 137 | 0.609 | 0.324 | 0.642 |
| qwen2.5-0.5b | FULL | cooperative | 133 | 0.323 | 0.309 | 0.492 |
| qwen2.5-0.5b | FULL | clarifying | 134 | 0.530 | 0.426 | 0.552 |
| qwen2.5-0.5b | FULL | resistant | 136 | 0.714 | 0.439 | 0.636 |
| qwen2.5-0.5b | FULL | suspicious | 137 | 0.696 | 0.515 | 0.586 |
| qwen2.5-0.5b | PREFIX_ONLY | cooperative | 133 | 0.585 | 0.485 | 0.549 |
| qwen2.5-0.5b | PREFIX_ONLY | clarifying | 134 | 0.576 | 0.485 | 0.545 |
| qwen2.5-0.5b | PREFIX_ONLY | resistant | 136 | 0.586 | 0.500 | 0.543 |
| qwen2.5-0.5b | PREFIX_ONLY | suspicious | 137 | 0.594 | 0.485 | 0.554 |
| qwen2.5-0.5b | PREFIX_PLUS_REACTION | cooperative | 133 | 0.569 | 0.250 | 0.658 |
| qwen2.5-0.5b | PREFIX_PLUS_REACTION | clarifying | 134 | 0.485 | 0.265 | 0.605 |
| qwen2.5-0.5b | PREFIX_PLUS_REACTION | resistant | 136 | 0.529 | 0.318 | 0.602 |
| qwen2.5-0.5b | PREFIX_PLUS_REACTION | suspicious | 137 | 0.609 | 0.338 | 0.635 |
| qwen2.5-1.5b | FULL | cooperative | 133 | 0.431 | 0.147 | 0.628 |
| qwen2.5-1.5b | FULL | clarifying | 134 | 0.833 | 0.265 | 0.783 |
| qwen2.5-1.5b | FULL | resistant | 136 | 0.743 | 0.273 | 0.735 |
| qwen2.5-1.5b | FULL | suspicious | 137 | 0.797 | 0.309 | 0.744 |
| qwen2.5-1.5b | PREFIX_ONLY | cooperative | 133 | 0.800 | 0.426 | 0.681 |
| qwen2.5-1.5b | PREFIX_ONLY | clarifying | 134 | 0.803 | 0.426 | 0.683 |
| qwen2.5-1.5b | PREFIX_ONLY | resistant | 136 | 0.800 | 0.455 | 0.669 |
| qwen2.5-1.5b | PREFIX_ONLY | suspicious | 137 | 0.812 | 0.426 | 0.689 |
| qwen2.5-1.5b | PREFIX_PLUS_REACTION | cooperative | 133 | 0.662 | 0.221 | 0.720 |
| qwen2.5-1.5b | PREFIX_PLUS_REACTION | clarifying | 134 | 0.894 | 0.397 | 0.742 |
| qwen2.5-1.5b | PREFIX_PLUS_REACTION | resistant | 136 | 0.714 | 0.303 | 0.706 |
| qwen2.5-1.5b | PREFIX_PLUS_REACTION | suspicious | 137 | 0.710 | 0.676 | 0.499 |
| qwen2.5-1.5b | STREAMING | cooperative | 133 | 0.431 | 0.147 | 0.628 |
| qwen2.5-1.5b | STREAMING | clarifying | 134 | 0.833 | 0.265 | 0.783 |
| qwen2.5-1.5b | STREAMING | resistant | 136 | 0.743 | 0.273 | 0.735 |
| qwen2.5-1.5b | STREAMING | suspicious | 137 | 0.797 | 0.309 | 0.744 |
| qwen3-0.6b | FULL | cooperative | 133 | 0.431 | 0.324 | 0.548 |
| qwen3-0.6b | FULL | clarifying | 134 | 0.515 | 0.265 | 0.621 |
| qwen3-0.6b | FULL | resistant | 136 | 0.500 | 0.439 | 0.529 |
| qwen3-0.6b | FULL | suspicious | 137 | 0.232 | 0.265 | 0.448 |
| qwen3-0.6b | PREFIX_ONLY | cooperative | 133 | 0.446 | 0.221 | 0.604 |
| qwen3-0.6b | PREFIX_ONLY | clarifying | 134 | 0.394 | 0.206 | 0.579 |
| qwen3-0.6b | PREFIX_ONLY | resistant | 136 | 0.400 | 0.197 | 0.581 |
| qwen3-0.6b | PREFIX_ONLY | suspicious | 137 | 0.420 | 0.206 | 0.592 |
| qwen3-0.6b | PREFIX_PLUS_REACTION | cooperative | 133 | 0.431 | 0.353 | 0.535 |
| qwen3-0.6b | PREFIX_PLUS_REACTION | clarifying | 134 | 0.303 | 0.221 | 0.515 |
| qwen3-0.6b | PREFIX_PLUS_REACTION | resistant | 136 | 0.443 | 0.394 | 0.520 |
| qwen3-0.6b | PREFIX_PLUS_REACTION | suspicious | 137 | 0.304 | 0.235 | 0.508 |
| qwen3-1.7b | FULL | cooperative | 133 | 0.815 | 0.353 | 0.728 |
| qwen3-1.7b | FULL | clarifying | 134 | 0.742 | 0.368 | 0.686 |
| qwen3-1.7b | FULL | resistant | 136 | 0.557 | 0.227 | 0.659 |
| qwen3-1.7b | FULL | suspicious | 137 | 0.797 | 0.485 | 0.649 |
| qwen3-1.7b | PREFIX_ONLY | cooperative | 133 | 0.754 | 0.441 | 0.652 |
| qwen3-1.7b | PREFIX_ONLY | clarifying | 134 | 0.727 | 0.441 | 0.640 |
| qwen3-1.7b | PREFIX_ONLY | resistant | 136 | 0.743 | 0.455 | 0.642 |
| qwen3-1.7b | PREFIX_ONLY | suspicious | 137 | 0.754 | 0.441 | 0.653 |
| qwen3-1.7b | PREFIX_PLUS_REACTION | cooperative | 133 | 0.785 | 0.250 | 0.767 |
| qwen3-1.7b | PREFIX_PLUS_REACTION | clarifying | 134 | 0.636 | 0.221 | 0.707 |
| qwen3-1.7b | PREFIX_PLUS_REACTION | resistant | 136 | 0.600 | 0.242 | 0.675 |
| qwen3-1.7b | PREFIX_PLUS_REACTION | suspicious | 137 | 0.768 | 0.515 | 0.620 |
| qwen3-1.7b | STREAMING | cooperative | 133 | 0.815 | 0.353 | 0.728 |
| qwen3-1.7b | STREAMING | clarifying | 134 | 0.742 | 0.368 | 0.686 |
| qwen3-1.7b | STREAMING | resistant | 136 | 0.557 | 0.227 | 0.659 |
| qwen3-1.7b | STREAMING | suspicious | 137 | 0.797 | 0.485 | 0.649 |
| smollm2-1.7b | FULL | cooperative | 133 | 0.646 | 0.574 | 0.529 |
| smollm2-1.7b | FULL | clarifying | 134 | 0.455 | 0.338 | 0.554 |
| smollm2-1.7b | FULL | resistant | 136 | 0.657 | 0.591 | 0.528 |
| smollm2-1.7b | FULL | suspicious | 137 | 0.710 | 0.456 | 0.625 |
| smollm2-1.7b | PREFIX_ONLY | cooperative | 133 | 0.800 | 0.824 | 0.430 |
| smollm2-1.7b | PREFIX_ONLY | clarifying | 134 | 0.788 | 0.824 | 0.427 |
| smollm2-1.7b | PREFIX_ONLY | resistant | 136 | 0.800 | 0.833 | 0.430 |
| smollm2-1.7b | PREFIX_ONLY | suspicious | 137 | 0.797 | 0.824 | 0.433 |
| smollm2-1.7b | PREFIX_PLUS_REACTION | cooperative | 133 | 0.615 | 0.603 | 0.499 |
| smollm2-1.7b | PREFIX_PLUS_REACTION | clarifying | 134 | 0.727 | 0.603 | 0.549 |
| smollm2-1.7b | PREFIX_PLUS_REACTION | resistant | 136 | 0.714 | 0.712 | 0.480 |
| smollm2-1.7b | PREFIX_PLUS_REACTION | suspicious | 137 | 0.739 | 0.588 | 0.564 |
| smollm2-135m | FULL | cooperative | 133 | 0.723 | 0.647 | 0.519 |
| smollm2-135m | FULL | clarifying | 134 | 0.864 | 0.735 | 0.519 |
| smollm2-135m | FULL | resistant | 136 | 0.757 | 0.742 | 0.478 |
| smollm2-135m | FULL | suspicious | 137 | 0.754 | 0.721 | 0.489 |
| smollm2-135m | PREFIX_ONLY | cooperative | 133 | 0.600 | 0.529 | 0.533 |
| smollm2-135m | PREFIX_ONLY | clarifying | 134 | 0.636 | 0.529 | 0.550 |
| smollm2-135m | PREFIX_ONLY | resistant | 136 | 0.600 | 0.545 | 0.526 |
| smollm2-135m | PREFIX_ONLY | suspicious | 137 | 0.609 | 0.529 | 0.538 |
| smollm2-135m | PREFIX_PLUS_REACTION | cooperative | 133 | 0.738 | 0.735 | 0.469 |
| smollm2-135m | PREFIX_PLUS_REACTION | clarifying | 134 | 0.803 | 0.809 | 0.443 |
| smollm2-135m | PREFIX_PLUS_REACTION | resistant | 136 | 0.757 | 0.788 | 0.447 |
| smollm2-135m | PREFIX_PLUS_REACTION | suspicious | 137 | 0.754 | 0.691 | 0.508 |
| smollm2-360m | FULL | cooperative | 133 | 0.631 | 0.485 | 0.571 |
| smollm2-360m | FULL | clarifying | 134 | 0.727 | 0.529 | 0.591 |
| smollm2-360m | FULL | resistant | 136 | 0.457 | 0.409 | 0.521 |
| smollm2-360m | FULL | suspicious | 137 | 0.478 | 0.529 | 0.474 |
| smollm2-360m | PREFIX_ONLY | cooperative | 133 | 0.446 | 0.485 | 0.480 |
| smollm2-360m | PREFIX_ONLY | clarifying | 134 | 0.485 | 0.500 | 0.492 |
| smollm2-360m | PREFIX_ONLY | resistant | 136 | 0.500 | 0.485 | 0.507 |
| smollm2-360m | PREFIX_ONLY | suspicious | 137 | 0.493 | 0.485 | 0.504 |
| smollm2-360m | PREFIX_PLUS_REACTION | cooperative | 133 | 0.492 | 0.485 | 0.504 |
| smollm2-360m | PREFIX_PLUS_REACTION | clarifying | 134 | 0.591 | 0.515 | 0.536 |
| smollm2-360m | PREFIX_PLUS_REACTION | resistant | 136 | 0.700 | 0.636 | 0.521 |
| smollm2-360m | PREFIX_PLUS_REACTION | suspicious | 137 | 0.681 | 0.441 | 0.619 |
| stablelm2-1.6b | FULL | cooperative | 133 | 0.369 | 0.353 | 0.500 |
| stablelm2-1.6b | FULL | clarifying | 134 | 0.424 | 0.235 | 0.584 |
| stablelm2-1.6b | FULL | resistant | 136 | 0.386 | 0.318 | 0.521 |
| stablelm2-1.6b | FULL | suspicious | 137 | 0.348 | 0.338 | 0.492 |
| stablelm2-1.6b | PREFIX_ONLY | cooperative | 133 | 0.723 | 0.662 | 0.510 |
| stablelm2-1.6b | PREFIX_ONLY | clarifying | 134 | 0.697 | 0.647 | 0.509 |
| stablelm2-1.6b | PREFIX_ONLY | resistant | 136 | 0.700 | 0.682 | 0.493 |
| stablelm2-1.6b | PREFIX_ONLY | suspicious | 137 | 0.696 | 0.647 | 0.511 |
| stablelm2-1.6b | PREFIX_PLUS_REACTION | cooperative | 133 | 0.369 | 0.368 | 0.493 |
| stablelm2-1.6b | PREFIX_PLUS_REACTION | clarifying | 134 | 0.515 | 0.529 | 0.492 |
| stablelm2-1.6b | PREFIX_PLUS_REACTION | resistant | 136 | 0.357 | 0.288 | 0.517 |
| stablelm2-1.6b | PREFIX_PLUS_REACTION | suspicious | 137 | 0.406 | 0.338 | 0.526 |
| tinyllama-1.1b | FULL | cooperative | 133 | 0.615 | 0.662 | 0.465 |
| tinyllama-1.1b | FULL | clarifying | 134 | 0.500 | 0.500 | 0.500 |
| tinyllama-1.1b | FULL | resistant | 136 | 0.586 | 0.606 | 0.486 |
| tinyllama-1.1b | FULL | suspicious | 137 | 0.536 | 0.471 | 0.533 |
| tinyllama-1.1b | PREFIX_ONLY | cooperative | 133 | 0.462 | 0.559 | 0.451 |
| tinyllama-1.1b | PREFIX_ONLY | clarifying | 134 | 0.470 | 0.559 | 0.455 |
| tinyllama-1.1b | PREFIX_ONLY | resistant | 136 | 0.486 | 0.561 | 0.463 |
| tinyllama-1.1b | PREFIX_ONLY | suspicious | 137 | 0.478 | 0.574 | 0.452 |
| tinyllama-1.1b | PREFIX_PLUS_REACTION | cooperative | 133 | 0.415 | 0.618 | 0.398 |
| tinyllama-1.1b | PREFIX_PLUS_REACTION | clarifying | 134 | 0.561 | 0.676 | 0.433 |
| tinyllama-1.1b | PREFIX_PLUS_REACTION | resistant | 136 | 0.671 | 0.621 | 0.517 |
| tinyllama-1.1b | PREFIX_PLUS_REACTION | suspicious | 137 | 0.565 | 0.574 | 0.494 |

## Branch metrics (complete sibling sets, test)

| model | view | n histories | disagreement | score spread (mean) | all-correct |
|---|---|---:|---:|---:|---:|
| danube2-1.8b | FULL | 122 | 0.680 | 0.025 | 0.230 |
| danube2-1.8b | PREFIX_ONLY | 122 | 0.000 | 0.000 | 0.516 |
| danube2-1.8b | PREFIX_PLUS_REACTION | 122 | 0.549 | 0.035 | 0.303 |
| danube2-1.8b | STREAMING | 0 | — | — | — |
| granite-1b | FULL | 122 | 0.820 | 0.392 | 0.123 |
| granite-1b | PREFIX_ONLY | 122 | 0.008 | 0.000 | 0.533 |
| granite-1b | PREFIX_PLUS_REACTION | 122 | 0.795 | 0.395 | 0.107 |
| granite-1b | STREAMING | 0 | — | — | — |
| qwen2-0.5b | FULL | 122 | 0.582 | 0.028 | 0.262 |
| qwen2-0.5b | PREFIX_ONLY | 122 | 0.000 | 0.000 | 0.443 |
| qwen2-0.5b | PREFIX_PLUS_REACTION | 122 | 0.426 | 0.020 | 0.328 |
| qwen2-0.5b | STREAMING | 0 | — | — | — |
| qwen2-1.5b | FULL | 122 | 0.475 | 0.236 | 0.443 |
| qwen2-1.5b | PREFIX_ONLY | 122 | 0.000 | 0.000 | 0.754 |
| qwen2-1.5b | PREFIX_PLUS_REACTION | 122 | 0.402 | 0.185 | 0.492 |
| qwen2-1.5b | STREAMING | 0 | — | — | — |
| qwen2.5-0.5b | FULL | 122 | 0.623 | 0.377 | 0.230 |
| qwen2.5-0.5b | PREFIX_ONLY | 122 | 0.000 | 0.003 | 0.533 |
| qwen2.5-0.5b | PREFIX_PLUS_REACTION | 122 | 0.336 | 0.250 | 0.467 |
| qwen2.5-0.5b | STREAMING | 0 | — | — | — |
| qwen2.5-1.5b | FULL | 122 | 0.467 | 0.450 | 0.475 |
| qwen2.5-1.5b | PREFIX_ONLY | 122 | 0.000 | 0.003 | 0.689 |
| qwen2.5-1.5b | PREFIX_PLUS_REACTION | 122 | 0.549 | 0.524 | 0.352 |
| qwen2.5-1.5b | STREAMING | 122 | 0.467 | 0.450 | 0.475 |
| qwen3-0.6b | FULL | 122 | 0.664 | 0.171 | 0.189 |
| qwen3-0.6b | PREFIX_ONLY | 122 | 0.000 | 0.000 | 0.615 |
| qwen3-0.6b | PREFIX_PLUS_REACTION | 122 | 0.484 | 0.140 | 0.279 |
| qwen3-0.6b | STREAMING | 0 | — | — | — |
| qwen3-1.7b | FULL | 122 | 0.607 | 0.000 | 0.336 |
| qwen3-1.7b | PREFIX_ONLY | 122 | 0.008 | 0.000 | 0.648 |
| qwen3-1.7b | PREFIX_PLUS_REACTION | 122 | 0.426 | 0.000 | 0.459 |
| qwen3-1.7b | STREAMING | 122 | 0.607 | 0.000 | 0.336 |
| smollm2-1.7b | FULL | 122 | 0.525 | 0.063 | 0.295 |
| smollm2-1.7b | PREFIX_ONLY | 122 | 0.000 | 0.000 | 0.467 |
| smollm2-1.7b | PREFIX_PLUS_REACTION | 122 | 0.336 | 0.053 | 0.369 |
| smollm2-1.7b | STREAMING | 0 | — | — | — |
| smollm2-135m | FULL | 122 | 0.328 | 0.023 | 0.361 |
| smollm2-135m | PREFIX_ONLY | 122 | 0.000 | 0.000 | 0.549 |
| smollm2-135m | PREFIX_PLUS_REACTION | 122 | 0.254 | 0.015 | 0.361 |
| smollm2-135m | STREAMING | 0 | — | — | — |
| smollm2-360m | FULL | 122 | 0.631 | 0.252 | 0.221 |
| smollm2-360m | PREFIX_ONLY | 122 | 0.000 | 0.001 | 0.492 |
| smollm2-360m | PREFIX_PLUS_REACTION | 122 | 0.533 | 0.247 | 0.262 |
| smollm2-360m | STREAMING | 0 | — | — | — |
| stablelm2-1.6b | FULL | 122 | 0.557 | 0.247 | 0.246 |
| stablelm2-1.6b | PREFIX_ONLY | 122 | 0.000 | 0.000 | 0.508 |
| stablelm2-1.6b | PREFIX_PLUS_REACTION | 122 | 0.492 | 0.176 | 0.295 |
| stablelm2-1.6b | STREAMING | 0 | — | — | — |
| tinyllama-1.1b | FULL | 122 | 0.500 | 0.015 | 0.246 |
| tinyllama-1.1b | PREFIX_ONLY | 122 | 0.000 | 0.000 | 0.451 |
| tinyllama-1.1b | PREFIX_PLUS_REACTION | 122 | 0.377 | 0.011 | 0.270 |
| tinyllama-1.1b | STREAMING | 0 | — | — | — |

## PREFIX_PLUS_REACTION − PREFIX_ONLY (mean Δ P(fraud), test)

| model | condition / label | mean Δ | 95% CI |
|---|---|---:|---|
| danube2-1.8b | all | -0.020 | |
| danube2-1.8b | cooperative | -0.026 | |
| danube2-1.8b | clarifying | -0.023 | |
| danube2-1.8b | resistant | -0.024 | |
| danube2-1.8b | suspicious | -0.008 | |
| danube2-1.8b | label=fraud | -0.013 | |
| danube2-1.8b | label=legitimate | -0.027 | |
| danube2-1.8b | cooperative|fraud | -0.023 |  [-0.035, -0.011] |
| danube2-1.8b | cooperative|legitimate | -0.029 |  [-0.039, -0.019] |
| danube2-1.8b | clarifying|fraud | -0.017 |  [-0.032, -0.005] |
| danube2-1.8b | clarifying|legitimate | -0.029 |  [-0.044, -0.015] |
| danube2-1.8b | resistant|fraud | -0.017 |  [-0.031, -0.005] |
| danube2-1.8b | resistant|legitimate | -0.030 |  [-0.044, -0.018] |
| danube2-1.8b | suspicious|fraud | 0.004 |  [-0.012, 0.018] |
| danube2-1.8b | suspicious|legitimate | -0.021 |  [-0.036, -0.008] |
| granite-1b | all | -0.223 | |
| granite-1b | cooperative | -0.262 | |
| granite-1b | clarifying | -0.119 | |
| granite-1b | resistant | -0.291 | |
| granite-1b | suspicious | -0.219 | |
| granite-1b | label=fraud | -0.258 | |
| granite-1b | label=legitimate | -0.188 | |
| granite-1b | cooperative|fraud | -0.236 |  [-0.298, -0.179] |
| granite-1b | cooperative|legitimate | -0.288 |  [-0.349, -0.215] |
| granite-1b | clarifying|fraud | -0.151 |  [-0.210, -0.089] |
| granite-1b | clarifying|legitimate | -0.088 |  [-0.164, -0.017] |
| granite-1b | resistant|fraud | -0.340 |  [-0.403, -0.279] |
| granite-1b | resistant|legitimate | -0.239 |  [-0.314, -0.153] |
| granite-1b | suspicious|fraud | -0.297 |  [-0.353, -0.234] |
| granite-1b | suspicious|legitimate | -0.140 |  [-0.224, -0.050] |
| qwen2-0.5b | all | -0.005 | |
| qwen2-0.5b | cooperative | -0.002 | |
| qwen2-0.5b | clarifying | -0.004 | |
| qwen2-0.5b | resistant | -0.004 | |
| qwen2-0.5b | suspicious | -0.011 | |
| qwen2-0.5b | label=fraud | -0.001 | |
| qwen2-0.5b | label=legitimate | -0.009 | |
| qwen2-0.5b | cooperative|fraud | 0.003 |  [0.001, 0.006] |
| qwen2-0.5b | cooperative|legitimate | -0.007 |  [-0.013, -0.002] |
| qwen2-0.5b | clarifying|fraud | -0.001 |  [-0.004, 0.002] |
| qwen2-0.5b | clarifying|legitimate | -0.006 |  [-0.012, -0.002] |
| qwen2-0.5b | resistant|fraud | -0.003 |  [-0.006, -0.001] |
| qwen2-0.5b | resistant|legitimate | -0.004 |  [-0.007, -0.001] |
| qwen2-0.5b | suspicious|fraud | -0.004 |  [-0.009, 0.001] |
| qwen2-0.5b | suspicious|legitimate | -0.018 |  [-0.027, -0.009] |
| qwen2-1.5b | all | -0.057 | |
| qwen2-1.5b | cooperative | -0.088 | |
| qwen2-1.5b | clarifying | -0.016 | |
| qwen2-1.5b | resistant | -0.077 | |
| qwen2-1.5b | suspicious | -0.046 | |
| qwen2-1.5b | label=fraud | -0.040 | |
| qwen2-1.5b | label=legitimate | -0.073 | |
| qwen2-1.5b | cooperative|fraud | -0.035 |  [-0.073, 0.001] |
| qwen2-1.5b | cooperative|legitimate | -0.138 |  [-0.198, -0.080] |
| qwen2-1.5b | clarifying|fraud | -0.001 |  [-0.023, 0.022] |
| qwen2-1.5b | clarifying|legitimate | -0.032 |  [-0.065, 0.006] |
| qwen2-1.5b | resistant|fraud | -0.061 |  [-0.098, -0.024] |
| qwen2-1.5b | resistant|legitimate | -0.094 |  [-0.130, -0.054] |
| qwen2-1.5b | suspicious|fraud | -0.062 |  [-0.102, -0.019] |
| qwen2-1.5b | suspicious|legitimate | -0.030 |  [-0.071, 0.010] |
| qwen2.5-0.5b | all | -0.003 | |
| qwen2.5-0.5b | cooperative | -0.009 | |
| qwen2.5-0.5b | clarifying | -0.018 | |
| qwen2.5-0.5b | resistant | -0.014 | |
| qwen2.5-0.5b | suspicious | 0.028 | |
| qwen2.5-0.5b | label=fraud | 0.011 | |
| qwen2.5-0.5b | label=legitimate | -0.017 | |
| qwen2.5-0.5b | cooperative|fraud | 0.031 |  [-0.007, 0.072] |
| qwen2.5-0.5b | cooperative|legitimate | -0.047 |  [-0.132, 0.027] |
| qwen2.5-0.5b | clarifying|fraud | -0.002 |  [-0.067, 0.067] |
| qwen2.5-0.5b | clarifying|legitimate | -0.033 |  [-0.115, 0.041] |
| qwen2.5-0.5b | resistant|fraud | 0.001 |  [-0.046, 0.060] |
| qwen2.5-0.5b | resistant|legitimate | -0.030 |  [-0.108, 0.032] |
| qwen2.5-0.5b | suspicious|fraud | 0.016 |  [-0.070, 0.109] |
| qwen2.5-0.5b | suspicious|legitimate | 0.041 |  [-0.038, 0.120] |
| qwen2.5-1.5b | all | -0.096 | |
| qwen2.5-1.5b | cooperative | -0.224 | |
| qwen2.5-1.5b | clarifying | -0.021 | |
| qwen2.5-1.5b | resistant | -0.169 | |
| qwen2.5-1.5b | suspicious | 0.027 | |
| qwen2.5-1.5b | label=fraud | -0.089 | |
| qwen2.5-1.5b | label=legitimate | -0.103 | |
| qwen2.5-1.5b | cooperative|fraud | -0.202 |  [-0.319, -0.090] |
| qwen2.5-1.5b | cooperative|legitimate | -0.244 |  [-0.412, -0.092] |
| qwen2.5-1.5b | clarifying|fraud | 0.074 |  [-0.028, 0.174] |
| qwen2.5-1.5b | clarifying|legitimate | -0.114 |  [-0.245, -0.009] |
| qwen2.5-1.5b | resistant|fraud | -0.155 |  [-0.270, -0.039] |
| qwen2.5-1.5b | resistant|legitimate | -0.184 |  [-0.314, -0.060] |
| qwen2.5-1.5b | suspicious|fraud | -0.072 |  [-0.171, 0.047] |
| qwen2.5-1.5b | suspicious|legitimate | 0.127 |  [-0.025, 0.273] |
| qwen3-0.6b | all | -0.025 | |
| qwen3-0.6b | cooperative | -0.015 | |
| qwen3-0.6b | clarifying | -0.023 | |
| qwen3-0.6b | resistant | 0.009 | |
| qwen3-0.6b | suspicious | -0.069 | |
| qwen3-0.6b | label=fraud | -0.031 | |
| qwen3-0.6b | label=legitimate | -0.019 | |
| qwen3-0.6b | cooperative|fraud | -0.016 |  [-0.035, 0.001] |
| qwen3-0.6b | cooperative|legitimate | -0.014 |  [-0.048, 0.022] |
| qwen3-0.6b | clarifying|fraud | -0.030 |  [-0.047, -0.010] |
| qwen3-0.6b | clarifying|legitimate | -0.017 |  [-0.044, 0.009] |
| qwen3-0.6b | resistant|fraud | -0.001 |  [-0.014, 0.011] |
| qwen3-0.6b | resistant|legitimate | 0.019 |  [-0.005, 0.043] |
| qwen3-0.6b | suspicious|fraud | -0.076 |  [-0.129, -0.024] |
| qwen3-0.6b | suspicious|legitimate | -0.061 |  [-0.123, -0.003] |
| qwen3-1.7b | all | -0.000 | |
| qwen3-1.7b | cooperative | -0.000 | |
| qwen3-1.7b | clarifying | -0.000 | |
| qwen3-1.7b | resistant | -0.000 | |
| qwen3-1.7b | suspicious | 0.000 | |
| qwen3-1.7b | label=fraud | -0.000 | |
| qwen3-1.7b | label=legitimate | -0.000 | |
| qwen3-1.7b | cooperative|fraud | 0.000 |  [-0.000, 0.000] |
| qwen3-1.7b | cooperative|legitimate | -0.000 |  [-0.000, -0.000] |
| qwen3-1.7b | clarifying|fraud | 0.000 |  [0.000, 0.000] |
| qwen3-1.7b | clarifying|legitimate | -0.000 |  [-0.000, 0.000] |
| qwen3-1.7b | resistant|fraud | -0.000 |  [-0.000, 0.000] |
| qwen3-1.7b | resistant|legitimate | -0.000 |  [-0.001, 0.000] |
| qwen3-1.7b | suspicious|fraud | 0.000 |  [-0.000, 0.000] |
| qwen3-1.7b | suspicious|legitimate | 0.000 |  [0.000, 0.000] |
| smollm2-1.7b | all | -0.025 | |
| smollm2-1.7b | cooperative | -0.030 | |
| smollm2-1.7b | clarifying | -0.028 | |
| smollm2-1.7b | resistant | -0.019 | |
| smollm2-1.7b | suspicious | -0.021 | |
| smollm2-1.7b | label=fraud | -0.018 | |
| smollm2-1.7b | label=legitimate | -0.032 | |
| smollm2-1.7b | cooperative|fraud | -0.024 |  [-0.032, -0.014] |
| smollm2-1.7b | cooperative|legitimate | -0.036 |  [-0.044, -0.028] |
| smollm2-1.7b | clarifying|fraud | -0.017 |  [-0.029, -0.005] |
| smollm2-1.7b | clarifying|legitimate | -0.040 |  [-0.052, -0.030] |
| smollm2-1.7b | resistant|fraud | -0.016 |  [-0.027, -0.007] |
| smollm2-1.7b | resistant|legitimate | -0.023 |  [-0.033, -0.014] |
| smollm2-1.7b | suspicious|fraud | -0.014 |  [-0.027, -0.001] |
| smollm2-1.7b | suspicious|legitimate | -0.028 |  [-0.042, -0.017] |
| smollm2-135m | all | -0.002 | |
| smollm2-135m | cooperative | -0.003 | |
| smollm2-135m | clarifying | 0.001 | |
| smollm2-135m | resistant | -0.002 | |
| smollm2-135m | suspicious | -0.004 | |
| smollm2-135m | label=fraud | -0.003 | |
| smollm2-135m | label=legitimate | -0.001 | |
| smollm2-135m | cooperative|fraud | -0.004 |  [-0.008, -0.001] |
| smollm2-135m | cooperative|legitimate | -0.002 |  [-0.006, 0.001] |
| smollm2-135m | clarifying|fraud | -0.000 |  [-0.004, 0.004] |
| smollm2-135m | clarifying|legitimate | 0.002 |  [-0.004, 0.006] |
| smollm2-135m | resistant|fraud | -0.003 |  [-0.007, 0.001] |
| smollm2-135m | resistant|legitimate | 0.000 |  [-0.004, 0.004] |
| smollm2-135m | suspicious|fraud | -0.004 |  [-0.008, -0.000] |
| smollm2-135m | suspicious|legitimate | -0.003 |  [-0.008, 0.001] |
| smollm2-360m | all | 0.064 | |
| smollm2-360m | cooperative | 0.037 | |
| smollm2-360m | clarifying | 0.058 | |
| smollm2-360m | resistant | 0.108 | |
| smollm2-360m | suspicious | 0.053 | |
| smollm2-360m | label=fraud | 0.083 | |
| smollm2-360m | label=legitimate | 0.046 | |
| smollm2-360m | cooperative|fraud | 0.054 |  [-0.007, 0.110] |
| smollm2-360m | cooperative|legitimate | 0.022 |  [-0.038, 0.076] |
| smollm2-360m | clarifying|fraud | 0.072 |  [0.004, 0.143] |
| smollm2-360m | clarifying|legitimate | 0.045 |  [-0.015, 0.098] |
| smollm2-360m | resistant|fraud | 0.098 |  [0.035, 0.154] |
| smollm2-360m | resistant|legitimate | 0.120 |  [0.057, 0.175] |
| smollm2-360m | suspicious|fraud | 0.105 |  [0.027, 0.180] |
| smollm2-360m | suspicious|legitimate | 0.000 |  [-0.070, 0.061] |
| stablelm2-1.6b | all | -0.061 | |
| stablelm2-1.6b | cooperative | -0.067 | |
| stablelm2-1.6b | clarifying | -0.024 | |
| stablelm2-1.6b | resistant | -0.076 | |
| stablelm2-1.6b | suspicious | -0.076 | |
| stablelm2-1.6b | label=fraud | -0.066 | |
| stablelm2-1.6b | label=legitimate | -0.056 | |
| stablelm2-1.6b | cooperative|fraud | -0.073 |  [-0.114, -0.034] |
| stablelm2-1.6b | cooperative|legitimate | -0.060 |  [-0.088, -0.033] |
| stablelm2-1.6b | clarifying|fraud | -0.016 |  [-0.058, 0.032] |
| stablelm2-1.6b | clarifying|legitimate | -0.032 |  [-0.055, -0.010] |
| stablelm2-1.6b | resistant|fraud | -0.089 |  [-0.122, -0.048] |
| stablelm2-1.6b | resistant|legitimate | -0.062 |  [-0.092, -0.036] |
| stablelm2-1.6b | suspicious|fraud | -0.083 |  [-0.120, -0.049] |
| stablelm2-1.6b | suspicious|legitimate | -0.069 |  [-0.095, -0.045] |
| tinyllama-1.1b | all | 0.012 | |
| tinyllama-1.1b | cooperative | 0.010 | |
| tinyllama-1.1b | clarifying | 0.013 | |
| tinyllama-1.1b | resistant | 0.015 | |
| tinyllama-1.1b | suspicious | 0.010 | |
| tinyllama-1.1b | label=fraud | 0.016 | |
| tinyllama-1.1b | label=legitimate | 0.008 | |
| tinyllama-1.1b | cooperative|fraud | 0.012 |  [0.001, 0.025] |
| tinyllama-1.1b | cooperative|legitimate | 0.008 |  [0.003, 0.016] |
| tinyllama-1.1b | clarifying|fraud | 0.017 |  [0.006, 0.031] |
| tinyllama-1.1b | clarifying|legitimate | 0.009 |  [0.003, 0.015] |
| tinyllama-1.1b | resistant|fraud | 0.018 |  [0.008, 0.032] |
| tinyllama-1.1b | resistant|legitimate | 0.011 |  [0.005, 0.018] |
| tinyllama-1.1b | suspicious|fraud | 0.016 |  [0.004, 0.031] |
| tinyllama-1.1b | suspicious|legitimate | 0.004 |  [-0.002, 0.013] |

## Complete-set slice (test)

| model | view | n | macro-F1 | disagreement | all-correct |
|---|---|---:|---:|---:|---:|
| danube2-1.8b | FULL | 488 | 0.596 | 0.680 | 0.230 |
| danube2-1.8b | PREFIX_ONLY | 488 | 0.504 | 0.000 | 0.516 |
| danube2-1.8b | PREFIX_PLUS_REACTION | 488 | 0.600 | 0.549 | 0.303 |
| danube2-1.8b | STREAMING | 0 | — | — | — |
| granite-1b | FULL | 488 | 0.534 | 0.820 | 0.123 |
| granite-1b | PREFIX_ONLY | 488 | 0.510 | 0.008 | 0.533 |
| granite-1b | PREFIX_PLUS_REACTION | 488 | 0.491 | 0.795 | 0.107 |
| granite-1b | STREAMING | 0 | — | — | — |
| qwen2-0.5b | FULL | 488 | 0.555 | 0.582 | 0.262 |
| qwen2-0.5b | PREFIX_ONLY | 488 | 0.442 | 0.000 | 0.443 |
| qwen2-0.5b | PREFIX_PLUS_REACTION | 488 | 0.541 | 0.426 | 0.328 |
| qwen2-0.5b | STREAMING | 0 | — | — | — |
| qwen2-1.5b | FULL | 488 | 0.688 | 0.475 | 0.443 |
| qwen2-1.5b | PREFIX_ONLY | 488 | 0.754 | 0.000 | 0.754 |
| qwen2-1.5b | PREFIX_PLUS_REACTION | 488 | 0.709 | 0.402 | 0.492 |
| qwen2-1.5b | STREAMING | 0 | — | — | — |
| qwen2.5-0.5b | FULL | 488 | 0.572 | 0.623 | 0.230 |
| qwen2.5-0.5b | PREFIX_ONLY | 488 | 0.532 | 0.000 | 0.533 |
| qwen2.5-0.5b | PREFIX_PLUS_REACTION | 488 | 0.629 | 0.336 | 0.467 |
| qwen2.5-0.5b | STREAMING | 0 | — | — | — |
| qwen2.5-1.5b | FULL | 488 | 0.739 | 0.467 | 0.475 |
| qwen2.5-1.5b | PREFIX_ONLY | 488 | 0.685 | 0.000 | 0.689 |
| qwen2.5-1.5b | PREFIX_PLUS_REACTION | 488 | 0.669 | 0.549 | 0.352 |
| qwen2.5-1.5b | STREAMING | 488 | 0.739 | 0.467 | 0.475 |
| qwen3-0.6b | FULL | 488 | 0.541 | 0.664 | 0.189 |
| qwen3-0.6b | PREFIX_ONLY | 488 | 0.598 | 0.000 | 0.615 |
| qwen3-0.6b | PREFIX_PLUS_REACTION | 488 | 0.531 | 0.484 | 0.279 |
| qwen3-0.6b | STREAMING | 0 | — | — | — |
| qwen3-1.7b | FULL | 488 | 0.703 | 0.607 | 0.336 |
| qwen3-1.7b | PREFIX_ONLY | 488 | 0.648 | 0.008 | 0.648 |
| qwen3-1.7b | PREFIX_PLUS_REACTION | 488 | 0.709 | 0.426 | 0.459 |
| qwen3-1.7b | STREAMING | 488 | 0.703 | 0.607 | 0.336 |
| smollm2-1.7b | FULL | 488 | 0.557 | 0.525 | 0.295 |
| smollm2-1.7b | PREFIX_ONLY | 488 | 0.419 | 0.000 | 0.467 |
| smollm2-1.7b | PREFIX_PLUS_REACTION | 488 | 0.526 | 0.336 | 0.369 |
| smollm2-1.7b | STREAMING | 0 | — | — | — |
| smollm2-135m | FULL | 488 | 0.498 | 0.328 | 0.361 |
| smollm2-135m | PREFIX_ONLY | 488 | 0.548 | 0.000 | 0.549 |
| smollm2-135m | PREFIX_PLUS_REACTION | 488 | 0.463 | 0.254 | 0.361 |
| smollm2-135m | STREAMING | 0 | — | — | — |
| smollm2-360m | FULL | 488 | 0.543 | 0.631 | 0.221 |
| smollm2-360m | PREFIX_ONLY | 488 | 0.491 | 0.000 | 0.492 |
| smollm2-360m | PREFIX_PLUS_REACTION | 488 | 0.542 | 0.533 | 0.262 |
| smollm2-360m | STREAMING | 0 | — | — | — |
| stablelm2-1.6b | FULL | 488 | 0.527 | 0.557 | 0.246 |
| stablelm2-1.6b | PREFIX_ONLY | 488 | 0.495 | 0.000 | 0.508 |
| stablelm2-1.6b | PREFIX_PLUS_REACTION | 488 | 0.524 | 0.492 | 0.295 |
| stablelm2-1.6b | STREAMING | 0 | — | — | — |
| tinyllama-1.1b | FULL | 488 | 0.475 | 0.500 | 0.246 |
| tinyllama-1.1b | PREFIX_ONLY | 488 | 0.451 | 0.000 | 0.451 |
| tinyllama-1.1b | PREFIX_PLUS_REACTION | 488 | 0.445 | 0.377 | 0.270 |
| tinyllama-1.1b | STREAMING | 0 | — | — | — |

## Streaming delay (test; misses counted)

| model | eligible | detected | miss rate | mean delay (hits) | pre-evidence alert rate |
|---|---:|---:|---:|---:|---:|
| danube2-1.8b | 0 | 0 | — | — | — |
| granite-1b | 0 | 0 | — | — | — |
| qwen2-0.5b | 0 | 0 | — | — | — |
| qwen2-1.5b | 0 | 0 | — | — | — |
| qwen2.5-0.5b | 0 | 0 | — | — | — |
| qwen2.5-1.5b | 540 | 497 | 0.080 | -2.308 | 0.724 |
| qwen3-0.6b | 0 | 0 | — | — | — |
| qwen3-1.7b | 540 | 345 | 0.361 | 2.055 | 0.048 |
| smollm2-1.7b | 0 | 0 | — | — | — |
| smollm2-135m | 0 | 0 | — | — | — |
| smollm2-360m | 0 | 0 | — | — | — |
| stablelm2-1.6b | 0 | 0 | — | — | — |
| tinyllama-1.1b | 0 | 0 | — | — | — |

## Runtime projection (full panel)

- median per-inference: 0.051 s
- mean turns: 9.94
- assumed test+dev conversations: 540+540
- one model, 4 views: 0.20 h
- 5 models × 4 views: 3.77 h
- note: STREAMING costs one inference per turn; the other three views cost one each. Naive 4×views undercounts STREAMING.
