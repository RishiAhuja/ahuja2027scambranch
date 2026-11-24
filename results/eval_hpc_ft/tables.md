# Detector eval `eval_hpc_ft`

- run_dir: `runs/full_v1`
- n_dev_scored: 540
- n_test_scored: 540
- views: FULL, PREFIX_ONLY, PREFIX_PLUS_REACTION, CALLER_ONLY
- models: qwen2.5-0.5b+branched-seed0, qwen2.5-0.5b+branched-seed1, qwen2.5-0.5b+branched-seed2, qwen2.5-0.5b+branched_consistency-seed0, qwen2.5-0.5b+branched_consistency-seed1, qwen2.5-0.5b+branched_consistency-seed2, qwen2.5-0.5b+unbranched-seed0, qwen2.5-0.5b+unbranched-seed1

## Thresholds (fit on DEV, max macro-F1)

| model | view | threshold |
|---|---|---|
| qwen2.5-0.5b+branched-seed0 | FULL | 0.5700 |
| qwen2.5-0.5b+branched-seed0 | PREFIX_ONLY | 0.3208 |
| qwen2.5-0.5b+branched-seed0 | PREFIX_PLUS_REACTION | 0.3775 |
| qwen2.5-0.5b+branched-seed0 | CALLER_ONLY | 0.4378 |
| qwen2.5-0.5b+branched-seed1 | FULL | 0.3775 |
| qwen2.5-0.5b+branched-seed1 | PREFIX_ONLY | 0.4073 |
| qwen2.5-0.5b+branched-seed1 | PREFIX_PLUS_REACTION | 0.3775 |
| qwen2.5-0.5b+branched-seed1 | CALLER_ONLY | 0.4073 |
| qwen2.5-0.5b+branched-seed2 | FULL | 0.3630 |
| qwen2.5-0.5b+branched-seed2 | PREFIX_ONLY | 0.3174 |
| qwen2.5-0.5b+branched-seed2 | PREFIX_PLUS_REACTION | 0.3849 |
| qwen2.5-0.5b+branched-seed2 | CALLER_ONLY | 0.3849 |
| qwen2.5-0.5b+branched_consistency-seed0 | FULL | 0.8397 |
| qwen2.5-0.5b+branched_consistency-seed0 | PREFIX_ONLY | 0.7100 |
| qwen2.5-0.5b+branched_consistency-seed0 | PREFIX_PLUS_REACTION | 0.8000 |
| qwen2.5-0.5b+branched_consistency-seed0 | CALLER_ONLY | 0.7200 |
| qwen2.5-0.5b+branched_consistency-seed1 | FULL | 0.2814 |
| qwen2.5-0.5b+branched_consistency-seed1 | PREFIX_ONLY | 0.3486 |
| qwen2.5-0.5b+branched_consistency-seed1 | PREFIX_PLUS_REACTION | 0.2942 |
| qwen2.5-0.5b+branched_consistency-seed1 | CALLER_ONLY | 0.2227 |
| qwen2.5-0.5b+branched_consistency-seed2 | FULL | 0.2942 |
| qwen2.5-0.5b+branched_consistency-seed2 | PREFIX_ONLY | 0.7500 |
| qwen2.5-0.5b+branched_consistency-seed2 | PREFIX_PLUS_REACTION | 0.4073 |
| qwen2.5-0.5b+branched_consistency-seed2 | CALLER_ONLY | 0.0953 |
| qwen2.5-0.5b+unbranched-seed0 | FULL | 0.9987 |
| qwen2.5-0.5b+unbranched-seed0 | PREFIX_ONLY | 0.9991 |
| qwen2.5-0.5b+unbranched-seed0 | PREFIX_PLUS_REACTION | 0.9999 |
| qwen2.5-0.5b+unbranched-seed0 | CALLER_ONLY | 0.9960 |
| qwen2.5-0.5b+unbranched-seed1 | FULL | 0.7900 |
| qwen2.5-0.5b+unbranched-seed1 | PREFIX_ONLY | 0.7600 |
| qwen2.5-0.5b+unbranched-seed1 | PREFIX_PLUS_REACTION | 0.9646 |
| qwen2.5-0.5b+unbranched-seed1 | CALLER_ONLY | 0.9379 |

## Test quality (pooled)

| model | view | n | macro-F1 | fraud recall | legit FPR | ROC-AUC | invalid | worst-cond F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen2.5-0.5b+branched-seed0 | FULL | 540 | 0.830 [0.727, 0.920] | 0.752 [0.551, 0.912] | 0.089 [0.000, 0.213] | 0.920 | 0.459 | 0.807 |
| qwen2.5-0.5b+branched-seed0 | PREFIX_ONLY | 540 | 0.798 [0.691, 0.892] | 0.789 [0.550, 1.000] | 0.193 [0.059, 0.321] | 0.869 | 0.333 | 0.794 |
| qwen2.5-0.5b+branched-seed0 | PREFIX_PLUS_REACTION | 540 | 0.809 [0.722, 0.888] | 0.807 [0.620, 0.952] | 0.189 [0.074, 0.333] | 0.885 | 0.317 | 0.757 |
| qwen2.5-0.5b+branched-seed0 | CALLER_ONLY | 540 | 0.806 [0.682, 0.908] | 0.678 [0.441, 0.870] | 0.059 [0.000, 0.152] | 0.918 | 0.228 | 0.766 |
| qwen2.5-0.5b+branched-seed1 | FULL | 540 | 0.727 [0.631, 0.808] | 0.633 [0.523, 0.741] | 0.174 [0.035, 0.342] | 0.805 | 0.004 | 0.595 |
| qwen2.5-0.5b+branched-seed1 | PREFIX_ONLY | 540 | 0.736 [0.647, 0.829] | 0.667 [0.550, 0.782] | 0.193 [0.030, 0.376] | 0.792 | 0.000 | 0.727 |
| qwen2.5-0.5b+branched-seed1 | PREFIX_PLUS_REACTION | 540 | 0.748 [0.639, 0.839] | 0.774 [0.647, 0.887] | 0.278 [0.130, 0.452] | 0.791 | 0.006 | 0.691 |
| qwen2.5-0.5b+branched-seed1 | CALLER_ONLY | 540 | 0.738 [0.654, 0.819] | 0.696 [0.578, 0.807] | 0.219 [0.080, 0.371] | 0.801 | 0.002 | 0.703 |
| qwen2.5-0.5b+branched-seed2 | FULL | 540 | 0.753 [0.630, 0.860] | 0.581 [0.368, 0.781] | 0.059 [0.008, 0.126] | 0.887 | 1.000 | 0.720 |
| qwen2.5-0.5b+branched-seed2 | PREFIX_ONLY | 540 | 0.754 [0.622, 0.869] | 0.633 [0.412, 0.843] | 0.119 [0.000, 0.275] | 0.849 | 1.000 | 0.747 |
| qwen2.5-0.5b+branched-seed2 | PREFIX_PLUS_REACTION | 540 | 0.762 [0.643, 0.869] | 0.630 [0.396, 0.835] | 0.096 [0.023, 0.206] | 0.861 | 1.000 | 0.728 |
| qwen2.5-0.5b+branched-seed2 | CALLER_ONLY | 540 | 0.748 [0.614, 0.864] | 0.552 [0.332, 0.769] | 0.033 [0.000, 0.082] | 0.868 | 1.000 | 0.727 |
| qwen2.5-0.5b+branched_consistency-seed0 | FULL | 540 | 0.750 [0.648, 0.844] | 0.659 [0.467, 0.829] | 0.156 [0.046, 0.294] | 0.830 | 0.111 | 0.705 |
| qwen2.5-0.5b+branched_consistency-seed0 | PREFIX_ONLY | 540 | 0.716 [0.609, 0.814] | 0.689 [0.428, 0.912] | 0.256 [0.096, 0.428] | 0.786 | 0.022 | 0.708 |
| qwen2.5-0.5b+branched_consistency-seed0 | PREFIX_PLUS_REACTION | 540 | 0.737 [0.635, 0.822] | 0.693 [0.513, 0.857] | 0.219 [0.082, 0.391] | 0.817 | 0.052 | 0.685 |
| qwen2.5-0.5b+branched_consistency-seed0 | CALLER_ONLY | 540 | 0.739 [0.630, 0.838] | 0.659 [0.430, 0.847] | 0.178 [0.066, 0.316] | 0.796 | 0.022 | 0.718 |
| qwen2.5-0.5b+branched_consistency-seed1 | FULL | 540 | 0.722 [0.599, 0.837] | 0.615 [0.388, 0.828] | 0.163 [0.000, 0.414] | 0.801 | 0.574 | 0.695 |
| qwen2.5-0.5b+branched_consistency-seed1 | PREFIX_ONLY | 540 | 0.621 [0.503, 0.732] | 0.481 [0.244, 0.724] | 0.222 [0.029, 0.470] | 0.738 | 0.759 | 0.611 |
| qwen2.5-0.5b+branched_consistency-seed1 | PREFIX_PLUS_REACTION | 540 | 0.702 [0.587, 0.809] | 0.630 [0.401, 0.833] | 0.222 [0.043, 0.459] | 0.760 | 0.656 | 0.691 |
| qwen2.5-0.5b+branched_consistency-seed1 | CALLER_ONLY | 540 | 0.697 [0.581, 0.814] | 0.630 [0.429, 0.822] | 0.233 [0.029, 0.489] | 0.758 | 0.580 | 0.691 |
| qwen2.5-0.5b+branched_consistency-seed2 | FULL | 540 | 0.787 [0.692, 0.875] | 0.704 [0.534, 0.866] | 0.126 [0.039, 0.244] | 0.894 | 1.000 | 0.739 |
| qwen2.5-0.5b+branched_consistency-seed2 | PREFIX_ONLY | 540 | 0.717 [0.532, 0.859] | 0.493 [0.235, 0.750] | 0.026 [0.000, 0.065] | 0.864 | 1.000 | 0.706 |
| qwen2.5-0.5b+branched_consistency-seed2 | PREFIX_PLUS_REACTION | 540 | 0.770 [0.678, 0.856] | 0.685 [0.502, 0.856] | 0.141 [0.045, 0.268] | 0.872 | 1.000 | 0.753 |
| qwen2.5-0.5b+branched_consistency-seed2 | CALLER_ONLY | 540 | 0.788 [0.688, 0.864] | 0.844 [0.685, 0.951] | 0.267 [0.161, 0.407] | 0.873 | 1.000 | 0.740 |
| qwen2.5-0.5b+unbranched-seed0 | FULL | 540 | 0.800 [0.682, 0.900] | 0.637 [0.431, 0.831] | 0.026 [0.000, 0.081] | 0.962 | 0.935 | 0.743 |
| qwen2.5-0.5b+unbranched-seed0 | PREFIX_ONLY | 540 | 0.740 [0.563, 0.872] | 0.541 [0.265, 0.792] | 0.037 [0.000, 0.116] | 0.894 | 0.950 | 0.732 |
| qwen2.5-0.5b+unbranched-seed0 | PREFIX_PLUS_REACTION | 540 | 0.734 [0.559, 0.867] | 0.504 [0.241, 0.742] | 0.004 [0.000, 0.012] | 0.933 | 0.933 | 0.702 |
| qwen2.5-0.5b+unbranched-seed0 | CALLER_ONLY | 540 | 0.774 [0.645, 0.873] | 0.604 [0.388, 0.799] | 0.041 [0.000, 0.128] | 0.954 | 0.907 | 0.733 |
| qwen2.5-0.5b+unbranched-seed1 | FULL | 540 | 0.816 [0.697, 0.906] | 0.667 [0.452, 0.856] | 0.026 [0.000, 0.081] | 0.948 | 0.409 | 0.765 |
| qwen2.5-0.5b+unbranched-seed1 | PREFIX_ONLY | 540 | 0.777 [0.643, 0.886] | 0.596 [0.364, 0.812] | 0.026 [0.000, 0.081] | 0.908 | 0.344 | 0.765 |
| qwen2.5-0.5b+unbranched-seed1 | PREFIX_PLUS_REACTION | 540 | 0.736 [0.583, 0.860] | 0.522 [0.269, 0.751] | 0.022 [0.000, 0.070] | 0.935 | 0.435 | 0.713 |
| qwen2.5-0.5b+unbranched-seed1 | CALLER_ONLY | 540 | 0.735 [0.576, 0.860] | 0.515 [0.268, 0.750] | 0.015 [0.000, 0.047] | 0.963 | 0.187 | 0.695 |

## Per language (test, pooled views listed separately)

| model | view | language | n | macro-F1 | fraud recall | legit FPR |
|---|---|---|---:|---:|---:|---:|
| qwen2.5-0.5b+branched-seed0 | FULL | hi_en | 188 | 0.845 | 0.796 | 0.105 |
| qwen2.5-0.5b+branched-seed0 | FULL | ta_en | 177 | 0.818 | 0.727 | 0.090 |
| qwen2.5-0.5b+branched-seed0 | FULL | te_en | 175 | 0.827 | 0.730 | 0.070 |
| qwen2.5-0.5b+branched-seed0 | PREFIX_ONLY | hi_en | 188 | 0.760 | 0.806 | 0.284 |
| qwen2.5-0.5b+branched-seed0 | PREFIX_ONLY | ta_en | 177 | 0.808 | 0.773 | 0.157 |
| qwen2.5-0.5b+branched-seed0 | PREFIX_ONLY | te_en | 175 | 0.828 | 0.787 | 0.128 |
| qwen2.5-0.5b+branched-seed0 | PREFIX_PLUS_REACTION | hi_en | 188 | 0.775 | 0.860 | 0.305 |
| qwen2.5-0.5b+branched-seed0 | PREFIX_PLUS_REACTION | ta_en | 177 | 0.830 | 0.761 | 0.101 |
| qwen2.5-0.5b+branched-seed0 | PREFIX_PLUS_REACTION | te_en | 175 | 0.823 | 0.798 | 0.151 |
| qwen2.5-0.5b+branched-seed0 | CALLER_ONLY | hi_en | 188 | 0.788 | 0.656 | 0.074 |
| qwen2.5-0.5b+branched-seed0 | CALLER_ONLY | ta_en | 177 | 0.829 | 0.727 | 0.067 |
| qwen2.5-0.5b+branched-seed0 | CALLER_ONLY | te_en | 175 | 0.802 | 0.652 | 0.035 |
| qwen2.5-0.5b+branched-seed1 | FULL | hi_en | 188 | 0.670 | 0.473 | 0.105 |
| qwen2.5-0.5b+branched-seed1 | FULL | ta_en | 177 | 0.774 | 0.750 | 0.202 |
| qwen2.5-0.5b+branched-seed1 | FULL | te_en | 175 | 0.731 | 0.685 | 0.221 |
| qwen2.5-0.5b+branched-seed1 | PREFIX_ONLY | hi_en | 188 | 0.574 | 0.301 | 0.074 |
| qwen2.5-0.5b+branched-seed1 | PREFIX_ONLY | ta_en | 177 | 0.830 | 0.909 | 0.247 |
| qwen2.5-0.5b+branched-seed1 | PREFIX_ONLY | te_en | 175 | 0.771 | 0.809 | 0.267 |
| qwen2.5-0.5b+branched-seed1 | PREFIX_PLUS_REACTION | hi_en | 188 | 0.735 | 0.624 | 0.147 |
| qwen2.5-0.5b+branched-seed1 | PREFIX_PLUS_REACTION | ta_en | 177 | 0.742 | 0.875 | 0.382 |
| qwen2.5-0.5b+branched-seed1 | PREFIX_PLUS_REACTION | te_en | 175 | 0.758 | 0.831 | 0.314 |
| qwen2.5-0.5b+branched-seed1 | CALLER_ONLY | hi_en | 188 | 0.666 | 0.473 | 0.116 |
| qwen2.5-0.5b+branched-seed1 | CALLER_ONLY | ta_en | 177 | 0.767 | 0.852 | 0.315 |
| qwen2.5-0.5b+branched-seed1 | CALLER_ONLY | te_en | 175 | 0.771 | 0.775 | 0.233 |
| qwen2.5-0.5b+branched-seed2 | FULL | hi_en | 188 | 0.705 | 0.452 | 0.000 |
| qwen2.5-0.5b+branched-seed2 | FULL | ta_en | 177 | 0.756 | 0.602 | 0.079 |
| qwen2.5-0.5b+branched-seed2 | FULL | te_en | 175 | 0.793 | 0.697 | 0.105 |
| qwen2.5-0.5b+branched-seed2 | PREFIX_ONLY | hi_en | 188 | 0.733 | 0.591 | 0.116 |
| qwen2.5-0.5b+branched-seed2 | PREFIX_ONLY | ta_en | 177 | 0.762 | 0.602 | 0.067 |
| qwen2.5-0.5b+branched-seed2 | PREFIX_ONLY | te_en | 175 | 0.765 | 0.708 | 0.174 |
| qwen2.5-0.5b+branched-seed2 | PREFIX_PLUS_REACTION | hi_en | 188 | 0.753 | 0.591 | 0.074 |
| qwen2.5-0.5b+branched-seed2 | PREFIX_PLUS_REACTION | ta_en | 177 | 0.754 | 0.580 | 0.056 |
| qwen2.5-0.5b+branched-seed2 | PREFIX_PLUS_REACTION | te_en | 175 | 0.777 | 0.719 | 0.163 |
| qwen2.5-0.5b+branched-seed2 | CALLER_ONLY | hi_en | 188 | 0.671 | 0.398 | 0.000 |
| qwen2.5-0.5b+branched-seed2 | CALLER_ONLY | ta_en | 177 | 0.768 | 0.614 | 0.067 |
| qwen2.5-0.5b+branched-seed2 | CALLER_ONLY | te_en | 175 | 0.802 | 0.652 | 0.035 |
| qwen2.5-0.5b+branched_consistency-seed0 | FULL | hi_en | 188 | 0.770 | 0.710 | 0.168 |
| qwen2.5-0.5b+branched_consistency-seed0 | FULL | ta_en | 177 | 0.742 | 0.625 | 0.135 |
| qwen2.5-0.5b+branched_consistency-seed0 | FULL | te_en | 175 | 0.735 | 0.640 | 0.163 |
| qwen2.5-0.5b+branched_consistency-seed0 | PREFIX_ONLY | hi_en | 188 | 0.697 | 0.677 | 0.284 |
| qwen2.5-0.5b+branched_consistency-seed0 | PREFIX_ONLY | ta_en | 177 | 0.759 | 0.648 | 0.124 |
| qwen2.5-0.5b+branched_consistency-seed0 | PREFIX_ONLY | te_en | 175 | 0.690 | 0.742 | 0.360 |
| qwen2.5-0.5b+branched_consistency-seed0 | PREFIX_PLUS_REACTION | hi_en | 188 | 0.782 | 0.796 | 0.232 |
| qwen2.5-0.5b+branched_consistency-seed0 | PREFIX_PLUS_REACTION | ta_en | 177 | 0.737 | 0.636 | 0.157 |
| qwen2.5-0.5b+branched_consistency-seed0 | PREFIX_PLUS_REACTION | te_en | 175 | 0.685 | 0.640 | 0.267 |
| qwen2.5-0.5b+branched_consistency-seed0 | CALLER_ONLY | hi_en | 188 | 0.733 | 0.677 | 0.211 |
| qwen2.5-0.5b+branched_consistency-seed0 | CALLER_ONLY | ta_en | 177 | 0.755 | 0.659 | 0.146 |
| qwen2.5-0.5b+branched_consistency-seed0 | CALLER_ONLY | te_en | 175 | 0.730 | 0.640 | 0.174 |
| qwen2.5-0.5b+branched_consistency-seed1 | FULL | hi_en | 188 | 0.731 | 0.634 | 0.168 |
| qwen2.5-0.5b+branched_consistency-seed1 | FULL | ta_en | 177 | 0.742 | 0.636 | 0.146 |
| qwen2.5-0.5b+branched_consistency-seed1 | FULL | te_en | 175 | 0.693 | 0.573 | 0.174 |
| qwen2.5-0.5b+branched_consistency-seed1 | PREFIX_ONLY | hi_en | 188 | 0.673 | 0.591 | 0.242 |
| qwen2.5-0.5b+branched_consistency-seed1 | PREFIX_ONLY | ta_en | 177 | 0.628 | 0.477 | 0.202 |
| qwen2.5-0.5b+branched_consistency-seed1 | PREFIX_ONLY | te_en | 175 | 0.555 | 0.371 | 0.221 |
| qwen2.5-0.5b+branched_consistency-seed1 | PREFIX_PLUS_REACTION | hi_en | 188 | 0.717 | 0.667 | 0.232 |
| qwen2.5-0.5b+branched_consistency-seed1 | PREFIX_PLUS_REACTION | ta_en | 177 | 0.721 | 0.636 | 0.191 |
| qwen2.5-0.5b+branched_consistency-seed1 | PREFIX_PLUS_REACTION | te_en | 175 | 0.667 | 0.584 | 0.244 |
| qwen2.5-0.5b+branched_consistency-seed1 | CALLER_ONLY | hi_en | 188 | 0.673 | 0.591 | 0.242 |
| qwen2.5-0.5b+branched_consistency-seed1 | CALLER_ONLY | ta_en | 177 | 0.723 | 0.682 | 0.236 |
| qwen2.5-0.5b+branched_consistency-seed1 | CALLER_ONLY | te_en | 175 | 0.696 | 0.618 | 0.221 |
| qwen2.5-0.5b+branched_consistency-seed2 | FULL | hi_en | 188 | 0.739 | 0.548 | 0.053 |
| qwen2.5-0.5b+branched_consistency-seed2 | FULL | ta_en | 177 | 0.802 | 0.773 | 0.169 |
| qwen2.5-0.5b+branched_consistency-seed2 | FULL | te_en | 175 | 0.817 | 0.798 | 0.163 |
| qwen2.5-0.5b+branched_consistency-seed2 | PREFIX_ONLY | hi_en | 188 | 0.685 | 0.419 | 0.000 |
| qwen2.5-0.5b+branched_consistency-seed2 | PREFIX_ONLY | ta_en | 177 | 0.732 | 0.523 | 0.034 |
| qwen2.5-0.5b+branched_consistency-seed2 | PREFIX_ONLY | te_en | 175 | 0.733 | 0.539 | 0.047 |
| qwen2.5-0.5b+branched_consistency-seed2 | PREFIX_PLUS_REACTION | hi_en | 188 | 0.719 | 0.548 | 0.095 |
| qwen2.5-0.5b+branched_consistency-seed2 | PREFIX_PLUS_REACTION | ta_en | 177 | 0.790 | 0.727 | 0.146 |
| qwen2.5-0.5b+branched_consistency-seed2 | PREFIX_PLUS_REACTION | te_en | 175 | 0.800 | 0.787 | 0.186 |
| qwen2.5-0.5b+branched_consistency-seed2 | CALLER_ONLY | hi_en | 188 | 0.818 | 0.753 | 0.116 |
| qwen2.5-0.5b+branched_consistency-seed2 | CALLER_ONLY | ta_en | 177 | 0.742 | 0.864 | 0.371 |
| qwen2.5-0.5b+branched_consistency-seed2 | CALLER_ONLY | te_en | 175 | 0.796 | 0.921 | 0.326 |
| qwen2.5-0.5b+unbranched-seed0 | FULL | hi_en | 188 | 0.821 | 0.688 | 0.042 |
| qwen2.5-0.5b+unbranched-seed0 | FULL | ta_en | 177 | 0.778 | 0.602 | 0.034 |
| qwen2.5-0.5b+unbranched-seed0 | FULL | te_en | 175 | 0.799 | 0.618 | 0.000 |
| qwen2.5-0.5b+unbranched-seed0 | PREFIX_ONLY | hi_en | 188 | 0.704 | 0.505 | 0.074 |
| qwen2.5-0.5b+unbranched-seed0 | PREFIX_ONLY | ta_en | 177 | 0.725 | 0.511 | 0.034 |
| qwen2.5-0.5b+unbranched-seed0 | PREFIX_ONLY | te_en | 175 | 0.793 | 0.607 | 0.000 |
| qwen2.5-0.5b+unbranched-seed0 | PREFIX_PLUS_REACTION | hi_en | 188 | 0.725 | 0.484 | 0.000 |
| qwen2.5-0.5b+unbranched-seed0 | PREFIX_PLUS_REACTION | ta_en | 177 | 0.715 | 0.477 | 0.011 |
| qwen2.5-0.5b+unbranched-seed0 | PREFIX_PLUS_REACTION | te_en | 175 | 0.761 | 0.551 | 0.000 |
| qwen2.5-0.5b+unbranched-seed0 | CALLER_ONLY | hi_en | 188 | 0.787 | 0.645 | 0.063 |
| qwen2.5-0.5b+unbranched-seed0 | CALLER_ONLY | ta_en | 177 | 0.720 | 0.511 | 0.045 |
| qwen2.5-0.5b+unbranched-seed0 | CALLER_ONLY | te_en | 175 | 0.813 | 0.652 | 0.012 |
| qwen2.5-0.5b+unbranched-seed1 | FULL | hi_en | 188 | 0.826 | 0.699 | 0.042 |
| qwen2.5-0.5b+unbranched-seed1 | FULL | ta_en | 177 | 0.803 | 0.648 | 0.034 |
| qwen2.5-0.5b+unbranched-seed1 | FULL | te_en | 175 | 0.818 | 0.652 | 0.000 |
| qwen2.5-0.5b+unbranched-seed1 | PREFIX_ONLY | hi_en | 188 | 0.768 | 0.591 | 0.042 |
| qwen2.5-0.5b+unbranched-seed1 | PREFIX_ONLY | ta_en | 177 | 0.784 | 0.614 | 0.034 |
| qwen2.5-0.5b+unbranched-seed1 | PREFIX_ONLY | te_en | 175 | 0.780 | 0.584 | 0.000 |
| qwen2.5-0.5b+unbranched-seed1 | PREFIX_PLUS_REACTION | hi_en | 188 | 0.736 | 0.527 | 0.032 |
| qwen2.5-0.5b+unbranched-seed1 | PREFIX_PLUS_REACTION | ta_en | 177 | 0.732 | 0.523 | 0.034 |
| qwen2.5-0.5b+unbranched-seed1 | PREFIX_PLUS_REACTION | te_en | 175 | 0.741 | 0.517 | 0.000 |
| qwen2.5-0.5b+unbranched-seed1 | CALLER_ONLY | hi_en | 188 | 0.740 | 0.516 | 0.011 |
| qwen2.5-0.5b+unbranched-seed1 | CALLER_ONLY | ta_en | 177 | 0.719 | 0.500 | 0.034 |
| qwen2.5-0.5b+unbranched-seed1 | CALLER_ONLY | te_en | 175 | 0.747 | 0.528 | 0.000 |

## Per recipient condition (test)

| model | view | condition | n | fraud recall | legit FPR | macro-F1 |
|---|---|---|---:|---:|---:|---:|
| qwen2.5-0.5b+branched-seed0 | FULL | cooperative | 133 | 0.754 | 0.088 | 0.833 |
| qwen2.5-0.5b+branched-seed0 | FULL | clarifying | 134 | 0.773 | 0.118 | 0.828 |
| qwen2.5-0.5b+branched-seed0 | FULL | resistant | 136 | 0.686 | 0.061 | 0.807 |
| qwen2.5-0.5b+branched-seed0 | FULL | suspicious | 137 | 0.797 | 0.088 | 0.854 |
| qwen2.5-0.5b+branched-seed0 | PREFIX_ONLY | cooperative | 133 | 0.800 | 0.191 | 0.804 |
| qwen2.5-0.5b+branched-seed0 | PREFIX_ONLY | clarifying | 134 | 0.788 | 0.191 | 0.798 |
| qwen2.5-0.5b+branched-seed0 | PREFIX_ONLY | resistant | 136 | 0.786 | 0.197 | 0.794 |
| qwen2.5-0.5b+branched-seed0 | PREFIX_ONLY | suspicious | 137 | 0.783 | 0.191 | 0.796 |
| qwen2.5-0.5b+branched-seed0 | PREFIX_PLUS_REACTION | cooperative | 133 | 0.846 | 0.162 | 0.842 |
| qwen2.5-0.5b+branched-seed0 | PREFIX_PLUS_REACTION | clarifying | 134 | 0.833 | 0.250 | 0.791 |
| qwen2.5-0.5b+branched-seed0 | PREFIX_PLUS_REACTION | resistant | 136 | 0.686 | 0.167 | 0.757 |
| qwen2.5-0.5b+branched-seed0 | PREFIX_PLUS_REACTION | suspicious | 137 | 0.870 | 0.176 | 0.847 |
| qwen2.5-0.5b+branched-seed0 | CALLER_ONLY | cooperative | 133 | 0.754 | 0.059 | 0.848 |
| qwen2.5-0.5b+branched-seed0 | CALLER_ONLY | clarifying | 134 | 0.758 | 0.074 | 0.842 |
| qwen2.5-0.5b+branched-seed0 | CALLER_ONLY | resistant | 136 | 0.600 | 0.045 | 0.766 |
| qwen2.5-0.5b+branched-seed0 | CALLER_ONLY | suspicious | 137 | 0.609 | 0.059 | 0.768 |
| qwen2.5-0.5b+branched-seed1 | FULL | cooperative | 133 | 0.692 | 0.162 | 0.765 |
| qwen2.5-0.5b+branched-seed1 | FULL | clarifying | 134 | 0.379 | 0.147 | 0.595 |
| qwen2.5-0.5b+branched-seed1 | FULL | resistant | 136 | 0.614 | 0.227 | 0.690 |
| qwen2.5-0.5b+branched-seed1 | FULL | suspicious | 137 | 0.841 | 0.162 | 0.839 |
| qwen2.5-0.5b+branched-seed1 | PREFIX_ONLY | cooperative | 133 | 0.677 | 0.191 | 0.743 |
| qwen2.5-0.5b+branched-seed1 | PREFIX_ONLY | clarifying | 134 | 0.667 | 0.191 | 0.737 |
| qwen2.5-0.5b+branched-seed1 | PREFIX_ONLY | resistant | 136 | 0.657 | 0.197 | 0.727 |
| qwen2.5-0.5b+branched-seed1 | PREFIX_ONLY | suspicious | 137 | 0.667 | 0.191 | 0.736 |
| qwen2.5-0.5b+branched-seed1 | PREFIX_PLUS_REACTION | cooperative | 133 | 0.800 | 0.206 | 0.797 |
| qwen2.5-0.5b+branched-seed1 | PREFIX_PLUS_REACTION | clarifying | 134 | 0.697 | 0.309 | 0.694 |
| qwen2.5-0.5b+branched-seed1 | PREFIX_PLUS_REACTION | resistant | 136 | 0.657 | 0.273 | 0.691 |
| qwen2.5-0.5b+branched-seed1 | PREFIX_PLUS_REACTION | suspicious | 137 | 0.942 | 0.324 | 0.806 |
| qwen2.5-0.5b+branched-seed1 | CALLER_ONLY | cooperative | 133 | 0.738 | 0.221 | 0.759 |
| qwen2.5-0.5b+branched-seed1 | CALLER_ONLY | clarifying | 134 | 0.576 | 0.162 | 0.703 |
| qwen2.5-0.5b+branched-seed1 | CALLER_ONLY | resistant | 136 | 0.757 | 0.303 | 0.727 |
| qwen2.5-0.5b+branched-seed1 | CALLER_ONLY | suspicious | 137 | 0.710 | 0.191 | 0.759 |
| qwen2.5-0.5b+branched-seed2 | FULL | cooperative | 133 | 0.646 | 0.044 | 0.798 |
| qwen2.5-0.5b+branched-seed2 | FULL | clarifying | 134 | 0.576 | 0.059 | 0.752 |
| qwen2.5-0.5b+branched-seed2 | FULL | resistant | 136 | 0.543 | 0.076 | 0.720 |
| qwen2.5-0.5b+branched-seed2 | FULL | suspicious | 137 | 0.565 | 0.059 | 0.743 |
| qwen2.5-0.5b+branched-seed2 | PREFIX_ONLY | cooperative | 133 | 0.646 | 0.118 | 0.763 |
| qwen2.5-0.5b+branched-seed2 | PREFIX_ONLY | clarifying | 134 | 0.636 | 0.118 | 0.757 |
| qwen2.5-0.5b+branched-seed2 | PREFIX_ONLY | resistant | 136 | 0.629 | 0.121 | 0.747 |
| qwen2.5-0.5b+branched-seed2 | PREFIX_ONLY | suspicious | 137 | 0.623 | 0.118 | 0.748 |
| qwen2.5-0.5b+branched-seed2 | PREFIX_PLUS_REACTION | cooperative | 133 | 0.692 | 0.103 | 0.794 |
| qwen2.5-0.5b+branched-seed2 | PREFIX_PLUS_REACTION | clarifying | 134 | 0.667 | 0.132 | 0.766 |
| qwen2.5-0.5b+branched-seed2 | PREFIX_PLUS_REACTION | resistant | 136 | 0.557 | 0.076 | 0.728 |
| qwen2.5-0.5b+branched-seed2 | PREFIX_PLUS_REACTION | suspicious | 137 | 0.609 | 0.074 | 0.761 |
| qwen2.5-0.5b+branched-seed2 | CALLER_ONLY | cooperative | 133 | 0.538 | 0.000 | 0.760 |
| qwen2.5-0.5b+branched-seed2 | CALLER_ONLY | clarifying | 134 | 0.576 | 0.029 | 0.766 |
| qwen2.5-0.5b+branched-seed2 | CALLER_ONLY | resistant | 136 | 0.557 | 0.045 | 0.742 |
| qwen2.5-0.5b+branched-seed2 | CALLER_ONLY | suspicious | 137 | 0.536 | 0.059 | 0.727 |
| qwen2.5-0.5b+branched_consistency-seed0 | FULL | cooperative | 133 | 0.646 | 0.162 | 0.741 |
| qwen2.5-0.5b+branched_consistency-seed0 | FULL | clarifying | 134 | 0.636 | 0.221 | 0.707 |
| qwen2.5-0.5b+branched_consistency-seed0 | FULL | resistant | 136 | 0.529 | 0.091 | 0.705 |
| qwen2.5-0.5b+branched_consistency-seed0 | FULL | suspicious | 137 | 0.826 | 0.147 | 0.839 |
| qwen2.5-0.5b+branched_consistency-seed0 | PREFIX_ONLY | cooperative | 133 | 0.708 | 0.250 | 0.729 |
| qwen2.5-0.5b+branched_consistency-seed0 | PREFIX_ONLY | clarifying | 134 | 0.682 | 0.250 | 0.716 |
| qwen2.5-0.5b+branched_consistency-seed0 | PREFIX_ONLY | resistant | 136 | 0.686 | 0.258 | 0.713 |
| qwen2.5-0.5b+branched_consistency-seed0 | PREFIX_ONLY | suspicious | 137 | 0.681 | 0.265 | 0.708 |
| qwen2.5-0.5b+branched_consistency-seed0 | PREFIX_PLUS_REACTION | cooperative | 133 | 0.677 | 0.235 | 0.721 |
| qwen2.5-0.5b+branched_consistency-seed0 | PREFIX_PLUS_REACTION | clarifying | 134 | 0.636 | 0.265 | 0.685 |
| qwen2.5-0.5b+branched_consistency-seed0 | PREFIX_PLUS_REACTION | resistant | 136 | 0.557 | 0.091 | 0.721 |
| qwen2.5-0.5b+branched_consistency-seed0 | PREFIX_PLUS_REACTION | suspicious | 137 | 0.899 | 0.279 | 0.808 |
| qwen2.5-0.5b+branched_consistency-seed0 | CALLER_ONLY | cooperative | 133 | 0.708 | 0.147 | 0.780 |
| qwen2.5-0.5b+branched_consistency-seed0 | CALLER_ONLY | clarifying | 134 | 0.712 | 0.250 | 0.731 |
| qwen2.5-0.5b+branched_consistency-seed0 | CALLER_ONLY | resistant | 136 | 0.600 | 0.152 | 0.718 |
| qwen2.5-0.5b+branched_consistency-seed0 | CALLER_ONLY | suspicious | 137 | 0.623 | 0.162 | 0.727 |
| qwen2.5-0.5b+branched_consistency-seed1 | FULL | cooperative | 133 | 0.631 | 0.147 | 0.740 |
| qwen2.5-0.5b+branched_consistency-seed1 | FULL | clarifying | 134 | 0.591 | 0.176 | 0.704 |
| qwen2.5-0.5b+branched_consistency-seed1 | FULL | resistant | 136 | 0.571 | 0.167 | 0.695 |
| qwen2.5-0.5b+branched_consistency-seed1 | FULL | suspicious | 137 | 0.667 | 0.162 | 0.750 |
| qwen2.5-0.5b+branched_consistency-seed1 | PREFIX_ONLY | cooperative | 133 | 0.508 | 0.221 | 0.638 |
| qwen2.5-0.5b+branched_consistency-seed1 | PREFIX_ONLY | clarifying | 134 | 0.470 | 0.221 | 0.617 |
| qwen2.5-0.5b+branched_consistency-seed1 | PREFIX_ONLY | resistant | 136 | 0.471 | 0.227 | 0.611 |
| qwen2.5-0.5b+branched_consistency-seed1 | PREFIX_ONLY | suspicious | 137 | 0.478 | 0.221 | 0.620 |
| qwen2.5-0.5b+branched_consistency-seed1 | PREFIX_PLUS_REACTION | cooperative | 133 | 0.677 | 0.250 | 0.713 |
| qwen2.5-0.5b+branched_consistency-seed1 | PREFIX_PLUS_REACTION | clarifying | 134 | 0.606 | 0.221 | 0.691 |
| qwen2.5-0.5b+branched_consistency-seed1 | PREFIX_PLUS_REACTION | resistant | 136 | 0.571 | 0.167 | 0.695 |
| qwen2.5-0.5b+branched_consistency-seed1 | PREFIX_PLUS_REACTION | suspicious | 137 | 0.667 | 0.250 | 0.708 |
| qwen2.5-0.5b+branched_consistency-seed1 | CALLER_ONLY | cooperative | 133 | 0.631 | 0.235 | 0.697 |
| qwen2.5-0.5b+branched_consistency-seed1 | CALLER_ONLY | clarifying | 134 | 0.636 | 0.235 | 0.700 |
| qwen2.5-0.5b+branched_consistency-seed1 | CALLER_ONLY | resistant | 136 | 0.629 | 0.242 | 0.691 |
| qwen2.5-0.5b+branched_consistency-seed1 | CALLER_ONLY | suspicious | 137 | 0.623 | 0.221 | 0.699 |
| qwen2.5-0.5b+branched_consistency-seed2 | FULL | cooperative | 133 | 0.754 | 0.118 | 0.818 |
| qwen2.5-0.5b+branched_consistency-seed2 | FULL | clarifying | 134 | 0.742 | 0.162 | 0.790 |
| qwen2.5-0.5b+branched_consistency-seed2 | FULL | resistant | 136 | 0.614 | 0.121 | 0.739 |
| qwen2.5-0.5b+branched_consistency-seed2 | FULL | suspicious | 137 | 0.710 | 0.103 | 0.801 |
| qwen2.5-0.5b+branched_consistency-seed2 | PREFIX_ONLY | cooperative | 133 | 0.523 | 0.015 | 0.744 |
| qwen2.5-0.5b+branched_consistency-seed2 | PREFIX_ONLY | clarifying | 134 | 0.485 | 0.029 | 0.713 |
| qwen2.5-0.5b+branched_consistency-seed2 | PREFIX_ONLY | resistant | 136 | 0.486 | 0.030 | 0.706 |
| qwen2.5-0.5b+branched_consistency-seed2 | PREFIX_ONLY | suspicious | 137 | 0.478 | 0.029 | 0.706 |
| qwen2.5-0.5b+branched_consistency-seed2 | PREFIX_PLUS_REACTION | cooperative | 133 | 0.738 | 0.162 | 0.789 |
| qwen2.5-0.5b+branched_consistency-seed2 | PREFIX_PLUS_REACTION | clarifying | 134 | 0.742 | 0.191 | 0.776 |
| qwen2.5-0.5b+branched_consistency-seed2 | PREFIX_PLUS_REACTION | resistant | 136 | 0.600 | 0.076 | 0.753 |
| qwen2.5-0.5b+branched_consistency-seed2 | PREFIX_PLUS_REACTION | suspicious | 137 | 0.667 | 0.132 | 0.764 |
| qwen2.5-0.5b+branched_consistency-seed2 | CALLER_ONLY | cooperative | 133 | 0.831 | 0.250 | 0.789 |
| qwen2.5-0.5b+branched_consistency-seed2 | CALLER_ONLY | clarifying | 134 | 0.894 | 0.294 | 0.797 |
| qwen2.5-0.5b+branched_consistency-seed2 | CALLER_ONLY | resistant | 136 | 0.814 | 0.333 | 0.740 |
| qwen2.5-0.5b+branched_consistency-seed2 | CALLER_ONLY | suspicious | 137 | 0.841 | 0.191 | 0.825 |
| qwen2.5-0.5b+unbranched-seed0 | FULL | cooperative | 133 | 0.723 | 0.000 | 0.861 |
| qwen2.5-0.5b+unbranched-seed0 | FULL | clarifying | 134 | 0.561 | 0.059 | 0.743 |
| qwen2.5-0.5b+unbranched-seed0 | FULL | resistant | 136 | 0.643 | 0.000 | 0.812 |
| qwen2.5-0.5b+unbranched-seed0 | FULL | suspicious | 137 | 0.623 | 0.044 | 0.783 |
| qwen2.5-0.5b+unbranched-seed0 | PREFIX_ONLY | cooperative | 133 | 0.554 | 0.029 | 0.754 |
| qwen2.5-0.5b+unbranched-seed0 | PREFIX_ONLY | clarifying | 134 | 0.545 | 0.044 | 0.742 |
| qwen2.5-0.5b+unbranched-seed0 | PREFIX_ONLY | resistant | 136 | 0.529 | 0.030 | 0.732 |
| qwen2.5-0.5b+unbranched-seed0 | PREFIX_ONLY | suspicious | 137 | 0.536 | 0.044 | 0.733 |
| qwen2.5-0.5b+unbranched-seed0 | PREFIX_PLUS_REACTION | cooperative | 133 | 0.569 | 0.000 | 0.777 |
| qwen2.5-0.5b+unbranched-seed0 | PREFIX_PLUS_REACTION | clarifying | 134 | 0.485 | 0.015 | 0.720 |
| qwen2.5-0.5b+unbranched-seed0 | PREFIX_PLUS_REACTION | resistant | 136 | 0.457 | 0.000 | 0.702 |
| qwen2.5-0.5b+unbranched-seed0 | PREFIX_PLUS_REACTION | suspicious | 137 | 0.507 | 0.000 | 0.737 |
| qwen2.5-0.5b+unbranched-seed0 | CALLER_ONLY | cooperative | 133 | 0.677 | 0.029 | 0.822 |
| qwen2.5-0.5b+unbranched-seed0 | CALLER_ONLY | clarifying | 134 | 0.652 | 0.059 | 0.793 |
| qwen2.5-0.5b+unbranched-seed0 | CALLER_ONLY | resistant | 136 | 0.557 | 0.030 | 0.749 |
| qwen2.5-0.5b+unbranched-seed0 | CALLER_ONLY | suspicious | 137 | 0.536 | 0.044 | 0.733 |
| qwen2.5-0.5b+unbranched-seed1 | FULL | cooperative | 133 | 0.708 | 0.015 | 0.846 |
| qwen2.5-0.5b+unbranched-seed1 | FULL | clarifying | 134 | 0.652 | 0.029 | 0.808 |
| qwen2.5-0.5b+unbranched-seed1 | FULL | resistant | 136 | 0.586 | 0.030 | 0.765 |
| qwen2.5-0.5b+unbranched-seed1 | FULL | suspicious | 137 | 0.725 | 0.029 | 0.845 |
| qwen2.5-0.5b+unbranched-seed1 | PREFIX_ONLY | cooperative | 133 | 0.631 | 0.015 | 0.805 |
| qwen2.5-0.5b+unbranched-seed1 | PREFIX_ONLY | clarifying | 134 | 0.591 | 0.029 | 0.774 |
| qwen2.5-0.5b+unbranched-seed1 | PREFIX_ONLY | resistant | 136 | 0.586 | 0.030 | 0.765 |
| qwen2.5-0.5b+unbranched-seed1 | PREFIX_ONLY | suspicious | 137 | 0.580 | 0.029 | 0.765 |
| qwen2.5-0.5b+unbranched-seed1 | PREFIX_PLUS_REACTION | cooperative | 133 | 0.569 | 0.015 | 0.770 |
| qwen2.5-0.5b+unbranched-seed1 | PREFIX_PLUS_REACTION | clarifying | 134 | 0.485 | 0.029 | 0.713 |
| qwen2.5-0.5b+unbranched-seed1 | PREFIX_PLUS_REACTION | resistant | 136 | 0.500 | 0.015 | 0.722 |
| qwen2.5-0.5b+unbranched-seed1 | PREFIX_PLUS_REACTION | suspicious | 137 | 0.536 | 0.029 | 0.740 |
| qwen2.5-0.5b+unbranched-seed1 | CALLER_ONLY | cooperative | 133 | 0.600 | 0.000 | 0.795 |
| qwen2.5-0.5b+unbranched-seed1 | CALLER_ONLY | clarifying | 134 | 0.515 | 0.015 | 0.738 |
| qwen2.5-0.5b+unbranched-seed1 | CALLER_ONLY | resistant | 136 | 0.457 | 0.015 | 0.695 |
| qwen2.5-0.5b+unbranched-seed1 | CALLER_ONLY | suspicious | 137 | 0.493 | 0.029 | 0.714 |

## Branch metrics (complete sibling sets, test)

| model | view | n histories | disagreement | score spread (mean) | all-correct |
|---|---|---:|---:|---:|---:|
| qwen2.5-0.5b+branched-seed0 | FULL | 122 | 0.230 | 0.221 | 0.713 |
| qwen2.5-0.5b+branched-seed0 | PREFIX_ONLY | 122 | 0.000 | 0.000 | 0.811 |
| qwen2.5-0.5b+branched-seed0 | PREFIX_PLUS_REACTION | 122 | 0.246 | 0.239 | 0.672 |
| qwen2.5-0.5b+branched-seed0 | CALLER_ONLY | 122 | 0.164 | 0.182 | 0.738 |
| qwen2.5-0.5b+branched-seed1 | FULL | 122 | 0.451 | 0.163 | 0.484 |
| qwen2.5-0.5b+branched-seed1 | PREFIX_ONLY | 122 | 0.000 | 0.000 | 0.754 |
| qwen2.5-0.5b+branched-seed1 | PREFIX_PLUS_REACTION | 122 | 0.418 | 0.148 | 0.516 |
| qwen2.5-0.5b+branched-seed1 | CALLER_ONLY | 122 | 0.385 | 0.113 | 0.516 |
| qwen2.5-0.5b+branched-seed2 | FULL | 122 | 0.213 | 0.179 | 0.664 |
| qwen2.5-0.5b+branched-seed2 | PREFIX_ONLY | 122 | 0.000 | 0.000 | 0.779 |
| qwen2.5-0.5b+branched-seed2 | PREFIX_PLUS_REACTION | 122 | 0.197 | 0.163 | 0.689 |
| qwen2.5-0.5b+branched-seed2 | CALLER_ONLY | 122 | 0.131 | 0.142 | 0.705 |
| qwen2.5-0.5b+branched_consistency-seed0 | FULL | 122 | 0.410 | 0.255 | 0.533 |
| qwen2.5-0.5b+branched_consistency-seed0 | PREFIX_ONLY | 122 | 0.000 | 0.000 | 0.721 |
| qwen2.5-0.5b+branched_consistency-seed0 | PREFIX_PLUS_REACTION | 122 | 0.377 | 0.215 | 0.549 |
| qwen2.5-0.5b+branched_consistency-seed0 | CALLER_ONLY | 122 | 0.303 | 0.198 | 0.582 |
| qwen2.5-0.5b+branched_consistency-seed1 | FULL | 122 | 0.148 | 0.196 | 0.672 |
| qwen2.5-0.5b+branched_consistency-seed1 | PREFIX_ONLY | 122 | 0.000 | 0.000 | 0.648 |
| qwen2.5-0.5b+branched_consistency-seed1 | PREFIX_PLUS_REACTION | 122 | 0.246 | 0.192 | 0.582 |
| qwen2.5-0.5b+branched_consistency-seed1 | CALLER_ONLY | 122 | 0.238 | 0.187 | 0.582 |
| qwen2.5-0.5b+branched_consistency-seed2 | FULL | 122 | 0.246 | 0.183 | 0.680 |
| qwen2.5-0.5b+branched_consistency-seed2 | PREFIX_ONLY | 122 | 0.000 | 0.000 | 0.746 |
| qwen2.5-0.5b+branched_consistency-seed2 | PREFIX_PLUS_REACTION | 122 | 0.221 | 0.181 | 0.672 |
| qwen2.5-0.5b+branched_consistency-seed2 | CALLER_ONLY | 122 | 0.270 | 0.156 | 0.648 |
| qwen2.5-0.5b+unbranched-seed0 | FULL | 122 | 0.189 | 0.138 | 0.713 |
| qwen2.5-0.5b+unbranched-seed0 | PREFIX_ONLY | 122 | 0.000 | 0.000 | 0.770 |
| qwen2.5-0.5b+unbranched-seed0 | PREFIX_PLUS_REACTION | 122 | 0.074 | 0.269 | 0.730 |
| qwen2.5-0.5b+unbranched-seed0 | CALLER_ONLY | 122 | 0.172 | 0.174 | 0.705 |
| qwen2.5-0.5b+unbranched-seed1 | FULL | 122 | 0.090 | 0.117 | 0.787 |
| qwen2.5-0.5b+unbranched-seed1 | PREFIX_ONLY | 122 | 0.000 | 0.000 | 0.803 |
| qwen2.5-0.5b+unbranched-seed1 | PREFIX_PLUS_REACTION | 122 | 0.066 | 0.115 | 0.738 |
| qwen2.5-0.5b+unbranched-seed1 | CALLER_ONLY | 122 | 0.107 | 0.141 | 0.730 |

## PREFIX_PLUS_REACTION − PREFIX_ONLY (mean Δ P(fraud), test)

| model | condition / label | mean Δ | 95% CI |
|---|---|---:|---|
| qwen2.5-0.5b+branched-seed0 | all | 0.049 | |
| qwen2.5-0.5b+branched-seed0 | cooperative | 0.033 | |
| qwen2.5-0.5b+branched-seed0 | clarifying | 0.083 | |
| qwen2.5-0.5b+branched-seed0 | resistant | -0.010 | |
| qwen2.5-0.5b+branched-seed0 | suspicious | 0.089 | |
| qwen2.5-0.5b+branched-seed0 | label=fraud | 0.066 | |
| qwen2.5-0.5b+branched-seed0 | label=legitimate | 0.031 | |
| qwen2.5-0.5b+branched-seed0 | cooperative|fraud | 0.072 |  [0.027, 0.126] |
| qwen2.5-0.5b+branched-seed0 | cooperative|legitimate | -0.005 |  [-0.087, 0.107] |
| qwen2.5-0.5b+branched-seed0 | clarifying|fraud | 0.075 |  [0.030, 0.138] |
| qwen2.5-0.5b+branched-seed0 | clarifying|legitimate | 0.091 |  [0.011, 0.162] |
| qwen2.5-0.5b+branched-seed0 | resistant|fraud | -0.013 |  [-0.052, 0.018] |
| qwen2.5-0.5b+branched-seed0 | resistant|legitimate | -0.007 |  [-0.058, 0.034] |
| qwen2.5-0.5b+branched-seed0 | suspicious|fraud | 0.132 |  [0.041, 0.254] |
| qwen2.5-0.5b+branched-seed0 | suspicious|legitimate | 0.046 |  [-0.021, 0.116] |
| qwen2.5-0.5b+branched-seed1 | all | 0.015 | |
| qwen2.5-0.5b+branched-seed1 | cooperative | 0.019 | |
| qwen2.5-0.5b+branched-seed1 | clarifying | 0.004 | |
| qwen2.5-0.5b+branched-seed1 | resistant | -0.007 | |
| qwen2.5-0.5b+branched-seed1 | suspicious | 0.042 | |
| qwen2.5-0.5b+branched-seed1 | label=fraud | 0.023 | |
| qwen2.5-0.5b+branched-seed1 | label=legitimate | 0.006 | |
| qwen2.5-0.5b+branched-seed1 | cooperative|fraud | 0.060 |  [0.010, 0.105] |
| qwen2.5-0.5b+branched-seed1 | cooperative|legitimate | -0.020 |  [-0.050, 0.013] |
| qwen2.5-0.5b+branched-seed1 | clarifying|fraud | -0.005 |  [-0.030, 0.020] |
| qwen2.5-0.5b+branched-seed1 | clarifying|legitimate | 0.013 |  [-0.016, 0.045] |
| qwen2.5-0.5b+branched-seed1 | resistant|fraud | -0.014 |  [-0.040, 0.012] |
| qwen2.5-0.5b+branched-seed1 | resistant|legitimate | 0.001 |  [-0.017, 0.015] |
| qwen2.5-0.5b+branched-seed1 | suspicious|fraud | 0.052 |  [0.025, 0.081] |
| qwen2.5-0.5b+branched-seed1 | suspicious|legitimate | 0.031 |  [0.001, 0.060] |
| qwen2.5-0.5b+branched-seed2 | all | 0.001 | |
| qwen2.5-0.5b+branched-seed2 | cooperative | 0.021 | |
| qwen2.5-0.5b+branched-seed2 | clarifying | 0.039 | |
| qwen2.5-0.5b+branched-seed2 | resistant | -0.036 | |
| qwen2.5-0.5b+branched-seed2 | suspicious | -0.018 | |
| qwen2.5-0.5b+branched-seed2 | label=fraud | 0.013 | |
| qwen2.5-0.5b+branched-seed2 | label=legitimate | -0.010 | |
| qwen2.5-0.5b+branched-seed2 | cooperative|fraud | 0.048 |  [0.009, 0.087] |
| qwen2.5-0.5b+branched-seed2 | cooperative|legitimate | -0.005 |  [-0.038, 0.028] |
| qwen2.5-0.5b+branched-seed2 | clarifying|fraud | 0.049 |  [0.017, 0.091] |
| qwen2.5-0.5b+branched-seed2 | clarifying|legitimate | 0.029 |  [-0.014, 0.068] |
| qwen2.5-0.5b+branched-seed2 | resistant|fraud | -0.037 |  [-0.081, -0.005] |
| qwen2.5-0.5b+branched-seed2 | resistant|legitimate | -0.035 |  [-0.078, 0.019] |
| qwen2.5-0.5b+branched-seed2 | suspicious|fraud | -0.006 |  [-0.037, 0.025] |
| qwen2.5-0.5b+branched-seed2 | suspicious|legitimate | -0.031 |  [-0.075, 0.009] |
| qwen2.5-0.5b+branched_consistency-seed0 | all | 0.049 | |
| qwen2.5-0.5b+branched_consistency-seed0 | cooperative | 0.039 | |
| qwen2.5-0.5b+branched_consistency-seed0 | clarifying | 0.067 | |
| qwen2.5-0.5b+branched_consistency-seed0 | resistant | -0.013 | |
| qwen2.5-0.5b+branched_consistency-seed0 | suspicious | 0.103 | |
| qwen2.5-0.5b+branched_consistency-seed0 | label=fraud | 0.063 | |
| qwen2.5-0.5b+branched_consistency-seed0 | label=legitimate | 0.036 | |
| qwen2.5-0.5b+branched_consistency-seed0 | cooperative|fraud | 0.063 |  [0.034, 0.096] |
| qwen2.5-0.5b+branched_consistency-seed0 | cooperative|legitimate | 0.017 |  [-0.046, 0.084] |
| qwen2.5-0.5b+branched_consistency-seed0 | clarifying|fraud | 0.052 |  [0.021, 0.084] |
| qwen2.5-0.5b+branched_consistency-seed0 | clarifying|legitimate | 0.082 |  [0.041, 0.125] |
| qwen2.5-0.5b+branched_consistency-seed0 | resistant|fraud | 0.004 |  [-0.019, 0.026] |
| qwen2.5-0.5b+branched_consistency-seed0 | resistant|legitimate | -0.031 |  [-0.065, -0.001] |
| qwen2.5-0.5b+branched_consistency-seed0 | suspicious|fraud | 0.133 |  [0.076, 0.198] |
| qwen2.5-0.5b+branched_consistency-seed0 | suspicious|legitimate | 0.073 |  [0.015, 0.138] |
| qwen2.5-0.5b+branched_consistency-seed1 | all | 0.018 | |
| qwen2.5-0.5b+branched_consistency-seed1 | cooperative | 0.055 | |
| qwen2.5-0.5b+branched_consistency-seed1 | clarifying | 0.020 | |
| qwen2.5-0.5b+branched_consistency-seed1 | resistant | -0.025 | |
| qwen2.5-0.5b+branched_consistency-seed1 | suspicious | 0.023 | |
| qwen2.5-0.5b+branched_consistency-seed1 | label=fraud | 0.042 | |
| qwen2.5-0.5b+branched_consistency-seed1 | label=legitimate | -0.006 | |
| qwen2.5-0.5b+branched_consistency-seed1 | cooperative|fraud | 0.133 |  [0.049, 0.222] |
| qwen2.5-0.5b+branched_consistency-seed1 | cooperative|legitimate | -0.020 |  [-0.062, 0.030] |
| qwen2.5-0.5b+branched_consistency-seed1 | clarifying|fraud | 0.019 |  [-0.027, 0.063] |
| qwen2.5-0.5b+branched_consistency-seed1 | clarifying|legitimate | 0.021 |  [-0.022, 0.061] |
| qwen2.5-0.5b+branched_consistency-seed1 | resistant|fraud | -0.019 |  [-0.061, 0.039] |
| qwen2.5-0.5b+branched_consistency-seed1 | resistant|legitimate | -0.031 |  [-0.045, -0.018] |
| qwen2.5-0.5b+branched_consistency-seed1 | suspicious|fraud | 0.041 |  [-0.003, 0.090] |
| qwen2.5-0.5b+branched_consistency-seed1 | suspicious|legitimate | 0.005 |  [-0.021, 0.038] |
| qwen2.5-0.5b+branched_consistency-seed2 | all | 0.029 | |
| qwen2.5-0.5b+branched_consistency-seed2 | cooperative | 0.043 | |
| qwen2.5-0.5b+branched_consistency-seed2 | clarifying | 0.072 | |
| qwen2.5-0.5b+branched_consistency-seed2 | resistant | -0.019 | |
| qwen2.5-0.5b+branched_consistency-seed2 | suspicious | 0.020 | |
| qwen2.5-0.5b+branched_consistency-seed2 | label=fraud | 0.035 | |
| qwen2.5-0.5b+branched_consistency-seed2 | label=legitimate | 0.022 | |
| qwen2.5-0.5b+branched_consistency-seed2 | cooperative|fraud | 0.051 |  [0.006, 0.107] |
| qwen2.5-0.5b+branched_consistency-seed2 | cooperative|legitimate | 0.035 |  [-0.007, 0.077] |
| qwen2.5-0.5b+branched_consistency-seed2 | clarifying|fraud | 0.076 |  [0.023, 0.143] |
| qwen2.5-0.5b+branched_consistency-seed2 | clarifying|legitimate | 0.067 |  [0.020, 0.125] |
| qwen2.5-0.5b+branched_consistency-seed2 | resistant|fraud | -0.022 |  [-0.065, 0.014] |
| qwen2.5-0.5b+branched_consistency-seed2 | resistant|legitimate | -0.016 |  [-0.044, 0.018] |
| qwen2.5-0.5b+branched_consistency-seed2 | suspicious|fraud | 0.037 |  [-0.014, 0.095] |
| qwen2.5-0.5b+branched_consistency-seed2 | suspicious|legitimate | 0.002 |  [-0.049, 0.054] |
| qwen2.5-0.5b+unbranched-seed0 | all | -0.000 | |
| qwen2.5-0.5b+unbranched-seed0 | cooperative | 0.028 | |
| qwen2.5-0.5b+unbranched-seed0 | clarifying | 0.027 | |
| qwen2.5-0.5b+unbranched-seed0 | resistant | -0.049 | |
| qwen2.5-0.5b+unbranched-seed0 | suspicious | -0.007 | |
| qwen2.5-0.5b+unbranched-seed0 | label=fraud | 0.055 | |
| qwen2.5-0.5b+unbranched-seed0 | label=legitimate | -0.056 | |
| qwen2.5-0.5b+unbranched-seed0 | cooperative|fraud | 0.109 |  [0.018, 0.241] |
| qwen2.5-0.5b+unbranched-seed0 | cooperative|legitimate | -0.049 |  [-0.160, 0.034] |
| qwen2.5-0.5b+unbranched-seed0 | clarifying|fraud | 0.037 |  [0.007, 0.078] |
| qwen2.5-0.5b+unbranched-seed0 | clarifying|legitimate | 0.018 |  [-0.160, 0.159] |
| qwen2.5-0.5b+unbranched-seed0 | resistant|fraud | 0.037 |  [-0.000, 0.086] |
| qwen2.5-0.5b+unbranched-seed0 | resistant|legitimate | -0.140 |  [-0.306, -0.015] |
| qwen2.5-0.5b+unbranched-seed0 | suspicious|fraud | 0.041 |  [-0.011, 0.104] |
| qwen2.5-0.5b+unbranched-seed0 | suspicious|legitimate | -0.056 |  [-0.188, 0.069] |
| qwen2.5-0.5b+unbranched-seed1 | all | 0.005 | |
| qwen2.5-0.5b+unbranched-seed1 | cooperative | 0.024 | |
| qwen2.5-0.5b+unbranched-seed1 | clarifying | -0.002 | |
| qwen2.5-0.5b+unbranched-seed1 | resistant | -0.018 | |
| qwen2.5-0.5b+unbranched-seed1 | suspicious | 0.018 | |
| qwen2.5-0.5b+unbranched-seed1 | label=fraud | 0.045 | |
| qwen2.5-0.5b+unbranched-seed1 | label=legitimate | -0.035 | |
| qwen2.5-0.5b+unbranched-seed1 | cooperative|fraud | 0.065 |  [0.030, 0.107] |
| qwen2.5-0.5b+unbranched-seed1 | cooperative|legitimate | -0.015 |  [-0.067, 0.039] |
| qwen2.5-0.5b+unbranched-seed1 | clarifying|fraud | 0.029 |  [-0.003, 0.068] |
| qwen2.5-0.5b+unbranched-seed1 | clarifying|legitimate | -0.032 |  [-0.093, 0.022] |
| qwen2.5-0.5b+unbranched-seed1 | resistant|fraud | 0.008 |  [-0.018, 0.037] |
| qwen2.5-0.5b+unbranched-seed1 | resistant|legitimate | -0.047 |  [-0.107, 0.003] |
| qwen2.5-0.5b+unbranched-seed1 | suspicious|fraud | 0.080 |  [0.023, 0.144] |
| qwen2.5-0.5b+unbranched-seed1 | suspicious|legitimate | -0.045 |  [-0.106, 0.011] |

## Complete-set slice (test)

| model | view | n | macro-F1 | disagreement | all-correct |
|---|---|---:|---:|---:|---:|
| qwen2.5-0.5b+branched-seed0 | FULL | 488 | 0.838 | 0.230 | 0.713 |
| qwen2.5-0.5b+branched-seed0 | PREFIX_ONLY | 488 | 0.811 | 0.000 | 0.811 |
| qwen2.5-0.5b+branched-seed0 | PREFIX_PLUS_REACTION | 488 | 0.813 | 0.246 | 0.672 |
| qwen2.5-0.5b+branched-seed0 | CALLER_ONLY | 488 | 0.814 | 0.164 | 0.738 |
| qwen2.5-0.5b+branched-seed1 | FULL | 488 | 0.743 | 0.451 | 0.484 |
| qwen2.5-0.5b+branched-seed1 | PREFIX_ONLY | 488 | 0.751 | 0.000 | 0.754 |
| qwen2.5-0.5b+branched-seed1 | PREFIX_PLUS_REACTION | 488 | 0.750 | 0.418 | 0.516 |
| qwen2.5-0.5b+branched-seed1 | CALLER_ONLY | 488 | 0.746 | 0.385 | 0.516 |
| qwen2.5-0.5b+branched-seed2 | FULL | 488 | 0.768 | 0.213 | 0.664 |
| qwen2.5-0.5b+branched-seed2 | PREFIX_ONLY | 488 | 0.772 | 0.000 | 0.779 |
| qwen2.5-0.5b+branched-seed2 | PREFIX_PLUS_REACTION | 488 | 0.787 | 0.197 | 0.689 |
| qwen2.5-0.5b+branched-seed2 | CALLER_ONLY | 488 | 0.751 | 0.131 | 0.705 |
| qwen2.5-0.5b+branched_consistency-seed0 | FULL | 488 | 0.758 | 0.410 | 0.533 |
| qwen2.5-0.5b+branched_consistency-seed0 | PREFIX_ONLY | 488 | 0.720 | 0.000 | 0.721 |
| qwen2.5-0.5b+branched_consistency-seed0 | PREFIX_PLUS_REACTION | 488 | 0.744 | 0.377 | 0.549 |
| qwen2.5-0.5b+branched_consistency-seed0 | CALLER_ONLY | 488 | 0.748 | 0.303 | 0.582 |
| qwen2.5-0.5b+branched_consistency-seed1 | FULL | 488 | 0.742 | 0.148 | 0.672 |
| qwen2.5-0.5b+branched_consistency-seed1 | PREFIX_ONLY | 488 | 0.635 | 0.000 | 0.648 |
| qwen2.5-0.5b+branched_consistency-seed1 | PREFIX_PLUS_REACTION | 488 | 0.718 | 0.246 | 0.582 |
| qwen2.5-0.5b+branched_consistency-seed1 | CALLER_ONLY | 488 | 0.718 | 0.238 | 0.582 |
| qwen2.5-0.5b+branched_consistency-seed2 | FULL | 488 | 0.809 | 0.246 | 0.680 |
| qwen2.5-0.5b+branched_consistency-seed2 | PREFIX_ONLY | 488 | 0.726 | 0.000 | 0.746 |
| qwen2.5-0.5b+branched_consistency-seed2 | PREFIX_PLUS_REACTION | 488 | 0.790 | 0.221 | 0.672 |
| qwen2.5-0.5b+branched_consistency-seed2 | CALLER_ONLY | 488 | 0.801 | 0.270 | 0.648 |
| qwen2.5-0.5b+unbranched-seed0 | FULL | 488 | 0.805 | 0.189 | 0.713 |
| qwen2.5-0.5b+unbranched-seed0 | PREFIX_ONLY | 488 | 0.756 | 0.000 | 0.770 |
| qwen2.5-0.5b+unbranched-seed0 | PREFIX_PLUS_REACTION | 488 | 0.741 | 0.074 | 0.730 |
| qwen2.5-0.5b+unbranched-seed0 | CALLER_ONLY | 488 | 0.785 | 0.172 | 0.705 |
| qwen2.5-0.5b+unbranched-seed1 | FULL | 488 | 0.824 | 0.090 | 0.787 |
| qwen2.5-0.5b+unbranched-seed1 | PREFIX_ONLY | 488 | 0.794 | 0.000 | 0.803 |
| qwen2.5-0.5b+unbranched-seed1 | PREFIX_PLUS_REACTION | 488 | 0.753 | 0.066 | 0.738 |
| qwen2.5-0.5b+unbranched-seed1 | CALLER_ONLY | 488 | 0.754 | 0.107 | 0.730 |

## Streaming delay (test; misses counted)

| model | eligible | detected | miss rate | mean delay (hits) | pre-evidence alert rate |
|---|---:|---:|---:|---:|---:|

## Runtime projection (full panel)

- median per-inference: 0.081 s
- mean turns: 9.94
- assumed test+dev conversations: 540+540
- one model, 4 views: 0.32 h
- 5 models × 4 views: 6.01 h
- note: STREAMING costs one inference per turn; the other three views cost one each. Naive 4×views undercounts STREAMING.
