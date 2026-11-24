# Detector eval `eval_hpc_ft_rest`

- run_dir: `runs/full_v1`
- n_dev_scored: 540
- n_test_scored: 540
- views: FULL, PREFIX_ONLY, PREFIX_PLUS_REACTION, CALLER_ONLY
- models: qwen2.5-0.5b+unbranched-seed2, qwen2.5-1.5b+branched-seed0, qwen2.5-1.5b+branched-seed1, qwen2.5-1.5b+branched-seed2, qwen2.5-1.5b+branched_consistency-seed0, qwen2.5-1.5b+branched_consistency-seed1, qwen2.5-1.5b+branched_consistency-seed2, qwen2.5-1.5b+unbranched-seed0, qwen2.5-1.5b+unbranched-seed1, qwen2.5-1.5b+unbranched-seed2

## Thresholds (fit on DEV, max macro-F1)

| model | view | threshold |
|---|---|---|
| qwen2.5-0.5b+unbranched-seed2 | FULL | 0.3775 |
| qwen2.5-0.5b+unbranched-seed2 | PREFIX_ONLY | 0.5400 |
| qwen2.5-0.5b+unbranched-seed2 | PREFIX_PLUS_REACTION | 0.5700 |
| qwen2.5-0.5b+unbranched-seed2 | CALLER_ONLY | 0.5000 |
| qwen2.5-1.5b+branched-seed0 | FULL | 0.8595 |
| qwen2.5-1.5b+branched-seed0 | PREFIX_ONLY | 0.8000 |
| qwen2.5-1.5b+branched-seed0 | PREFIX_PLUS_REACTION | 0.8400 |
| qwen2.5-1.5b+branched-seed0 | CALLER_ONLY | 0.5000 |
| qwen2.5-1.5b+branched-seed1 | FULL | 0.3486 |
| qwen2.5-1.5b+branched-seed1 | PREFIX_ONLY | 0.6300 |
| qwen2.5-1.5b+branched-seed1 | PREFIX_PLUS_REACTION | 0.7600 |
| qwen2.5-1.5b+branched-seed1 | CALLER_ONLY | 0.0759 |
| qwen2.5-1.5b+branched-seed2 | FULL | 0.8000 |
| qwen2.5-1.5b+branched-seed2 | PREFIX_ONLY | 0.8700 |
| qwen2.5-1.5b+branched-seed2 | PREFIX_PLUS_REACTION | 0.9400 |
| qwen2.5-1.5b+branched-seed2 | CALLER_ONLY | 0.7600 |
| qwen2.5-1.5b+branched_consistency-seed0 | FULL | 0.8300 |
| qwen2.5-1.5b+branched_consistency-seed0 | PREFIX_ONLY | 0.8400 |
| qwen2.5-1.5b+branched_consistency-seed0 | PREFIX_PLUS_REACTION | 0.8595 |
| qwen2.5-1.5b+branched_consistency-seed0 | CALLER_ONLY | 0.4688 |
| qwen2.5-1.5b+branched_consistency-seed1 | FULL | 0.9981 |
| qwen2.5-1.5b+branched_consistency-seed1 | PREFIX_ONLY | 0.9992 |
| qwen2.5-1.5b+branched_consistency-seed1 | PREFIX_PLUS_REACTION | 0.9994 |
| qwen2.5-1.5b+branched_consistency-seed1 | CALLER_ONLY | 0.8400 |
| qwen2.5-1.5b+branched_consistency-seed2 | FULL | 0.8990 |
| qwen2.5-1.5b+branched_consistency-seed2 | PREFIX_ONLY | 0.8600 |
| qwen2.5-1.5b+branched_consistency-seed2 | PREFIX_PLUS_REACTION | 0.9883 |
| qwen2.5-1.5b+branched_consistency-seed2 | CALLER_ONLY | 0.4378 |
| qwen2.5-1.5b+unbranched-seed0 | FULL | 0.5000 |
| qwen2.5-1.5b+unbranched-seed0 | PREFIX_ONLY | 0.3775 |
| qwen2.5-1.5b+unbranched-seed0 | PREFIX_PLUS_REACTION | 0.6300 |
| qwen2.5-1.5b+unbranched-seed0 | CALLER_ONLY | 0.3486 |
| qwen2.5-1.5b+unbranched-seed1 | FULL | 0.5400 |
| qwen2.5-1.5b+unbranched-seed1 | PREFIX_ONLY | 0.6600 |
| qwen2.5-1.5b+unbranched-seed1 | PREFIX_PLUS_REACTION | 0.9362 |
| qwen2.5-1.5b+unbranched-seed1 | CALLER_ONLY | 0.0260 |
| qwen2.5-1.5b+unbranched-seed2 | FULL | 0.3775 |
| qwen2.5-1.5b+unbranched-seed2 | PREFIX_ONLY | 0.6800 |
| qwen2.5-1.5b+unbranched-seed2 | PREFIX_PLUS_REACTION | 0.8990 |
| qwen2.5-1.5b+unbranched-seed2 | CALLER_ONLY | 0.0601 |

## Test quality (pooled)

| model | view | n | macro-F1 | fraud recall | legit FPR | ROC-AUC | invalid | worst-cond F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen2.5-0.5b+unbranched-seed2 | FULL | 540 | 0.781 [0.709, 0.846] | 0.774 [0.689, 0.848] | 0.211 [0.099, 0.349] | 0.851 | 0.004 | 0.686 |
| qwen2.5-0.5b+unbranched-seed2 | PREFIX_ONLY | 540 | 0.659 [0.523, 0.780] | 0.615 [0.360, 0.835] | 0.296 [0.136, 0.478] | 0.775 | 0.000 | 0.654 |
| qwen2.5-0.5b+unbranched-seed2 | PREFIX_PLUS_REACTION | 540 | 0.758 [0.658, 0.839] | 0.681 [0.533, 0.809] | 0.163 [0.074, 0.267] | 0.851 | 0.009 | 0.688 |
| qwen2.5-0.5b+unbranched-seed2 | CALLER_ONLY | 540 | 0.725 [0.651, 0.790] | 0.670 [0.576, 0.755] | 0.219 [0.092, 0.372] | 0.812 | 0.002 | 0.689 |
| qwen2.5-1.5b+branched-seed0 | FULL | 540 | 0.810 [0.731, 0.883] | 0.726 [0.584, 0.841] | 0.104 [0.015, 0.230] | 0.920 | 0.000 | 0.763 |
| qwen2.5-1.5b+branched-seed0 | PREFIX_ONLY | 540 | 0.828 [0.737, 0.908] | 0.848 [0.682, 0.978] | 0.193 [0.031, 0.377] | 0.902 | 0.000 | 0.823 |
| qwen2.5-1.5b+branched-seed0 | PREFIX_PLUS_REACTION | 540 | 0.829 [0.750, 0.897] | 0.781 [0.646, 0.889] | 0.122 [0.019, 0.264] | 0.912 | 0.000 | 0.779 |
| qwen2.5-1.5b+branched-seed0 | CALLER_ONLY | 540 | 0.774 [0.694, 0.847] | 0.696 [0.534, 0.834] | 0.144 [0.022, 0.288] | 0.897 | 0.000 | 0.750 |
| qwen2.5-1.5b+branched-seed1 | FULL | 540 | 0.879 [0.774, 0.962] | 0.837 [0.697, 0.943] | 0.078 [0.004, 0.219] | 0.939 | 0.000 | 0.850 |
| qwen2.5-1.5b+branched-seed1 | PREFIX_ONLY | 540 | 0.775 [0.632, 0.901] | 0.633 [0.423, 0.830] | 0.074 [0.000, 0.233] | 0.890 | 0.000 | 0.768 |
| qwen2.5-1.5b+branched-seed1 | PREFIX_PLUS_REACTION | 540 | 0.838 [0.740, 0.922] | 0.756 [0.608, 0.889] | 0.078 [0.004, 0.218] | 0.916 | 0.000 | 0.777 |
| qwen2.5-1.5b+branched-seed1 | CALLER_ONLY | 540 | 0.832 [0.730, 0.915] | 0.748 [0.577, 0.888] | 0.081 [0.000, 0.207] | 0.941 | 0.000 | 0.808 |
| qwen2.5-1.5b+branched-seed2 | FULL | 540 | 0.900 [0.851, 0.942] | 0.881 [0.798, 0.947] | 0.081 [0.019, 0.160] | 0.975 | 0.009 | 0.855 |
| qwen2.5-1.5b+branched-seed2 | PREFIX_ONLY | 540 | 0.752 [0.638, 0.858] | 0.719 [0.519, 0.889] | 0.215 [0.064, 0.411] | 0.873 | 0.030 | 0.744 |
| qwen2.5-1.5b+branched-seed2 | PREFIX_PLUS_REACTION | 540 | 0.855 [0.790, 0.910] | 0.811 [0.687, 0.912] | 0.100 [0.035, 0.194] | 0.937 | 0.002 | 0.835 |
| qwen2.5-1.5b+branched-seed2 | CALLER_ONLY | 540 | 0.812 [0.720, 0.889] | 0.704 [0.529, 0.850] | 0.074 [0.011, 0.159] | 0.937 | 0.006 | 0.759 |
| qwen2.5-1.5b+branched_consistency-seed0 | FULL | 540 | 0.808 [0.737, 0.873] | 0.741 [0.625, 0.838] | 0.122 [0.022, 0.257] | 0.921 | 0.000 | 0.763 |
| qwen2.5-1.5b+branched_consistency-seed0 | PREFIX_ONLY | 540 | 0.793 [0.680, 0.892] | 0.722 [0.508, 0.902] | 0.133 [0.000, 0.297] | 0.900 | 0.000 | 0.786 |
| qwen2.5-1.5b+branched_consistency-seed0 | PREFIX_PLUS_REACTION | 540 | 0.791 [0.710, 0.868] | 0.667 [0.522, 0.799] | 0.078 [0.000, 0.188] | 0.911 | 0.000 | 0.738 |
| qwen2.5-1.5b+branched_consistency-seed0 | CALLER_ONLY | 540 | 0.772 [0.693, 0.847] | 0.722 [0.567, 0.851] | 0.178 [0.048, 0.332] | 0.899 | 0.000 | 0.761 |
| qwen2.5-1.5b+branched_consistency-seed1 | FULL | 540 | 0.790 [0.697, 0.878] | 0.681 [0.484, 0.847] | 0.096 [0.000, 0.247] | 0.920 | 0.000 | 0.754 |
| qwen2.5-1.5b+branched_consistency-seed1 | PREFIX_ONLY | 540 | 0.733 [0.586, 0.861] | 0.574 [0.323, 0.808] | 0.093 [0.000, 0.246] | 0.886 | 0.000 | 0.723 |
| qwen2.5-1.5b+branched_consistency-seed1 | PREFIX_PLUS_REACTION | 540 | 0.728 [0.608, 0.836] | 0.593 [0.362, 0.813] | 0.126 [0.000, 0.275] | 0.894 | 0.000 | 0.686 |
| qwen2.5-1.5b+branched_consistency-seed1 | CALLER_ONLY | 540 | 0.816 [0.708, 0.907] | 0.752 [0.540, 0.920] | 0.119 [0.012, 0.256] | 0.924 | 0.000 | 0.795 |
| qwen2.5-1.5b+branched_consistency-seed2 | FULL | 540 | 0.933 [0.889, 0.966] | 0.930 [0.886, 0.967] | 0.063 [0.007, 0.159] | 0.980 | 0.002 | 0.909 |
| qwen2.5-1.5b+branched_consistency-seed2 | PREFIX_ONLY | 540 | 0.820 [0.734, 0.903] | 0.856 [0.689, 0.985] | 0.215 [0.070, 0.388] | 0.908 | 0.000 | 0.813 |
| qwen2.5-1.5b+branched_consistency-seed2 | PREFIX_PLUS_REACTION | 540 | 0.872 [0.827, 0.913] | 0.811 [0.717, 0.886] | 0.067 [0.014, 0.140] | 0.946 | 0.006 | 0.854 |
| qwen2.5-1.5b+branched_consistency-seed2 | CALLER_ONLY | 540 | 0.841 [0.770, 0.906] | 0.759 [0.620, 0.884] | 0.074 [0.000, 0.163] | 0.957 | 0.002 | 0.808 |
| qwen2.5-1.5b+unbranched-seed0 | FULL | 540 | 0.844 [0.776, 0.900] | 0.822 [0.735, 0.896] | 0.133 [0.037, 0.267] | 0.936 | 0.000 | 0.781 |
| qwen2.5-1.5b+unbranched-seed0 | PREFIX_ONLY | 540 | 0.727 [0.607, 0.824] | 0.830 [0.636, 0.984] | 0.370 [0.182, 0.572] | 0.839 | 0.000 | 0.724 |
| qwen2.5-1.5b+unbranched-seed0 | PREFIX_PLUS_REACTION | 540 | 0.826 [0.740, 0.900] | 0.804 [0.690, 0.908] | 0.152 [0.055, 0.282] | 0.905 | 0.000 | 0.804 |
| qwen2.5-1.5b+unbranched-seed0 | CALLER_ONLY | 540 | 0.795 [0.698, 0.883] | 0.726 [0.552, 0.872] | 0.133 [0.021, 0.275] | 0.904 | 0.000 | 0.779 |
| qwen2.5-1.5b+unbranched-seed1 | FULL | 540 | 0.894 [0.839, 0.943] | 0.878 [0.772, 0.955] | 0.089 [0.018, 0.195] | 0.959 | 0.000 | 0.864 |
| qwen2.5-1.5b+unbranched-seed1 | PREFIX_ONLY | 540 | 0.812 [0.733, 0.881] | 0.748 [0.580, 0.873] | 0.122 [0.029, 0.240] | 0.913 | 0.000 | 0.801 |
| qwen2.5-1.5b+unbranched-seed1 | PREFIX_PLUS_REACTION | 540 | 0.817 [0.744, 0.885] | 0.689 [0.544, 0.829] | 0.048 [0.000, 0.135] | 0.950 | 0.000 | 0.744 |
| qwen2.5-1.5b+unbranched-seed1 | CALLER_ONLY | 540 | 0.846 [0.758, 0.924] | 0.796 [0.654, 0.914] | 0.104 [0.007, 0.239] | 0.940 | 0.000 | 0.816 |
| qwen2.5-1.5b+unbranched-seed2 | FULL | 540 | 0.852 [0.790, 0.904] | 0.848 [0.757, 0.922] | 0.144 [0.034, 0.278] | 0.937 | 0.000 | 0.779 |
| qwen2.5-1.5b+unbranched-seed2 | PREFIX_ONLY | 540 | 0.767 [0.669, 0.863] | 0.785 [0.573, 0.954] | 0.252 [0.075, 0.451] | 0.857 | 0.000 | 0.764 |
| qwen2.5-1.5b+unbranched-seed2 | PREFIX_PLUS_REACTION | 540 | 0.807 [0.740, 0.869] | 0.748 [0.598, 0.871] | 0.133 [0.039, 0.250] | 0.898 | 0.000 | 0.790 |
| qwen2.5-1.5b+unbranched-seed2 | CALLER_ONLY | 540 | 0.813 [0.738, 0.884] | 0.841 [0.720, 0.927] | 0.215 [0.058, 0.389] | 0.896 | 0.000 | 0.797 |

## Per language (test, pooled views listed separately)

| model | view | language | n | macro-F1 | fraud recall | legit FPR |
|---|---|---|---:|---:|---:|---:|
| qwen2.5-0.5b+unbranched-seed2 | FULL | hi_en | 188 | 0.777 | 0.796 | 0.242 |
| qwen2.5-0.5b+unbranched-seed2 | FULL | ta_en | 177 | 0.785 | 0.830 | 0.258 |
| qwen2.5-0.5b+unbranched-seed2 | FULL | te_en | 175 | 0.782 | 0.697 | 0.128 |
| qwen2.5-0.5b+unbranched-seed2 | PREFIX_ONLY | hi_en | 188 | 0.526 | 0.505 | 0.453 |
| qwen2.5-0.5b+unbranched-seed2 | PREFIX_ONLY | ta_en | 177 | 0.783 | 0.682 | 0.112 |
| qwen2.5-0.5b+unbranched-seed2 | PREFIX_ONLY | te_en | 175 | 0.674 | 0.663 | 0.314 |
| qwen2.5-0.5b+unbranched-seed2 | PREFIX_PLUS_REACTION | hi_en | 188 | 0.782 | 0.785 | 0.221 |
| qwen2.5-0.5b+unbranched-seed2 | PREFIX_PLUS_REACTION | ta_en | 177 | 0.752 | 0.614 | 0.101 |
| qwen2.5-0.5b+unbranched-seed2 | PREFIX_PLUS_REACTION | te_en | 175 | 0.735 | 0.640 | 0.163 |
| qwen2.5-0.5b+unbranched-seed2 | CALLER_ONLY | hi_en | 188 | 0.706 | 0.634 | 0.221 |
| qwen2.5-0.5b+unbranched-seed2 | CALLER_ONLY | ta_en | 177 | 0.768 | 0.773 | 0.236 |
| qwen2.5-0.5b+unbranched-seed2 | CALLER_ONLY | te_en | 175 | 0.701 | 0.607 | 0.198 |
| qwen2.5-1.5b+branched-seed0 | FULL | hi_en | 188 | 0.877 | 0.828 | 0.074 |
| qwen2.5-1.5b+branched-seed0 | FULL | ta_en | 177 | 0.790 | 0.727 | 0.146 |
| qwen2.5-1.5b+branched-seed0 | FULL | te_en | 175 | 0.756 | 0.618 | 0.093 |
| qwen2.5-1.5b+branched-seed0 | PREFIX_ONLY | hi_en | 188 | 0.899 | 0.871 | 0.074 |
| qwen2.5-1.5b+branched-seed0 | PREFIX_ONLY | ta_en | 177 | 0.807 | 0.864 | 0.247 |
| qwen2.5-1.5b+branched-seed0 | PREFIX_ONLY | te_en | 175 | 0.771 | 0.809 | 0.267 |
| qwen2.5-1.5b+branched-seed0 | PREFIX_PLUS_REACTION | hi_en | 188 | 0.931 | 0.925 | 0.063 |
| qwen2.5-1.5b+branched-seed0 | PREFIX_PLUS_REACTION | ta_en | 177 | 0.744 | 0.659 | 0.169 |
| qwen2.5-1.5b+branched-seed0 | PREFIX_PLUS_REACTION | te_en | 175 | 0.805 | 0.753 | 0.140 |
| qwen2.5-1.5b+branched-seed0 | CALLER_ONLY | hi_en | 188 | 0.887 | 0.785 | 0.011 |
| qwen2.5-1.5b+branched-seed0 | CALLER_ONLY | ta_en | 177 | 0.708 | 0.602 | 0.180 |
| qwen2.5-1.5b+branched-seed0 | CALLER_ONLY | te_en | 175 | 0.720 | 0.697 | 0.256 |
| qwen2.5-1.5b+branched-seed1 | FULL | hi_en | 188 | 0.888 | 0.828 | 0.053 |
| qwen2.5-1.5b+branched-seed1 | FULL | ta_en | 177 | 0.887 | 0.841 | 0.067 |
| qwen2.5-1.5b+branched-seed1 | FULL | te_en | 175 | 0.863 | 0.843 | 0.116 |
| qwen2.5-1.5b+branched-seed1 | PREFIX_ONLY | hi_en | 188 | 0.777 | 0.634 | 0.074 |
| qwen2.5-1.5b+branched-seed1 | PREFIX_ONLY | ta_en | 177 | 0.730 | 0.545 | 0.067 |
| qwen2.5-1.5b+branched-seed1 | PREFIX_ONLY | te_en | 175 | 0.816 | 0.719 | 0.081 |
| qwen2.5-1.5b+branched-seed1 | PREFIX_PLUS_REACTION | hi_en | 188 | 0.860 | 0.774 | 0.053 |
| qwen2.5-1.5b+branched-seed1 | PREFIX_PLUS_REACTION | ta_en | 177 | 0.811 | 0.693 | 0.067 |
| qwen2.5-1.5b+branched-seed1 | PREFIX_PLUS_REACTION | te_en | 175 | 0.840 | 0.798 | 0.116 |
| qwen2.5-1.5b+branched-seed1 | CALLER_ONLY | hi_en | 188 | 0.898 | 0.828 | 0.032 |
| qwen2.5-1.5b+branched-seed1 | CALLER_ONLY | ta_en | 177 | 0.788 | 0.670 | 0.090 |
| qwen2.5-1.5b+branched-seed1 | CALLER_ONLY | te_en | 175 | 0.805 | 0.742 | 0.128 |
| qwen2.5-1.5b+branched-seed2 | FULL | hi_en | 188 | 0.952 | 0.989 | 0.084 |
| qwen2.5-1.5b+branched-seed2 | FULL | ta_en | 177 | 0.874 | 0.773 | 0.022 |
| qwen2.5-1.5b+branched-seed2 | FULL | te_en | 175 | 0.869 | 0.876 | 0.140 |
| qwen2.5-1.5b+branched-seed2 | PREFIX_ONLY | hi_en | 188 | 0.803 | 0.806 | 0.200 |
| qwen2.5-1.5b+branched-seed2 | PREFIX_ONLY | ta_en | 177 | 0.689 | 0.557 | 0.169 |
| qwen2.5-1.5b+branched-seed2 | PREFIX_ONLY | te_en | 175 | 0.754 | 0.787 | 0.279 |
| qwen2.5-1.5b+branched-seed2 | PREFIX_PLUS_REACTION | hi_en | 188 | 0.915 | 0.946 | 0.116 |
| qwen2.5-1.5b+branched-seed2 | PREFIX_PLUS_REACTION | ta_en | 177 | 0.820 | 0.670 | 0.022 |
| qwen2.5-1.5b+branched-seed2 | PREFIX_PLUS_REACTION | te_en | 175 | 0.823 | 0.809 | 0.163 |
| qwen2.5-1.5b+branched-seed2 | CALLER_ONLY | hi_en | 188 | 0.871 | 0.796 | 0.053 |
| qwen2.5-1.5b+branched-seed2 | CALLER_ONLY | ta_en | 177 | 0.758 | 0.568 | 0.034 |
| qwen2.5-1.5b+branched-seed2 | CALLER_ONLY | te_en | 175 | 0.800 | 0.742 | 0.140 |
| qwen2.5-1.5b+branched_consistency-seed0 | FULL | hi_en | 188 | 0.910 | 0.914 | 0.095 |
| qwen2.5-1.5b+branched_consistency-seed0 | FULL | ta_en | 177 | 0.778 | 0.705 | 0.146 |
| qwen2.5-1.5b+branched_consistency-seed0 | FULL | te_en | 175 | 0.727 | 0.596 | 0.128 |
| qwen2.5-1.5b+branched_consistency-seed0 | PREFIX_ONLY | hi_en | 188 | 0.888 | 0.849 | 0.074 |
| qwen2.5-1.5b+branched_consistency-seed0 | PREFIX_ONLY | ta_en | 177 | 0.761 | 0.682 | 0.157 |
| qwen2.5-1.5b+branched_consistency-seed0 | PREFIX_ONLY | te_en | 175 | 0.724 | 0.629 | 0.174 |
| qwen2.5-1.5b+branched_consistency-seed0 | PREFIX_PLUS_REACTION | hi_en | 188 | 0.920 | 0.892 | 0.053 |
| qwen2.5-1.5b+branched_consistency-seed0 | PREFIX_PLUS_REACTION | ta_en | 177 | 0.709 | 0.545 | 0.112 |
| qwen2.5-1.5b+branched_consistency-seed0 | PREFIX_PLUS_REACTION | te_en | 175 | 0.729 | 0.551 | 0.070 |
| qwen2.5-1.5b+branched_consistency-seed0 | CALLER_ONLY | hi_en | 188 | 0.877 | 0.796 | 0.042 |
| qwen2.5-1.5b+branched_consistency-seed0 | CALLER_ONLY | ta_en | 177 | 0.705 | 0.636 | 0.225 |
| qwen2.5-1.5b+branched_consistency-seed0 | CALLER_ONLY | te_en | 175 | 0.726 | 0.730 | 0.279 |
| qwen2.5-1.5b+branched_consistency-seed1 | FULL | hi_en | 188 | 0.849 | 0.753 | 0.053 |
| qwen2.5-1.5b+branched_consistency-seed1 | FULL | ta_en | 177 | 0.770 | 0.648 | 0.101 |
| qwen2.5-1.5b+branched_consistency-seed1 | FULL | te_en | 175 | 0.746 | 0.640 | 0.140 |
| qwen2.5-1.5b+branched_consistency-seed1 | PREFIX_ONLY | hi_en | 188 | 0.792 | 0.634 | 0.042 |
| qwen2.5-1.5b+branched_consistency-seed1 | PREFIX_ONLY | ta_en | 177 | 0.709 | 0.545 | 0.112 |
| qwen2.5-1.5b+branched_consistency-seed1 | PREFIX_ONLY | te_en | 175 | 0.696 | 0.539 | 0.128 |
| qwen2.5-1.5b+branched_consistency-seed1 | PREFIX_PLUS_REACTION | hi_en | 188 | 0.763 | 0.591 | 0.053 |
| qwen2.5-1.5b+branched_consistency-seed1 | PREFIX_PLUS_REACTION | ta_en | 177 | 0.723 | 0.591 | 0.135 |
| qwen2.5-1.5b+branched_consistency-seed1 | PREFIX_PLUS_REACTION | te_en | 175 | 0.695 | 0.596 | 0.198 |
| qwen2.5-1.5b+branched_consistency-seed1 | CALLER_ONLY | hi_en | 188 | 0.888 | 0.828 | 0.053 |
| qwen2.5-1.5b+branched_consistency-seed1 | CALLER_ONLY | ta_en | 177 | 0.784 | 0.705 | 0.135 |
| qwen2.5-1.5b+branched_consistency-seed1 | CALLER_ONLY | te_en | 175 | 0.771 | 0.719 | 0.174 |
| qwen2.5-1.5b+branched_consistency-seed2 | FULL | hi_en | 188 | 0.936 | 0.978 | 0.105 |
| qwen2.5-1.5b+branched_consistency-seed2 | FULL | ta_en | 177 | 0.932 | 0.886 | 0.022 |
| qwen2.5-1.5b+branched_consistency-seed2 | FULL | te_en | 175 | 0.931 | 0.921 | 0.058 |
| qwen2.5-1.5b+branched_consistency-seed2 | PREFIX_ONLY | hi_en | 188 | 0.834 | 0.914 | 0.242 |
| qwen2.5-1.5b+branched_consistency-seed2 | PREFIX_ONLY | ta_en | 177 | 0.825 | 0.818 | 0.169 |
| qwen2.5-1.5b+branched_consistency-seed2 | PREFIX_ONLY | te_en | 175 | 0.800 | 0.831 | 0.233 |
| qwen2.5-1.5b+branched_consistency-seed2 | PREFIX_PLUS_REACTION | hi_en | 188 | 0.904 | 0.935 | 0.126 |
| qwen2.5-1.5b+branched_consistency-seed2 | PREFIX_PLUS_REACTION | ta_en | 177 | 0.852 | 0.761 | 0.056 |
| qwen2.5-1.5b+branched_consistency-seed2 | PREFIX_PLUS_REACTION | te_en | 175 | 0.855 | 0.730 | 0.012 |
| qwen2.5-1.5b+branched_consistency-seed2 | CALLER_ONLY | hi_en | 188 | 0.904 | 0.882 | 0.074 |
| qwen2.5-1.5b+branched_consistency-seed2 | CALLER_ONLY | ta_en | 177 | 0.774 | 0.625 | 0.067 |
| qwen2.5-1.5b+branched_consistency-seed2 | CALLER_ONLY | te_en | 175 | 0.839 | 0.764 | 0.081 |
| qwen2.5-1.5b+unbranched-seed0 | FULL | hi_en | 188 | 0.860 | 0.774 | 0.053 |
| qwen2.5-1.5b+unbranched-seed0 | FULL | ta_en | 177 | 0.842 | 0.886 | 0.202 |
| qwen2.5-1.5b+unbranched-seed0 | FULL | te_en | 175 | 0.829 | 0.809 | 0.151 |
| qwen2.5-1.5b+unbranched-seed0 | PREFIX_ONLY | hi_en | 188 | 0.803 | 0.806 | 0.200 |
| qwen2.5-1.5b+unbranched-seed0 | PREFIX_ONLY | ta_en | 177 | 0.620 | 0.773 | 0.517 |
| qwen2.5-1.5b+unbranched-seed0 | PREFIX_ONLY | te_en | 175 | 0.747 | 0.910 | 0.407 |
| qwen2.5-1.5b+unbranched-seed0 | PREFIX_PLUS_REACTION | hi_en | 188 | 0.871 | 0.796 | 0.053 |
| qwen2.5-1.5b+unbranched-seed0 | PREFIX_PLUS_REACTION | ta_en | 177 | 0.808 | 0.807 | 0.191 |
| qwen2.5-1.5b+unbranched-seed0 | PREFIX_PLUS_REACTION | te_en | 175 | 0.794 | 0.809 | 0.221 |
| qwen2.5-1.5b+unbranched-seed0 | CALLER_ONLY | hi_en | 188 | 0.821 | 0.688 | 0.042 |
| qwen2.5-1.5b+unbranched-seed0 | CALLER_ONLY | ta_en | 177 | 0.774 | 0.739 | 0.191 |
| qwen2.5-1.5b+unbranched-seed0 | CALLER_ONLY | te_en | 175 | 0.788 | 0.753 | 0.174 |
| qwen2.5-1.5b+unbranched-seed1 | FULL | hi_en | 188 | 0.936 | 0.946 | 0.074 |
| qwen2.5-1.5b+unbranched-seed1 | FULL | ta_en | 177 | 0.881 | 0.875 | 0.112 |
| qwen2.5-1.5b+unbranched-seed1 | FULL | te_en | 175 | 0.863 | 0.809 | 0.081 |
| qwen2.5-1.5b+unbranched-seed1 | PREFIX_ONLY | hi_en | 188 | 0.838 | 0.720 | 0.042 |
| qwen2.5-1.5b+unbranched-seed1 | PREFIX_ONLY | ta_en | 177 | 0.761 | 0.682 | 0.157 |
| qwen2.5-1.5b+unbranched-seed1 | PREFIX_ONLY | te_en | 175 | 0.834 | 0.843 | 0.174 |
| qwen2.5-1.5b+unbranched-seed1 | PREFIX_PLUS_REACTION | hi_en | 188 | 0.893 | 0.828 | 0.042 |
| qwen2.5-1.5b+unbranched-seed1 | PREFIX_PLUS_REACTION | ta_en | 177 | 0.716 | 0.523 | 0.067 |
| qwen2.5-1.5b+unbranched-seed1 | PREFIX_PLUS_REACTION | te_en | 175 | 0.832 | 0.708 | 0.035 |
| qwen2.5-1.5b+unbranched-seed1 | CALLER_ONLY | hi_en | 188 | 0.909 | 0.839 | 0.021 |
| qwen2.5-1.5b+unbranched-seed1 | CALLER_ONLY | ta_en | 177 | 0.790 | 0.727 | 0.146 |
| qwen2.5-1.5b+unbranched-seed1 | CALLER_ONLY | te_en | 175 | 0.834 | 0.820 | 0.151 |
| qwen2.5-1.5b+unbranched-seed2 | FULL | hi_en | 188 | 0.915 | 0.946 | 0.116 |
| qwen2.5-1.5b+unbranched-seed2 | FULL | ta_en | 177 | 0.819 | 0.795 | 0.157 |
| qwen2.5-1.5b+unbranched-seed2 | FULL | te_en | 175 | 0.817 | 0.798 | 0.163 |
| qwen2.5-1.5b+unbranched-seed2 | PREFIX_ONLY | hi_en | 188 | 0.835 | 0.871 | 0.200 |
| qwen2.5-1.5b+unbranched-seed2 | PREFIX_ONLY | ta_en | 177 | 0.718 | 0.727 | 0.292 |
| qwen2.5-1.5b+unbranched-seed2 | PREFIX_ONLY | te_en | 175 | 0.743 | 0.753 | 0.267 |
| qwen2.5-1.5b+unbranched-seed2 | PREFIX_PLUS_REACTION | hi_en | 188 | 0.888 | 0.882 | 0.105 |
| qwen2.5-1.5b+unbranched-seed2 | PREFIX_PLUS_REACTION | ta_en | 177 | 0.749 | 0.648 | 0.146 |
| qwen2.5-1.5b+unbranched-seed2 | PREFIX_PLUS_REACTION | te_en | 175 | 0.776 | 0.708 | 0.151 |
| qwen2.5-1.5b+unbranched-seed2 | CALLER_ONLY | hi_en | 188 | 0.888 | 0.914 | 0.137 |
| qwen2.5-1.5b+unbranched-seed2 | CALLER_ONLY | ta_en | 177 | 0.762 | 0.716 | 0.191 |
| qwen2.5-1.5b+unbranched-seed2 | CALLER_ONLY | te_en | 175 | 0.780 | 0.888 | 0.326 |

## Per recipient condition (test)

| model | view | condition | n | fraud recall | legit FPR | macro-F1 |
|---|---|---|---:|---:|---:|---:|
| qwen2.5-0.5b+unbranched-seed2 | FULL | cooperative | 133 | 0.569 | 0.191 | 0.686 |
| qwen2.5-0.5b+unbranched-seed2 | FULL | clarifying | 134 | 0.712 | 0.118 | 0.797 |
| qwen2.5-0.5b+unbranched-seed2 | FULL | resistant | 136 | 0.829 | 0.288 | 0.771 |
| qwen2.5-0.5b+unbranched-seed2 | FULL | suspicious | 137 | 0.971 | 0.250 | 0.859 |
| qwen2.5-0.5b+unbranched-seed2 | PREFIX_ONLY | cooperative | 133 | 0.615 | 0.294 | 0.660 |
| qwen2.5-0.5b+unbranched-seed2 | PREFIX_ONLY | clarifying | 134 | 0.606 | 0.294 | 0.655 |
| qwen2.5-0.5b+unbranched-seed2 | PREFIX_ONLY | resistant | 136 | 0.614 | 0.303 | 0.654 |
| qwen2.5-0.5b+unbranched-seed2 | PREFIX_ONLY | suspicious | 137 | 0.623 | 0.294 | 0.664 |
| qwen2.5-0.5b+unbranched-seed2 | PREFIX_PLUS_REACTION | cooperative | 133 | 0.677 | 0.132 | 0.772 |
| qwen2.5-0.5b+unbranched-seed2 | PREFIX_PLUS_REACTION | clarifying | 134 | 0.500 | 0.103 | 0.688 |
| qwen2.5-0.5b+unbranched-seed2 | PREFIX_PLUS_REACTION | resistant | 136 | 0.671 | 0.212 | 0.728 |
| qwen2.5-0.5b+unbranched-seed2 | PREFIX_PLUS_REACTION | suspicious | 137 | 0.870 | 0.206 | 0.832 |
| qwen2.5-0.5b+unbranched-seed2 | CALLER_ONLY | cooperative | 133 | 0.569 | 0.176 | 0.693 |
| qwen2.5-0.5b+unbranched-seed2 | CALLER_ONLY | clarifying | 134 | 0.576 | 0.191 | 0.689 |
| qwen2.5-0.5b+unbranched-seed2 | CALLER_ONLY | resistant | 136 | 0.814 | 0.333 | 0.740 |
| qwen2.5-0.5b+unbranched-seed2 | CALLER_ONLY | suspicious | 137 | 0.710 | 0.176 | 0.766 |
| qwen2.5-1.5b+branched-seed0 | FULL | cooperative | 133 | 0.615 | 0.059 | 0.775 |
| qwen2.5-1.5b+branched-seed0 | FULL | clarifying | 134 | 0.773 | 0.132 | 0.820 |
| qwen2.5-1.5b+branched-seed0 | FULL | resistant | 136 | 0.671 | 0.136 | 0.763 |
| qwen2.5-1.5b+branched-seed0 | FULL | suspicious | 137 | 0.841 | 0.088 | 0.876 |
| qwen2.5-1.5b+branched-seed0 | PREFIX_ONLY | cooperative | 133 | 0.862 | 0.191 | 0.835 |
| qwen2.5-1.5b+branched-seed0 | PREFIX_ONLY | clarifying | 134 | 0.848 | 0.191 | 0.828 |
| qwen2.5-1.5b+branched-seed0 | PREFIX_ONLY | resistant | 136 | 0.843 | 0.197 | 0.823 |
| qwen2.5-1.5b+branched-seed0 | PREFIX_ONLY | suspicious | 137 | 0.841 | 0.191 | 0.825 |
| qwen2.5-1.5b+branched-seed0 | PREFIX_PLUS_REACTION | cooperative | 133 | 0.815 | 0.088 | 0.864 |
| qwen2.5-1.5b+branched-seed0 | PREFIX_PLUS_REACTION | clarifying | 134 | 0.803 | 0.118 | 0.843 |
| qwen2.5-1.5b+branched-seed0 | PREFIX_PLUS_REACTION | resistant | 136 | 0.700 | 0.136 | 0.779 |
| qwen2.5-1.5b+branched-seed0 | PREFIX_PLUS_REACTION | suspicious | 137 | 0.812 | 0.147 | 0.832 |
| qwen2.5-1.5b+branched-seed0 | CALLER_ONLY | cooperative | 133 | 0.646 | 0.132 | 0.755 |
| qwen2.5-1.5b+branched-seed0 | CALLER_ONLY | clarifying | 134 | 0.712 | 0.147 | 0.782 |
| qwen2.5-1.5b+branched-seed0 | CALLER_ONLY | resistant | 136 | 0.700 | 0.197 | 0.750 |
| qwen2.5-1.5b+branched-seed0 | CALLER_ONLY | suspicious | 137 | 0.725 | 0.103 | 0.809 |
| qwen2.5-1.5b+branched-seed1 | FULL | cooperative | 133 | 0.831 | 0.074 | 0.879 |
| qwen2.5-1.5b+branched-seed1 | FULL | clarifying | 134 | 0.773 | 0.074 | 0.850 |
| qwen2.5-1.5b+branched-seed1 | FULL | resistant | 136 | 0.829 | 0.091 | 0.868 |
| qwen2.5-1.5b+branched-seed1 | FULL | suspicious | 137 | 0.913 | 0.074 | 0.920 |
| qwen2.5-1.5b+branched-seed1 | PREFIX_ONLY | cooperative | 133 | 0.646 | 0.074 | 0.784 |
| qwen2.5-1.5b+branched-seed1 | PREFIX_ONLY | clarifying | 134 | 0.636 | 0.074 | 0.778 |
| qwen2.5-1.5b+branched-seed1 | PREFIX_ONLY | resistant | 136 | 0.629 | 0.076 | 0.768 |
| qwen2.5-1.5b+branched-seed1 | PREFIX_ONLY | suspicious | 137 | 0.623 | 0.074 | 0.769 |
| qwen2.5-1.5b+branched-seed1 | PREFIX_PLUS_REACTION | cooperative | 133 | 0.815 | 0.074 | 0.871 |
| qwen2.5-1.5b+branched-seed1 | PREFIX_PLUS_REACTION | clarifying | 134 | 0.682 | 0.059 | 0.810 |
| qwen2.5-1.5b+branched-seed1 | PREFIX_PLUS_REACTION | resistant | 136 | 0.657 | 0.091 | 0.777 |
| qwen2.5-1.5b+branched-seed1 | PREFIX_PLUS_REACTION | suspicious | 137 | 0.870 | 0.088 | 0.890 |
| qwen2.5-1.5b+branched-seed1 | CALLER_ONLY | cooperative | 133 | 0.769 | 0.074 | 0.848 |
| qwen2.5-1.5b+branched-seed1 | CALLER_ONLY | clarifying | 134 | 0.758 | 0.088 | 0.834 |
| qwen2.5-1.5b+branched-seed1 | CALLER_ONLY | resistant | 136 | 0.729 | 0.106 | 0.808 |
| qwen2.5-1.5b+branched-seed1 | CALLER_ONLY | suspicious | 137 | 0.739 | 0.059 | 0.838 |
| qwen2.5-1.5b+branched-seed2 | FULL | cooperative | 133 | 0.754 | 0.044 | 0.855 |
| qwen2.5-1.5b+branched-seed2 | FULL | clarifying | 134 | 0.848 | 0.088 | 0.880 |
| qwen2.5-1.5b+branched-seed2 | FULL | resistant | 136 | 0.929 | 0.091 | 0.919 |
| qwen2.5-1.5b+branched-seed2 | FULL | suspicious | 137 | 0.986 | 0.103 | 0.941 |
| qwen2.5-1.5b+branched-seed2 | PREFIX_ONLY | cooperative | 133 | 0.723 | 0.206 | 0.759 |
| qwen2.5-1.5b+branched-seed2 | PREFIX_ONLY | clarifying | 134 | 0.727 | 0.221 | 0.753 |
| qwen2.5-1.5b+branched-seed2 | PREFIX_ONLY | resistant | 136 | 0.714 | 0.212 | 0.750 |
| qwen2.5-1.5b+branched-seed2 | PREFIX_ONLY | suspicious | 137 | 0.710 | 0.221 | 0.744 |
| qwen2.5-1.5b+branched-seed2 | PREFIX_PLUS_REACTION | cooperative | 133 | 0.723 | 0.015 | 0.854 |
| qwen2.5-1.5b+branched-seed2 | PREFIX_PLUS_REACTION | clarifying | 134 | 0.803 | 0.132 | 0.835 |
| qwen2.5-1.5b+branched-seed2 | PREFIX_PLUS_REACTION | resistant | 136 | 0.829 | 0.061 | 0.882 |
| qwen2.5-1.5b+branched-seed2 | PREFIX_PLUS_REACTION | suspicious | 137 | 0.884 | 0.191 | 0.846 |
| qwen2.5-1.5b+branched-seed2 | CALLER_ONLY | cooperative | 133 | 0.600 | 0.074 | 0.759 |
| qwen2.5-1.5b+branched-seed2 | CALLER_ONLY | clarifying | 134 | 0.848 | 0.074 | 0.888 |
| qwen2.5-1.5b+branched-seed2 | CALLER_ONLY | resistant | 136 | 0.686 | 0.106 | 0.785 |
| qwen2.5-1.5b+branched-seed2 | CALLER_ONLY | suspicious | 137 | 0.681 | 0.044 | 0.814 |
| qwen2.5-1.5b+branched_consistency-seed0 | FULL | cooperative | 133 | 0.646 | 0.118 | 0.763 |
| qwen2.5-1.5b+branched_consistency-seed0 | FULL | clarifying | 134 | 0.788 | 0.162 | 0.813 |
| qwen2.5-1.5b+branched_consistency-seed0 | FULL | resistant | 136 | 0.700 | 0.121 | 0.786 |
| qwen2.5-1.5b+branched_consistency-seed0 | FULL | suspicious | 137 | 0.826 | 0.088 | 0.868 |
| qwen2.5-1.5b+branched_consistency-seed0 | PREFIX_ONLY | cooperative | 133 | 0.738 | 0.132 | 0.803 |
| qwen2.5-1.5b+branched_consistency-seed0 | PREFIX_ONLY | clarifying | 134 | 0.727 | 0.132 | 0.797 |
| qwen2.5-1.5b+branched_consistency-seed0 | PREFIX_ONLY | resistant | 136 | 0.714 | 0.136 | 0.786 |
| qwen2.5-1.5b+branched_consistency-seed0 | PREFIX_ONLY | suspicious | 137 | 0.710 | 0.132 | 0.787 |
| qwen2.5-1.5b+branched_consistency-seed0 | PREFIX_PLUS_REACTION | cooperative | 133 | 0.708 | 0.059 | 0.824 |
| qwen2.5-1.5b+branched_consistency-seed0 | PREFIX_PLUS_REACTION | clarifying | 134 | 0.682 | 0.074 | 0.802 |
| qwen2.5-1.5b+branched_consistency-seed0 | PREFIX_PLUS_REACTION | resistant | 136 | 0.586 | 0.091 | 0.738 |
| qwen2.5-1.5b+branched_consistency-seed0 | PREFIX_PLUS_REACTION | suspicious | 137 | 0.696 | 0.088 | 0.801 |
| qwen2.5-1.5b+branched_consistency-seed0 | CALLER_ONLY | cooperative | 133 | 0.692 | 0.162 | 0.765 |
| qwen2.5-1.5b+branched_consistency-seed0 | CALLER_ONLY | clarifying | 134 | 0.727 | 0.206 | 0.761 |
| qwen2.5-1.5b+branched_consistency-seed0 | CALLER_ONLY | resistant | 136 | 0.729 | 0.197 | 0.765 |
| qwen2.5-1.5b+branched_consistency-seed0 | CALLER_ONLY | suspicious | 137 | 0.739 | 0.147 | 0.795 |
| qwen2.5-1.5b+branched_consistency-seed1 | FULL | cooperative | 133 | 0.723 | 0.088 | 0.817 |
| qwen2.5-1.5b+branched_consistency-seed1 | FULL | clarifying | 134 | 0.636 | 0.103 | 0.764 |
| qwen2.5-1.5b+branched_consistency-seed1 | FULL | resistant | 136 | 0.629 | 0.106 | 0.754 |
| qwen2.5-1.5b+branched_consistency-seed1 | FULL | suspicious | 137 | 0.739 | 0.088 | 0.824 |
| qwen2.5-1.5b+branched_consistency-seed1 | PREFIX_ONLY | cooperative | 133 | 0.585 | 0.088 | 0.744 |
| qwen2.5-1.5b+branched_consistency-seed1 | PREFIX_ONLY | clarifying | 134 | 0.576 | 0.088 | 0.738 |
| qwen2.5-1.5b+branched_consistency-seed1 | PREFIX_ONLY | resistant | 136 | 0.571 | 0.106 | 0.723 |
| qwen2.5-1.5b+branched_consistency-seed1 | PREFIX_ONLY | suspicious | 137 | 0.565 | 0.088 | 0.730 |
| qwen2.5-1.5b+branched_consistency-seed1 | PREFIX_PLUS_REACTION | cooperative | 133 | 0.646 | 0.074 | 0.784 |
| qwen2.5-1.5b+branched_consistency-seed1 | PREFIX_PLUS_REACTION | clarifying | 134 | 0.545 | 0.147 | 0.693 |
| qwen2.5-1.5b+branched_consistency-seed1 | PREFIX_PLUS_REACTION | resistant | 136 | 0.543 | 0.152 | 0.686 |
| qwen2.5-1.5b+branched_consistency-seed1 | PREFIX_PLUS_REACTION | suspicious | 137 | 0.638 | 0.132 | 0.749 |
| qwen2.5-1.5b+branched_consistency-seed1 | CALLER_ONLY | cooperative | 133 | 0.723 | 0.103 | 0.810 |
| qwen2.5-1.5b+branched_consistency-seed1 | CALLER_ONLY | clarifying | 134 | 0.773 | 0.088 | 0.842 |
| qwen2.5-1.5b+branched_consistency-seed1 | CALLER_ONLY | resistant | 136 | 0.771 | 0.136 | 0.816 |
| qwen2.5-1.5b+branched_consistency-seed1 | CALLER_ONLY | suspicious | 137 | 0.739 | 0.147 | 0.795 |
| qwen2.5-1.5b+branched_consistency-seed2 | FULL | cooperative | 133 | 0.846 | 0.029 | 0.909 |
| qwen2.5-1.5b+branched_consistency-seed2 | FULL | clarifying | 134 | 0.970 | 0.074 | 0.948 |
| qwen2.5-1.5b+branched_consistency-seed2 | FULL | resistant | 136 | 0.914 | 0.091 | 0.912 |
| qwen2.5-1.5b+branched_consistency-seed2 | FULL | suspicious | 137 | 0.986 | 0.059 | 0.963 |
| qwen2.5-1.5b+branched_consistency-seed2 | PREFIX_ONLY | cooperative | 133 | 0.862 | 0.206 | 0.827 |
| qwen2.5-1.5b+branched_consistency-seed2 | PREFIX_ONLY | clarifying | 134 | 0.848 | 0.221 | 0.813 |
| qwen2.5-1.5b+branched_consistency-seed2 | PREFIX_ONLY | resistant | 136 | 0.857 | 0.212 | 0.823 |
| qwen2.5-1.5b+branched_consistency-seed2 | PREFIX_ONLY | suspicious | 137 | 0.855 | 0.221 | 0.817 |
| qwen2.5-1.5b+branched_consistency-seed2 | PREFIX_PLUS_REACTION | cooperative | 133 | 0.723 | 0.015 | 0.854 |
| qwen2.5-1.5b+branched_consistency-seed2 | PREFIX_PLUS_REACTION | clarifying | 134 | 0.894 | 0.059 | 0.918 |
| qwen2.5-1.5b+branched_consistency-seed2 | PREFIX_PLUS_REACTION | resistant | 136 | 0.757 | 0.030 | 0.859 |
| qwen2.5-1.5b+branched_consistency-seed2 | PREFIX_PLUS_REACTION | suspicious | 137 | 0.870 | 0.162 | 0.854 |
| qwen2.5-1.5b+branched_consistency-seed2 | CALLER_ONLY | cooperative | 133 | 0.708 | 0.044 | 0.831 |
| qwen2.5-1.5b+branched_consistency-seed2 | CALLER_ONLY | clarifying | 134 | 0.818 | 0.103 | 0.858 |
| qwen2.5-1.5b+branched_consistency-seed2 | CALLER_ONLY | resistant | 136 | 0.743 | 0.121 | 0.808 |
| qwen2.5-1.5b+branched_consistency-seed2 | CALLER_ONLY | suspicious | 137 | 0.768 | 0.029 | 0.867 |
| qwen2.5-1.5b+unbranched-seed0 | FULL | cooperative | 133 | 0.723 | 0.162 | 0.781 |
| qwen2.5-1.5b+unbranched-seed0 | FULL | clarifying | 134 | 0.788 | 0.176 | 0.806 |
| qwen2.5-1.5b+unbranched-seed0 | FULL | resistant | 136 | 0.857 | 0.136 | 0.860 |
| qwen2.5-1.5b+unbranched-seed0 | FULL | suspicious | 137 | 0.913 | 0.059 | 0.927 |
| qwen2.5-1.5b+unbranched-seed0 | PREFIX_ONLY | cooperative | 133 | 0.831 | 0.368 | 0.727 |
| qwen2.5-1.5b+unbranched-seed0 | PREFIX_ONLY | clarifying | 134 | 0.833 | 0.368 | 0.729 |
| qwen2.5-1.5b+unbranched-seed0 | PREFIX_ONLY | resistant | 136 | 0.829 | 0.379 | 0.724 |
| qwen2.5-1.5b+unbranched-seed0 | PREFIX_ONLY | suspicious | 137 | 0.826 | 0.368 | 0.727 |
| qwen2.5-1.5b+unbranched-seed0 | PREFIX_PLUS_REACTION | cooperative | 133 | 0.785 | 0.176 | 0.804 |
| qwen2.5-1.5b+unbranched-seed0 | PREFIX_PLUS_REACTION | clarifying | 134 | 0.773 | 0.118 | 0.828 |
| qwen2.5-1.5b+unbranched-seed0 | PREFIX_PLUS_REACTION | resistant | 136 | 0.771 | 0.152 | 0.809 |
| qwen2.5-1.5b+unbranched-seed0 | PREFIX_PLUS_REACTION | suspicious | 137 | 0.884 | 0.162 | 0.861 |
| qwen2.5-1.5b+unbranched-seed0 | CALLER_ONLY | cooperative | 133 | 0.738 | 0.147 | 0.796 |
| qwen2.5-1.5b+unbranched-seed0 | CALLER_ONLY | clarifying | 134 | 0.788 | 0.162 | 0.813 |
| qwen2.5-1.5b+unbranched-seed0 | CALLER_ONLY | resistant | 136 | 0.729 | 0.167 | 0.779 |
| qwen2.5-1.5b+unbranched-seed0 | CALLER_ONLY | suspicious | 137 | 0.652 | 0.059 | 0.792 |
| qwen2.5-1.5b+unbranched-seed1 | FULL | cooperative | 133 | 0.800 | 0.074 | 0.864 |
| qwen2.5-1.5b+unbranched-seed1 | FULL | clarifying | 134 | 0.909 | 0.147 | 0.881 |
| qwen2.5-1.5b+unbranched-seed1 | FULL | resistant | 136 | 0.857 | 0.091 | 0.882 |
| qwen2.5-1.5b+unbranched-seed1 | FULL | suspicious | 137 | 0.942 | 0.044 | 0.949 |
| qwen2.5-1.5b+unbranched-seed1 | PREFIX_ONLY | cooperative | 133 | 0.754 | 0.118 | 0.818 |
| qwen2.5-1.5b+unbranched-seed1 | PREFIX_ONLY | clarifying | 134 | 0.758 | 0.118 | 0.820 |
| qwen2.5-1.5b+unbranched-seed1 | PREFIX_ONLY | resistant | 136 | 0.743 | 0.136 | 0.801 |
| qwen2.5-1.5b+unbranched-seed1 | PREFIX_ONLY | suspicious | 137 | 0.739 | 0.118 | 0.809 |
| qwen2.5-1.5b+unbranched-seed1 | PREFIX_PLUS_REACTION | cooperative | 133 | 0.600 | 0.029 | 0.780 |
| qwen2.5-1.5b+unbranched-seed1 | PREFIX_PLUS_REACTION | clarifying | 134 | 0.803 | 0.044 | 0.880 |
| qwen2.5-1.5b+unbranched-seed1 | PREFIX_PLUS_REACTION | resistant | 136 | 0.586 | 0.076 | 0.744 |
| qwen2.5-1.5b+unbranched-seed1 | PREFIX_PLUS_REACTION | suspicious | 137 | 0.768 | 0.044 | 0.860 |
| qwen2.5-1.5b+unbranched-seed1 | CALLER_ONLY | cooperative | 133 | 0.754 | 0.103 | 0.826 |
| qwen2.5-1.5b+unbranched-seed1 | CALLER_ONLY | clarifying | 134 | 0.848 | 0.074 | 0.888 |
| qwen2.5-1.5b+unbranched-seed1 | CALLER_ONLY | resistant | 136 | 0.771 | 0.136 | 0.816 |
| qwen2.5-1.5b+unbranched-seed1 | CALLER_ONLY | suspicious | 137 | 0.812 | 0.103 | 0.854 |
| qwen2.5-1.5b+unbranched-seed2 | FULL | cooperative | 133 | 0.677 | 0.118 | 0.779 |
| qwen2.5-1.5b+unbranched-seed2 | FULL | clarifying | 134 | 0.833 | 0.162 | 0.836 |
| qwen2.5-1.5b+unbranched-seed2 | FULL | resistant | 136 | 0.900 | 0.197 | 0.852 |
| qwen2.5-1.5b+unbranched-seed2 | FULL | suspicious | 137 | 0.971 | 0.103 | 0.934 |
| qwen2.5-1.5b+unbranched-seed2 | PREFIX_ONLY | cooperative | 133 | 0.785 | 0.250 | 0.767 |
| qwen2.5-1.5b+unbranched-seed2 | PREFIX_ONLY | clarifying | 134 | 0.788 | 0.250 | 0.769 |
| qwen2.5-1.5b+unbranched-seed2 | PREFIX_ONLY | resistant | 136 | 0.786 | 0.258 | 0.764 |
| qwen2.5-1.5b+unbranched-seed2 | PREFIX_ONLY | suspicious | 137 | 0.783 | 0.250 | 0.766 |
| qwen2.5-1.5b+unbranched-seed2 | PREFIX_PLUS_REACTION | cooperative | 133 | 0.631 | 0.044 | 0.790 |
| qwen2.5-1.5b+unbranched-seed2 | PREFIX_PLUS_REACTION | clarifying | 134 | 0.727 | 0.118 | 0.804 |
| qwen2.5-1.5b+unbranched-seed2 | PREFIX_PLUS_REACTION | resistant | 136 | 0.743 | 0.136 | 0.801 |
| qwen2.5-1.5b+unbranched-seed2 | PREFIX_PLUS_REACTION | suspicious | 137 | 0.884 | 0.235 | 0.824 |
| qwen2.5-1.5b+unbranched-seed2 | CALLER_ONLY | cooperative | 133 | 0.800 | 0.206 | 0.797 |
| qwen2.5-1.5b+unbranched-seed2 | CALLER_ONLY | clarifying | 134 | 0.833 | 0.176 | 0.828 |
| qwen2.5-1.5b+unbranched-seed2 | CALLER_ONLY | resistant | 136 | 0.857 | 0.258 | 0.800 |
| qwen2.5-1.5b+unbranched-seed2 | CALLER_ONLY | suspicious | 137 | 0.870 | 0.221 | 0.824 |

## Branch metrics (complete sibling sets, test)

| model | view | n histories | disagreement | score spread (mean) | all-correct |
|---|---|---:|---:|---:|---:|
| qwen2.5-0.5b+unbranched-seed2 | FULL | 122 | 0.500 | 0.356 | 0.484 |
| qwen2.5-0.5b+unbranched-seed2 | PREFIX_ONLY | 122 | 0.000 | 0.000 | 0.656 |
| qwen2.5-0.5b+unbranched-seed2 | PREFIX_PLUS_REACTION | 122 | 0.459 | 0.311 | 0.516 |
| qwen2.5-0.5b+unbranched-seed2 | CALLER_ONLY | 122 | 0.508 | 0.288 | 0.443 |
| qwen2.5-1.5b+branched-seed0 | FULL | 122 | 0.287 | 0.179 | 0.672 |
| qwen2.5-1.5b+branched-seed0 | PREFIX_ONLY | 122 | 0.000 | 0.000 | 0.852 |
| qwen2.5-1.5b+branched-seed0 | PREFIX_PLUS_REACTION | 122 | 0.172 | 0.136 | 0.762 |
| qwen2.5-1.5b+branched-seed0 | CALLER_ONLY | 122 | 0.238 | 0.175 | 0.656 |
| qwen2.5-1.5b+branched-seed1 | FULL | 122 | 0.156 | 0.197 | 0.795 |
| qwen2.5-1.5b+branched-seed1 | PREFIX_ONLY | 122 | 0.000 | 0.000 | 0.803 |
| qwen2.5-1.5b+branched-seed1 | PREFIX_PLUS_REACTION | 122 | 0.205 | 0.185 | 0.746 |
| qwen2.5-1.5b+branched-seed1 | CALLER_ONLY | 122 | 0.197 | 0.158 | 0.730 |
| qwen2.5-1.5b+branched-seed2 | FULL | 122 | 0.213 | 0.220 | 0.779 |
| qwen2.5-1.5b+branched-seed2 | PREFIX_ONLY | 122 | 0.000 | 0.000 | 0.754 |
| qwen2.5-1.5b+branched-seed2 | PREFIX_PLUS_REACTION | 122 | 0.254 | 0.298 | 0.705 |
| qwen2.5-1.5b+branched-seed2 | CALLER_ONLY | 122 | 0.205 | 0.220 | 0.713 |
| qwen2.5-1.5b+branched_consistency-seed0 | FULL | 122 | 0.328 | 0.179 | 0.631 |
| qwen2.5-1.5b+branched_consistency-seed0 | PREFIX_ONLY | 122 | 0.000 | 0.000 | 0.820 |
| qwen2.5-1.5b+branched_consistency-seed0 | PREFIX_PLUS_REACTION | 122 | 0.197 | 0.141 | 0.705 |
| qwen2.5-1.5b+branched_consistency-seed0 | CALLER_ONLY | 122 | 0.254 | 0.165 | 0.631 |
| qwen2.5-1.5b+branched_consistency-seed1 | FULL | 122 | 0.189 | 0.096 | 0.713 |
| qwen2.5-1.5b+branched_consistency-seed1 | PREFIX_ONLY | 122 | 0.000 | 0.000 | 0.754 |
| qwen2.5-1.5b+branched_consistency-seed1 | PREFIX_PLUS_REACTION | 122 | 0.180 | 0.103 | 0.664 |
| qwen2.5-1.5b+branched_consistency-seed1 | CALLER_ONLY | 122 | 0.180 | 0.162 | 0.730 |
| qwen2.5-1.5b+branched_consistency-seed2 | FULL | 122 | 0.172 | 0.162 | 0.820 |
| qwen2.5-1.5b+branched_consistency-seed2 | PREFIX_ONLY | 122 | 0.000 | 0.000 | 0.820 |
| qwen2.5-1.5b+branched_consistency-seed2 | PREFIX_PLUS_REACTION | 122 | 0.270 | 0.354 | 0.705 |
| qwen2.5-1.5b+branched_consistency-seed2 | CALLER_ONLY | 122 | 0.246 | 0.225 | 0.705 |
| qwen2.5-1.5b+unbranched-seed0 | FULL | 122 | 0.352 | 0.275 | 0.648 |
| qwen2.5-1.5b+unbranched-seed0 | PREFIX_ONLY | 122 | 0.000 | 0.001 | 0.730 |
| qwen2.5-1.5b+unbranched-seed0 | PREFIX_PLUS_REACTION | 122 | 0.369 | 0.289 | 0.615 |
| qwen2.5-1.5b+unbranched-seed0 | CALLER_ONLY | 122 | 0.262 | 0.208 | 0.648 |
| qwen2.5-1.5b+unbranched-seed1 | FULL | 122 | 0.197 | 0.208 | 0.787 |
| qwen2.5-1.5b+unbranched-seed1 | PREFIX_ONLY | 122 | 0.000 | 0.000 | 0.828 |
| qwen2.5-1.5b+unbranched-seed1 | PREFIX_PLUS_REACTION | 122 | 0.238 | 0.206 | 0.721 |
| qwen2.5-1.5b+unbranched-seed1 | CALLER_ONLY | 122 | 0.230 | 0.230 | 0.730 |
| qwen2.5-1.5b+unbranched-seed2 | FULL | 122 | 0.320 | 0.315 | 0.680 |
| qwen2.5-1.5b+unbranched-seed2 | PREFIX_ONLY | 122 | 0.000 | 0.000 | 0.779 |
| qwen2.5-1.5b+unbranched-seed2 | PREFIX_PLUS_REACTION | 122 | 0.287 | 0.291 | 0.664 |
| qwen2.5-1.5b+unbranched-seed2 | CALLER_ONLY | 122 | 0.262 | 0.241 | 0.664 |

## PREFIX_PLUS_REACTION − PREFIX_ONLY (mean Δ P(fraud), test)

| model | condition / label | mean Δ | 95% CI |
|---|---|---:|---|
| qwen2.5-0.5b+unbranched-seed2 | all | -0.021 | |
| qwen2.5-0.5b+unbranched-seed2 | cooperative | -0.038 | |
| qwen2.5-0.5b+unbranched-seed2 | clarifying | -0.085 | |
| qwen2.5-0.5b+unbranched-seed2 | resistant | -0.035 | |
| qwen2.5-0.5b+unbranched-seed2 | suspicious | 0.073 | |
| qwen2.5-0.5b+unbranched-seed2 | label=fraud | 0.026 | |
| qwen2.5-0.5b+unbranched-seed2 | label=legitimate | -0.067 | |
| qwen2.5-0.5b+unbranched-seed2 | cooperative|fraud | 0.036 |  [-0.005, 0.073] |
| qwen2.5-0.5b+unbranched-seed2 | cooperative|legitimate | -0.108 |  [-0.175, -0.043] |
| qwen2.5-0.5b+unbranched-seed2 | clarifying|fraud | -0.089 |  [-0.152, -0.026] |
| qwen2.5-0.5b+unbranched-seed2 | clarifying|legitimate | -0.080 |  [-0.133, -0.025] |
| qwen2.5-0.5b+unbranched-seed2 | resistant|fraud | -0.003 |  [-0.059, 0.066] |
| qwen2.5-0.5b+unbranched-seed2 | resistant|legitimate | -0.069 |  [-0.125, -0.012] |
| qwen2.5-0.5b+unbranched-seed2 | suspicious|fraud | 0.155 |  [0.087, 0.232] |
| qwen2.5-0.5b+unbranched-seed2 | suspicious|legitimate | -0.010 |  [-0.078, 0.052] |
| qwen2.5-1.5b+branched-seed0 | all | -0.008 | |
| qwen2.5-1.5b+branched-seed0 | cooperative | -0.027 | |
| qwen2.5-1.5b+branched-seed0 | clarifying | -0.004 | |
| qwen2.5-1.5b+branched-seed0 | resistant | -0.031 | |
| qwen2.5-1.5b+branched-seed0 | suspicious | 0.032 | |
| qwen2.5-1.5b+branched-seed0 | label=fraud | 0.001 | |
| qwen2.5-1.5b+branched-seed0 | label=legitimate | -0.016 | |
| qwen2.5-1.5b+branched-seed0 | cooperative|fraud | 0.005 |  [-0.026, 0.041] |
| qwen2.5-1.5b+branched-seed0 | cooperative|legitimate | -0.058 |  [-0.091, -0.027] |
| qwen2.5-1.5b+branched-seed0 | clarifying|fraud | 0.008 |  [-0.022, 0.035] |
| qwen2.5-1.5b+branched-seed0 | clarifying|legitimate | -0.016 |  [-0.057, 0.021] |
| qwen2.5-1.5b+branched-seed0 | resistant|fraud | -0.025 |  [-0.061, 0.003] |
| qwen2.5-1.5b+branched-seed0 | resistant|legitimate | -0.039 |  [-0.066, -0.013] |
| qwen2.5-1.5b+branched-seed0 | suspicious|fraud | 0.016 |  [-0.010, 0.045] |
| qwen2.5-1.5b+branched-seed0 | suspicious|legitimate | 0.047 |  [-0.006, 0.099] |
| qwen2.5-1.5b+branched-seed1 | all | 0.066 | |
| qwen2.5-1.5b+branched-seed1 | cooperative | 0.075 | |
| qwen2.5-1.5b+branched-seed1 | clarifying | 0.034 | |
| qwen2.5-1.5b+branched-seed1 | resistant | 0.032 | |
| qwen2.5-1.5b+branched-seed1 | suspicious | 0.120 | |
| qwen2.5-1.5b+branched-seed1 | label=fraud | 0.119 | |
| qwen2.5-1.5b+branched-seed1 | label=legitimate | 0.012 | |
| qwen2.5-1.5b+branched-seed1 | cooperative|fraud | 0.163 |  [0.074, 0.266] |
| qwen2.5-1.5b+branched-seed1 | cooperative|legitimate | -0.008 |  [-0.043, 0.024] |
| qwen2.5-1.5b+branched-seed1 | clarifying|fraud | 0.067 |  [-0.042, 0.180] |
| qwen2.5-1.5b+branched-seed1 | clarifying|legitimate | 0.003 |  [-0.031, 0.034] |
| qwen2.5-1.5b+branched-seed1 | resistant|fraud | 0.058 |  [-0.047, 0.154] |
| qwen2.5-1.5b+branched-seed1 | resistant|legitimate | 0.005 |  [-0.018, 0.033] |
| qwen2.5-1.5b+branched-seed1 | suspicious|fraud | 0.190 |  [0.087, 0.325] |
| qwen2.5-1.5b+branched-seed1 | suspicious|legitimate | 0.050 |  [0.013, 0.085] |
| qwen2.5-1.5b+branched-seed2 | all | 0.044 | |
| qwen2.5-1.5b+branched-seed2 | cooperative | -0.039 | |
| qwen2.5-1.5b+branched-seed2 | clarifying | 0.067 | |
| qwen2.5-1.5b+branched-seed2 | resistant | -0.010 | |
| qwen2.5-1.5b+branched-seed2 | suspicious | 0.156 | |
| qwen2.5-1.5b+branched-seed2 | label=fraud | 0.062 | |
| qwen2.5-1.5b+branched-seed2 | label=legitimate | 0.027 | |
| qwen2.5-1.5b+branched-seed2 | cooperative|fraud | 0.034 |  [0.003, 0.078] |
| qwen2.5-1.5b+branched-seed2 | cooperative|legitimate | -0.108 |  [-0.180, -0.023] |
| qwen2.5-1.5b+branched-seed2 | clarifying|fraud | 0.062 |  [0.025, 0.114] |
| qwen2.5-1.5b+branched-seed2 | clarifying|legitimate | 0.071 |  [-0.028, 0.154] |
| qwen2.5-1.5b+branched-seed2 | resistant|fraud | 0.059 |  [0.026, 0.101] |
| qwen2.5-1.5b+branched-seed2 | resistant|legitimate | -0.082 |  [-0.160, 0.009] |
| qwen2.5-1.5b+branched-seed2 | suspicious|fraud | 0.090 |  [0.035, 0.166] |
| qwen2.5-1.5b+branched-seed2 | suspicious|legitimate | 0.222 |  [0.102, 0.362] |
| qwen2.5-1.5b+branched_consistency-seed0 | all | -0.003 | |
| qwen2.5-1.5b+branched_consistency-seed0 | cooperative | -0.024 | |
| qwen2.5-1.5b+branched_consistency-seed0 | clarifying | 0.005 | |
| qwen2.5-1.5b+branched_consistency-seed0 | resistant | -0.030 | |
| qwen2.5-1.5b+branched_consistency-seed0 | suspicious | 0.037 | |
| qwen2.5-1.5b+branched_consistency-seed0 | label=fraud | 0.005 | |
| qwen2.5-1.5b+branched_consistency-seed0 | label=legitimate | -0.011 | |
| qwen2.5-1.5b+branched_consistency-seed0 | cooperative|fraud | 0.010 |  [-0.027, 0.052] |
| qwen2.5-1.5b+branched_consistency-seed0 | cooperative|legitimate | -0.058 |  [-0.089, -0.028] |
| qwen2.5-1.5b+branched_consistency-seed0 | clarifying|fraud | 0.016 |  [-0.022, 0.049] |
| qwen2.5-1.5b+branched_consistency-seed0 | clarifying|legitimate | -0.006 |  [-0.041, 0.025] |
| qwen2.5-1.5b+branched_consistency-seed0 | resistant|fraud | -0.022 |  [-0.061, 0.010] |
| qwen2.5-1.5b+branched_consistency-seed0 | resistant|legitimate | -0.039 |  [-0.068, -0.012] |
| qwen2.5-1.5b+branched_consistency-seed0 | suspicious|fraud | 0.018 |  [-0.009, 0.046] |
| qwen2.5-1.5b+branched_consistency-seed0 | suspicious|legitimate | 0.057 |  [0.011, 0.104] |
| qwen2.5-1.5b+branched_consistency-seed1 | all | 0.029 | |
| qwen2.5-1.5b+branched_consistency-seed1 | cooperative | 0.011 | |
| qwen2.5-1.5b+branched_consistency-seed1 | clarifying | 0.022 | |
| qwen2.5-1.5b+branched_consistency-seed1 | resistant | 0.004 | |
| qwen2.5-1.5b+branched_consistency-seed1 | suspicious | 0.076 | |
| qwen2.5-1.5b+branched_consistency-seed1 | label=fraud | 0.053 | |
| qwen2.5-1.5b+branched_consistency-seed1 | label=legitimate | 0.004 | |
| qwen2.5-1.5b+branched_consistency-seed1 | cooperative|fraud | 0.066 |  [0.007, 0.141] |
| qwen2.5-1.5b+branched_consistency-seed1 | cooperative|legitimate | -0.042 |  [-0.099, 0.011] |
| qwen2.5-1.5b+branched_consistency-seed1 | clarifying|fraud | 0.043 |  [0.001, 0.096] |
| qwen2.5-1.5b+branched_consistency-seed1 | clarifying|legitimate | 0.002 |  [-0.053, 0.067] |
| qwen2.5-1.5b+branched_consistency-seed1 | resistant|fraud | 0.039 |  [-0.020, 0.104] |
| qwen2.5-1.5b+branched_consistency-seed1 | resistant|legitimate | -0.032 |  [-0.093, 0.027] |
| qwen2.5-1.5b+branched_consistency-seed1 | suspicious|fraud | 0.065 |  [0.012, 0.130] |
| qwen2.5-1.5b+branched_consistency-seed1 | suspicious|legitimate | 0.087 |  [-0.002, 0.189] |
| qwen2.5-1.5b+branched_consistency-seed2 | all | 0.041 | |
| qwen2.5-1.5b+branched_consistency-seed2 | cooperative | -0.027 | |
| qwen2.5-1.5b+branched_consistency-seed2 | clarifying | 0.086 | |
| qwen2.5-1.5b+branched_consistency-seed2 | resistant | -0.031 | |
| qwen2.5-1.5b+branched_consistency-seed2 | suspicious | 0.134 | |
| qwen2.5-1.5b+branched_consistency-seed2 | label=fraud | 0.061 | |
| qwen2.5-1.5b+branched_consistency-seed2 | label=legitimate | 0.020 | |
| qwen2.5-1.5b+branched_consistency-seed2 | cooperative|fraud | 0.040 |  [-0.014, 0.116] |
| qwen2.5-1.5b+branched_consistency-seed2 | cooperative|legitimate | -0.092 |  [-0.254, 0.037] |
| qwen2.5-1.5b+branched_consistency-seed2 | clarifying|fraud | 0.085 |  [0.023, 0.179] |
| qwen2.5-1.5b+branched_consistency-seed2 | clarifying|legitimate | 0.087 |  [-0.090, 0.249] |
| qwen2.5-1.5b+branched_consistency-seed2 | resistant|fraud | 0.038 |  [-0.027, 0.116] |
| qwen2.5-1.5b+branched_consistency-seed2 | resistant|legitimate | -0.105 |  [-0.250, 0.027] |
| qwen2.5-1.5b+branched_consistency-seed2 | suspicious|fraud | 0.082 |  [0.019, 0.174] |
| qwen2.5-1.5b+branched_consistency-seed2 | suspicious|legitimate | 0.186 |  [-0.018, 0.395] |
| qwen2.5-1.5b+unbranched-seed0 | all | 0.098 | |
| qwen2.5-1.5b+unbranched-seed0 | cooperative | 0.067 | |
| qwen2.5-1.5b+unbranched-seed0 | clarifying | 0.082 | |
| qwen2.5-1.5b+unbranched-seed0 | resistant | 0.084 | |
| qwen2.5-1.5b+unbranched-seed0 | suspicious | 0.157 | |
| qwen2.5-1.5b+unbranched-seed0 | label=fraud | 0.164 | |
| qwen2.5-1.5b+unbranched-seed0 | label=legitimate | 0.032 | |
| qwen2.5-1.5b+unbranched-seed0 | cooperative|fraud | 0.142 |  [0.058, 0.223] |
| qwen2.5-1.5b+unbranched-seed0 | cooperative|legitimate | -0.004 |  [-0.039, 0.036] |
| qwen2.5-1.5b+unbranched-seed0 | clarifying|fraud | 0.132 |  [0.064, 0.188] |
| qwen2.5-1.5b+unbranched-seed0 | clarifying|legitimate | 0.032 |  [-0.004, 0.068] |
| qwen2.5-1.5b+unbranched-seed0 | resistant|fraud | 0.153 |  [0.083, 0.227] |
| qwen2.5-1.5b+unbranched-seed0 | resistant|legitimate | 0.010 |  [-0.058, 0.084] |
| qwen2.5-1.5b+unbranched-seed0 | suspicious|fraud | 0.226 |  [0.141, 0.317] |
| qwen2.5-1.5b+unbranched-seed0 | suspicious|legitimate | 0.087 |  [0.014, 0.166] |
| qwen2.5-1.5b+unbranched-seed1 | all | 0.028 | |
| qwen2.5-1.5b+unbranched-seed1 | cooperative | -0.034 | |
| qwen2.5-1.5b+unbranched-seed1 | clarifying | 0.048 | |
| qwen2.5-1.5b+unbranched-seed1 | resistant | -0.004 | |
| qwen2.5-1.5b+unbranched-seed1 | suspicious | 0.099 | |
| qwen2.5-1.5b+unbranched-seed1 | label=fraud | 0.082 | |
| qwen2.5-1.5b+unbranched-seed1 | label=legitimate | -0.027 | |
| qwen2.5-1.5b+unbranched-seed1 | cooperative|fraud | 0.032 |  [-0.029, 0.086] |
| qwen2.5-1.5b+unbranched-seed1 | cooperative|legitimate | -0.097 |  [-0.189, -0.017] |
| qwen2.5-1.5b+unbranched-seed1 | clarifying|fraud | 0.130 |  [0.046, 0.205] |
| qwen2.5-1.5b+unbranched-seed1 | clarifying|legitimate | -0.031 |  [-0.084, 0.014] |
| qwen2.5-1.5b+unbranched-seed1 | resistant|fraud | 0.045 |  [-0.023, 0.105] |
| qwen2.5-1.5b+unbranched-seed1 | resistant|legitimate | -0.056 |  [-0.123, -0.003] |
| qwen2.5-1.5b+unbranched-seed1 | suspicious|fraud | 0.121 |  [0.066, 0.178] |
| qwen2.5-1.5b+unbranched-seed1 | suspicious|legitimate | 0.076 |  [0.017, 0.130] |
| qwen2.5-1.5b+unbranched-seed2 | all | 0.030 | |
| qwen2.5-1.5b+unbranched-seed2 | cooperative | -0.041 | |
| qwen2.5-1.5b+unbranched-seed2 | clarifying | 0.032 | |
| qwen2.5-1.5b+unbranched-seed2 | resistant | -0.020 | |
| qwen2.5-1.5b+unbranched-seed2 | suspicious | 0.146 | |
| qwen2.5-1.5b+unbranched-seed2 | label=fraud | 0.060 | |
| qwen2.5-1.5b+unbranched-seed2 | label=legitimate | -0.000 | |
| qwen2.5-1.5b+unbranched-seed2 | cooperative|fraud | 0.012 |  [-0.039, 0.078] |
| qwen2.5-1.5b+unbranched-seed2 | cooperative|legitimate | -0.091 |  [-0.186, -0.023] |
| qwen2.5-1.5b+unbranched-seed2 | clarifying|fraud | 0.076 |  [0.016, 0.148] |
| qwen2.5-1.5b+unbranched-seed2 | clarifying|legitimate | -0.010 |  [-0.103, 0.058] |
| qwen2.5-1.5b+unbranched-seed2 | resistant|fraud | 0.034 |  [-0.020, 0.097] |
| qwen2.5-1.5b+unbranched-seed2 | resistant|legitimate | -0.076 |  [-0.148, -0.014] |
| qwen2.5-1.5b+unbranched-seed2 | suspicious|fraud | 0.119 |  [0.025, 0.233] |
| qwen2.5-1.5b+unbranched-seed2 | suspicious|legitimate | 0.174 |  [0.100, 0.254] |

## Complete-set slice (test)

| model | view | n | macro-F1 | disagreement | all-correct |
|---|---|---:|---:|---:|---:|
| qwen2.5-0.5b+unbranched-seed2 | FULL | 488 | 0.801 | 0.500 | 0.484 |
| qwen2.5-0.5b+unbranched-seed2 | PREFIX_ONLY | 488 | 0.652 | 0.000 | 0.656 |
| qwen2.5-0.5b+unbranched-seed2 | PREFIX_PLUS_REACTION | 488 | 0.769 | 0.459 | 0.516 |
| qwen2.5-0.5b+unbranched-seed2 | CALLER_ONLY | 488 | 0.737 | 0.508 | 0.443 |
| qwen2.5-1.5b+branched-seed0 | FULL | 488 | 0.831 | 0.287 | 0.672 |
| qwen2.5-1.5b+branched-seed0 | PREFIX_ONLY | 488 | 0.852 | 0.000 | 0.852 |
| qwen2.5-1.5b+branched-seed0 | PREFIX_PLUS_REACTION | 488 | 0.853 | 0.172 | 0.762 |
| qwen2.5-1.5b+branched-seed0 | CALLER_ONLY | 488 | 0.773 | 0.238 | 0.656 |
| qwen2.5-1.5b+branched-seed1 | FULL | 488 | 0.897 | 0.156 | 0.795 |
| qwen2.5-1.5b+branched-seed1 | PREFIX_ONLY | 488 | 0.795 | 0.000 | 0.803 |
| qwen2.5-1.5b+branched-seed1 | PREFIX_PLUS_REACTION | 488 | 0.858 | 0.205 | 0.746 |
| qwen2.5-1.5b+branched-seed1 | CALLER_ONLY | 488 | 0.843 | 0.197 | 0.730 |
| qwen2.5-1.5b+branched-seed2 | FULL | 488 | 0.903 | 0.213 | 0.779 |
| qwen2.5-1.5b+branched-seed2 | PREFIX_ONLY | 488 | 0.753 | 0.000 | 0.754 |
| qwen2.5-1.5b+branched-seed2 | PREFIX_PLUS_REACTION | 488 | 0.855 | 0.254 | 0.705 |
| qwen2.5-1.5b+branched-seed2 | CALLER_ONLY | 488 | 0.832 | 0.205 | 0.713 |
| qwen2.5-1.5b+branched_consistency-seed0 | FULL | 488 | 0.832 | 0.328 | 0.631 |
| qwen2.5-1.5b+branched_consistency-seed0 | PREFIX_ONLY | 488 | 0.817 | 0.000 | 0.820 |
| qwen2.5-1.5b+branched_consistency-seed0 | PREFIX_PLUS_REACTION | 488 | 0.815 | 0.197 | 0.705 |
| qwen2.5-1.5b+branched_consistency-seed0 | CALLER_ONLY | 488 | 0.770 | 0.254 | 0.631 |
| qwen2.5-1.5b+branched_consistency-seed1 | FULL | 488 | 0.805 | 0.189 | 0.713 |
| qwen2.5-1.5b+branched_consistency-seed1 | PREFIX_ONLY | 488 | 0.742 | 0.000 | 0.754 |
| qwen2.5-1.5b+branched_consistency-seed1 | PREFIX_PLUS_REACTION | 488 | 0.746 | 0.180 | 0.664 |
| qwen2.5-1.5b+branched_consistency-seed1 | CALLER_ONLY | 488 | 0.828 | 0.180 | 0.730 |
| qwen2.5-1.5b+branched_consistency-seed2 | FULL | 488 | 0.947 | 0.172 | 0.820 |
| qwen2.5-1.5b+branched_consistency-seed2 | PREFIX_ONLY | 488 | 0.820 | 0.000 | 0.820 |
| qwen2.5-1.5b+branched_consistency-seed2 | PREFIX_PLUS_REACTION | 488 | 0.880 | 0.270 | 0.705 |
| qwen2.5-1.5b+branched_consistency-seed2 | CALLER_ONLY | 488 | 0.846 | 0.246 | 0.705 |
| qwen2.5-1.5b+unbranched-seed0 | FULL | 488 | 0.864 | 0.352 | 0.648 |
| qwen2.5-1.5b+unbranched-seed0 | PREFIX_ONLY | 488 | 0.729 | 0.000 | 0.730 |
| qwen2.5-1.5b+unbranched-seed0 | PREFIX_PLUS_REACTION | 488 | 0.837 | 0.369 | 0.615 |
| qwen2.5-1.5b+unbranched-seed0 | CALLER_ONLY | 488 | 0.798 | 0.262 | 0.648 |
| qwen2.5-1.5b+unbranched-seed1 | FULL | 488 | 0.918 | 0.197 | 0.787 |
| qwen2.5-1.5b+unbranched-seed1 | PREFIX_ONLY | 488 | 0.826 | 0.000 | 0.828 |
| qwen2.5-1.5b+unbranched-seed1 | PREFIX_PLUS_REACTION | 488 | 0.836 | 0.238 | 0.721 |
| qwen2.5-1.5b+unbranched-seed1 | CALLER_ONLY | 488 | 0.857 | 0.230 | 0.730 |
| qwen2.5-1.5b+unbranched-seed2 | FULL | 488 | 0.868 | 0.320 | 0.680 |
| qwen2.5-1.5b+unbranched-seed2 | PREFIX_ONLY | 488 | 0.779 | 0.000 | 0.779 |
| qwen2.5-1.5b+unbranched-seed2 | PREFIX_PLUS_REACTION | 488 | 0.818 | 0.287 | 0.664 |
| qwen2.5-1.5b+unbranched-seed2 | CALLER_ONLY | 488 | 0.820 | 0.262 | 0.664 |

## Streaming delay (test; misses counted)

| model | eligible | detected | miss rate | mean delay (hits) | pre-evidence alert rate |
|---|---:|---:|---:|---:|---:|

## Runtime projection (full panel)

- median per-inference: 0.082 s
- mean turns: 9.94
- assumed test+dev conversations: 540+540
- one model, 4 views: 0.32 h
- 5 models × 4 views: 6.04 h
- note: STREAMING costs one inference per turn; the other three views cost one each. Naive 4×views undercounts STREAMING.
