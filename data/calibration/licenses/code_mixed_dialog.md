# Code-Mixed-Dialog (DSTC2) — license record

Recorded: 2026-09-25

Local files:

- `data/calibration/raw/spoken/code_mixed_dialog/hi_en/dialog-dstc2-{trn,dev,tst}.txt`
- `data/calibration/raw/spoken/code_mixed_dialog/ta_en/dialog-dstc2-{trn,dev,tst}.txt`
- `data/calibration/raw/spoken/code_mixed_dialog/{hi_en,ta_en}/vocab_splits.json`
- `data/calibration/raw/spoken/code_mixed_dialog/LICENSE` (repo copy)
- `data/calibration/raw/spoken/code_mixed_dialog/README.md`
- earlier 200-line samples kept beside the full files

## Source

- Official repository: https://github.com/sumanbanerjee1/Code-Mixed-Dialog
- Paper: Banerjee, Moghe, Arora, Khapra, *A Dataset for Building Code-Mixed Goal Oriented Conversation Systems*, COLING 2018, https://aclanthology.org/C18-1319/ (arXiv:1806.05997)
- Languages kept here: Hindi–English and Tamil–English only (repo also has Bengali, Gujarati, English).

## License (two statements; both permissive)

1. **Paper footnote (COLING 2018 / ACL Anthology C18-1319):** “This work is licensed under a Creative Commons Attribution 4.0 International License. License details: http://creativecommons.org/licenses/by/4.0/”
2. **GitHub `LICENSE` file (present in the official repo as of 2026-09-25):** **Apache License 2.0** (full text copied to `raw/spoken/code_mixed_dialog/LICENSE`). A previous calibration note that the repo had no LICENSE file is outdated.

English DSTC2 source conversations (Henderson et al. 2014; Bordes & Weston 2017 adaptation) are a separate corpus. This record covers the IIT Madras code-mixed translations.

## What was downloaded

Full `dialog-dstc2-trn.txt`, `dialog-dstc2-dev.txt`, `dialog-dstc2-tst.txt` plus `vocab_splits.json` for Hindi and Tamil. **No candidate-response files** exist in those language folders. The 170k-class synthetic sets are not part of this repo.

## Citation

Banerjee, Suman; Moghe, Nikita; Arora, Siddhartha; Khapra, Mitesh M. 2018. A Dataset for Building Code-Mixed Goal Oriented Conversation Systems. In *Proceedings of the 27th International Conference on Computational Linguistics*, pages 3766–3780.
