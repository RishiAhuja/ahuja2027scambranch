# Hinglish-TOP — license record

Recorded: 2026-09-25
Local files:

- `data/calibration/raw/spoken/hinglish_top/train.tsv`
- `data/calibration/raw/spoken/hinglish_top/validation.tsv`
- `data/calibration/raw/spoken/hinglish_top/test.tsv`
- `data/calibration/raw/spoken/hinglish_top/LICENSE.md` (repo copy)
- `data/calibration/raw/spoken/hinglish_top/README.md`
- earlier `train_sample_200.tsv` kept beside the full files

## Source

- Official repository: https://github.com/google-research-datasets/Hinglish-TOP-Dataset
- Hugging Face mirror: https://huggingface.co/datasets/WillHeld/hinglish_top
- Paper: Agarwal et al., *CST5: Data Augmentation for Code-Switched Semantic Parsing*, arXiv:2211.07514

## License

**Apache License 2.0**, verified from `LICENSE.md` in the official GitHub repository (copy stored next to the TSVs).

## What was downloaded

The **human-annotated** split only (`Dataset/Human Annotated Data/{train,validation,test}.tsv`). Columns: `en_query`, `cs_query`, `en_parse`, `cs_parse`, `domain`.

The ~170k **synthetically generated** CST5/mT5 file under `Dataset/Synthetically Generated Data/` was **not** downloaded and must not be used for human-register calibration.

## Citation

Agarwal, Anmol; Gupta, Jigar; Goel, Rahul; Upadhyay, Shyam; Joshi, Pankaj; Aravamudhan, Rengarajan. 2022. CST5: Data Augmentation for Code-Switched Semantic Parsing. arXiv:2211.07514.
