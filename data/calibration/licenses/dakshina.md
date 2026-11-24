# Dakshina dataset — license record

Recorded: 2026-09-25
Local extract: `data/calibration/raw/dakshina/` (Hindi, Tamil, Telugu lexicons and romanized folders only)

## Source

- Official repository: https://github.com/google-research-datasets/dakshina
- Tarball: https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar
- Version: 1.0 (2020-05-27, initial data release)
- Paper: Roark et al., LREC 2020, https://aclanthology.org/2020.lrec-1.294/

## License

**CC BY-SA 4.0**

Verified from official Dakshina files listed below.

Files inspected for license/README text:

- `dakshina_dataset_v1.0/README.md`

Files that contained an explicit CC BY-SA 4.0 (or Creative Commons Attribution-ShareAlike 4.0) statement:

- `dakshina_dataset_v1.0/README.md`

The official dataset README states:

> The dataset is licensed under [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/).
> Any third party content or data is provided "As Is" without any warranty, express or implied.

Verification used the v1.0 tarball `README.md`. There is **no standalone LICENSE file** in the tarball or in the GitHub repository (confirmed: GitHub tree contains only `README.md`). The license statement lives in that README.

## Citation

@inproceedings{roark-etal-2020-processing,
    title = "Processing {South} {Asian} Languages Written in the {Latin} Script:
    the {Dakshina} Dataset",
    author = "Roark, Brian and
      Wolf-Sonkin, Lawrence and
      Kirov, Christo and
      Mielke, Sabrina J. and
      Johny, Cibu and
      Demir{\c{s}}ahin, I{\c{s}}in and
      Hall, Keith",
    booktitle = "Proceedings of The 12th Language Resources and Evaluation Conference (LREC)",
    year = "2020",
    url = "https://www.aclweb.org/anthology/2020.lrec-1.294",
    pages = "2413--2423"
}

## Redistribution terms (CC BY-SA 4.0)

- You may share and adapt the material.
- **Attribution** is required (credit Roark et al. 2020 / Google Research Dakshina; link the license; note changes).
- **ShareAlike**: adapted material, including derived lexicons and variant tables produced from this dataset, must be licensed under CC BY-SA 4.0 (or a compatible license).
- Native-script Wikipedia strings in Dakshina are derived from Wikipedia and remain subject to Wikipedia/CC BY-SA provenance in addition to the dataset license.
- This repository keeps only `hi`/`ta`/`te` `lexicons/` and `romanized/` folders, not the full 12-language tarball. Do not treat processed TSVs as public-domain spelling lists.

## What we extracted

Only romanization lexicons and sentence-level romanized Wikipedia strings for `hi`, `ta`, and `te`. Native-script Wikipedia dumps for those languages were not kept.
