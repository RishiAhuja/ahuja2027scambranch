# lince (Hindi-English LID + POS)

- **source_id:** `lince`, `lince_pos`
- **URL:** https://huggingface.co/datasets/lince-benchmark/lince (configs `lid_hineng`, `pos_hineng`); official site https://ritual.uh.edu/lince/ (login-walled; site unreachable at harvest). Data taken from the Hugging Face `refs/convert/parquet` conversion.
- **version:** LinCE benchmark 1.0.0 / HF parquet conversion harvested 2026-09-25. LID splits used: train 4823 + validation 744 (test 1854 has empty LID labels and was skipped). POS splits used: train 1030 + validation 160 + test 299 (test POS labels empty; LID kept).
- **size:** raw parquet 872959 bytes; processed utterances: LID 5566, POS 1489.
- **underlying corpora:** LID = Mave, Maharjan & Solorio 2018 (Facebook/Twitter + ICON 2016 reuse). POS = Singh, Sen & Kumaraguru 2018 (Hindi-English Twitter POS).
- **stated terms:** Hugging Face dataset card has no license field ("Licensing Information: More Information Needed"). The official LinCE datasets page states that Twitter-derived data is **distributed for non-commercial and for research purposes only, following Twitter's own Developer Agreement and Policy**.
- **redistribution:** **NOT allowed** for raw tweet text (Twitter Developer Agreement + official non-commercial research restriction). **UNCLEAR - stats only, do not redistribute** the raw utterances outside this local calibration folder.
- **citation:**

```
@inproceedings{aguilar-etal-2020-lince,
  title = "{L}in{CE}: A Centralized Benchmark for Linguistic Code-switching Evaluation",
  author = "Aguilar, Gustavo and Kar, Sudipta and Solorio, Thamar",
  booktitle = "Proceedings of The 12th Language Resources and Evaluation Conference",
  year = 2020,
  pages = "1803--1813"
}

@inproceedings{mave-etal-2018-language,
  title = "Language Identification and Analysis of Code-Switched Social Media Text",
  author = "Mave, Deepthi and Maharjan, Suraj and Solorio, Thamar",
  booktitle = "Proceedings of the Third Workshop on Computational Approaches to Linguistic Code-Switching",
  year = 2018,
  pages = "51--61"
}

@inproceedings{singh-etal-2018-twitter,
  title = "A Twitter Corpus for {H}indi-{E}nglish Code Mixed {POS} Tagging",
  author = "Singh, Kushagra and Sen, Indira and Kumaraguru, Ponnurangam",
  booktitle = "Proceedings of the Sixth International Workshop on Natural Language Processing for Social Media",
  year = 2018,
  pages = "12--17"
}
```

Download status: {'ok': True}
