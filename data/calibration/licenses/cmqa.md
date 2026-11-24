# CMQA (Chandu et al., CALCS 2018) — license record

Recorded: 2026-09-25
Local raw files: `data/calibration/raw/cmqa/`

## Source

- Paper: Chandu et al., *Code-Mixed Question Answering Challenge: Crowd-sourcing Data and Techniques*, CALCS @ ACL 2018. https://aclanthology.org/W18-3204/
- Official host named in `COPYING`: http://tts.speech.cs.cmu.edu/codemixedqa/
  - Live status (this run): **DEAD / unreachable**
  - The 2018 site was a challenge registration page, not a public JSON dump. Training data was released to registered teams. Wayback snapshots (2017–2019) archive HTML only; no question files.
- Live public copy used here: https://github.com/khyathiraghavi/code_switched_QA (created 2020-05-04)
- GitHub files downloaded: yes

## License

**CMU 2018 custom permissive data license** (file `COPYING` in the author repository).

Not UNCLEAR. Redistribution of the original files is allowed if the copyright notice, conditions, and disclaimer are retained; modifications must be marked; original authors' names must not be deleted. The grant is “use, copy, modify … for any purpose” including commercial. The paper’s prose (“freely available for research purposes”) is narrower than `COPYING`; we follow `COPYING`.

Processed `questions.jsonl` is a derived extract (roman-script questions + provided tags). Treat it as a modification of the CMU data and keep this notice.

### `COPYING` text

```
This data is free for use for any purpose (commercial or otherwise)
subject to the pretty light restrictions detailed below.

############################################################################
###                                                                       ##
###                     Carnegie Mellon University                        ##
###                         Copyright (c) 2018                            ##
###                        All Rights Reserved.                           ##
###                                                                       ##
###  Permission to use, copy, modify this data and its 			  ##
###  documentation for any purpose, is hereby granted without fee,        ##
###  subject to the following conditions:                                 ##
###   1. The data must retain the above copyright notice, this list of    ##
###      conditions and the following disclaimer.                         ##
###   2. Any modifications must be clearly marked as such.                ##
###   3. Original authors' names are not deleted.                         ##
###                                                                       ##
###  THE AUTHORS OF THIS WORK DISCLAIM ALL WARRANTIES WITH REGARD TO      ##
###  THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY   ##
###  AND FITNESS, IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY         ##
###  SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES            ##
###  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN   ##
###  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,          ##
###  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF       ##
###  THIS DATA.                                                           ##
###                                                                       ##
############################################################################
###                                                                       ##
###  See http://tts.speech.cs.cmu.edu/codemixedqa/ for more details       ##
###                                                                       ##
############################################################################
```

## Citation

@inproceedings{chandu-etal-2018-code,
    title = "Code-Mixed Question Answering Challenge: Crowd-sourcing Data and Techniques",
    author = "Chandu, Khyathi  and
      Loginova, Ekaterina  and
      Gupta, Vishal  and
      van Genabith, Josef  and
      Neumann, G{\"u}nter  and
      Chinnakotla, Manoj  and
      Nyberg, Eric  and
      Black, Alan W.",
    booktitle = "Proceedings of the Third Workshop on Computational Approaches to Linguistic Code-Switching",
    month = jul,
    year = "2018",
    address = "Melbourne, Australia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W18-3204/",
    doi = "10.18653/v1/W18-3204",
    pages = "29--38"
}

## Related resources (not used as the main dump)

| Resource | What we found | Redistribution |
|---|---|---|
| Raghavi / Chandu et al. 2015, *Answer ka type kya he?* (WWW companion) | ~1,000 En–Hi question-classification items. No public dump on GitHub, author page, ACM, or the WWW 2015 PDF host. | UNCLEAR → stats only, do not redistribute |
| Chandu et al. 2017, *WebShodh* (CLEF) | Demo at `https://tts.speech.cs.cmu.edu/webshodh/cmqa.php` (host dead). Wayback has the interactive page only; no question file. Microsoft India blog describes the system, no download. | UNCLEAR → stats only, do not redistribute |
| Microsoft / Manoj Chinnakotla | Co-author of the 2015/2017/2018 papers. No separate MSR Open Data “code-mixed QA” release found. | — |
| Microsoft GLUECoS QA_EN_HI | Downloads `https://raw.githubusercontent.com/khyathiraghavi/code_switched_QA/master/code_mixed_qa_train.json` (same GitHub train file) and builds a 313-triple En–Hi subset with DrQA contexts. Not a new human question set. | same as CMU `COPYING` for the questions |
| `smitpandit/hinglish-cs-qa` (Hugging Face) | Unofficial 100-row Hinglish sample. Not used. | UNCLEAR → stats only, do not redistribute |

## What we keep

Challenge JSON (train + three test batches with answers) from the author GitHub repo, plus this license record and a roman-script question extract.
