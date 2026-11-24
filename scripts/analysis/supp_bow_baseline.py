#!/usr/bin/env python3
"""Bag-of-words (TF-IDF + logistic regression) baseline on the ScamBranch test split.

Trains on (a) the full training split and (b) the exact 408-call selections used by
each LoRA arm (seeds 0-2). C is chosen on the dev split by macro-F1.
Writes results/supplement/baseline/{baseline.json,baseline.tex}.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from scambranch.eval.data import load_usable_conversations  # noqa: E402
from scambranch.ft.data import FT_ARMS, select_arm  # noqa: E402

RUN = ROOT / "runs" / "full_v1"
OUT = ROOT / "results" / "supplement" / "baseline"
CS = [0.25, 1.0, 4.0, 16.0]
VIEWS = ("opening", "opening_reply", "full")


def texts(conv_id: str) -> dict[str, str]:
    d = json.loads((RUN / "conversations" / f"{conv_id}.json").read_text(encoding="utf-8"))
    k = d["private_annotations"]["branch_after_turn"]
    t = [x["text"] for x in d["turns"]]
    return {"opening": " ".join(t[:k]), "opening_reply": " ".join(t[: k + 1]), "full": " ".join(t)}


def label(conv) -> int:
    return 0 if conv.label == "legitimate" else 1


def evaluate(train, dev, test, view, cache):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    x_tr = vec.fit_transform([cache[c.conversation_id][view] for c in train])
    y_tr = [label(c) for c in train]
    x_dv = vec.transform([cache[c.conversation_id][view] for c in dev])
    y_dv = [label(c) for c in dev]
    best = None
    for c_val in CS:
        m = LogisticRegression(C=c_val, max_iter=5000).fit(x_tr, y_tr)
        s = f1_score(y_dv, m.predict(x_dv), average="macro")
        if best is None or s > best[0]:
            best = (s, m)
    model = best[1]
    pred = model.predict(vec.transform([cache[c.conversation_id][view] for c in test]))
    y_te = [label(c) for c in test]
    out = {"macro_f1": f1_score(y_te, pred, average="macro")}
    legit = [p for c, p in zip(test, pred) if label(c) == 0]
    out["false_alarm"] = float(np.mean(legit))
    by_cond_fa, by_cond_rec = defaultdict(list), defaultdict(list)
    groups = defaultdict(list)
    for c, p in zip(test, pred):
        (by_cond_fa if label(c) == 0 else by_cond_rec)[c.recipient_condition].append(p)
        groups[c.history_key].append((int(p), label(c)))
    out["false_alarm_suspicious"] = float(np.mean(by_cond_fa["suspicious"]))
    out["recall_cooperative"] = float(np.mean(by_cond_rec["cooperative"]))
    complete = [g for g in groups.values() if len(g) == 4]
    out["disagreement"] = float(np.mean([len({p for p, _ in g}) > 1 for g in complete]))
    out["all_four_correct"] = float(np.mean([all(p == y for p, y in g) for g in complete]))
    out["n_complete_sets"] = len(complete)
    return out


def main() -> int:
    train = load_usable_conversations(RUN, split="train", complete_sets_path=None, use_complete_sets_if_present=False)
    dev = load_usable_conversations(RUN, split="dev", complete_sets_path=None, use_complete_sets_if_present=False)
    test = load_usable_conversations(RUN, split="test", complete_sets_path=None, use_complete_sets_if_present=False)
    cache = {c.conversation_id: texts(c.conversation_id) for c in [*train, *dev, *test]}
    results: dict = {"n_train_full": len(train), "n_dev": len(dev), "n_test": len(test), "runs": {}}
    results["runs"]["full_train"] = {v: evaluate(train, dev, test, v, cache) for v in VIEWS}
    for arm in FT_ARMS:
        per_seed = []
        for seed in (0, 1, 2):
            sel = select_arm(train, arm, seed=seed)
            per_seed.append({v: evaluate(sel.conversations, dev, test, v, cache) for v in VIEWS} | {"n": sel.n})
        results["runs"][arm] = per_seed
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "baseline.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    def fmt(x):
        return f"{x:.2f}".lstrip("0") if x < 1 else f"{x:.2f}"

    def row(name, stats):
        cells = []
        for key in ("macro_f1", "false_alarm", "false_alarm_suspicious", "recall_cooperative"):
            vals = [s["full"][key] for s in stats]
            cells.append(fmt(st.mean(vals)) + (f"{{\\scriptsize$\\pm${fmt(st.stdev(vals))}}}" if len(vals) > 1 else ""))
        d1 = [s["opening_reply"]["disagreement"] for s in stats]
        d = [s["full"]["disagreement"] for s in stats]
        cells += [fmt(st.mean(d1)), fmt(st.mean(d))]
        return f"{name} & " + " & ".join(cells) + r" \\"

    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\begin{tabular}{lcccccc}", r"\toprule",
        r"\rowcolor{headblue}",
        r"\textbf{Training data} & \textbf{F1}$\uparrow$ & \textbf{FA}$\downarrow$ & \textbf{FA}$_{\mathrm{S}}\!\downarrow$ & \textbf{R}$_{\mathrm{C}}\!\uparrow$ & \textbf{D}$_1\!\downarrow$ & \textbf{D}$\downarrow$ \\",
        r"\midrule",
        row(f"Full training split ({len(train):,})", [results["runs"]["full_train"]]),
        row("Unbranched selection (408)", results["runs"]["unbranched"]),
        row("Branched selection (408)", results["runs"]["branched"]),
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Bag-of-words baseline (TF-IDF word 1--2-grams, logistic regression, $C$ chosen on the development split). The 408-call rows use exactly the training calls of the corresponding fine-tuning arm, seeds 0--2 (mean$\pm$sd); the consistency arm uses the branched selection. Columns as in Table~1 of the abstract.}",
        r"\label{tab:bow-baseline}", r"\end{table}",
    ]
    (OUT / "baseline.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: (v if k != "runs" else None) for k, v in results.items()}, indent=2))
    for k, v in results["runs"].items():
        seeds = v if isinstance(v, list) else [v]
        f1 = [s["full"]["macro_f1"] for s in seeds]
        d = [s["full"]["disagreement"] for s in seeds]
        d1 = [s["opening_reply"]["disagreement"] for s in seeds]
        fa = [s["full"]["false_alarm"] for s in seeds]
        rc = [s["full"]["recall_cooperative"] for s in seeds]
        print(f"{k:22} F1 {st.mean(f1):.3f} (seeds {[round(x,3) for x in f1]}) FA {st.mean(fa):.3f} Rc {st.mean(rc):.3f} D1 {st.mean(d1):.3f} D {st.mean(d):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
