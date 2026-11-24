"""Forest plot of the key bootstrap contrasts reported in the supplement."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[2] / "paper" / "supplement" / "figures" / "forest.pdf"

GROUPS = [
    ("One reply splits the verdicts", [
        ("Qwen2.5-1.5B", .55, .47, .63),
        ("Gemma-3-1B", .61, .52, .69),
        ("Llama-3.2-1B", .30, .21, .38),
    ]),
    ("Stance effect (full call)", [
        ("Qwen2.5-1.5B  FA susp. − coop.", .16, .06, .31),
        ("Qwen2.5-1.5B  recall coop. − others", -.36, -.52, -.19),
        ("Gemma-3-1B  FA susp. − coop.", .37, .24, .50),
        ("Gemma-3-1B  recall coop. − others", -.35, -.47, -.23),
        ("Qwen2.5-0.5B  recall coop. − others", -.32, -.41, -.22),
        ("Qwen3-1.7B  recall coop. − others", .12, .05, .19),
    ]),
    ("Language gap (macro-F1)", [
        ("Qwen2.5-1.5B  Hindi − Tamil", .17, .08, .29),
        ("Qwen2.5-1.5B  Hindi − Telugu", .14, .06, .23),
        ("Qwen3-1.7B  Hindi − Tamil", .19, .11, .27),
        ("Gemma-3-1B  Hindi − Tamil", .15, .08, .22),
        ("Granite-1B  Hindi − Tamil", .15, .06, .26),
    ]),
    ("Hiding the victim (D)", [
        ("Gemma-3-1B", -.32, -.39, -.25),
        ("Qwen2.5-0.5B", -.14, -.21, -.08),
    ]),
    ("Fine-tuning Qwen2.5-1.5B", [
        ("Unbranched − zero-shot  F1", .14, .09, .18),
        ("Branched − zero-shot  F1", .14, .07, .20),
        ("Branched − unbranched  F1", -.00, -.03, .03),
        ("Branched − unbranched  D full", -.07, -.11, -.03),
        ("Branched − unbranched  D +reply", -.09, -.13, -.04),
    ]),
]


def main() -> None:
    labels, ys, heads = [], [], []
    y = 0
    rows = []
    for title, items in GROUPS:
        heads.append((y, title))
        y += 1
        for name, est, lo, hi in items:
            rows.append((y, est, lo, hi))
            labels.append((y, name))
            y += 1
        y += 0.4

    fig, ax = plt.subplots(figsize=(6.6, 7.4))
    for yy, est, lo, hi in rows:
        crosses = lo <= 0 <= hi
        color = "#9aa0a6" if crosses else "#1f4e8c"
        ax.plot([lo, hi], [yy, yy], color=color, lw=1.6, solid_capstyle="butt")
        ax.plot(est, yy, "o", color=color, ms=4.5)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_yticks([yy for yy, _ in labels] + [yy for yy, _ in heads])
    ax.set_yticklabels([n for _, n in labels] + [t for _, t in heads], fontsize=7.5)
    for tick, (_, _) in zip(ax.get_yticklabels()[len(labels):], heads):
        tick.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlabel("Estimate with 95% family-clustered bootstrap interval", fontsize=8)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color="#e6e6e6", lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
