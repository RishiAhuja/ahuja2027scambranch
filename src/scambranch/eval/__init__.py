"""Pretrained detector evaluation (plan §§12–14, E3)."""

from scambranch.eval.data import (
    assert_no_private_leak,
    build_detector_input,
    load_usable_conversations,
    render_transcript,
)
from scambranch.eval.metrics import (
    all_branches_correct_rate,
    branch_disagreement_rate,
    branch_score_spread,
    evidence_relative_delay,
    family_bootstrap,
    fit_threshold,
    legitimate_fpr,
    macro_f1,
)

__all__ = [
    "assert_no_private_leak",
    "build_detector_input",
    "load_usable_conversations",
    "render_transcript",
    "all_branches_correct_rate",
    "branch_disagreement_rate",
    "branch_score_spread",
    "evidence_relative_delay",
    "family_bootstrap",
    "fit_threshold",
    "legitimate_fpr",
    "macro_f1",
]
