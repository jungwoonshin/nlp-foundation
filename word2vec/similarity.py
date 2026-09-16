"""Word similarity: Spearman correlation with human ratings (FastText eval.py)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from scipy import stats
from torch.nn import functional as F


@dataclass(frozen=True)
class SimilarityPair:
    a: str
    b: str
    score: float


@dataclass(frozen=True)
class SimilarityReport:
    rho: float
    scored: int
    skipped: int

    @property
    def rho_x100(self) -> float:
        return 100.0 * self.rho


def load_rw(path: str | Path) -> list[SimilarityPair]:
    """Parse Stanford Rare Words `rw.txt` (`word1 word2 mean_score ...`)."""
    pairs: list[SimilarityPair] = []
    with Path(path).open(encoding="utf-8") as handle:
        for raw in handle:
            parts = raw.split()
            if len(parts) < 3:
                raise ValueError(f"Expected word1 word2 score, got {parts!r}")
            pairs.append(
                SimilarityPair(a=parts[0].lower(), b=parts[1].lower(), score=float(parts[2]))
            )
    return pairs


def spearman_from_vectors(
    left: torch.Tensor,
    right: torch.Tensor,
    scores: torch.Tensor,
) -> SimilarityReport:
    """Cosine vs human scores; drop pairs whose vectors are exactly zero (FastText eval.py)."""
    if left.shape != right.shape:
        raise ValueError("left and right must have the same shape")
    norms = left.norm(dim=1) * right.norm(dim=1)
    usable = norms > 0
    skipped = int((~usable).sum().item())
    if int(usable.sum().item()) < 2:
        return SimilarityReport(rho=0.0, scored=int(usable.sum().item()), skipped=skipped)
    cosine = F.cosine_similarity(left[usable], right[usable], dim=1)
    rho, _ = stats.spearmanr(
        cosine.detach().float().cpu().numpy(),
        scores[usable].detach().float().cpu().numpy(),
    )
    if rho != rho:  # NaN
        rho = 0.0
    return SimilarityReport(rho=float(rho), scored=int(usable.sum().item()), skipped=skipped)
