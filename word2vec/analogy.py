"""Word analogical reasoning (Mikolov et al. questions-words.txt)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

SEMANTIC_SECTIONS = frozenset(
    {
        "capital-common-countries",
        "capital-world",
        "currency",
        "city-in-state",
        "family",
    }
)


@dataclass(frozen=True)
class AnalogyQuestion:
    section: str
    a: str
    b: str
    c: str
    d: str


@dataclass(frozen=True)
class SectionScore:
    name: str
    correct: int
    answered: int
    skipped: int

    @property
    def accuracy(self) -> float:
        if self.answered == 0:
            return 0.0
        return self.correct / self.answered


@dataclass(frozen=True)
class AnalogyReport:
    sections: tuple[SectionScore, ...]

    def _sum(self, names: frozenset[str] | None) -> SectionScore:
        matched = [
            section
            for section in self.sections
            if names is None or section.name in names
        ]
        return SectionScore(
            name="total" if names is None else "group",
            correct=sum(section.correct for section in matched),
            answered=sum(section.answered for section in matched),
            skipped=sum(section.skipped for section in matched),
        )

    @property
    def semantic(self) -> SectionScore:
        return self._sum(SEMANTIC_SECTIONS)

    @property
    def syntactic(self) -> SectionScore:
        syntactic = frozenset(
            section.name
            for section in self.sections
            if section.name not in SEMANTIC_SECTIONS
        )
        return self._sum(syntactic)

    @property
    def total(self) -> SectionScore:
        return self._sum(None)


def load_questions(path: str | Path) -> list[AnalogyQuestion]:
    """Parse `questions-words.txt` (`: section` headers, four words per line)."""
    questions: list[AnalogyQuestion] = []
    section = ""
    with Path(path).open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(":"):
                section = line[1:].strip()
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"Expected 4 words, got {parts!r} in section {section!r}")
            if not section:
                raise ValueError(f"Question {parts!r} appeared before a section header")
            questions.append(
                AnalogyQuestion(
                    section=section,
                    a=parts[0],
                    b=parts[1],
                    c=parts[2],
                    d=parts[3],
                )
            )
    return questions


def evaluate_analogies(
    embeddings: torch.Tensor,
    word_to_id: dict[str, int],
    questions: list[AnalogyQuestion],
    *,
    restrict_vocab: int | None = None,
    batch_size: int = 512,
) -> AnalogyReport:
    """3CosAdd exact-match accuracy, skipping OOV and (optionally) rare words.

    For `a : b :: c : d`, the predicted word is the cosine nearest neighbor of
    `vec(b) - vec(a) + vec(c)` among the search vocab, excluding `a`, `b`, and
    `c`. Matches `compute-accuracy.c` and Mikolov et al. Table 1.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D (V, D), got {tuple(embeddings.shape)}")
    vocab_size = embeddings.shape[0]
    search_size = vocab_size if restrict_vocab is None else min(restrict_vocab, vocab_size)
    vectors = F.normalize(embeddings[:search_size].detach().float(), dim=1)

    usable: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
    skipped: dict[str, int] = defaultdict(int)
    seen_sections: list[str] = []
    seen: set[str] = set()
    for question in questions:
        if question.section not in seen:
            seen.add(question.section)
            seen_sections.append(question.section)
        ids = []
        in_search = True
        for word in (question.a, question.b, question.c, question.d):
            index = word_to_id.get(word)
            if index is None:
                index = word_to_id.get(word.lower())
            if index is None or index >= search_size:
                in_search = False
                break
            ids.append(index)
        if not in_search:
            skipped[question.section] += 1
            continue
        usable[question.section].append((ids[0], ids[1], ids[2], ids[3]))

    scores: list[SectionScore] = []
    device = vectors.device
    for section in seen_sections:
        rows = usable.get(section, [])
        correct = 0
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            a = torch.tensor([row[0] for row in batch], dtype=torch.long, device=device)
            b = torch.tensor([row[1] for row in batch], dtype=torch.long, device=device)
            c = torch.tensor([row[2] for row in batch], dtype=torch.long, device=device)
            d = torch.tensor([row[3] for row in batch], dtype=torch.long, device=device)
            query = F.normalize(vectors[b] - vectors[a] + vectors[c], dim=1)
            logits = query @ vectors.T
            logits[torch.arange(a.shape[0], device=device), a] = float("-inf")
            logits[torch.arange(b.shape[0], device=device), b] = float("-inf")
            logits[torch.arange(c.shape[0], device=device), c] = float("-inf")
            correct += int((logits.argmax(dim=1) == d).sum().item())
        scores.append(
            SectionScore(
                name=section,
                correct=correct,
                answered=len(rows),
                skipped=skipped[section],
            )
        )
    return AnalogyReport(sections=tuple(scores))


def format_report(report: AnalogyReport, *, title: str = "") -> str:
    lines = []
    if title:
        lines.append(title)
    for section in report.sections:
        lines.append(
            f"  {section.name:28s}  "
            f"{100.0 * section.accuracy:6.2f}%  "
            f"({section.correct}/{section.answered}, skip {section.skipped})"
        )
    for label, group in (
        ("semantic", report.semantic),
        ("syntactic", report.syntactic),
        ("total", report.total),
    ):
        lines.append(
            f"  {label:28s}  "
            f"{100.0 * group.accuracy:6.2f}%  "
            f"({group.correct}/{group.answered}, skip {group.skipped})"
        )
    return "\n".join(lines)
