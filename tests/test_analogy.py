from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from word2vec.analogy import (
    AnalogyQuestion,
    evaluate_analogies,
    load_questions,
)


class AnalogyEvalTests(unittest.TestCase):
    def test_king_man_woman_queen(self) -> None:
        man = torch.tensor([1.0, 0.0, 0.0])
        woman = torch.nn.functional.normalize(torch.tensor([1.0, 1.0, 0.0]), dim=0)
        king = torch.nn.functional.normalize(torch.tensor([1.0, 0.0, 1.0]), dim=0)
        queen = torch.nn.functional.normalize(woman - man + king, dim=0)
        apple = torch.tensor([0.0, 0.0, 1.0])
        word_to_id = {"man": 0, "woman": 1, "king": 2, "queen": 3, "apple": 4}
        embeddings = torch.stack([man, woman, king, queen, apple])
        questions = [
            AnalogyQuestion("family", "man", "woman", "king", "queen"),
            AnalogyQuestion("gram8-plural", "man", "woman", "king", "queen"),
        ]
        report = evaluate_analogies(embeddings, word_to_id, questions)
        self.assertEqual(report.total.correct, 2)
        self.assertEqual(report.total.answered, 2)
        self.assertEqual(report.total.skipped, 0)
        self.assertAlmostEqual(report.semantic.accuracy, 1.0)
        self.assertAlmostEqual(report.syntactic.accuracy, 1.0)

    def test_skips_oov_and_restricted_vocab(self) -> None:
        word_to_id = {"a": 0, "b": 1, "c": 2, "d": 3}
        embeddings = torch.eye(4)
        questions = [
            AnalogyQuestion("family", "a", "b", "c", "d"),
            AnalogyQuestion("family", "a", "b", "c", "missing"),
        ]
        report = evaluate_analogies(embeddings, word_to_id, questions, restrict_vocab=3)
        self.assertEqual(report.total.answered, 0)
        self.assertEqual(report.total.skipped, 2)

    def test_load_questions_parses_sections(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "questions-words.txt"
            path.write_text(
                ": family\n"
                "king queen man woman\n"
                ": gram8-plural\n"
                "dollar dollars mouse mice\n",
                encoding="utf-8",
            )
            questions = load_questions(path)
        self.assertEqual(len(questions), 2)
        self.assertEqual(questions[0].section, "family")
        self.assertEqual(questions[1].d, "mice")

    def test_one_score_row_per_section(self) -> None:
        word_to_id = {"a": 0, "b": 1, "c": 2, "d": 3}
        embeddings = torch.eye(4)
        questions = [
            AnalogyQuestion("family", "a", "b", "c", "d"),
            AnalogyQuestion("family", "a", "b", "c", "d"),
            AnalogyQuestion("gram8-plural", "a", "b", "c", "d"),
        ]
        report = evaluate_analogies(embeddings, word_to_id, questions)
        self.assertEqual([section.name for section in report.sections], ["family", "gram8-plural"])

    def test_lookup_is_case_insensitive(self) -> None:
        man = torch.tensor([1.0, 0.0, 0.0])
        woman = torch.nn.functional.normalize(torch.tensor([1.0, 1.0, 0.0]), dim=0)
        king = torch.nn.functional.normalize(torch.tensor([1.0, 0.0, 1.0]), dim=0)
        queen = torch.nn.functional.normalize(woman - man + king, dim=0)
        embeddings = torch.stack([man, woman, king, queen])
        word_to_id = {"man": 0, "woman": 1, "king": 2, "queen": 3}
        questions = [AnalogyQuestion("family", "Man", "Woman", "King", "Queen")]
        report = evaluate_analogies(embeddings, word_to_id, questions)
        self.assertEqual(report.total.answered, 1)
        self.assertEqual(report.total.correct, 1)


if __name__ == "__main__":
    unittest.main()
