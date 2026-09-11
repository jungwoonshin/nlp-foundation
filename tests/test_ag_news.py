from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from prepare_ag_news import _tokenize, prepare_ag_news
from word2vec.config import ProcessingConfig
from word2vec.corpus import LABEL_PREFIX, WhitespaceCorpus
from word2vec.pipeline import process_corpus


class AgNewsPrepareTests(unittest.TestCase):
    def test_tokenize_drops_punctuation_and_case(self) -> None:
        self.assertEqual(
            _tokenize("Wall St. Bears (Reuters)"),
            ["wall", "st", "bears", "reuters"],
        )

    def test_prepare_writes_one_labeled_line_per_article(self) -> None:
        csv_text = (
            '"3","Wall St. Bears (Reuters)","Short-sellers are seeing green."\n'
            '"2","Sports Win","The team scored twice."\n'
        )
        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            raw = data_dir / "raw"
            raw.mkdir()
            (raw / "ag_news_train.csv").write_text(csv_text, encoding="utf-8")
            out = prepare_ag_news(data_dir)
            lines = out.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(
            lines[0].split(),
            [
                f"{LABEL_PREFIX}3",
                "wall",
                "st",
                "bears",
                "reuters",
                "short",
                "sellers",
                "are",
                "seeing",
                "green",
            ],
        )
        self.assertEqual(lines[1].split()[:2], [f"{LABEL_PREFIX}2", "sports"])
        self.assertNotEqual(lines[0], lines[1])


class CorpusReadLimitTests(unittest.TestCase):
    def test_reader_stops_after_max_sentences(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "docs.txt"
            path.write_text("".join(f"word{i} extra\n" for i in range(150)), encoding="utf-8")
            sentences = WhitespaceCorpus(path, max_sentences=100).sentences()
        self.assertEqual(len(sentences), 100)
        self.assertEqual(sentences[0], ["word0", "extra"])
        self.assertEqual(sentences[99], ["word99", "extra"])
        self.assertTrue(all(len(s) >= 1 for s in sentences))

    def test_process_uses_only_the_first_100_documents(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "docs.txt"
            lines = [f"token{i} token{i}\n" for i in range(150)]
            lines[0] = "alpha alpha alpha\n"
            lines[99] = "omega omega omega\n"
            lines[100] = "unseenuniquetoken unseenuniquetoken unseenuniquetoken\n"
            path.write_text("".join(lines), encoding="utf-8")
            processed = process_corpus(
                path,
                ProcessingConfig(
                    min_count=1,
                    subsample_threshold=1.0,
                    max_sentences=100,
                    architecture="skipgram",
                ),
            )
        self.assertEqual(len(processed.sentences), 100)
        self.assertEqual(processed.raw_token_count, 202)
        self.assertIn("alpha", processed.vocab.word_to_id)
        self.assertIn("omega", processed.vocab.word_to_id)
        self.assertNotIn("unseenuniquetoken", processed.vocab.word_to_id)
        self.assertEqual(processed.vocab.decode(processed.sentences[0][0]), "alpha")
        self.assertEqual(processed.vocab.decode(processed.sentences[99][0]), "omega")
        self.assertEqual(processed.labels, [None] * 100)

    def test_process_keeps_class_labels_and_one_document_per_line(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "docs.txt"
            path.write_text(
                f"{LABEL_PREFIX}3 wall st bears\n{LABEL_PREFIX}2 sports win\n",
                encoding="utf-8",
            )
            processed = process_corpus(
                path,
                ProcessingConfig(
                    min_count=1,
                    subsample_threshold=1.0,
                    architecture="skipgram",
                ),
            )
        self.assertEqual(len(processed.sentences), 2)
        self.assertEqual(processed.labels, ["3", "2"])
        self.assertEqual(
            [processed.vocab.decode(i) for i in processed.sentences[0]],
            ["wall", "st", "bears"],
        )
        self.assertEqual(
            [processed.vocab.decode(i) for i in processed.sentences[1]],
            ["sports", "win"],
        )
        self.assertNotIn(f"{LABEL_PREFIX}3", processed.vocab.word_to_id)
        self.assertNotIn("3", processed.vocab.word_to_id)
        self.assertEqual(processed.label_to_id, {"2": 0, "3": 1})

    def test_fasttext_examples_are_normalized_frequencies_and_labels(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "docs.txt"
            path.write_text(
                f"{LABEL_PREFIX}3 alpha beta alpha\n{LABEL_PREFIX}2 sports win\n",
                encoding="utf-8",
            )
            processed = process_corpus(
                path,
                ProcessingConfig(
                    min_count=1,
                    subsample_threshold=1.0,
                    architecture="fasttext",
                ),
            )
        examples = processed.rebuild_examples(epoch=1)
        self.assertEqual(len(examples), 2)
        self.assertEqual(processed.labels, ["3", "2"])
        alpha, beta = (processed.vocab.word_to_id[w] for w in ("alpha", "beta"))
        first = examples[0]
        present = first["features"] >= 0
        ids = first["features"][present].tolist()
        freqs = first["weights"][present].tolist()
        self.assertEqual(set(ids), {alpha, beta})
        by_id = dict(zip(ids, freqs))
        self.assertAlmostEqual(by_id[alpha], 2 / 3)
        self.assertAlmostEqual(by_id[beta], 1 / 3)
        self.assertAlmostEqual(sum(freqs), 1.0)
        self.assertEqual(int(first["label"]), processed.label_to_id["3"])
        self.assertEqual(int(examples[1]["label"]), processed.label_to_id["2"])
        batch = next(
            iter(processed.dataloader(batch_size=2, epoch=1, shuffle=False, with_negatives=False))
        )
        self.assertIn("features", batch)
        self.assertIn("weights", batch)
        self.assertIn("label", batch)
        self.assertNotIn("negatives", batch)
        self.assertNotIn("center", batch)

    def test_fasttext_rejects_unlabeled_documents(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "docs.txt"
            path.write_text("alpha beta\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                process_corpus(
                    path,
                    ProcessingConfig(min_count=1, architecture="fasttext"),
                )


if __name__ == "__main__":
    unittest.main()
