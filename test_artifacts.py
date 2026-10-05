"""Checks for the failures that could silently misattribute an index."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from artifacts import checkpoint_valid, chunk_ranges, digest, load_corpus, save_vectors
from workload import deadline


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.rows = [{"id": "source-a/shared-entity"}, {"id": "source-b/shared-entity"}]
        self.vectors = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
        self.path = self.root / "chunk.npy"

    def tearDown(self):
        self.temporary.cleanup()

    def test_checkpoint_requires_matching_identity_and_order(self):
        save_vectors(self.path, self.vectors, self.rows)
        self.assertTrue(checkpoint_valid(self.path, self.rows, 2))
        with self.assertRaises(ValueError):
            checkpoint_valid(self.path, list(reversed(self.rows)), 2)

    def test_corrupt_vectors_are_not_resumed(self):
        save_vectors(self.path, self.vectors, self.rows)
        with self.path.open("ab") as stream:
            stream.write(b"corrupt")
        with self.assertRaises(ValueError):
            checkpoint_valid(self.path, self.rows, 2)

    def test_interrupted_commit_is_incomplete(self):
        save_vectors(self.path, self.vectors, self.rows)
        self.path.with_suffix(".json").unlink()
        self.assertFalse(checkpoint_valid(self.path, self.rows, 2))

    def test_nonfinite_or_unnormalized_outputs_are_rejected(self):
        with self.assertRaises(ValueError):
            save_vectors(self.path, np.array([[np.nan, 0.], [0., 1.]]), self.rows)
        with self.assertRaises(ValueError):
            save_vectors(self.path, self.vectors * 2, self.rows)

    def test_corpus_tampering_fails(self):
        corpus = self.root / "corpus.jsonl"
        corpus.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        (self.root / "manifest.json").write_text(json.dumps({"sha256": digest(corpus), "records": 2}))
        self.assertEqual(load_corpus(self.root)[1], self.rows)
        corpus.write_text(corpus.read_text() + "{}\n")
        with self.assertRaises(ValueError):
            load_corpus(self.root)

    def test_ranks_cover_all_records_once_for_uneven_sizes(self):
        parts = chunk_ranges(222925)
        for world in [1, 2, 8]:
            assigned = [p for rank in range(world) for i, p in enumerate(parts) if i % world == rank]
            flattened = [i for start, end in assigned for i in range(start, end)]
            self.assertEqual(sorted(flattened), list(range(222925)))

    def test_deadline_requires_timezone_and_preserves_instant(self):
        with self.assertRaises(ValueError):
            deadline("2026-10-08T18:00:00")
        self.assertEqual(deadline("2026-10-08T18:00:00-04:00"), deadline("2026-10-08T22:00:00Z"))


if __name__ == "__main__":
    unittest.main()
