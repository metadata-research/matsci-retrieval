"""Checks of the record summary on a small output directory written by hand."""
import json
from pathlib import Path
import tempfile
import unittest

import report


def stamp(second):
    return f"2026-10-05T08:{second // 60:02d}:{second % 60:02d}.000000+00:00"


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.out = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, attempt, records, cut_off=""):
        body = "".join(json.dumps({**record, "rank": 0}) + "\n" for record in records) + cut_off
        (self.out / f"measurements-rank-0-{attempt}.jsonl").write_text(body)
        (self.out / f"environment-rank-0-{attempt}.json").write_text(json.dumps({
            "architecture": "x86_64", "python": "3.12.3", "cuda": "13.3", "gpu": "NVIDIA RTX A6000",
            "gpu_bytes": 48 * report.GIB, "packages": {"torch": "2.13.0", "numpy": "2.5.3"}}))

    def test_summary_covers_restarts_stages_trials_and_telemetry(self):
        trial = "bge-large-r16-lr1e-4-balanced-s1"
        encode = {"precision": "float32", "batch_size": 32, "rows": 1024, "seconds": 2.0, "peak_allocated_bytes": 3 * report.GIB}
        self.write(1, [
            {"stage": "start", "stop_at": "2026-10-05T09:26:35+00:00", "time": stamp(0)},
            {"stage": "resume", "index": ".", "chunks": {"bge-large": {"assigned": 3, "present": 0}}, "time": stamp(5)},
            {"stage": "build", "model": "bge-large", "start": 0, **encode, "rows_per_second": 500.0, "time": stamp(10)},
            {"stage": "build", "model": "bge-large", "start": 1024, **encode, "rows_per_second": 600.0, "time": stamp(20)},
            {"stage": "train", "trial": trial, "status": "started", "step": 0, "steps": 469, "batch_size": 128,
             "precision": "bfloat16", "time": stamp(30)},
            {"stage": "train", "trial": trial, "step": 98, "steps": 469, "loss": 0.8, "pairs_per_second": 170.0,
             "peak_allocated_bytes": 20 * report.GIB, "time": stamp(60)},
            {"stage": "signal", "signal": "SIGTERM", "time": stamp(70)},
            {"stage": "train", "trial": trial, "status": "checkpoint", "step": 110, "seconds": 0.08, "time": stamp(71)},
            {"stage": "finished", "status": "interrupted", "time": stamp(72)},
        ], cut_off='{"stage": "train", "trial": "bge-la')
        self.write(2, [
            {"stage": "start", "stop_at": "2026-10-05T09:26:35+00:00", "time": stamp(120)},
            {"stage": "resume", "index": ".", "chunks": {"bge-large": {"assigned": 3, "present": 3}}, "time": stamp(125)},
            {"stage": "train", "trial": trial, "status": "resumed", "step": 110, "steps": 469, "batch_size": 128,
             "precision": "bfloat16", "time": stamp(130)},
            {"stage": "train", "trial": trial, "step": 140, "steps": 469, "loss": 0.7, "pairs_per_second": 190.0,
             "peak_allocated_bytes": 21 * report.GIB, "time": stamp(140)},
            {"stage": "train", "trial": trial, "status": "complete", "steps": 469, "baseline_mrr": 0.5, "final_mrr": 0.6,
             "time": stamp(200)},
            {"stage": "train", "status": "queue ended", "reason": "the trial limit of the job is reached", "finished": 1,
             "time": stamp(201)},
            {"stage": "adapted", "status": "selected", "trial": trial, "score": 0.61, "time": stamp(202)},
            {"stage": "sweep", "model": "minilm", "max_length": 128, "distribution": "natural", "warmup": True,
             **encode, "rows_per_second": 4000.0, "time": stamp(210)},
            {"stage": "sweep", "model": "minilm", "max_length": 128, "distribution": "natural", "warmup": False,
             **encode, "rows_per_second": 9000.0, "time": stamp(212)},
            {"stage": "sustained", "model": "bge-large", "distribution": "natural", **encode, "rows_per_second": 700.0,
             "time": stamp(220)},
            {"stage": "finished", "status": "worker finished", "time": stamp(230)},
        ])
        header = ("timestamp, uuid, name, utilization.gpu [%], utilization.memory [%], memory.used [MiB], "
                  "memory.total [MiB], power.draw [W], temperature.gpu\n")
        (self.out / "gpu-node-1.csv").write_text(header + "".join(
            f"2026/10/05 08:{second // 60:02d}:{second % 60:02d}.000, GPU-1, NVIDIA RTX A6000, {use}, 40, 20480, 49140, {power}, 70\n"
            for second, use, power in [(12, 95, "280.5"), (22, 85, "[N/A]"), (40, 100, "290.0"), (65, 98, "291.0")]
        ) + "2026/10/05 08:01:1")
        text = "\n".join(report.lines(self.out))
        for expected in [
            "environment | x86_64 | python 3.12.3 | torch 2.13.0 | cuda 13.3 | NVIDIA RTX A6000 48.0 GiB | numpy 2.5.3 | attempts 2",
            "resume | 08:02:05 | bge-large 3/3",
            "build | bge-large | chunks 2 | batch 32 | median 550 rows/s | peak 3.0 GiB",
            "signal | 08:01:10 | SIGTERM",
            "finished | 08:03:50 | worker finished",
            f"trial | {trial} | batch 128 | bfloat16 | steps 469 | median 180 pairs/s | peak 21.0 GiB | "
            "checkpoints 1, slowest 0.08 s | resumed at step [110] | 170 s from start to result | mrr 0.500 -> 0.600",
            "queue ended | 08:03:21 | the trial limit of the job is reached | finished 1",
            f"adapted | selected | {trial} | 08:03:22 | 0.61",
            "sweep | cases started 1 of 72 | cases with a measurement after the warmup 1",
            "sweep | minilm float32 | best 9000 rows/s at batch 32, length 128, natural",
            "sustained | calls 1 | median 700 rows/s",
            "gpu | first sample +12 s after the first record",
            "gpu | build bge-large | 0.2 min | utilization mean 90% | at least 90% in 50% of samples | memory max 20.0 GiB "
            "| power mean 280 W | max 70 C",
            "gpu | train bge-large | 0.2 min | utilization mean 99% | at least 90% in 100% of samples",
        ]:
            self.assertIn(expected, text)

    def test_trial_without_result_and_empty_output(self):
        self.assertEqual(list(report.lines(self.out)), [])
        self.write(1, [{"stage": "train", "trial": "minilm-r8-lr5e-5-all-s2", "status": "started", "step": 0, "steps": 118,
                        "batch_size": 512, "precision": "bfloat16", "time": stamp(1)},
                       {"stage": "train", "trial": "minilm-r8-lr5e-5-all-s2", "status": "oom", "batch_size": 512,
                        "time": stamp(2)}])
        text = "\n".join(report.lines(self.out))
        self.assertIn("NO RESULT | other records: ['oom']", text)
        self.assertIn("attention | ", text)


if __name__ == "__main__":
    unittest.main()
