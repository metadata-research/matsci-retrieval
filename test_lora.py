"""Checks of the training data split, the trial queue and, where the libraries exist, a real trial.

The first class needs no machine-learning library. The second class trains
MiniLM for a few steps on CPU and is skipped unless torch, sentence-transformers,
peft and the pinned MiniLM snapshot are present, as they are after prepare.py.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import unittest.mock

import lora

CODE = Path(__file__).resolve().parent
SPECS = json.loads((CODE / "models.json").read_text())
SUBSTANCES = ["steel", "copper", "glass", "nickel", "titanium", "polymer", "ceramic", "silicon", "zinc", "cobalt"]
PROPERTIES = ["hardness", "toughness", "density", "porosity", "conductivity", "ductility", "viscosity", "stiffness"]


def row(number, source="pmdco", **changes):
    """A description with a label and a definition of more than eight words."""
    substance, measured = SUBSTANCES[number % 10], PROPERTIES[number // 10 % 8]
    label = f"{substance} {measured} {number}"
    value = {"id": f"https://example.org/entries/{source}/{number:05d}", "iri": f"https://example.org/{source}#{number}",
             "source": source, "version": "1", "license": "CC-BY-4.0", "label": label, "definition_language": "en",
             "definition": f"The {measured} of a {substance} sample, recorded as value {number} under the standard test conditions."}
    value.update(changes)
    value["text"] = value["label"] + "\n" + value["definition"]
    return value


def write_corpus(directory, rows):
    directory.mkdir(parents=True)
    body = "".join(json.dumps(item) + "\n" for item in rows)
    (directory / "corpus.jsonl").write_text(body)
    (directory / "manifest.json").write_text(json.dumps(
        {"sha256": hashlib.sha256(body.encode()).hexdigest(), "records": len(rows)}))


def tiny_trials(path, **changes):
    document = json.loads((CODE / "trials.json").read_text())
    document["trials"] = [{"id": "minilm-test", "parent": "minilm", "rank": 4, "learning_rate": 0.001, "data": "all",
                           "seed": 1, "pairs_budget": 64, "batch_size": 16, "distractors": 50, **changes}]
    path.write_text(json.dumps(document))
    return path


class PairAndQueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def members(self, rows):
        return sorted(sorted(rows[index]["id"][-5:] for index in component["members"])
                      for component in lora.build_pairs(rows))

    def test_candidates_need_an_english_or_untagged_definition_of_eight_words(self):
        rows = [row(1), row(2, definition_language=""), row(3, definition_language="en-US"),
                row(4, definition_language="de"), row(5, definition="Too short to train on."), row(6, definition="")]
        self.assertEqual(self.members(rows), [["00001"], ["00002"], ["00003"]])

    def test_records_that_share_an_iri_a_definition_or_a_label_stay_together(self):
        rows = [row(1), row(2, source="emmo", iri=row(1)["iri"]),
                row(3), row(4, definition="  " + row(3)["definition"].upper().replace(",", " ;")),
                row(5), row(6, label=row(5)["label"].title().replace(" ", "")),
                row(7), row(8, source="chebi"), row(9, source="chebi", label=row(1)["label"]),
                row(10, label="(+)"), row(11, label="(-)")]
        self.assertEqual(self.members(rows), [["00001", "00002", "00009"], ["00003", "00004"], ["00005", "00006"],
                                              ["00007"], ["00008"], ["00010"], ["00011"]])
        groups = {tuple(sorted(rows[index]["source"] for index in component["members"])): component["group"]
                  for component in lora.build_pairs(rows)}
        self.assertEqual(groups[("chebi",)], "chebi")
        self.assertEqual(groups[("chebi", "emmo", "pmdco")], "materials")

    def test_split_ignores_row_order_and_keeps_shared_text_on_one_side(self):
        rows = [row(number, source="chebi" if number % 3 else "pmdco") for number in range(600)]
        rows += [row(1000 + number, source="emmo", label=rows[number]["label"]) for number in range(40)]
        first = {rows[index]["id"]: component["split"] for component in lora.build_pairs(rows)
                 for index in component["members"]}
        shuffled = rows[:]
        random.Random(7).shuffle(shuffled)
        second = {shuffled[index]["id"]: component["split"] for component in lora.build_pairs(shuffled)
                  for index in component["members"]}
        self.assertEqual(first, second)
        self.assertEqual(set(first.values()), {"train", "validation", "test"})
        for field in ("iri", "label", "definition"):
            seen = {}
            for item in rows:
                seen.setdefault(lora.normalized(item[field]), set()).add(first[item["id"]])
            self.assertTrue(all(len(splits) == 1 for splits in seen.values()), field)

    def test_manifest_identity_follows_the_data(self):
        rows = [row(number) for number in range(50)]
        whole = lora.pairs_manifest(rows, lora.build_pairs(rows), "corpus")
        fewer = lora.pairs_manifest(rows[1:], lora.build_pairs(rows[1:]), "corpus")
        self.assertNotEqual(whole["sha256"], fewer["sha256"])
        self.assertEqual(sum(split["components"] for split in whole["splits"].values()), 50)
        self.assertEqual(whole, lora.pairs_manifest(rows, lora.build_pairs(rows), "corpus"))

    def test_balanced_epochs_hold_every_materials_component_and_rotate_chebi(self):
        train = [{"group": "materials"}] * 5 + [{"group": "chebi"}] * 50
        trial = {"data": "balanced", "seed": 3, "chebi_ratio": 4}
        epochs = [lora.epoch_components(train, trial, epoch) for epoch in range(3)]
        for chosen in epochs:
            self.assertEqual(len(chosen), 25)
            self.assertEqual(len(set(chosen)), 25)
            self.assertTrue(set(range(5)) <= set(chosen))
        self.assertEqual(set().union(*epochs), set(range(55)))
        self.assertEqual(epochs[0], lora.epoch_components(train, trial, 0))
        self.assertNotEqual(epochs[0], lora.epoch_components(train, {**trial, "seed": 4}, 0))
        self.assertEqual(sorted(lora.epoch_components(train, {**trial, "data": "all"}, 0)), list(range(55)))
        self.assertEqual(sorted(lora.epoch_components(train[5:], trial, 0)), list(range(50)))

    def test_distractors_hold_every_materials_definition_before_chebi(self):
        rows = [row(number, source="chebi") for number in range(200)] + [row(500 + number) for number in range(40)]
        session = lora.Session(rows, "corpus", SPECS, self.root, "cpu", None, None)
        materials = sum(component["group"] == "materials" for component in session.train)
        chosen = session.distractors(materials + 10)
        self.assertGreater(materials, 20)
        self.assertEqual([component["group"] for component in chosen], ["materials"] * materials + ["chebi"] * 10)
        self.assertEqual(len(session.distractors(5)), 5)

    def test_trial_queue_is_valid_and_shards_partition_it(self):
        queue = lora.load_trials(CODE / "trials.json", SPECS)
        self.assertEqual(len(queue), 216)
        self.assertEqual([trial["parent"] for trial in queue[:3]], ["bge-base", "minilm", "bge-large"])
        self.assertEqual(queue[0]["batch_size"], 256)
        self.assertEqual(queue[0]["alpha"], 2 * queue[0]["rank"])
        shards = [lora.load_trials(CODE / "trials.json", SPECS, f"{index}/4") for index in range(4)]
        self.assertEqual(sorted(trial["id"] for shard in shards for trial in shard), sorted(trial["id"] for trial in queue))
        self.assertEqual(len(lora.load_trials(CODE / "trials.json", SPECS, "1/4", limit=6)), 6)
        self.assertEqual(lora.load_trials(CODE / "trials.json", SPECS, limit=0), [])

    def test_bad_trial_files_and_shards_are_refused(self):
        for changes in ({"parent": "unpinned"}, {"id": "../escape"}, {"data": "some"}, {"rank": 0}):
            with self.assertRaises(ValueError, msg=changes):
                lora.load_trials(tiny_trials(self.root / "trials.json", **changes), SPECS)
        document = json.loads(tiny_trials(self.root / "trials.json").read_text())
        document["trials"] *= 2
        (self.root / "twice.json").write_text(json.dumps(document))
        with self.assertRaises(ValueError):
            lora.load_trials(self.root / "twice.json", SPECS)
        for shard in ("4/4", "1", "a/b"):
            with self.assertRaises(ValueError, msg=shard):
                lora.load_trials(CODE / "trials.json", SPECS, shard)

    def test_ranks_of_a_job_divide_the_shard_of_the_job(self):
        # A job with two GPUs and shard 0/2 beside a single-GPU job with shard 1/2.
        shares = [lora.rank_shard("0/2", 0, 2), lora.rank_shard("0/2", 1, 2), lora.rank_shard("1/2", 0, 1)]
        self.assertEqual(shares, ["0/4", "2/4", "1/2"])
        taken = [trial["id"] for share in shares for trial in lora.load_trials(CODE / "trials.json", SPECS, share)]
        self.assertEqual(sorted(taken), sorted(trial["id"] for trial in lora.load_trials(CODE / "trials.json", SPECS)))
        self.assertEqual(lora.rank_shard("3/4", 0, 1), "3/4")

    def test_newest_checkpoint_needs_a_matching_receipt(self):
        def checkpoint(step, content, receipt=True, spec="spec", sha256=None):
            path = self.root / f"checkpoint-{step:07d}.pt"
            path.write_bytes(content)
            if receipt:
                path.with_suffix(".json").write_text(json.dumps(
                    {"step": step, "spec_sha256": spec, "sha256": sha256 or hashlib.sha256(content).hexdigest()}))
        self.assertIsNone(lora.latest_checkpoint(self.root, "spec"))
        checkpoint(2, b"complete")
        checkpoint(3, b"changed after the receipt", sha256="0" * 64)
        checkpoint(4, b"killed before the receipt", receipt=False)
        checkpoint(5, b"another trial", spec="other")
        (self.root / "checkpoint-0000006.json").write_text("{")
        self.assertEqual(lora.latest_checkpoint(self.root, "spec"), (2, self.root / "checkpoint-0000002.pt"))

    def test_selection_uses_the_mean_of_the_group_scores(self):
        def result(name, materials, chebi):
            directory = self.root / "trials" / name
            directory.mkdir(parents=True)
            scores = {"materials": {"mrr@10": materials}, "chebi": {"mrr@10": chebi}}
            (directory / "result.json").write_text(json.dumps({
                "status": "complete", "trial": {"id": name}, "training": {"pairs_per_second": 10, "seconds": 5},
                "validation": {"baseline": scores, "final": scores}}))
        self.assertIsNone(lora.best_trial(self.root))
        result("chemistry-specialist", 0.50, 0.99)
        result("balanced", 0.80, 0.75)
        (self.root / "trials" / "unfinished").mkdir()
        self.assertEqual(lora.best_trial(self.root)["trial"]["id"], "balanced")
        self.assertEqual([entry["trial"] for entry in lora.summary([self.root])], ["balanced", "chemistry-specialist"])


def minilm_ready():
    try:
        import peft  # noqa: F401
        from sentence_transformers import SentenceTransformer
        SentenceTransformer(SPECS[0]["id"], revision=SPECS[0]["revision"], device="cpu", local_files_only=True)
    except Exception:  # A missing library or a missing model snapshot.
        return False
    return True


@unittest.skipUnless(minilm_ready(), "torch, sentence-transformers, peft or the MiniLM snapshot is missing")
class RealTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.rows = [row(number, source="chebi" if number % 4 == 0 else "pmdco") for number in range(300)]
        self.events = []
        self.trials = lora.load_trials(tiny_trials(self.root / "trials.json"), SPECS)

    def tearDown(self):
        self.temporary.cleanup()

    def run_queue(self, available=lambda: True, out="out", checkpoint_seconds=3600):
        return lora.run_trials(self.rows, "corpus", SPECS, self.trials, self.root / out, "cpu",
                               lambda **event: self.events.append(event), available, time.time() + 3600,
                               checkpoint_seconds=checkpoint_seconds, log_seconds=0)

    def steps(self):
        return [event["step"] for event in self.events if event.get("stage") == "train" and "loss" in event]

    def statuses(self, status):
        return [event["step"] for event in self.events if event.get("status") == status]

    def result(self, out="out"):
        return json.loads((self.root / out / "trials" / "minilm-test" / "result.json").read_text())

    def test_stops_and_restarts_give_the_adapter_of_an_uninterrupted_trial(self):
        from sentence_transformers import SentenceTransformer
        directory = self.root / "out" / "trials" / "minilm-test"
        # First attempt: a stop between two steps.
        self.assertEqual(self.run_queue(lambda: len(self.steps()) < 2), [])
        self.assertEqual((self.steps(), self.statuses("checkpoint")), ([1, 2], [2]))
        self.assertFalse((directory / "result.json").exists())
        # Second attempt: a stop before a new step ends. The checkpoint of step 2 is
        # on disk and must not be written again.
        seen = len(self.events)
        self.assertEqual(self.run_queue(lambda: not any(event["stage"] == "eval" for event in self.events[seen:])), [])
        self.assertEqual((self.steps(), self.statuses("resumed"), self.statuses("checkpoint")), ([1, 2], [2], [2]))
        # Third attempt: a stop during the evaluation that follows the last step.
        self.assertEqual(self.run_queue(lambda: 4 not in self.steps()), [])
        self.assertEqual((self.steps(), self.statuses("resumed"), self.statuses("checkpoint")),
                         ([1, 2, 3, 4], [2, 2], [2, 4]))
        self.assertFalse((directory / "result.json").exists())
        # Fourth attempt: nothing is left to train.
        self.assertEqual(self.run_queue(), ["minilm-test"])
        self.assertEqual((self.steps(), self.statuses("resumed")), ([1, 2, 3, 4], [2, 2, 4]))
        interrupted = self.result()
        self.assertEqual((interrupted["training"]["steps"], interrupted["training"]["attempts"],
                          interrupted["training"]["pairs"]), (4, 4, 64))
        self.assertEqual(sorted(path.name for path in directory.iterdir()), ["adapter", "model.json", "result.json"])
        self.assertLess(interrupted["lora"]["trainable_parameters"], interrupted["lora"]["total_parameters"] // 100)

        # The same trial without a stop, with a checkpoint after every step.
        self.events.clear()
        self.assertEqual(self.run_queue(out="whole", checkpoint_seconds=0), ["minilm-test"])
        self.assertEqual(self.statuses("checkpoint"), [1, 2, 3])
        whole = self.result("whole")
        self.assertEqual(whole["adapter_sha256"], interrupted["adapter_sha256"])
        self.assertEqual(whole["validation"], interrupted["validation"])

        # The parent was measured with the trained adapter switched off, and the
        # exported adapter gives the scores of the trained model.
        session = lora.Session(self.rows, "corpus", SPECS, self.root / "out", "cpu", lambda **_event: None, lambda: True)

        def model():
            return SentenceTransformer(SPECS[0]["id"], revision=SPECS[0]["revision"], device="cpu", local_files_only=True)

        def score(metrics):
            return metrics["all"]["positive_score"]

        baseline, final = interrupted["validation"]["baseline"], interrupted["validation"]["final"]
        self.assertGreater(abs(score(final) - score(baseline)), 1e-4)
        self.assertAlmostEqual(score(session.evaluate(model(), SPECS[0], self.trials[0])), score(baseline), places=6)
        adapted = lora.attach_adapter(model(), directory / "adapter", interrupted["adapter_sha256"])
        self.assertAlmostEqual(score(session.evaluate(adapted, SPECS[0], self.trials[0])), score(final), places=5)
        self.assertEqual(len(adapted.state_dict()), len(model().state_dict()))
        with self.assertRaises(ValueError):
            lora.attach_adapter(model(), directory / "adapter", "0" * 64)

        self.events.clear()
        self.assertEqual(self.run_queue(), [])
        self.assertEqual([event["status"] for event in self.events[-2:]], ["already complete", "queue ended"])

        write_corpus(self.root / "corpus", self.rows)
        report = lora.test_split(self.root / "corpus", self.root / "out", "minilm-test", "cpu")
        self.assertEqual(report, json.loads((directory / "test.json").read_text()))
        self.assertGreater(abs(score(report["adapted"]) - score(report["parent"])), 1e-4)

    def test_metrics_rank_the_positive_and_count_ties_against_the_query(self):
        import torch
        documents = torch.eye(4)
        queries = torch.tensor([[1., 0., 0., 0.], [0.6, 0.8, 0., 0.], [0.8, 0., 0.6, 0.]])
        result = lora.metrics(queries, documents, ["materials", "chebi", "chebi"])
        self.assertEqual(result["materials"], {"queries": 1, "recall@1": 1.0, "recall@10": 1.0, "mrr@10": 1.0,
                                               "positive_score": 1.0})
        self.assertEqual((result["chebi"]["recall@1"], result["chebi"]["mrr@10"]), (0.5, 0.75))
        self.assertAlmostEqual(result["all"]["positive_score"], 0.8, places=6)
        collapsed = lora.metrics(torch.ones(3, 4) / 2, torch.ones(4, 4) / 2, ["chebi"] * 3)
        self.assertEqual((collapsed["all"]["recall@1"], collapsed["all"]["mrr@10"]), (0.0, 0.25))
        self.assertNotIn("materials", collapsed)

    def test_out_of_memory_halves_the_batch_and_starts_the_trial_again(self):
        import torch
        loss = torch.nn.functional.cross_entropy

        def limited(scores, target):
            if scores.shape[0] > 8:
                raise torch.cuda.OutOfMemoryError("stand-in for a full GPU")
            return loss(scores, target)

        with unittest.mock.patch.object(torch.nn.functional, "cross_entropy", limited):
            self.assertEqual(self.run_queue(), ["minilm-test"])
        result = self.result()
        self.assertEqual((result["training"]["batch_size"], result["training"]["steps"], result["trial"]["batch_size"]),
                         (8, 8, 16))
        self.assertEqual([event["batch_size"] for event in self.events if event.get("status") == "oom"], [16])
        self.assertFalse((self.root / "out" / "trials" / "minilm-test" / "batch-size.json").exists())

    def test_embeddings_that_are_not_finite_fail_the_trial(self):
        import torch
        whole = torch.nn.functional.normalize

        def broken(tensor, **options):
            return whole(tensor, **options) * float("nan") if not tensor.requires_grad else whole(tensor, **options)

        with unittest.mock.patch.object(torch.nn.functional, "normalize", broken):
            self.assertEqual(self.run_queue(), [])
        self.assertEqual([event["status"] for event in self.events if event.get("status") in ("failed", "complete")],
                         ["failed"])
        self.assertIn("Nonfinite", next(event["error"] for event in self.events if event.get("status") == "failed"))
        self.assertIsNone(lora.best_trial(self.root / "out"))

    def test_job_trains_builds_the_adapted_index_and_search_reads_it(self):
        write_corpus(self.root / "corpus", self.rows)
        (self.root / "models.json").write_text(json.dumps(SPECS[:1]))
        out = self.root / "out"
        stop = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 900))
        worker = subprocess.Popen(
            [sys.executable, str(CODE / "workload.py"), "--allow-cpu", "--corpus", str(self.root / "corpus"),
             "--out", str(out), "--stop-at", stop, "--case-seconds", "1", "--models", str(self.root / "models.json"),
             "--trials", str(tiny_trials(self.root / "trials.json")), "--reserve-seconds", "0"],
            env=dict(os.environ, RANK="0", WORLD_SIZE="1", LOCAL_RANK="0", LOCAL_WORLD_SIZE="1",
                     CUDA_VISIBLE_DEVICES=""),  # The test is about the stages. It stays on CPU.
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)

        def events():
            return [json.loads(line) for path in sorted(out.glob("measurements-rank-0-*.jsonl"))
                    for line in path.read_text().splitlines()]

        try:
            limit = time.time() + 600
            while not any(event["stage"] == "sustained" for event in events()):
                self.assertIsNone(worker.poll(), worker.stderr.read() if worker.poll() is not None else "")
                self.assertLess(time.time(), limit, "The worker did not reach the last stage")
                time.sleep(0.5)
            worker.send_signal(signal.SIGTERM)
            self.assertEqual(worker.wait(timeout=120), 0, worker.stderr.read())
        finally:
            if worker.poll() is None:
                worker.kill()
                worker.wait()
            worker.stderr.close()
        stages = [event["stage"] for event in events()]
        order = [stages.index(name) for name in ("build", "train", "adapted", "sweep", "sustained", "finished")]
        self.assertEqual(order, sorted(order))
        self.assertTrue(all(event["precision"] == "float32" for event in events() if event["stage"] == "sweep"))
        index = out / "adapted" / "minilm-test"
        contract = json.loads((index / "manifest.json").read_text())
        self.assertEqual(contract["models"][0]["adapter"]["sha256"],
                         json.loads((out / "trials" / "minilm-test" / "result.json").read_text())["adapter_sha256"])
        self.assertEqual(len(list((index / "minilm-test").glob("*.npy"))), 1)

        def search(directory, name):
            done = subprocess.run(
                [sys.executable, str(CODE / "search.py"), "--corpus", str(self.root / "corpus"), "--index", str(directory),
                 "--model", name, "--query", self.rows[17]["definition"], "--top-k", "3"],
                capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
            return json.loads(done.stdout)["queries"][0]["results"]

        parent, adapted = search(out, "minilm"), search(index, "minilm-test")
        self.assertEqual(parent[0]["id"], self.rows[17]["id"])
        self.assertEqual(adapted[0]["id"], self.rows[17]["id"])
        self.assertNotEqual([hit["score"] for hit in parent], [hit["score"] for hit in adapted])


if __name__ == "__main__":
    unittest.main()
