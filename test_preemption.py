"""Checks that a stop is recorded, a restart resumes, and a late job ends quietly.

The worker and the submission wrapper run as real subprocesses. Small stand-ins
replace torch and sentence-transformers, so no GPU or model download is needed.
"""
import contextlib
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import submit

CODE = Path(__file__).resolve().parent

FAKE_TORCH = '''
import contextlib, os, time, types
time.sleep(float(os.environ.get("FAKE_IMPORT_SECONDS", "0")))
class OutOfMemoryError(RuntimeError):
    pass
cuda = types.SimpleNamespace(
    is_available=lambda: True, set_device=lambda *_: None, get_device_name=lambda *_: "FAKE GPU",
    get_device_properties=lambda *_: types.SimpleNamespace(total_memory=1),
    reset_peak_memory_stats=lambda *_: None, synchronize=lambda *_: None,
    max_memory_allocated=lambda *_: 0, empty_cache=lambda: None, OutOfMemoryError=OutOfMemoryError)
version = types.SimpleNamespace(cuda="fake")
set_num_threads = manual_seed = lambda _value: None
inference_mode = contextlib.nullcontext
'''

FAKE_SENTENCE_TRANSFORMERS = '''
import hashlib, os, time
import numpy as np
DIMENSIONS = {"sentence-transformers/all-MiniLM-L6-v2": 384, "BAAI/bge-base-en-v1.5": 768,
              "BAAI/bge-large-en-v1.5": 1024}
class SentenceTransformer:
    def __init__(self, model_id, **_options):
        self.dimension, self.max_seq_length = DIMENSIONS[model_id], None
    def float(self): return self
    def half(self): return self
    def eval(self): return self
    def get_sentence_embedding_dimension(self): return self.dimension
    def encode(self, texts, **_options):
        time.sleep(len(texts) * float(os.environ.get("FAKE_ROW_SECONDS", "0")))
        seed = int.from_bytes(hashlib.sha256(texts[0].encode()).digest()[:4], "little")
        vectors = np.random.default_rng(seed).standard_normal((len(texts), self.dimension)).astype(np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
'''


def sdk_installed():
    try:
        import kubeflow.trainer  # noqa: F401
    except Exception:  # Absent, or another distribution that owns the name.
        return False
    return True


def stop_time(seconds):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + seconds))


def wait_for(condition, seconds=30):
    limit = time.time() + seconds
    while time.time() < limit:
        value = condition()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("Timed out waiting for the worker")


class PreemptionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "bundle"
        fakes = Path(self.temporary.name) / "fakes"
        for name, source in [("torch", FAKE_TORCH), ("sentence_transformers", FAKE_SENTENCE_TRANSFORMERS)]:
            (fakes / name).mkdir(parents=True)
            (fakes / name / "__init__.py").write_text(source)
        for name in ["torch", "sentence_transformers", "transformers"]:
            (fakes / f"{name}-0.0.dist-info").mkdir()
            (fakes / f"{name}-0.0.dist-info" / "METADATA").write_text(
                f"Metadata-Version: 2.1\nName: {name.replace('_', '-')}\nVersion: 0.0\n")
        (self.root / "code").mkdir(parents=True)
        for name in ["workload.py", "artifacts.py", "lora.py", "models.json", "trials.json"]:
            shutil.copy(CODE / name, self.root / "code" / name)
        (self.root / "corpus").mkdir()
        self.write_corpus([{"id": f"test/{i}", "text": f"text {i}"} for i in range(2500)])
        self.out = self.root / "results"
        self.environment = dict(os.environ, PYTHONPATH=os.pathsep.join([str(fakes), str(CODE)]),
                                RANK="0", WORLD_SIZE="1", LOCAL_RANK="0", LOCAL_WORLD_SIZE="1")
        self.processes = []

    def tearDown(self):
        for process in self.processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            process.stderr.close()
        # The wrapper starts the worker in a session of its own, so a failed
        # wrapper test can leave the worker behind. Stop it before the files go.
        for attempt in self.attempts() if self.out.exists() else []:
            for event in attempt:
                try:
                    command = Path(f"/proc/{event.get('pid')}/cmdline").read_bytes()
                except OSError:
                    continue
                if event["stage"] == "start" and str(self.root).encode() in command:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(event["pid"], signal.SIGKILL)
        self.temporary.cleanup()

    def write_corpus(self, rows):
        body = "".join(json.dumps(row) + "\n" for row in rows)
        (self.root / "corpus" / "corpus.jsonl").write_text(body)
        (self.root / "corpus" / "manifest.json").write_text(json.dumps(
            {"sha256": hashlib.sha256(body.encode()).hexdigest(), "records": len(rows)}))

    def start(self, command, **variables):
        process = subprocess.Popen(command, env=dict(self.environment, **variables), start_new_session=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        self.processes.append(process)
        return process

    def worker(self, *options, **variables):
        return self.start([sys.executable, str(self.root / "code/workload.py"), "--corpus", str(self.root / "corpus"),
                           "--out", str(self.out), "--stop-at", stop_time(600), *options], **variables)

    def attempts(self):
        """Records of every attempt, oldest attempt first."""
        return [[json.loads(line) for line in path.read_text().splitlines()]
                for path in sorted(self.out.glob("measurements-rank-0-*.jsonl"))]

    def stages(self, attempt=-1):
        return [event["stage"] for event in self.attempts()[attempt]]

    def committed(self):
        return sorted(str(path.relative_to(self.out)) for path in self.out.glob("*/*.npy")
                      if path.with_suffix(".json").exists())

    def test_stop_during_build_is_recorded_and_the_restart_resumes(self):
        first = self.worker("--mode", "build", FAKE_ROW_SECONDS="0.0003")
        wait_for(self.committed)
        os.kill(first.pid, signal.SIGTERM)
        self.assertEqual(first.wait(timeout=30), 0, first.stderr.read())
        stages = self.stages()
        self.assertEqual(stages[0], "start")
        self.assertIn("signal", stages)
        self.assertEqual(self.attempts()[-1][-1]["status"], "interrupted")
        before = self.committed()
        self.assertLess(len(before), 9)
        # A chunk whose receipt is missing does not count as finished.
        unfinished = Path(before[0])
        (self.out / unfinished).with_suffix(".json").unlink()

        second = self.worker("--mode", "build")
        self.assertEqual(second.wait(timeout=60), 0, second.stderr.read())
        self.assertEqual(len(self.committed()), 9)
        earlier, later = self.attempts()
        resume = next(event for event in later if event["stage"] == "resume")
        present = sum(model["present"] for model in resume["chunks"].values())
        self.assertEqual(present, len(before) - 1)
        self.assertEqual(sum(model["assigned"] for model in resume["chunks"].values()), 9)
        built = [(event["model"], event["start"]) for event in earlier + later
                 if event["stage"] == "build" and "start" in event]
        repeated = sorted(chunk for chunk in set(built) if built.count(chunk) > 1)
        self.assertEqual(repeated, [(unfinished.parent.name, int(unfinished.stem))])
        self.assertEqual(len(built), 10)

    def test_stop_is_recorded_when_the_worker_is_killed_inside_a_long_call(self):
        process = self.worker("--mode", "build", FAKE_ROW_SECONDS="0.01")
        wait_for(lambda: self.out.exists() and self.attempts() and "resume" in self.stages())
        time.sleep(1)  # The first encode call takes ten seconds and has started by now.
        os.kill(process.pid, signal.SIGTERM)
        wait_for(lambda: "signal" in self.stages())
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        stages = self.stages()
        self.assertEqual(stages[-1], "signal")
        self.assertNotIn("finished", stages)
        self.assertEqual(self.committed(), [])

    def test_stop_during_startup_is_recorded(self):
        process = self.worker("--mode", "build", FAKE_IMPORT_SECONDS="2")
        wait_for(lambda: self.out.exists() and self.attempts() and self.stages() == ["start"])
        os.kill(process.pid, signal.SIGTERM)
        self.assertEqual(process.wait(timeout=30), 0, process.stderr.read())
        stages = self.stages()
        self.assertEqual(stages[:2], ["start", "signal"])
        self.assertEqual(self.attempts()[-1][-1]["status"], "interrupted")
        self.assertEqual(self.committed(), [])

    def launch(self, stop_at, **variables):
        call = (f"import submit; submit.launch({str(self.root)!r}, {sys.executable!r}, {stop_at!r}, "
                f"'results', case_seconds=1, trial_shard='0/1', max_trials=0)")
        return self.start([sys.executable, "-c", call], **variables)

    def test_wrapper_passes_the_stop_on_and_the_case_duration_through(self):
        def sweep_cases():
            if not self.out.exists() or not self.attempts():
                return set()
            return {(event["model"], event["precision"], event["max_length"], event["distribution"],
                     event["batch_size"]) for event in self.attempts()[-1] if event["stage"] == "sweep"}

        wrapper = self.launch(stop_time(600), FAKE_ROW_SECONDS="0.0003")
        # A second case within seconds shows the one-second case duration reached the worker.
        wait_for(lambda: len(sweep_cases()) >= 2, seconds=40)
        os.kill(wrapper.pid, signal.SIGTERM)
        self.assertEqual(wrapper.wait(timeout=30), 128 + signal.SIGTERM)
        stages = self.stages()
        self.assertIn("signal", stages)
        self.assertEqual(stages[-1], "finished")
        self.assertNotIn("train", stages)  # The trial limit of zero reached the worker.
        self.assertEqual(len(self.committed()), 9)

    def test_failed_training_is_recorded_and_the_load_continues(self):
        # The stand-ins cannot train, so every trial fails. The job must go on to
        # the stages that keep the GPU busy.
        self.write_corpus([{
            "id": f"test/{i}", "iri": f"https://example.org/{i}", "source": "test", "label": f"term {i}",
            "definition": f"A description of term {i} that is long enough to count as a training pair.",
            "definition_language": "en", "text": f"term {i}"} for i in range(2500)])
        process = self.worker("--case-seconds", "1", "--trial-shard", "1/4", "--reserve-seconds", "0",
                              FAKE_ROW_SECONDS="0.0003")
        wait_for(lambda: self.out.exists() and self.attempts() and "sweep" in self.stages(), seconds=60)
        os.kill(process.pid, signal.SIGTERM)
        self.assertEqual(process.wait(timeout=30), 0, process.stderr.read())
        training = [event for event in self.attempts()[-1] if event["stage"] == "train"]
        self.assertEqual([event["status"] for event in training], ["queue", "failed", "failed", "failed", "abandoned"])
        self.assertEqual(training[0]["trials"], 54)
        self.assertEqual(training[0]["components"]["train"] + training[0]["components"]["validation"]
                         + training[0]["components"]["test"], 2500)
        stages = self.stages()
        self.assertLess(stages.index("train"), stages.index("sweep"))
        self.assertEqual(stages[-1], "finished")
        self.assertNotIn("adapted", stages)
        self.assertTrue((self.out / "pairs-manifest.json").exists())

    def test_job_admitted_too_late_ends_without_work_or_error(self):
        for seconds in [-3600, 60, 110]:
            wrapper = self.launch(stop_time(seconds))
            self.assertEqual(wrapper.wait(timeout=30), 0, wrapper.stderr.read())
            self.assertFalse(self.out.exists())

    def test_launch_source_survives_the_unquoted_here_document(self):
        source = inspect.getsource(submit.launch)
        for character in "$`\\":
            self.assertNotIn(character, source)
        self.assertNotIn("EOM", [line.strip() for line in source.splitlines()])

    def test_submit_refuses_values_the_here_document_would_expand(self):
        arguments = ["submit.py", "--root", "/personal/$USER/bundle", "--python", "/personal/bundle/bin/python",
                     "--runtime", "torch-rtxa6000", "--stop-at", stop_time(600)]
        with mock.patch.object(sys, "argv", arguments), contextlib.redirect_stderr(io.StringIO()) as printed, \
                self.assertRaises(SystemExit):
            submit.main()
        self.assertIn("must not contain", printed.getvalue())

    def test_submit_refuses_a_trial_shard_outside_its_range(self):
        for shard in ["4/4", "$(id)/4", "1"]:
            arguments = ["submit.py", "--root", "/personal/bundle", "--python", "/personal/bundle/bin/python",
                         "--runtime", "torch-rtxa6000", "--stop-at", stop_time(600), "--trial-shard", shard]
            with mock.patch.object(sys, "argv", arguments), contextlib.redirect_stderr(io.StringIO()) as printed, \
                    self.assertRaises(SystemExit):
                submit.main()
            self.assertIn("trial shard", printed.getvalue())

    @unittest.skipUnless(sdk_installed(), "Kubeflow SDK not installed")
    def test_execute_builds_the_job_with_the_installed_sdk(self):
        import kubeflow.trainer
        calls = []

        class Client:
            def __init__(self, config):
                calls.append(config)
                address.update({name: os.environ.get(name) for name in address})

            def train(self, **options):
                calls.append(options)
                return "job-name"

        arguments = ["submit.py", "--root", "/personal/bundle", "--python", "/personal/bundle/bin/python",
                     "--runtime", "torch-rtxa6000", "--stop-at", stop_time(600), "--case-seconds", "5",
                     "--output", "results-test", "--trial-shard", "2/4", "--max-trials", "6", "--execute"]
        # A shell opened through SSH inside a pod has the service account but not
        # the two variables that locate the cluster API.
        address = {"KUBERNETES_SERVICE_HOST": None, "KUBERNETES_SERVICE_PORT": None}
        without = {name: value for name, value in os.environ.items() if name not in address}
        with mock.patch.object(kubeflow.trainer, "TrainerClient", Client), mock.patch.object(sys, "argv", arguments), \
                mock.patch.object(submit.Path, "exists", return_value=True), \
                mock.patch.dict(os.environ, without, clear=True), \
                contextlib.redirect_stdout(io.StringIO()) as printed:
            submit.main()
        self.assertEqual(address, {"KUBERNETES_SERVICE_HOST": "kubernetes.default.svc", "KUBERNETES_SERVICE_PORT": "443"})
        config, options = calls
        self.assertEqual(config.namespace, "metadata-research-center")
        self.assertEqual(options["runtime"], "torch-rtxa6000")
        trainer = options["trainer"]
        self.assertIs(trainer.func, submit.launch)
        self.assertEqual(trainer.func_args, {"root": "/personal/bundle", "interpreter": "/personal/bundle/bin/python",
                                             "stop_at": arguments[8], "output": "results-test", "case_seconds": 5,
                                             "trial_shard": "2/4", "max_trials": 6})
        self.assertEqual(trainer.resources_per_node, {"cpu": 8, "memory": "32Gi", "gpu": 1})
        self.assertIn("Submitted TrainJob: job-name", printed.getvalue())


if __name__ == "__main__":
    unittest.main()
