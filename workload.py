"""Build portable indexes, run LoRA trials, then benchmark until an explicit wall-clock deadline.

One process per allocated GPU. torchrun's RANK/WORLD_SIZE split artifact
chunks and trials; no collectives are used, so this does not measure
interconnects. No network downloads are allowed in a timed run.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import gc
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import time

from artifacts import atomic_json, checkpoint_valid, chunk_ranges, digest, load_corpus, save_vectors
import lora


class Deadline(BaseException):
    """The alarm at --stop-at. Not an Exception, so no handler for ordinary errors swallows it."""


def deadline(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("--stop-at must include its UTC offset")
    return parsed.timestamp()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stop-at", required=True, help="ISO date/time including offset; nothing is scheduled")
    parser.add_argument("--models", type=Path, default=Path(__file__).with_name("models.json"))
    parser.add_argument("--mode", choices=["build", "all"], default="all")
    parser.add_argument("--case-seconds", type=int, default=45)
    parser.add_argument("--cpu-threads", type=int, default=1,
                        help="CPU threads per GPU worker; submit.py derives this from the job's CPU allocation")
    parser.add_argument("--trials", type=Path, default=Path(__file__).with_name("trials.json"))
    parser.add_argument("--trial-shard", default="0/1",
                        help="i/n: this job runs the trials whose position modulo n equals i")
    parser.add_argument("--max-trials", type=int, help="Upper limit of trials per GPU process; 0 skips the training")
    parser.add_argument("--reserve-seconds", type=int, default=900,
                        help="No trial starts this close to the end of work, which leaves time for the adapted index")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="Functional tests without a GPU; never for a timed run")
    args = parser.parse_args()
    stop = deadline(args.stop_at)
    if stop - time.time() < 90 or args.case_seconds <= 0 or args.cpu_threads < 1:
        parser.error("Allow at least 90 seconds until stop-at and a positive case duration and CPU thread count")
    shard = re.fullmatch(r"(\d+)/(\d+)", args.trial_shard)
    if not shard or not int(shard[1]) < int(shard[2]) or (args.max_trials or 0) < 0 or args.reserve_seconds < 0:
        parser.error("The trial shard has the form i/n with i below n; limits must not be negative")
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if not 0 <= rank < world:
        raise ValueError("Invalid process rank")
    attempt = str(time.time_ns())
    args.out.mkdir(parents=True, exist_ok=True)
    # Unbuffered and append-only: each record is one write, so the signal handler
    # can add a record, and a record written just before a kill is on disk.
    events = (args.out / f"measurements-rank-{rank}-{attempt}.jsonl").open("ab", buffering=0)
    interrupted = False

    def record(**event):
        event.update(time=datetime.now(timezone.utc).isoformat(), rank=rank)
        line = json.dumps(event)
        events.write((line + "\n").encode())
        return line

    def handle_signal(signum, _frame):
        nonlocal interrupted
        interrupted = True
        try:
            record(stage="signal", signal=signal.Signals(signum).name)
        except (OSError, ValueError):
            pass  # A failed write must not stop the worker; the file may be closed.

    # SIGTERM is handled before the slow imports and the corpus load, so that a
    # stop during startup is recorded. SIGINT keeps its default until the build.
    signal.signal(signal.SIGTERM, handle_signal)
    record(stage="start", pid=os.getpid(), world_size=world, local_rank=local, stop_at=args.stop_at)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.cpu_threads)
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    cuda = torch.cuda.is_available()
    if not cuda and not args.allow_cpu:
        raise RuntimeError("This workload requires CUDA; refusing a silent CPU run")
    if cuda:
        torch.cuda.set_device(local)
    device = f"cuda:{local}" if cuda else "cpu"
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(42)
    # A restart may resume; two writers with the same rank may not overlap.
    lock = (args.out / f"rank-{rank}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest, rows = load_corpus(args.corpus)
    specs = json.loads(args.models.read_text())
    ranges = chunk_ranges(len(rows))
    started = datetime.now(timezone.utc).isoformat()
    packages = {p: importlib.metadata.version(p) for p in ["torch", "sentence-transformers", "transformers", "numpy"]}
    contract = {
        "format": "matsci-ont-vectors-v1", "corpus_sha256": manifest["sha256"],
        "models_sha256": digest(args.models), "chunk_size": 1024,
        "models": specs, "storage_dtype": "float32", "normalized": True,
        "build_precision": "float32", "packages": packages,
        "worker_sha256": digest(Path(__file__)),
        "artifacts_sha256": digest(Path(__file__).with_name("artifacts.py")),
        "lora_sha256": digest(Path(lora.__file__)),
        "gpu": torch.cuda.get_device_name(local) if cuda else "cpu", "cuda": torch.version.cuda,
        "architecture": platform.machine()
    }

    def bind(root, expected):
        """Bind an index directory to its corpus, models and runtime."""
        # Serialize the short initialization across ranks, then work independently.
        with (root / "manifest.lock").open("a") as meta_lock:
            fcntl.flock(meta_lock, fcntl.LOCK_EX)
            path = root / "manifest.json"
            if path.exists() and json.loads(path.read_text()) != expected:
                raise ValueError("Output belongs to a different corpus/model/runtime; choose a new output directory")
            if not path.exists():
                atomic_json(path, expected)

    bind(args.out, contract)
    atomic_json(args.out / f"environment-rank-{rank}-{attempt}.json", {
        **contract, "started": started, "stop_at": args.stop_at,
        "rank": rank, "world_size": world, "local_rank": local,
        "gpu_bytes": torch.cuda.get_device_properties(local).total_memory if cuda else 0,
        "python": platform.python_version(), "cpu_threads": args.cpu_threads
    })
    measured = {}

    def log(**event):
        if event.get("stage") == "sweep" and not event.get("warmup", True) and "seconds" in event:
            key = (event["model"], event["precision"], event["max_length"],
                   event["distribution"], event["batch_size"])
            count, seconds = measured.get(key, (0, 0.0))
            measured[key] = (count + event["rows"], seconds + event["seconds"])
        print(record(**event), flush=True)

    # Set again here in case an imported library replaced the SIGTERM handler.
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    def available():
        return not interrupted and time.time() < stop - 60

    # The submission wrapper also has a process-level watchdog, since Python
    # signals can be delayed inside native CUDA calls.
    def timed_out(_signum, _frame):
        raise Deadline("GPU test deadline")

    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(max(1, int(stop - time.time())))
    telemetry = None
    telemetry_file = None
    try:
        if local == 0 and cuda:
            telemetry_file = (args.out / f"gpu-{platform.node()}-{attempt}.csv").open("w")
            try:
                telemetry = subprocess.Popen([
                    "nvidia-smi", "--query-gpu=timestamp,uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu",
                    "--format=csv,nounits", "--loop=5"
                ], stdout=telemetry_file, stderr=subprocess.STDOUT)
            except FileNotFoundError:
                log(stage="telemetry", status="nvidia-smi unavailable; request cluster dashboard export")

        def load(spec, root=args.out):
            if not available():
                return None
            model = SentenceTransformer(spec["id"], revision=spec["revision"], device=device,
                                        local_files_only=True, trust_remote_code=False)
            if "adapter" in spec:
                lora.attach_adapter(model, root / "adapter", spec["adapter"]["sha256"])
            model.float().eval()
            model.max_seq_length = spec["max_length"]
            # The method was renamed in sentence-transformers 6; the old name warns on every call.
            dimensions = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
            if dimensions() != spec["dimensions"]:
                raise ValueError("Model dimensions differ from pinned configuration")
            return model

        def encode(encoder, texts, batch, **context):
            if cuda:
                torch.cuda.reset_peak_memory_stats(local)
                torch.cuda.synchronize(local)
            begin = time.perf_counter()
            with torch.inference_mode():
                result = encoder.encode(texts, batch_size=batch, normalize_embeddings=True,
                                      show_progress_bar=False, convert_to_numpy=True)
            if cuda:
                torch.cuda.synchronize(local)
            elapsed = time.perf_counter() - begin
            if not np.isfinite(result).all():
                raise ValueError("Nonfinite GPU output")
            log(**context, batch_size=batch, rows=len(texts), seconds=elapsed,
                rows_per_second=len(texts) / elapsed,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(local) if cuda else 0)
            return result

        def build(root, model_specs, stage, share=(rank, world)):
            """Produce durable vectors. Each chunk is committed independently."""
            mine = [part for number, part in enumerate(ranges) if number % share[1] == share[0]]
            # One record per attempt shows what a restart found. The loop below
            # still verifies every chunk before it skips one.
            log(stage="resume", index=str(root.relative_to(args.out)), chunks={spec["name"]: {
                "assigned": len(mine),
                "present": sum((root / spec["name"] / f"{start:09d}.npy").exists()
                               and (root / spec["name"] / f"{start:09d}.json").exists() for start, _end in mine)
            } for spec in model_specs})
            for spec in model_specs:
                model = load(spec, root)
                if model is None:
                    break
                target = root / spec["name"]
                target.mkdir(exist_ok=True)
                batch = spec["batch_size"]
                for start, end in mine:
                    part = rows[start:end]
                    path = target / f"{start:09d}.npy"
                    if checkpoint_valid(path, part, spec["dimensions"]):
                        continue
                    if not available():
                        break
                    while available():
                        try:
                            vectors = encode(model, [r["text"] for r in part], batch,
                                             stage=stage, model=spec["name"], start=start, precision="float32")
                            save_vectors(path, vectors, part)
                            break
                        except torch.cuda.OutOfMemoryError:
                            log(stage=stage, model=spec["name"], status="oom", batch_size=batch)
                            torch.cuda.empty_cache()
                            if batch == 1:
                                raise
                            batch = max(1, batch // 2)
                del model
                torch.cuda.empty_cache()

        build(args.out, specs, "build")

        if args.mode == "all" and args.max_trials != 0 and available():
            # Training failures are recorded and the job continues, because the
            # stages below keep the GPU loaded until the deadline.
            try:
                trials = lora.load_trials(args.trials, specs, limit=args.max_trials,
                                          shard=lora.rank_shard(args.trial_shard, rank, world))
                lora.run_trials(rows, manifest["sha256"], specs, trials, args.out, device, log, available,
                                start_before=stop - 60 - args.reserve_seconds)
            except Exception as error:
                log(stage="train", status="failed", error=f"{type(error).__name__}: {error}"[:500])
            gc.collect()
            torch.cuda.empty_cache()
            try:
                # One adapted index per output: the trial that was best when the
                # first attempt reached this point. Rank zero builds all of it.
                chosen = sorted(path.parent.name for path in args.out.glob("adapted/*/manifest.json"))
                best = lora.best_trial(args.out)
                if rank == 0 and (chosen or best) and available():
                    trial_id = chosen[0] if chosen else best["trial"]["id"]
                    source = args.out / "trials" / trial_id
                    root = args.out / "adapted" / trial_id
                    root.mkdir(parents=True, exist_ok=True)
                    for stale in root.glob("adapter.*.partial"):
                        shutil.rmtree(stale)
                    if not (root / "adapter").exists():
                        staging = root / f"adapter.{os.getpid()}.partial"
                        shutil.copytree(source / "adapter", staging)
                        staging.replace(root / "adapter")
                    result = json.loads((source / "result.json").read_text())
                    adapted_specs = json.loads((source / "model.json").read_text())
                    bind(root, {**contract, "models": adapted_specs, "models_sha256": digest(source / "model.json"),
                                "trial": trial_id, "pairs_sha256": result["data"]["pairs_sha256"]})
                    log(stage="adapted", status="selected", trial=trial_id, score=lora.selection_score(result))
                    build(root, adapted_specs, "adapted", share=(0, 1))
            except Exception as error:
                log(stage="adapted", status="failed", error=f"{type(error).__name__}: {error}"[:500])
            gc.collect()
            torch.cuda.empty_cache()

        if args.mode == "all" and available():
            # Natural short and long records; max_length is a cap, not artificial padding.
            sample_size = min(len(rows), 4096)
            sample = [rows[i * len(rows) // sample_size] for i in range(sample_size)]
            long_rows = sorted(rows, key=lambda r: len(r["text"]), reverse=True)[:4096]
            for spec in specs:
                model = load(spec)
                if model is None:
                    break
                for precision in ["float32", "float16"] if cuda else ["float32"]:
                    if precision == "float16":
                        model.half()
                    for length, batch, distribution in itertools.product(
                        sorted({128, spec["max_length"]}), [32, 128, 512], ["natural", "long"]
                    ):
                        if not available():
                            break
                        model.max_seq_length = length
                        selection = sample if distribution == "natural" else long_rows
                        # Identical records across batch sizes keep throughput
                        # comparisons from being confounded by text length.
                        texts = [r["text"] for r in selection[:1024]]
                        until = min(stop - 60, time.time() + args.case_seconds)
                        first = True
                        while available() and time.time() < until:
                            try:
                                encode(model, texts, batch, stage="sweep", model=spec["name"],
                                       precision=precision, max_length=length, distribution=distribution, warmup=first)
                                first = False
                            except torch.cuda.OutOfMemoryError:
                                log(stage="sweep", model=spec["name"], status="oom", batch_size=batch,
                                    precision=precision, max_length=length, distribution=distribution)
                                torch.cuda.empty_cache()
                                break
                del model
                torch.cuda.empty_cache()

            # Repeat useful capacity measurements to occupy the remaining window.
            # Only the original indexes are retained; repeated vectors are discarded.
            spec = specs[-1]
            model = load(spec)
            if model is not None:
                precision = "float16" if cuda else "float32"
                if cuda:
                    model.half()
                batches = {"natural": 128, "long": 128}
                for distribution in batches:
                    candidates = [(n / seconds, key[4]) for key, (n, seconds) in measured.items()
                                  if key[:4] == (spec["name"], precision, spec["max_length"], distribution)]
                    if candidates:
                        batches[distribution] = max(candidates)[1]
                cycle = 0
                while available():
                    distribution = "long" if cycle % 2 else "natural"
                    selection = long_rows if cycle % 2 else sample
                    batch = batches[distribution]
                    texts = [r["text"] for r in selection]
                    try:
                        encode(model, texts, batch, stage="sustained", model=spec["name"],
                               precision=precision, distribution="long" if cycle % 2 else "natural")
                        cycle += 1
                    except torch.cuda.OutOfMemoryError:
                        log(stage="sustained", status="oom", batch_size=batch)
                        torch.cuda.empty_cache()
                        if batch == 1:
                            raise
                        batches[distribution] = max(1, batch // 2)
                del model
        log(stage="finished", status="interrupted" if interrupted else "worker finished",
            note="search.py independently requires every corpus chunk before serving results")
    finally:
        signal.alarm(0)
        if telemetry:
            telemetry.terminate()
            try:
                telemetry.wait(timeout=5)
            except subprocess.TimeoutExpired:
                telemetry.kill()
                telemetry.wait()
        if telemetry_file:
            telemetry_file.close()
        events.close()
        lock.close()


if __name__ == "__main__":
    main()
