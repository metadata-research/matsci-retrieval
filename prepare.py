"""Download and validate model snapshots before the GPU reservation."""
import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time

from artifacts import atomic_json, load_corpus
import lora


def lora_check(rows, corpus_sha256, specs, trials_path):
    """Two training steps per model on CPU, through the code path of the GPU job."""
    import numpy as np
    from sentence_transformers import SentenceTransformer
    sample = [row for row in rows if row["definition"]][::100]
    chosen = {}
    for trial in lora.load_trials(trials_path, specs):
        chosen.setdefault(trial["parent"], {**trial, "pairs_budget": 16, "batch_size": 8, "distractors": 200})
    events = []

    def log(**event):
        events.append(event)
        if event.get("status") in ("started", "complete", "failed"):
            print(f"Training check {event['trial']}: {event['status']}", flush=True)

    with tempfile.TemporaryDirectory() as directory:
        lora.run_trials(sample, corpus_sha256, specs, list(chosen.values()), Path(directory), "cpu",
                        log, lambda: True, time.time() + 3600)
        results = lora.completed_trials(directory)
        failed = [event for event in events if event.get("status") == "failed"]
        if failed or len(results) != len(chosen):
            raise ValueError(f"LoRA preparation check failed: {failed or events[-3:]}")
        report = []
        for result in results:
            spec = next(spec for spec in specs if spec["name"] == result["trial"]["parent"])
            model = SentenceTransformer(spec["id"], revision=spec["revision"], device="cpu", trust_remote_code=False)
            lora.attach_adapter(model, Path(directory) / "trials" / result["trial"]["id"] / "adapter",
                                result["adapter_sha256"])
            vectors = model.encode([row["text"] for row in sample[:4]], normalize_embeddings=True)
            if vectors.shape != (4, spec["dimensions"]) or not np.isfinite(vectors).all():
                raise ValueError(f"Adapter reload failed: {spec['name']}")
            report.append({"parent": spec["name"], "trial": result["trial"]["id"],
                           "trainable_parameters": result["lora"]["trainable_parameters"],
                           "steps": result["training"]["steps"], "seconds": result["training"]["seconds"],
                           "sample_components": result["data"]["train_components"]})
            del model
            gc.collect()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=Path(__file__).with_name("models.json"))
    parser.add_argument("--trials", type=Path, default=Path(__file__).with_name("trials.json"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))  # More threads than CPUs slow the checks down.
    manifest, rows = load_corpus(args.corpus)
    specs = json.loads(args.models.read_text())
    results = []
    for spec in specs:
        model = SentenceTransformer(spec["id"], revision=spec["revision"], device="cpu", trust_remote_code=False)
        model.max_seq_length = spec["max_length"]
        vectors = model.encode([r["text"] for r in rows[:8]], normalize_embeddings=True)
        if vectors.shape != (8, spec["dimensions"]) or not np.isfinite(vectors).all():
            raise ValueError(f"Model preflight failed: {spec['name']}")
        # Tokenize the complete corpus without padding so truncation is measured.
        lengths = []
        for start in range(0, len(rows), 1024):
            batch = model.tokenizer([r["text"] for r in rows[start:start + 1024]],
                                    truncation=False, padding=False, verbose=False)
            lengths.extend(len(x) for x in batch["input_ids"])
        results.append({**spec, "truncated_records": sum(n > spec["max_length"] for n in lengths),
                        "token_length_percentiles": dict(zip(["p50", "p95", "p99", "max"],
                            [float(x) for x in np.percentile(lengths, [50, 95, 99, 100])]))})
        del model
        gc.collect()
        print(f"{spec['name']}: snapshot, CPU inference and token lengths checked", flush=True)
    pairs = lora.pairs_manifest(rows, lora.build_pairs(rows), manifest["sha256"])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.report, {
        "corpus_sha256": manifest["sha256"], "models": results,
        "pairs": {"sha256": pairs["sha256"], "splits": pairs["splits"]},
        "lora": {"peft": importlib.metadata.version("peft"),
                 "checks": lora_check(rows, manifest["sha256"], specs, args.trials)},
        "architecture": platform.machine(), "python": platform.python_version(),
        "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "packages": {p: importlib.metadata.version(p) for p in ["sentence-transformers", "transformers", "numpy"]},
        "scope": "CPU model, corpus and training-code preparation only; GPU launch, GPU telemetry and resume still require preflight"
    })
    import sys
    frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    args.report.with_suffix(".freeze.txt").write_text(frozen)
    print(args.report)


if __name__ == "__main__":
    main()
