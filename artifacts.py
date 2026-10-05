"""Integrity and identity rules shared by the GPU writer and CPU reader."""
import hashlib
import json
import os
from pathlib import Path

CHUNK_SIZE = 1024


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.partial")
    with temporary.open("w") as target:
        json.dump(value, target, indent=2, ensure_ascii=False)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def load_corpus(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if digest(directory / "corpus.jsonl") != manifest["sha256"]:
        raise ValueError("Corpus hash mismatch")
    with (directory / "corpus.jsonl").open() as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != manifest["records"] or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Corpus record count or identity mismatch")
    return manifest, rows


def chunk_ranges(count):
    return [(start, min(start + CHUNK_SIZE, count)) for start in range(0, count, CHUNK_SIZE)]


def id_digest(rows):
    return hashlib.sha256("\n".join(r["id"] for r in rows).encode()).hexdigest()


def checkpoint_valid(path, rows, dimensions):
    path = Path(path)
    receipt = path.with_suffix(".json")
    if not path.exists() or not receipt.exists():
        return False
    saved = json.loads(receipt.read_text())
    if saved["ids_sha256"] != id_digest(rows) or saved["dimensions"] != dimensions:
        raise ValueError(f"Checkpoint identity mismatch: {path}")
    if saved["sha256"] != digest(path):
        raise ValueError(f"Checkpoint checksum mismatch: {path}")
    return True


def save_vectors(path, vectors, rows):
    import numpy as np
    path = Path(path)
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(rows) or not np.isfinite(vectors).all():
        raise ValueError("Invalid embedding output")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=0.005):
        raise ValueError("Embeddings are not normalized")
    temporary = path.with_suffix(f".{os.getpid()}.partial")
    with temporary.open("wb") as stream:
        np.save(stream, vectors, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    atomic_json(path.with_suffix(".json"), {
        "rows": len(rows), "dimensions": vectors.shape[1],
        "ids_sha256": id_digest(rows), "sha256": digest(path), "dtype": "float32"
    })
