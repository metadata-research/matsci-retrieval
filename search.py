"""Search a complete vector artifact on CPU, retaining source attribution."""
import argparse
import heapq
import json
import os
from pathlib import Path

from artifacts import atomic_json, checkpoint_valid, chunk_ranges, load_corpus
from lora import attach_adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--query")
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--source", help="Optional ontology source key")
    args = parser.parse_args()
    if bool(args.query) == bool(args.queries) or args.top_k < 1:
        parser.error("Specify one of --query or --queries, and a positive --top-k")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    manifest, rows = load_corpus(args.corpus)
    contract = json.loads((args.index / "manifest.json").read_text())
    if contract["corpus_sha256"] != manifest["sha256"]:
        raise ValueError("This index was built for another corpus")
    spec = next(s for s in contract["models"] if s["name"] == args.model)
    ranges = chunk_ranges(len(rows))
    for start, end in ranges:
        if not checkpoint_valid(args.index / args.model / f"{start:09d}.npy", rows[start:end], spec["dimensions"]):
            raise ValueError(f"Incomplete index; missing chunk at {start}")
    queries = json.loads(args.queries.read_text()) if args.queries else [{"id": "query", "query": args.query}]
    model = SentenceTransformer(spec["id"], revision=spec["revision"], device="cpu",
                                local_files_only=True, trust_remote_code=False)
    if "adapter" in spec:
        # An adapted index holds its adapter. Queries need the same weights as the documents.
        attach_adapter(model, args.index / "adapter", spec["adapter"]["sha256"])
    model.float().eval()
    model.max_seq_length = spec["max_length"]
    query_vectors = model.encode([spec["query_prefix"] + q["query"] for q in queries], normalize_embeddings=True)
    heaps = [[] for _ in queries]
    for start, end in ranges:
        vectors = np.load(args.index / args.model / f"{start:09d}.npy", mmap_mode="r", allow_pickle=False)
        scores = vectors @ query_vectors.T
        for qnumber, heap in enumerate(heaps):
            ranked = sorted(range(end - start), key=lambda i: (-float(scores[i, qnumber]), rows[start + i]["id"]))
            accepted = 0
            for i in ranked:
                if args.source and rows[start + i]["source"] != args.source:
                    continue
                candidate = (float(scores[i, qnumber]), start + i)
                if len(heap) < args.top_k:
                    heapq.heappush(heap, candidate)
                elif candidate > heap[0]:
                    heapq.heapreplace(heap, candidate)
                accepted += 1
                if accepted == args.top_k:
                    break
    result = {
        "corpus_sha256": manifest["sha256"], "model": spec, "method": "exact normalized dot product",
        "source_filter": args.source,
        "queries": [{**q, "results": [
            {"score": score, **{k: rows[i][k] for k in ["id", "iri", "source", "version", "license", "label", "definition"]}}
            for score, i in sorted(heap, reverse=True)
        ]} for q, heap in zip(queries, heaps)]
    }
    if args.out:
        atomic_json(args.out, result)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
