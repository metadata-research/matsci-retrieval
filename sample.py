"""Create a separate small, source-balanced corpus for the GPU preflight."""
import argparse
import json
from pathlib import Path

from artifacts import atomic_json, digest, load_corpus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest, rows = load_corpus(args.corpus)
    chosen = []
    for source in manifest["sources"]:
        group = [r for r in rows if r["source"] == source["key"]]
        longest = sorted(group, key=lambda r: (-len(r["text"]), r["id"]))[:16]
        long_ids = {r["id"] for r in longest}
        remainder = sorted((r for r in group if r["id"] not in long_ids), key=lambda r: r["id"])[:48]
        chosen.extend(longest + remainder)
    chosen.sort(key=lambda r: (r["source"], r["id"]))
    args.out.mkdir(parents=True, exist_ok=False)
    corpus = args.out / "corpus.jsonl"
    corpus.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in chosen))
    atomic_json(args.out / "manifest.json", {
        **manifest, "parent_sha256": manifest["sha256"], "sha256": digest(corpus),
        "records": len(chosen), "sample_recipe": "up to 16 longest + 48 by hashed entry IRI per source",
        "counts": {s["key"]: {"records": sum(r["source"] == s["key"] for r in chosen),
                               "missingDefinitions": sum(r["source"] == s["key"] and not r["definition"] for r in chosen)}
                   for s in manifest["sources"]}
    })
    print(f"Saved {len(chosen)} records to {args.out}")


if __name__ == "__main__":
    main()
