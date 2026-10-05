"""Pack the code and a corpus export into the bundle that is unpacked on FLAME."""
import argparse
from pathlib import Path
import tarfile

from artifacts import digest, load_corpus

ROOT = "matsci-gpu-test"
CODE = ["README.md", "artifacts.py", "lora.py", "models.json", "prepare.py", "queries.draft.json", "report.py",
        "requirements.txt", "sample.py", "search.py", "submit.py", "test_artifacts.py", "test_lora.py",
        "test_preemption.py", "test_report.py", "trials.json", "workload.py"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True, help="Export directory with corpus.jsonl and manifest.json")
    parser.add_argument("--out", type=Path, required=True, help="Bundle file with the suffix .tar.xz")
    args = parser.parse_args()
    if not args.out.name.endswith(".tar.xz"):
        parser.error("The bundle name must end with .tar.xz")
    here = Path(__file__).resolve().parent
    members = [(f"{ROOT}/code/{name}", here / name) for name in CODE]
    load_corpus(args.corpus)  # Refuses a corpus that does not match its manifest.
    members += [(f"{ROOT}/corpus/{path.name}", path) for path in sorted(args.corpus.iterdir()) if path.is_file()]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.out, "w:xz", preset=9) as archive:
        for name, path in members:
            entry = archive.gettarinfo(str(path), arcname=name)
            entry.uid = entry.gid = 0
            entry.uname = entry.gname = ""
            entry.mode = 0o644
            with path.open("rb") as stream:
                archive.addfile(entry, stream)
    checksum = args.out.with_name(args.out.name + ".sha256")
    checksum.write_text(f"{digest(args.out)}  {args.out.name}\n")
    print(f"{args.out} ({args.out.stat().st_size} bytes, {len(members)} files)")


if __name__ == "__main__":
    main()
