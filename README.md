# MatSci retrieval experiments

This repository holds GPU experiments on semantic search over the ontology
descriptions that MatSci-ONT publishes. The code builds vector indexes for three
pinned embedding models, runs LoRA training trials on label and definition pairs,
and searches the indexes on CPU. It runs as batch jobs on the FLAME cluster at
Drexel and keeps the GPUs busy until a stated deadline, so it also serves as a
load for cluster tests.

## Files

| File | Purpose |
| --- | --- |
| `models.json` | Pinned revisions of MiniLM, BGE Base and BGE Large |
| `trials.json` | Ordered queue of 216 LoRA trials with their default settings |
| `artifacts.py` | Corpus loading, vector chunks, receipts and checksums |
| `workload.py` | The GPU job (indexes, trials, adapted index, sweep, sustained load) |
| `lora.py` | Training pairs, trials, checkpoints, adapter export, summary and test split |
| `submit.py` | Preview of a Kubeflow TrainJob, submitted with `--execute` |
| `prepare.py` | Model download, token statistics and a training check on CPU |
| `search.py` | Exact cosine search over a complete index on CPU |
| `sample.py` | Small source-balanced corpus for a direct run of `workload.py` |
| `report.py` | Summary of the records and the GPU telemetry of a job |
| `bundle.py` | Packs the code and a corpus export for FLAME |
| `queries.draft.json` | Draft queries for `search.py` |
| `test_*.py` | Local tests |

## Corpus

The corpus comes from the MatSci-ONT repository. Its scripts
`pipeline/gpu-search/export.mjs` and `pipeline/gpu-search/lexical-baseline.mjs`
read the ONT store and write `corpus.jsonl`, `manifest.json` and
`lexical-baseline.json` to `build/gpu-search/corpus` in that repository. Every
script that reads the corpus checks it against the SHA-256 in its manifest.

The export of 2 October 2026 holds 222,925 descriptions, and 164,423 of them have
no definition. The three vector matrices for that export hold one normalized
float32 row per description and total about 1.81 GiB.

## Local tests

```bash
python3 -m unittest discover -p "test_*.py"
```

The tests need numpy. The submission test is skipped where the Kubeflow SDK is
absent. The tests that train MiniLM on CPU are skipped unless torch,
sentence-transformers, peft and the pinned MiniLM snapshot are present, which is
the case on FLAME after `prepare.py`.

## FLAME preparation

Build the bundle on the workstation and copy it to the home volume, which
notebooks see as `/home/ubuntu` and jobs as `/personal`.

```bash
NAME=matsci-gpu-test-$(date +%F).tar.xz
python3 bundle.py --corpus ../matsci-ont/build/gpu-search/corpus --out build/$NAME
scp build/$NAME flame:~/
```

```text
/personal/matsci-gpu-test/
  code/        the code, the JSON files and the README of this repository
  corpus/      corpus.jsonl, manifest.json, lexical-baseline.json
  results-*/   one directory per job, written during the run
```

Unpack with `tar -xf` in `/home/ubuntu`. Use the PyTorch runtime image of FLAME
and keep its CUDA build of torch. GH200 nodes are ARM64 and RTX A6000 nodes are
AMD64, so an environment serves one architecture and one image. The commands
below are for the A6000 node, and `uname -m` has to print `x86_64` in the shell
that runs them.

```bash
cd /home/ubuntu/matsci-gpu-test
uname -m
python3 -m venv --system-site-packages .venv-x86_64
PY="$PWD/.venv-x86_64/bin/python"
"$PY" -m pip install --dry-run -r code/requirements.txt
# Read the output. Torch must not be among the packages to install.
"$PY" -m pip install -r code/requirements.txt kubeflow==0.4.1
"$PY" -m pip check
"$PY" code/prepare.py --corpus corpus --report preflight-x86_64.json
"$PY" -m unittest discover -s code -p "test_*.py"
```

`prepare.py` downloads the three model revisions to the cache that jobs also
read (`XDG_CACHE_HOME=/personal/.cache` in the FLAME runtime), records token
lengths and truncation, counts the training pairs, and runs two LoRA training
steps per model on CPU through the code of the GPU job. It needs about 6 GB of
memory. The job itself is offline and fails if a model is missing from the
cache. The later commands in this file use `"$PY"` for this interpreter.

In the FLAME image of release 2026.07.2, `pip check` reports two conflicts in
packages that the pipeline never imports (numba asks for numpy below 2.5, and
triton-kernels asks for pytest).

The image also ships the Transformer Engine. The peft library imports that
package whenever it is installed, and the import fails in a notebook without a
GPU, where the GPU driver library is missing. `lora.py` hides the package from
Python, and every script that loads the libraries imports `lora` first. The
trials use standard LoRA layers.

A shell opened with `ssh flame` lands in the notebook pod without the two
variables that locate the cluster API. `submit.py` sets them itself. For
`kubectl` in such a shell, export them first.

```bash
export KUBERNETES_SERVICE_HOST=kubernetes.default.svc KUBERNETES_SERVICE_PORT=443
kubectl auth can-i create trainjobs.trainer.kubeflow.org
kubectl get clustertrainingruntimes
kubectl get clusterqueue <workspace> -o yaml
```

The cluster queue shows the guarantee of the workspace (`nominalQuota`) and its
borrowing limit for each GPU type.

## The job

`submit.py` prints the submission and sends nothing. With `--execute` it creates
the TrainJob at once.

```bash
"$PY" code/submit.py \
  --root /personal/matsci-gpu-test \
  --python /personal/matsci-gpu-test/.venv-x86_64/bin/python \
  --runtime torch-rtxa6000 --gpus-per-node 1 --nodes 1 \
  --cpus-per-node 4 --memory-per-node 32Gi \
  --output results-a6000-0 --trial-shard 0/4 \
  --stop-at 2026-10-08T18:00:00+02:00
```

The A6000 node has 31 CPUs for eight GPUs, so a job asks for four CPUs. The
worker limits torch to four threads.

Each process works through five stages and stops taking work one minute before
`--stop-at`.

1. Build the three parent indexes in float32. Every chunk of 1,024 records is
   saved with a checksum receipt.
2. Run LoRA trials from `trials.json` in order. `--trial-shard i/n` gives a job
   the trials whose position modulo n equals i, so that n jobs share the queue
   without overlap. `--max-trials` limits the count per GPU process, and 0 skips
   the training. No trial starts in the last 16 minutes before `--stop-at`, so
   that the next stage has time.
3. Build an index with the adapter of the best finished trial of this output
   directory, under `adapted/<trial>/`.
4. Sweep float32 and float16, input caps of 128 and the model maximum, batches of
   32, 128 and 512, and natural versus long records. A case lasts 45 seconds
   unless `--case-seconds` says otherwise. The first measurement of a case is
   marked as warmup.
5. Encode with BGE Large in float16 until the deadline. These vectors are
   discarded. The output of this stage is throughput, temperature and power data.

A trial that fails is recorded and the next trial starts. After three failures in
a row the job leaves the training and continues with the later stages, so a
fault in the training code does not end the GPU load. After an out-of-memory
error a trial starts again with half its batch.

Separate TrainJobs need separate `--output` directories, because every
single-GPU job is rank zero and takes the same rank lock. An output directory is
bound to its corpus, its models, the checksums of `workload.py`, `artifacts.py`
and `lora.py`, the package versions, the GPU model, the CUDA build and the
architecture. A change to one of them needs a new output directory.

### Stops and restarts

A job on borrowed GPUs can be preempted at any time. FLAME puts a preempted
workload back in the queue and restarts it when GPUs are free (FLAME preemption
guide). The restarted job verifies the finished chunks and builds only the
missing ones, skips finished trials, and resumes the interrupted trial from its
newest checkpoint. Do not resubmit a suspended job. Submit again to the same
output only after the earlier TrainJob has failed or been deleted, with the same
code, environment and deadline. A job admitted later than two minutes before
`--stop-at` ends without work and without an error.

On SIGTERM the wrapper gives the worker five seconds. A trial writes a checkpoint
as soon as its running step ends. That took half a second in the check on an
A6000. A trial also writes a checkpoint every 60 seconds, so a kill without that
checkpoint costs at most a minute of training. Each restart repeats the sweep
from its first case.

### Records

Each attempt writes `measurements-rank-<rank>-<attempt>.jsonl`, with one JSON
record per event of a stage, and `gpu-<node>-<attempt>.csv`, with utilization,
memory, power and temperature from `nvidia-smi` every five seconds. An attempt
that was inside a long library call when it was killed can end without a
`signal` or `finished` record. An attempt that failed during startup leaves a
file with only a `start` record.

```bash
"$PY" code/report.py results-a6000-0
```

`report.py` prints the restarts, the stages, the trials and the GPU utilization
of each stage from these files.

### First check on a new image or GPU type

Run one single-GPU job through `submit.py --execute` with `--case-seconds 5`,
`--max-trials 6`, an output directory of its own and a stop time 90 minutes
ahead. The first six trials cover the three parent models twice. Delete the pod
once during the build and once during a trial, and confirm with `report.py` that
the build skipped finished chunks and that the trial resumed at its checkpoint
step. Then run `search.py` on the parent index and on the adapted index.

## LoRA trials

Training pairs are publisher labels and definitions from the corpus. A record is
a candidate when its definition is English or untagged and has at least eight
words. Records that share an entity IRI, a definition or a label form one
component, where texts are compared by their letters and digits. A component lies
in one split (80% training, 10% validation, 10% test, by a hash of its lowest
record id), and a batch takes one record per component.
`python3 code/lora.py pairs --corpus corpus` prints the counts. The export of
2 October 2026 has 55,962 candidate records in 55,459 components, and ChEBI
supplies 53,356 of the records.

A trial attaches a LoRA adapter to the query and value projections of a pinned
parent model and trains it with an in-batch contrastive loss. The label is the
query, with the query instruction of the model, and the definition is the
passage, cut at 128 tokens. `trials.json` orders 216 trials, which vary the
parent model, the rank (8 to 64), the learning rate, the data (balanced or all)
and the seed. The comparison of rank 16 with rank 32 comes first. A balanced
epoch holds every materials component and a rotating share of ChEBI that is four
times as large. Every trial sees about 60,000 pairs.

Each trial directory under `trials/` ends with the adapter in PEFT format, a
`model.json` for index builds and a `result.json`. Validation retrieves the
definition of each held-out label among the held-out definitions and up to
20,000 training definitions (every materials definition, then ChEBI). It reports
recall, reciprocal rank and the mean similarity of the correct pair, for ChEBI
and for the materials sources separately. These pairs are weak supervision.

```bash
"$PY" code/lora.py summary results-a6000-0 results-a6000-1
"$PY" code/lora.py test --corpus corpus --out results-a6000-0 --trial <trial>
```

`summary` lists finished trials by the mean of the two group scores, which is
also the rule that picks the trial for the adapted index. The worker never reads
the test split. `test` evaluates one selected trial and its parent on it and
writes `test.json` beside the result.

## Search

The search runs on CPU and needs the pinned model cache.

```bash
"$PY" code/search.py --corpus corpus --index results-a6000-0 \
  --model bge-large --query "measuring surface topography at nanometre scale"
"$PY" code/search.py --corpus corpus --index results-a6000-0/adapted/<trial> \
  --model <trial> --queries code/queries.draft.json --out adapted-candidates.json
```

`search.py` refuses missing chunks, checksum failures and an index built for
another corpus. An adapted index holds a copy of its adapter, and the search
merges that adapter into the parent model before it encodes the query.
`--source chameo` restricts the results to one source. Every result keeps its
description id, publisher IRI, source, version, licence and original text.

Embedding similarity proposes related concepts. It does not establish ontology
equivalence, and nothing here writes a mapping back to an ontology or to curated
SAM content. The draft queries carry `relevant_ids: null`, which means unjudged.
A claim about search quality needs independently judged queries that cover each
source, records without definitions, ambiguous terms, exact identifiers and
queries without a suitable answer.

## Limits

One job ran on an RTX A6000 of FLAME on 5 October 2026, in the image
`pytorch:2026.07.2` (Python 3.12.3, torch 2.13.0a0 with CUDA 13.3). It built the
three indexes, trained trials in bfloat16 with the batch sizes of `trials.json`,
and resumed after two pod deletions, one in the build and one in a trial. A
preemption by Kueue, a job with several GPUs and the GH200 nodes are untested.
Whole-job throughput includes loading, tokenization, checkpoint writes and gaps,
so per-worker rates must not be summed and reported as cluster speedup.

## References

- [FLAME training runtimes](https://docs.flamecluster.io/reference/training-runtimes/)
- [FLAME batch guide](https://docs.flamecluster.io/guides/training-a-simple-vision-transformer/trainjob/)
- [FLAME preemption guide](https://docs.flamecluster.io/guides/designing-for-preemption/)
- [FLAME queueing reference](https://docs.flamecluster.io/reference/kueue/)
- [FLAME CLI access](https://docs.flamecluster.io/getting-started/cli-access/)
- [Kubeflow SDK API](https://sdk.kubeflow.org/en/stable/train/api.html)
- [Sentence Transformers PEFT training](https://www.sbert.net/examples/sentence_transformer/training/peft/README.html)
- [PEFT LoRA reference](https://huggingface.co/docs/peft/en/package_reference/lora)
- [BGE model card](https://huggingface.co/BAAI/bge-large-en-v1.5)
- [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
