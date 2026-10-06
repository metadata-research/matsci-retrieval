# MatSci retrieval experiments

This repository holds a GPU experiment on semantic search over the ontology
descriptions that MatSci-ONT publishes. The experiment trains LoRA adapters for
three embedding models and measures whether an adapted model finds the
definition of a term more often than its parent model. It runs as batch jobs on
the FLAME cluster at Drexel and keeps each GPU busy until a stated stop time, so
the same jobs supply the load for the preemption test of 8 October 2026.

## What the experiment does

The corpus is an export from the MatSci-ONT store with 222,925 descriptions of
ontology terms (export of 2 October 2026). A description holds the label of a
term, and about a quarter of the descriptions also hold a definition. A
training pair is a label with its definition. The pairs come from the 55,962
records with an English or untagged definition of at least eight words, and
ChEBI supplies 53,356 of them. Records that share a term, a label or a
definition stay together in one of three splits (80% training, 10% validation,
10% test).

The parent models are MiniLM (`all-MiniLM-L6-v2`), BGE Base
(`bge-base-en-v1.5`) and BGE Large (`bge-large-en-v1.5`), each pinned to one
revision in `models.json`. A LoRA adapter (low-rank adaptation) is a small set
of weights added to a parent model whose own weights stay frozen. A trial trains
one adapter. The label is the query and the definition is the passage, and an
in-batch contrastive loss moves a label toward its definition and away from the
other definitions in the batch.

`trials.json` orders 216 trials, which vary the parent model, the rank of the
adapter (8 to 64), the learning rate, the data (a balanced mix or all pairs) and
the seed. Every trial sees about 60,000 pairs. The comparison of rank 16 with
rank 32 comes first in the queue.

Validation retrieves the definition of each held-out label among the held-out
definitions and up to 20,000 training definitions. It reports recall and
reciprocal rank for ChEBI and for the materials sources separately. The test
split stays unread until one trial has been selected.

These pairs are weak supervision. A model that matches a label to its own
definition has not been shown to answer the queries that a user types, and a
claim about search quality needs independently judged queries. Embedding
similarity proposes related concepts and establishes no ontology equivalence.
Nothing here writes a mapping back to an ontology or to curated SAM content.

## What a job does

Each job is one process on one GPU. It works through five stages and stops
taking work one minute before its stop time.

| Stage | Work | Result |
| --- | --- | --- |
| 1 | Encode the whole corpus with each parent model | Three vector indexes, 1.81 GiB together |
| 2 | Run LoRA trials from the queue in order | One adapter with validation metrics per trial |
| 3 | Encode the corpus with the adapter of the best finished trial | One adapted index, 0.3 to 0.9 GiB |
| 4 | Sweep precision, input length and batch size, 45 seconds per case | Throughput measurements |
| 5 | Encode with BGE Large until the stop time | GPU load. The vectors are discarded |

Stages 1 to 3 produce the results for MatSci. Stages 4 and 5 fill the time that
remains, so a GPU stays loaded when the training ends early. A trial that fails
is recorded and the next trial starts. After three failures in a row the job
leaves the training and continues with the later stages. No trial starts in the
last 16 minutes before the stop time, which leaves time for stage 3.

## What a job asks of the cluster

| Item | Value |
| --- | --- |
| Object | One Kubeflow TrainJob per GPU, created by `submit.py` through the Kubeflow SDK |
| Runtime | `torch-rtxa6000` with its own image |
| Resources | 1 node, 1 GPU, 4 CPUs, 32Gi of memory |
| Queue | Local queue `default`, cluster queue `metadata-research-center`, cohort `gpu-pool` |
| Storage | The home volume, which a job sees at `/personal`. About 20 GiB for four jobs (estimate) |
| Software | A Python environment on the home volume, on top of the packages of the image |
| Network | None. The job runs offline and reads the models from the cache on the home volume |
| End | The worker exits by itself about one minute before `--stop-at` |

The cluster queue had a nominal quota of 0 for both GPU types on 4 October 2026,
so every GPU of a job is borrowed and a job can be preempted at any time.

One job ran on an A6000 on 5 October 2026 and gave these figures.

| Stage | GPU utilization, mean | GPU memory, maximum |
| --- | --- | --- |
| Index build, MiniLM | 47% | 1.3 GiB |
| Index build, BGE Base | 80% | 2.6 GiB |
| Index build, BGE Large | 92% | 3.2 GiB |
| Trial, MiniLM | 78% | 9.3 GiB |
| Trial, BGE Base | 93% | 15.9 GiB |
| Trial, BGE Large | 98% | 33.9 GiB |
| Adapted index, BGE Large | 89% | 2.7 GiB |
| Sweep | 30% to 82% | 21.0 GiB |
| Sustained encoding | 83% | 4.5 GiB |

The memory column is the value of `nvidia-smi` on a GPU with 47.4 GiB, and it
includes memory that PyTorch holds in reserve. The mean power draw was 232 to
294 W in the builds, the trials and the sustained encoding. The GPU reached 90
to 91 C in the build of the BGE Large index and in the trials, and at most 73 C
from the adapted index to the end at a similar power draw. Each pod deletion
started a new pod, and whether the last pod ran on another GPU is open.

## Stops and restarts

A preempted job receives SIGTERM. The wrapper gives the worker five seconds, and
a trial writes a checkpoint as soon as its running step ends, which took half a
second on the A6000. A trial also writes a checkpoint every 60 seconds, so a
kill without SIGTERM costs at most a minute of training.

FLAME puts a preempted workload back in the queue and restarts it when GPUs are
free (FLAME preemption guide). The restarted job verifies the finished chunks of
an index and builds only the missing ones, skips finished trials, and resumes
the interrupted trial at its newest checkpoint. The sweep starts again at its
first case. A job admitted later than two minutes before the stop time ends as
Complete without work.

Do not resubmit a suspended job. It returns by itself, and a second job on the
same output directory fails on a lock.

In the run of 5 October the pod was deleted by hand twice, once in an index
build and once in a trial. The job carried a grace period of 30 seconds, the
restart policy `OnFailure` and a `backoffLimit` of 6. The next pod wrote its
first record 23 and 26 seconds after SIGTERM, and the job ended as Complete.
Each deletion counted as one failure of the Job, so a job survives six stops of
that kind. A suspension by Kueue removes the pods through the job controller,
which does not count the pods that it deletes itself (Kubernetes source,
untested on FLAME).

## The run on 8 October 2026

The window is 9:00 to 12:00 in Philadelphia, which is 15:00 to 18:00 Central
European Summer Time and 13:00 to 16:00 UTC. Four jobs run side by side on the
A6000 node, each with one GPU and its own quarter of the trial queue. Four
single-GPU jobs run because one failed rank stops all ranks of a multi-GPU job
within seconds. Other runs use the same commands with another stop time and
other output names.

A job that is never stopped is expected to follow this course (calculated from
the run of 5 October, where every trial had rank 16).

| Minutes after the start | Work |
| --- | --- |
| 0 to 11 | Start of the pod and the three index builds |
| 11 to about 164 | 54 trials, 18 per parent model |
| About 164 to 179 | Adapted index (about 6 minutes), then the sweep |

The sustained encoding is not reached in that case. A job that is preempted and
resumed finishes fewer trials.

All commands run in `/home/ubuntu/matsci-gpu-test`, in a shell opened with
`ssh flame`. A job sees that directory as `/personal/matsci-gpu-test`. It holds
the code in `code/`, the corpus in `corpus/` and the Python environment
`.venv-x86_64`, which are in place for this run. The steps that set them up are
recorded in the internal documentation of MatSci-ONT
(`docs-internal/FLAME-PREPARATION.md`).

### Before the start

```bash
export KUBERNETES_SERVICE_HOST=kubernetes.default.svc KUBERNETES_SERVICE_PORT=443
PY="$PWD/.venv-x86_64/bin/python"
NS=$(cat /var/run/secrets/kubernetes.io/serviceaccount/namespace)
export WATCH_END=$(date -d '2026-10-08T18:15:00+02:00' +%s)
nohup bash -c 'while [ "$(date +%s)" -lt "$WATCH_END" ]; do date -u "+== %FT%TZ"; kubectl get workloads,trainjobs,pods; kubectl get jobs -o custom-columns=NAME:.metadata.name,ACTIVE:.status.active,FAILED:.status.failed,SUCCEEDED:.status.succeeded; sleep 30; done' >> watch-state.txt 2>&1 &
nohup bash -c 'while [ "$(date +%s)" -lt "$WATCH_END" ]; do timeout "$((WATCH_END - $(date +%s) + 5))" kubectl get events --watch -o custom-columns=CREATED:.metadata.creationTimestamp,LAST:.lastTimestamp,REASON:.reason,OBJECT:.involvedObject.name,MESSAGE:.message; sleep 5; done' >> watch-events.txt 2>&1 &
jobs    # both loops must show as Running
kubectl get workloads
```

A shell from `ssh flame` lands in the notebook pod without the two variables
that locate the cluster API, and the first line sets them for `kubectl`. The two
loops record the state of the workloads, the jobs and the pods every 30 seconds
and the events of the namespace, until 18:15. They hold the times at which Kueue
suspends and readmits a job, so they start before the jobs. Whether the notebook
account may list events is unchecked, and `watch-events.txt` shows the refusal
if it may not. The last line should show the notebook as the only unfinished
workload of the workspace.

### At the start

```bash
set -o pipefail
for i in 0 1 2 3; do
  "$PY" code/submit.py --root /personal/matsci-gpu-test \
    --python /personal/matsci-gpu-test/.venv-x86_64/bin/python \
    --runtime torch-rtxa6000 --gpus-per-node 1 --nodes 1 \
    --cpus-per-node 4 --memory-per-node 32Gi --namespace "$NS" \
    --output "results-1008-$i" --trial-shard "$i/4" \
    --stop-at 2026-10-08T18:00:00+02:00 \
    --execute | tee -a submitted.txt || break
done
```

`submit.py` prints a submission and sends nothing unless `--execute` is given.
`--trial-shard i/4` gives a job the trials whose position modulo 4 equals `i`,
so the four jobs share the queue without overlap. Each job needs an output
directory of its own. `submitted.txt` links each TrainJob name to its output
directory. The A6000 node has 31 CPUs for eight GPUs, so a job asks for four,
and the worker limits torch to four threads.

### During the window

```bash
kubectl get workloads,pods
tail -q -n 1 results-1008-*/measurements-rank-0-*.jsonl | cut -c1-200
tail -q -n 1 results-1008-*/gpu-*.csv
```

About a minute after the submission these lines show four admitted workloads,
four running pods, and the stage and last GPU sample of each job. The `STATE`
column of a TrainJob showed `Suspended` while its pod ran on 5 October, so the
workload and the pod are the places to read the state of a job.

The jobs run unattended, and a dropped SSH session does not stop them. Leave the
jobs, the code and the environment unchanged until the stop time. An output
directory is bound to its corpus, its models, its code and its environment, and
a change to one of them needs a new output directory.

### After the stop time

```bash
kubectl get trainjobs,workloads,pods
kubectl get jobs -o custom-columns=NAME:.metadata.name,BACKOFF:.spec.backoffLimit,FAILED:.status.failed,SUCCEEDED:.status.succeeded
kubectl get trainjob,jobset,job,pod,workloads -o yaml > "results-1008-specs-$(date -u +%H%M%S).yaml"
"$PY" code/report.py results-1008-0 results-1008-1 results-1008-2 results-1008-3 | tee report-1008.txt
"$PY" code/lora.py summary results-1008-0 results-1008-1 results-1008-2 results-1008-3 | tee summary-1008.txt
```

These lines show how each job ended and save the specifications together with
the workloads, whose conditions record the evictions (Kueue documentation, form
on FLAME unseen). They run before any job is deleted. Afterwards, delete every
job of the run that is not Complete, and keep the files.

## Records

| File | Content |
| --- | --- |
| `results-*/measurements-rank-<rank>-<attempt>.jsonl` | One JSON record per event of a stage, per attempt of a job |
| `results-*/gpu-<node>-<attempt>.csv` | Utilization, memory, power and temperature from `nvidia-smi` every five seconds |
| `results-*/trials/<trial>/` | The adapter in PEFT format and a `result.json` with the validation metrics |
| `watch-state.txt`, `watch-events.txt` | Cluster state and events from the two loops |
| `submitted.txt` | TrainJob name and output directory of each job |
| `results-1008-specs-*.yaml` | Specifications and workloads at the end |
| `report-1008.txt`, `summary-1008.txt` | Output of `report.py` and of `lora.py summary` |

`report.py` prints the restarts, the stages, the trials and the GPU utilization
of each stage. `lora.py summary` lists finished trials by the mean of the ChEBI
score and the materials score, which is also the rule that picks the trial for
the adapted index. An attempt that was inside a long library call when it was
killed can end without a `signal` or `finished` record.

## Evaluation after a run

```bash
OUT=results-1008-0    # the output directory of the selected trial
TRIAL=                # the name of the selected trial
"$PY" code/lora.py test --corpus corpus --out "$OUT" --trial "$TRIAL"
ADAPTED=$(ls "$OUT/adapted" | head -n 1)
"$PY" code/search.py --corpus corpus --index "$OUT/adapted/$ADAPTED" \
  --model "$ADAPTED" --queries code/queries.draft.json --out adapted-candidates.json
```

`lora.py test` evaluates one selected trial and its parent on the test split and
writes `test.json` beside the result. It runs once, for the trial that the
summary ranks first. `search.py` runs on CPU over a complete index and merges
the adapter into the parent model before it encodes a query. The draft queries
are unjudged.

## Files

| File | Purpose |
| --- | --- |
| `models.json` | Pinned revisions of MiniLM, BGE Base and BGE Large |
| `trials.json` | Ordered queue of 216 LoRA trials with their default settings |
| `workload.py` | The GPU job with its five stages |
| `lora.py` | Training pairs, trials, checkpoints, adapter export, summary and test split |
| `artifacts.py` | Corpus loading, vector chunks, receipts and checksums |
| `submit.py` | Preview of a Kubeflow TrainJob, submitted with `--execute` |
| `report.py` | Summary of the records and the GPU telemetry of a job |
| `search.py` | Exact cosine search over a complete index on CPU |
| `prepare.py` | Model download, token statistics and a training check on CPU |
| `bundle.py` | Packs the code and a corpus export for FLAME |
| `sample.py` | Small source-balanced corpus for a direct run of `workload.py` |
| `queries.draft.json` | Draft queries for `search.py` |
| `test_*.py` | Tests, run with `python3 -m unittest discover -p "test_*.py"` |

## Limits

One job has run on FLAME, on one RTX A6000, in the image `pytorch:2026.07.2`
(Python 3.12.3, torch 2.13.0a0 with CUDA 13.3). Four jobs side by side on the
node, a preemption by Kueue with its suspension and requeue, a job with several
GPUs and the GH200 nodes are untested. Whole-job throughput includes loading,
tokenization, checkpoint writes and gaps, so per-worker rates must not be summed
and reported as cluster speedup.

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
