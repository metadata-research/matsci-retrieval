# MatSci retrieval experiments

This repository holds a GPU experiment on semantic search over the ontology
descriptions that MatSci-ONT publishes. It trains LoRA adapters for three
embedding models and runs as batch jobs on FLAME. Each job keeps its GPU loaded
until a stated stop time, so the same jobs supply the load for the preemption
test of 8 October 2026.

## The experiment

The corpus is an export from the MatSci-ONT store with 222,925 descriptions of
ontology terms. A training pair is the label of a term with its definition. The
pairs come from the 55,962 records with a definition of at least eight words,
and ChEBI supplies 53,356 of them.

A trial trains one LoRA adapter on a frozen parent model, which is MiniLM, BGE
Base or BGE Large. The label is the query and the definition is the passage, and
a contrastive loss moves each label toward its own definition. `trials.json`
orders 216 trials over the parent model, the rank of the adapter (8 to 64), the
learning rate, the data mix and the seed.

Validation ranks the correct definition of each held-out label among the other
definitions. It reports recall and reciprocal rank for ChEBI and for the
materials sources separately.

## What a job does

Each job is one process on one GPU and works through five stages.

| Stage | Work | Result |
| --- | --- | --- |
| 1 | Encode the whole corpus with each parent model | Three vector indexes, 1.81 GiB together |
| 2 | Run LoRA trials from the queue in order | One adapter with validation metrics per trial |
| 3 | Encode the corpus with the adapter of the best finished trial | One adapted index, 0.3 to 0.9 GiB |
| 4 | Sweep precision, input length and batch size | Throughput measurements |
| 5 | Encode with BGE Large until the stop time | GPU load only |

Stages 1 to 3 produce the results for MatSci. Stages 4 and 5 keep the GPU loaded
for the time that remains. After three failed trials in a row a job skips to
stage 3, so a fault in the training does not leave a GPU idle.

## What a job requests

| Item | Value |
| --- | --- |
| Object | One TrainJob per GPU, created by `submit.py` through the Kubeflow SDK |
| Runtime | `torch-rtxa6000` |
| Resources | 1 GPU, 4 CPUs, 32Gi of memory |
| Storage | The home volume, about 20 GiB for four jobs (estimate) |
| Network | None |
| End | The worker exits about one minute before `--stop-at` |

In a trial of BGE Large the mean GPU utilization was 98% and the GPU memory in
use reached 33.9 GiB.

## Stops and restarts

On SIGTERM a trial writes a checkpoint as soon as its running step ends, and the
worker exits within five seconds. A trial also writes a checkpoint every 60
seconds.

A restarted job skips the finished parts of an index and the finished trials,
and it resumes the interrupted trial at its newest checkpoint. A job readmitted
less than two minutes before the stop time ends as Complete without work.

A suspended job returns by itself and must not be resubmitted.

## The run on 8 October 2026

The window is 9:00 to 12:00 in Philadelphia (15:00 to 18:00 Central European
Summer Time). Four single-GPU jobs run side by side on the A6000 node, each with
its own quarter of the trial queue and its own output directory.

A job that is never stopped builds its indexes in about 11 minutes and then
trains until 16 minutes before the stop time. The adapted index and a part of
the sweep fill the rest.

All commands run in `/home/ubuntu/matsci-gpu-test`, in a shell opened with
`ssh flame`. The setup of that directory is recorded in
`docs-internal/FLAME-PREPARATION.md` of MatSci-ONT.

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

The two loops record the state of the workloads, the jobs and the pods every 30
seconds, together with the events, until 18:15.

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
`submitted.txt` links each TrainJob name to its output directory.

### During the window

```bash
kubectl get workloads,pods
tail -q -n 1 results-1008-*/measurements-rank-0-*.jsonl | cut -c1-200
tail -q -n 1 results-1008-*/gpu-*.csv
```

These lines show the admitted workloads, the pods, and the stage and last GPU
sample of each job. The jobs run unattended. The code and the environment stay
unchanged until the stop time.

### After the stop time

```bash
kubectl get trainjobs,workloads,pods
kubectl get jobs -o custom-columns=NAME:.metadata.name,BACKOFF:.spec.backoffLimit,FAILED:.status.failed,SUCCEEDED:.status.succeeded
kubectl get trainjob,jobset,job,pod,workloads -o yaml > "results-1008-specs-$(date -u +%H%M%S).yaml"
"$PY" code/report.py results-1008-0 results-1008-1 results-1008-2 results-1008-3 | tee report-1008.txt
"$PY" code/lora.py summary results-1008-0 results-1008-1 results-1008-2 results-1008-3 | tee summary-1008.txt
```

These lines run before any job is deleted. They save the final state and print
the two summaries.

## Records

| File | Content |
| --- | --- |
| `results-*/measurements-rank-<rank>-<attempt>.jsonl` | One JSON record per event of a stage |
| `results-*/gpu-<node>-<attempt>.csv` | Utilization, memory, power and temperature every five seconds |
| `results-*/trials/<trial>/` | The adapter and a `result.json` with the validation metrics |
| `watch-state.txt`, `watch-events.txt` | Cluster state and events from the two loops |
| `report-1008.txt`, `summary-1008.txt` | Output of `report.py` and of `lora.py summary` |

`report.py` prints the restarts, the stages, the trials and the GPU utilization
of each stage. `lora.py summary` ranks the finished trials.

## Evaluation after a run

```bash
OUT=results-1008-0    # the output directory of the selected trial
TRIAL=                # the name of the selected trial
"$PY" code/lora.py test --corpus corpus --out "$OUT" --trial "$TRIAL"
ADAPTED=$(ls "$OUT/adapted" | head -n 1)
"$PY" code/search.py --corpus corpus --index "$OUT/adapted/$ADAPTED" \
  --model "$ADAPTED" --queries code/queries.draft.json --out adapted-candidates.json
```

`lora.py test` evaluates the selected trial and its parent on the test split,
which stays unread until then. `search.py` runs draft queries against the
adapted index on CPU.

## Files

| File | Purpose |
| --- | --- |
| `models.json` | Pinned revisions of MiniLM, BGE Base and BGE Large |
| `trials.json` | Ordered queue of 216 LoRA trials |
| `workload.py` | The GPU job with its five stages |
| `lora.py` | Training pairs, trials, checkpoints, summary and test split |
| `artifacts.py` | Corpus loading, vector chunks and checksums |
| `submit.py` | Preview of a TrainJob, submitted with `--execute` |
| `report.py` | Summary of the records and the GPU telemetry of a job |
| `search.py` | Exact cosine search over a complete index on CPU |
| `prepare.py` | Model download and a training check on CPU |
| `bundle.py` | Packs the code and a corpus export for FLAME |
| `sample.py` | Small corpus for a direct run of `workload.py` |
| `queries.draft.json` | Draft queries for `search.py` |
| `test_*.py` | Tests, run with `python3 -m unittest discover -p "test_*.py"` |

## Limits

The training pairs are weak supervision, and a claim about search quality needs
independently judged queries. Nothing here writes a mapping back to an ontology
or to curated SAM content.

A preemption by Kueue, four jobs side by side on one node, a job with several
GPUs and the GH200 nodes are untested.

## References

- [FLAME preemption guide](https://docs.flamecluster.io/guides/designing-for-preemption/)
- [FLAME training runtimes](https://docs.flamecluster.io/reference/training-runtimes/)
- [Kubeflow SDK API](https://sdk.kubeflow.org/en/stable/train/api.html)
- [Sentence Transformers PEFT training](https://www.sbert.net/examples/sentence_transformer/training/peft/README.html)
- [PEFT LoRA reference](https://huggingface.co/docs/peft/en/package_reference/lora)
- [BGE model card](https://huggingface.co/BAAI/bge-large-en-v1.5)
- [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
