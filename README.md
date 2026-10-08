# MatSci retrieval experiments

This experiment trains LoRA adapters for three
embedding models and runs as batch jobs on FLAME. Each job keeps its GPU loaded
until a stated stop time, so the same jobs supply the load for the preemption
test.

## The experiment

The corpus is an export from the MatSci-ONT store with 222,925 descriptions of
ontology terms. A training pair is the label of a term with its definition. The
pairs come from the 55,962 records with an English or untagged definition of at
least eight words, and ChEBI supplies 53,356 of them.

A trial trains one LoRA adapter on a frozen parent model, which is MiniLM, BGE
Base or BGE Large. The label is the query and the definition is the passage, and
a contrastive loss moves each label toward its own definition. `trials.json`
orders 216 trials over the parent model, the rank of the adapter (8 to 64), the
learning rate, the data mix and the seed.

Validation ranks the correct definition of each held-out label among the
held-out definitions and up to 20,000 training definitions. It reports recall
and reciprocal rank for ChEBI and for the materials sources separately.

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
for the time that remains. After three failed trials in a row a job moves on to
the later stages, so a fault in the training does not leave a GPU idle.

## What a job requests

| Item | Value |
| --- | --- |
| Object | One TrainJob per GPU, created by `submit.py` through the Kubeflow SDK |
| Runtime | `torch-rtxa6000` |
| Resources | 1 GPU, 1 CPU, 32Gi of memory |
| Storage | The home volume, about 25 GiB for five jobs (estimate) |
| Network | None |
| End | The worker exits about one minute before `--stop-at` |

## Stops and restarts

On SIGTERM a trial writes a checkpoint as soon as its running step ends, and the
wrapper allows the worker five seconds to exit. A trial also writes a checkpoint
every 60 seconds.

A restarted job skips the finished parts of an index and the finished trials,
and it resumes the interrupted trial at its newest checkpoint. A job readmitted
less than two minutes before the stop time ends as Complete without work.

A suspended job returns by itself and must not be resubmitted.

## Submit jobs

In a shell opened with `ssh flame`:

```bash
cd /home/ubuntu/matsci-gpu-test
bash code/run-1008.sh preview
bash code/run-1008.sh monitor
# At the start of the run:
bash code/run-1008.sh start
bash code/run-1008.sh status
```

`preview` prints the five job plans; `monitor` records cluster activity.
`start` submits the jobs from 15:00 Berlin time, with one CPU and one GPU each,
assignments `i/5`, at most 30 trials, and an 18:00 deadline. `status` shows
their progress. Logs are saved in `logs/1008/`.

## Evaluation after a run

```bash
OUT=results-1008-0    # the output directory of the selected trial
TRIAL=                # the name of the selected trial
PY="$PWD/.venv-x86_64/bin/python"
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
| `run-1008.sh` | Preview, monitoring, launch and status for the five-job run |
| `report.py` | Summary of the records and the GPU telemetry of a job |
| `search.py` | Exact cosine search over a complete index on CPU |
| `prepare.py` | Model download and a training check on CPU |
| `bundle.py` | Packs the code and a corpus export for FLAME |
| `sample.py` | Small corpus for a direct run of `workload.py` |
| `queries.draft.json` | Draft queries for `search.py` |
| `test_*.py` | Tests, run with `python3 -m unittest discover -p "test_*.py"` |

## References

- [FLAME preemption guide](https://docs.flamecluster.io/guides/designing-for-preemption/)
- [FLAME training runtimes](https://docs.flamecluster.io/reference/training-runtimes/)
- [Kubeflow SDK API](https://sdk.kubeflow.org/en/stable/train/api.html)
- [Sentence Transformers PEFT training](https://www.sbert.net/examples/sentence_transformer/training/peft/README.html)
- [PEFT LoRA reference](https://huggingface.co/docs/peft/en/package_reference/lora)
- [BGE model card](https://huggingface.co/BAAI/bge-large-en-v1.5)
- [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
