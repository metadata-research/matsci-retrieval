"""Summarize the records of job outputs: environment, restarts, stages, trials and GPU telemetry.

The script reads the files that workload.py writes into an output directory and
needs no library beyond Python itself, so it runs in any environment.
"""
import argparse
import bisect
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import statistics

GIB = 2 ** 30
# On a GPU the sweep has 24 cases per model: two precisions, two input caps,
# three batch sizes and two record sets.
SWEEP_CASES = 72


def when(text):
    return datetime.fromisoformat(text).timestamp()


def clock(record):
    return record["time"][11:19]


def number(text):
    try:
        return float(text)
    except ValueError:
        return None  # nvidia-smi prints [N/A] for a value that the GPU does not report.


def parent(name):
    """The model name of a build record, or the parent model in a trial name such as bge-base-r16-lr1e-4-all-s1."""
    return re.sub(r"-r\d+-.*", "", name)


def load_records(out):
    records = []
    for path in sorted(out.glob("measurements-rank-*.jsonl")):
        for line in path.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except ValueError:
                pass  # A line that a kill cut off.
    return sorted(records, key=lambda record: record["time"])


def trial_line(trial, mine):
    first = next((r for r in mine if r.get("status") == "started"), mine[0])
    done = next((r for r in mine if r.get("status") == "complete"), None)
    steps = [r for r in mine if "loss" in r]
    saves = [r["seconds"] for r in mine if r.get("status") == "checkpoint"]
    resumed = [r["step"] for r in mine if r.get("status") == "resumed"]
    other = sorted({r["status"] for r in mine
                    if r.get("status") not in (None, "started", "complete", "checkpoint", "resumed")})
    line = [f"trial | {trial}", f"batch {first.get('batch_size')}", str(first.get("precision")), f"steps {first.get('steps')}"]
    if steps:
        line += [f"median {statistics.median(r['pairs_per_second'] for r in steps):.0f} pairs/s",
                 f"peak {max(r['peak_allocated_bytes'] for r in steps) / GIB:.1f} GiB"]
    if saves:
        line.append(f"checkpoints {len(saves)}, slowest {max(saves):.2f} s")
    if resumed:
        line.append(f"resumed at step {resumed}")
    if done:
        # The span includes the evaluations and any restart in between.
        line += [f"{when(done['time']) - when(first['time']):.0f} s from start to result",
                 f"mrr {done['baseline_mrr']:.3f} -> {done['final_mrr']:.3f}"]
    else:
        line.append("NO RESULT")
    if other:
        line.append(f"other records: {other}")
    return " | ".join(line)


def telemetry_lines(out, records):
    """GPU utilization by stage. A sample belongs to the stage of the last record before it."""
    times = [when(r["time"]) for r in records]

    def phase(moment):
        position = bisect.bisect_right(times, moment) - 1
        r = records[position] if position >= 0 else {"stage": "none"}
        if r["stage"] in ("train", "eval") and (r.get("trial") or r.get("model")):
            return "train " + parent(r.get("trial") or r["model"])
        if r["stage"] in ("build", "adapted", "sweep", "sustained") and "model" in r:
            return f"{r['stage']} {parent(r['model'])}"
        return "loading and between stages"

    samples, first_sample = {}, None
    for path in sorted(out.glob("gpu-*.csv")):
        with path.open(newline="") as handle:
            for row in csv.reader(handle, skipinitialspace=True):
                try:
                    # nvidia-smi writes the local time of the pod. The samples are read as UTC,
                    # and the first telemetry line shows whether that fits the records.
                    moment = datetime.strptime(row[0], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
                    values = [number(row[column]) for column in (3, 5, 6, 7, 8)]
                except (ValueError, IndexError):
                    continue  # The header, or a line that a kill cut off.
                if values[0] is None:
                    continue
                first_sample = first_sample or moment
                samples.setdefault(phase(moment), []).append(values)
    if samples and times:
        # A large value means that the clock of the samples is not UTC, and the stages below are then wrong.
        yield f"gpu | first sample {first_sample - times[0]:+.0f} s after the first record"
    for name, rows in samples.items():
        use = [values[0] for values in rows]
        memory = [values[1] for values in rows if values[1] is not None]
        power = [values[3] for values in rows if values[3] is not None]
        heat = [values[4] for values in rows if values[4] is not None]
        line = [f"gpu | {name}", f"{len(rows) * 5 / 60:.1f} min", f"utilization mean {statistics.mean(use):.0f}%",
                f"at least 90% in {100 * sum(value >= 90 for value in use) / len(use):.0f}% of samples"]
        if memory:
            line.append(f"memory max {max(memory) / 1024:.1f} GiB")
        if power:
            line.append(f"power mean {statistics.mean(power):.0f} W")
        if heat:
            line.append(f"max {max(heat):.0f} C")
        yield " | ".join(line)


def lines(out):
    """The summary of one output directory, one line per fact."""
    out = Path(out)
    environments = [json.loads(path.read_text()) for path in sorted(out.glob("environment-rank-*.json"))]
    if environments:
        e = environments[-1]
        yield " | ".join(["environment", e["architecture"], f"python {e['python']}", f"torch {e['packages']['torch']}",
                          f"cuda {e['cuda']}", f"{e['gpu']} {e['gpu_bytes'] / GIB:.1f} GiB",
                          f"numpy {e['packages']['numpy']}", f"attempts {len(environments)}"])
    records = load_records(out)
    for r in records:
        if r["stage"] in ("start", "signal", "finished"):
            yield f"{r['stage']} | {clock(r)} | {r.get('status') or r.get('signal') or 'stop at ' + str(r.get('stop_at'))}"
        elif r["stage"] == "resume":
            yield f"resume | {clock(r)} | " + ", ".join(
                f"{model} {count['present']}/{count['assigned']}" for model, count in r["chunks"].items())
    for stage in ("build", "adapted"):
        for model in dict.fromkeys(r.get("model") for r in records if r["stage"] == stage and "rows_per_second" in r):
            rows = [r for r in records if r["stage"] == stage and r.get("model") == model and "rows_per_second" in r]
            yield (f"{stage} | {model} | chunks {len(rows)} | batch {rows[-1]['batch_size']} | median "
                   f"{statistics.median(r['rows_per_second'] for r in rows):.0f} rows/s | peak "
                   f"{max(r['peak_allocated_bytes'] for r in rows) / GIB:.1f} GiB | {clock(rows[0])} to {clock(rows[-1])}")
    for r in records:
        if r["stage"] == "adapted" and r.get("status"):
            yield f"adapted | {r['status']} | {r.get('trial', '')} | {clock(r)} | {str(r.get('score', r.get('error', '')))[:120]}"
    for trial in dict.fromkeys(r["trial"] for r in records if r["stage"] == "train" and "trial" in r):
        yield trial_line(trial, [r for r in records if r["stage"] == "train" and r.get("trial") == trial])
    for r in records:
        if r["stage"] == "train" and r.get("status") == "queue ended":
            yield f"queue ended | {clock(r)} | {r.get('reason')} | finished {r.get('finished')}"
        if r.get("status") in ("failed", "oom", "abandoned") or r["stage"] == "telemetry":
            yield "attention | " + json.dumps(r)[:300]
    calls = [r for r in records if r["stage"] == "sweep" and "rows_per_second" in r]
    measured = [r for r in calls if not r.get("warmup")]
    if calls:
        def case(r):
            return r["model"], r["precision"], r["max_length"], r["distribution"], r["batch_size"]
        yield (f"sweep | cases started {len({case(r) for r in calls})} of {SWEEP_CASES} | cases with a measurement "
               f"after the warmup {len({case(r) for r in measured})} | {clock(calls[0])} to {clock(calls[-1])}")
        for key in dict.fromkeys((r["model"], r["precision"]) for r in measured):
            best = max((r for r in measured if (r["model"], r["precision"]) == key), key=lambda r: r["rows_per_second"])
            yield (f"sweep | {key[0]} {key[1]} | best {best['rows_per_second']:.0f} rows/s at batch {best['batch_size']}, "
                   f"length {best['max_length']}, {best['distribution']} | peak {best['peak_allocated_bytes'] / GIB:.1f} GiB")
    sustained = [r for r in records if r["stage"] == "sustained" and "rows_per_second" in r]
    if sustained:
        yield (f"sustained | calls {len(sustained)} | median {statistics.median(r['rows_per_second'] for r in sustained):.0f}"
               f" rows/s | {clock(sustained[0])} to {clock(sustained[-1])}")
    yield from telemetry_lines(out, records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outputs", type=Path, nargs="+", help="Output directories of jobs")
    args = parser.parse_args()
    for out in args.outputs:
        print(f"== {out}")
        for line in lines(out):
            print(line)


if __name__ == "__main__":
    main()
