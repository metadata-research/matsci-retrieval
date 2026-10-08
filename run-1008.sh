#!/usr/bin/env bash
# Commands for the 8 October FLAME run. With no argument, only preview jobs.
set -euo pipefail

ROOT=${FLAME_TEST_ROOT:-/home/ubuntu/matsci-gpu-test}
START_AT=2026-10-08T15:00:00+02:00
STOP_AT=2026-10-08T18:00:00+02:00
WATCH_UNTIL=2026-10-08T18:15:00+02:00
ACTION=${1:-preview}
SCRIPT=$(realpath "${BASH_SOURCE[0]}")

fail() { printf '%s\n' "$*" >&2; exit 1; }

if [[ "$ACTION" == help || "$ACTION" == --help ]]; then
  cat <<'HELP'
Usage: bash code/run-1008.sh [command]
  preview  Print the five job plans; submit nothing (the default).
  monitor  Record cluster status and events until 18:15 Berlin time.
  start    Start monitoring and submit five jobs, from 15:00 Berlin time.
  status   Show current workloads, jobs, pods and latest worker progress.
  report   Save final cluster specifications and summarize all five outputs.
Logs and reports are saved in logs/1008/. Each job writes to results-1008-i/.
HELP
  exit 0
fi

case "$ACTION" in
  preview|monitor|start|status|report|_record-state|_record-events) ;;
  *) fail "Unknown command: $ACTION. Run with help for the available commands." ;;
esac

cd "$ROOT"
PY="$ROOT/.venv-x86_64/bin/python"
NS=${FLAME_NAMESPACE:-$(cat /var/run/secrets/kubernetes.io/serviceaccount/namespace)}
LOGS="$ROOT/logs/1008"
export FLAME_TEST_ROOT="$ROOT" FLAME_NAMESPACE="$NS"
export KUBERNETES_SERVICE_HOST=kubernetes.default.svc KUBERNETES_SERVICE_PORT=443

# All five plans use the same resources; only the folder and assignment change.
submit_job() {
  local i=$1
  shift
  "$PY" code/submit.py \
    --root "$ROOT" --python "$PY" \
    --runtime torch-rtxa6000 --gpus-per-node 1 --nodes 1 \
    --cpus-per-node 1 --memory-per-node 32Gi --namespace "$NS" \
    --output "results-1008-$i" --trial-shard "$i/5" --max-trials 30 \
    --stop-at "$STOP_AT" "$@"
}

check_files() {
  [[ -x "$PY" ]] || fail "Prepared Python not found: $PY"
  [[ -f corpus/corpus.jsonl && -f corpus/manifest.json ]] || fail "Corpus files are missing."
}

# Each recorder holds a lock while running, so repeating monitor is harmless.
monitor() {
  (( $(date +%s) < $(date -d "$WATCH_UNTIL" +%s) )) || fail "The recording window has ended."
  mkdir -p "$LOGS"
  for kind in state events; do
    if flock -n "$LOGS/.$kind.lock" true; then
      nohup bash "$SCRIPT" "_record-$kind" >> "$LOGS/watch-$kind.txt" 2>&1 </dev/null &
    fi
  done
  printf 'Recorders run until 18:15 Berlin time. Logs: %s\n' "$LOGS"
}

case "$ACTION" in
  preview)
    check_files
    for i in 0 1 2 3 4; do submit_job "$i"; done
    ;;
  monitor)
    monitor
    ;;
  start)
    check_files
    (( $(date +%s) >= $(date -d "$START_AT" +%s) )) || fail "The run starts at 15:00 Berlin time. Use preview until then."
    [[ ! -e "$LOGS/submitted.txt" ]] || fail "A launch was already attempted. Use status and inspect logs/1008/submitted.txt; do not resubmit suspended jobs."
    for i in 0 1 2 3 4; do
      [[ ! -e "results-1008-$i" ]] || fail "Output already exists: results-1008-$i. Use status before launching again."
      submit_job "$i" >/dev/null
    done
    monitor
    # Create the launch log exclusively to prevent a second launch racing us.
    (set -o noclobber; : > "$LOGS/submitted.txt") || fail "Another launch has already started."
    for i in 0 1 2 3 4; do
      if ! submit_job "$i" --execute | tee -a "$LOGS/submitted.txt"; then
        fail "Submission stopped at assignment $i/5. Inspect status and the submission log before any retry."
      fi
    done
    ;;
  status)
    kubectl -n "$NS" get trainjobs,workloads,pods
    for kind in state events; do
      if [[ -e "$LOGS/.$kind.lock" ]] && ! flock -n "$LOGS/.$kind.lock" true; then
        printf '%s recorder: running\n' "$kind"
      else
        printf '%s recorder: stopped\n' "$kind"
      fi
    done
    shopt -s nullglob
    files=(results-1008-*/measurements-rank-0-*.jsonl)
    if (( ${#files[@]} )); then tail -n 1 -v "${files[@]}"; fi
    ;;
  report)
    check_files
    mkdir -p "$LOGS"
    kubectl -n "$NS" get trainjob,jobset,job,pod,workloads -o yaml > "$LOGS/specs-$(date -u +%H%M%S).yaml"
    "$PY" code/report.py results-1008-{0,1,2,3,4} | tee "$LOGS/report.txt"
    "$PY" code/lora.py summary results-1008-{0,1,2,3,4} | tee "$LOGS/summary.txt"
    ;;
  _record-state)
    exec 9> "$LOGS/.state.lock"
    flock -n 9 || exit 0
    end=$(date -d "$WATCH_UNTIL" +%s)
    while (( $(date +%s) < end )); do
      date -u '+== %FT%TZ'
      kubectl -n "$NS" get workloads,trainjobs,pods || true
      kubectl -n "$NS" get jobs -o custom-columns=NAME:.metadata.name,ACTIVE:.status.active,FAILED:.status.failed,SUCCEEDED:.status.succeeded || true
      sleep 30
    done
    ;;
  _record-events)
    exec 9> "$LOGS/.events.lock"
    flock -n 9 || exit 0
    end=$(date -d "$WATCH_UNTIL" +%s)
    while (( (remaining = end - $(date +%s)) > 0 )); do
      timeout "$remaining" kubectl -n "$NS" get events --watch -o custom-columns=CREATED:.metadata.creationTimestamp,LAST:.lastTimestamp,REASON:.reason,OBJECT:.involvedObject.name,MESSAGE:.message || true
      sleep 5
    done
    ;;
esac
