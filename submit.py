"""Print a FLAME submission plan; --execute explicitly submits it now."""
import argparse
import json
import os
from pathlib import Path
import re
import time

from workload import deadline


def launch(root, interpreter, stop_at, output, case_seconds=45, trial_shard="0/1", max_trials=-1, cpu_threads=1):
    # Kubeflow serializes this function alone, so imports stay inside it.
    # The SDK embeds this source in an unquoted shell here-document: keep it
    # free of dollar signs, backticks and backslashes.
    import os
    import signal
    import subprocess
    import time
    from datetime import datetime
    from pathlib import Path
    root = Path(root)
    remaining = datetime.fromisoformat(stop_at.replace("Z", "+00:00")).timestamp() - time.time()
    if remaining < 120:
        # A job admitted this late, for example one requeued after a preemption,
        # ends as Complete. An error here would mark it Failed and start retries.
        print(f"Stop time {stop_at} is past or under two minutes away; no GPU work started", flush=True)
        return
    command = [interpreter, str(root / "code/workload.py"),
               "--corpus", str(root / "corpus"), "--out", str(root / output),
               "--stop-at", stop_at, "--case-seconds", str(case_seconds), "--trial-shard", trial_shard,
               "--cpu-threads", str(cpu_threads)]
    if max_trials >= 0:
        command += ["--max-trials", str(max_trials)]
    environment = dict(os.environ, OMP_NUM_THREADS=str(cpu_threads),
                       MKL_NUM_THREADS=str(cpu_threads), OPENBLAS_NUM_THREADS=str(cpu_threads))
    process = subprocess.Popen(command, start_new_session=True, env=environment)

    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    try:
        code = process.wait(timeout=remaining)
        if code:
            raise RuntimeError(f"GPU worker exited with code {code}")
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Absolute persistent FLAME bundle path")
    parser.add_argument("--python", required=True, help="Absolute Python path prepared for the target architecture/image")
    parser.add_argument("--runtime", choices=["torch-gh200", "torch-rtxa6000"], required=True)
    parser.add_argument("--gpus-per-node", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--cpus-per-node", type=int, default=1,
                        help="CPUs shared by this job's GPU workers; FLAME also reserves CPUs for storage and inference")
    parser.add_argument("--memory-per-node", default="32Gi")
    parser.add_argument("--namespace", default="metadata-research-center")
    parser.add_argument("--stop-at", required=True)
    parser.add_argument("--output", default="results")
    parser.add_argument("--case-seconds", type=int, default=45,
                        help="Seconds per sweep case, passed to the worker; a small value lets a short preflight reach every stage")
    parser.add_argument("--trial-shard", default="0/1",
                        help="i/n: this job runs the LoRA trials whose position modulo n equals i")
    parser.add_argument("--max-trials", type=int,
                        help="Upper limit of LoRA trials per GPU process; 0 skips the training")
    parser.add_argument("--image", help="Optional prepared custom image, preferably pinned by digest")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.root.is_absolute() or not Path(args.python).is_absolute():
        parser.error("Root and Python paths must be absolute")
    if min(args.nodes, args.gpus_per_node, args.cpus_per_node, args.case_seconds) < 1:
        parser.error("Resource counts and the case duration must be positive")
    shard = re.fullmatch(r"(\d+)/(\d+)", args.trial_shard)
    if not shard or not int(shard[1]) < int(shard[2]) or (args.max_trials or 0) < 0:
        parser.error("The trial shard has the form i/n with i below n; the trial limit must not be negative")
    if args.gpus_per_node > (1 if args.runtime == "torch-gh200" else 8):
        parser.error("GPU request exceeds the documented per-node runtime capacity")
    if Path(args.output).is_absolute() or ".." in Path(args.output).parts:
        parser.error("Output must be a relative directory under root")
    if any(character in value for character in "$`\\"
           for value in [str(args.root), args.python, args.stop_at, args.output]):
        # These values travel through the same unquoted here-document as launch().
        parser.error("Root, Python path, stop time and output must not contain $, a backtick or a backslash")
    if deadline(args.stop_at) - time.time() < 120:
        parser.error("Deadline must be at least two minutes in the future")
    plan = vars(args).copy()
    plan["root"] = str(args.root)
    print(json.dumps(plan, indent=2))
    if not args.execute:
        print("Preview only. Add --execute inside the approved window to submit immediately.")
        return
    from kubeflow.trainer import CustomTrainer, KubernetesBackendConfig, TrainerClient
    # A shell opened on FLAME through SSH lacks the two variables that locate the
    # cluster API from inside a pod. These are the standard in-cluster values.
    if Path("/var/run/secrets/kubernetes.io/serviceaccount/token").exists():
        os.environ.setdefault("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc")
        os.environ.setdefault("KUBERNETES_SERVICE_PORT", "443")
    client = TrainerClient(KubernetesBackendConfig(namespace=args.namespace))
    options = {"image": args.image} if args.image else {}
    job = client.train(runtime=args.runtime, trainer=CustomTrainer(
        func=launch, func_args={"root": str(args.root), "interpreter": args.python,
                                "stop_at": args.stop_at, "output": args.output,
                                "case_seconds": args.case_seconds, "trial_shard": args.trial_shard,
                                "max_trials": -1 if args.max_trials is None else args.max_trials,
                                "cpu_threads": max(1, args.cpus_per_node // args.gpus_per_node)},
        num_nodes=args.nodes,
        resources_per_node={"cpu": args.cpus_per_node, "memory": args.memory_per_node, "gpu": args.gpus_per_node},
        **options
    ))
    print(f"Submitted TrainJob: {job}")


if __name__ == "__main__":
    main()
