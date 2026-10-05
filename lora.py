"""LoRA trials for the pinned retrieval models.

Publisher labels and definitions are grouped, split and used as weak
supervision. Each trial is bounded, writes resumable checkpoints and ends with
an adapter export. Functions that need torch import it when they are called, so
the data split, the trial queue and the summary run in any Python environment.
"""
import argparse
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import time

from artifacts import atomic_json, digest, load_corpus

FORMAT = "matsci-retrieval-lora-trial-v1"
MIN_DEFINITION_WORDS = 8
SPLIT_BOUNDS = (("train", 0.80), ("validation", 0.90), ("test", 1.00))
SCALE = 20.0  # Similarity scale of the in-batch contrastive loss.
GROUPS = ("materials", "chebi")
NOTE = ("Validation pairs are publisher labels and definitions (weak supervision), "
        "not judged relevance. The worker does not read the test split.")


class Interrupted(Exception):
    """Work stopped because a signal arrived or the deadline is near."""


def normalized(text):
    """Letters and digits only, so that case, spacing and punctuation do not separate two versions of a text."""
    return re.sub(r"[\W_]+", "", text.casefold())


def build_pairs(rows):
    """Group label and definition pairs into components and give each component one split.

    Records join a component when they share an entity IRI, a definition or a
    label, compared by letters and digits. A batch takes one record per component,
    and a component lies in one split, so no two versions of a description meet
    as anchor and negative or on both sides of an evaluation.
    """
    candidates = [index for index, row in enumerate(rows)
                  if row.get("definition")
                  and row.get("definition_language", "").lower().split("-")[0] in ("", "en")
                  and len(re.findall(r"\w+", row["definition"])) >= MIN_DEFINITION_WORDS]
    parent = {index: index for index in candidates}

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    first = {}
    for index in candidates:
        row = rows[index]
        for key in (("iri", row["iri"]), ("definition", normalized(row["definition"])),
                    ("label", normalized(row["label"]))):
            if not key[1]:
                continue  # A label of punctuation alone must not join unrelated records.
            other = first.setdefault(key, index)
            if other != index:
                parent[find(index)] = find(other)
    grouped = {}
    for index in candidates:
        grouped.setdefault(find(index), []).append(index)
    components = []
    for members in grouped.values():
        members.sort(key=lambda index: rows[index]["id"])
        key = rows[members[0]]["id"]
        position = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 2 ** 32
        components.append({
            "key": key, "members": members,
            "split": next(name for name, bound in SPLIT_BOUNDS if position < bound),
            "group": "chebi" if all(rows[index]["source"] == "chebi" for index in members) else "materials"})
    components.sort(key=lambda component: component["key"])
    return components


def pairs_manifest(rows, components, corpus_sha256):
    """Counts and identity of the split; two outputs with one hash trained on the same data."""
    splits = {}
    identity = hashlib.sha256()
    for component in components:
        entry = splits.setdefault(component["split"], {"components": 0, "records": 0, "groups": {}, "sources": {}})
        entry["components"] += 1
        entry["records"] += len(component["members"])
        entry["groups"][component["group"]] = entry["groups"].get(component["group"], 0) + 1
        for index in component["members"]:
            source = rows[index]["source"]
            entry["sources"][source] = entry["sources"].get(source, 0) + 1
        identity.update(json.dumps([component["split"], [rows[index]["id"] for index in component["members"]]]).encode())
        identity.update(b"\n")
    return {"format": "matsci-retrieval-pairs-v1", "corpus_sha256": corpus_sha256,
            "rule": {"definition_language": "English or untagged",
                     "minimum_definition_words": MIN_DEFINITION_WORDS,
                     "component": "shared entity IRI, or shared letters and digits of the definition or of the label",
                     "split": "first 32 bits of the SHA-256 of the lowest record id in the component",
                     "bounds": dict(SPLIT_BOUNDS)},
            "splits": splits, "sha256": identity.hexdigest()}


def load_trials(path, specs, shard="0/1", limit=None):
    """Read the trial queue, apply the defaults and keep the trials of one shard."""
    document = json.loads(Path(path).read_text())
    defaults = document["defaults"]
    parents = {spec["name"]: spec for spec in specs}
    trials = []
    for entry in document["trials"]:
        trial = {**{key: value for key, value in defaults.items() if key != "batch_size"}, **entry}
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", trial["id"]):
            raise ValueError(f"Trial id is not a safe directory name: {trial['id']}")
        if trial["parent"] not in parents:
            raise ValueError(f"Trial {trial['id']} names a parent that models.json does not pin")
        if trial["data"] not in ("balanced", "all"):
            raise ValueError(f"Trial {trial['id']} has an unknown data selection")
        trial.setdefault("batch_size", defaults["batch_size"][trial["parent"]])
        trial.setdefault("alpha", 2 * trial["rank"])
        if min(trial["rank"], trial["batch_size"], trial["pairs_budget"], trial["seed"]) < 1 or trial["learning_rate"] <= 0:
            raise ValueError(f"Trial {trial['id']} has a value out of range")
        trials.append(trial)
    if len({trial["id"] for trial in trials}) != len(trials):
        raise ValueError("Trial ids are not unique")
    index, count = (int(value) for value in rank_shard(shard, 0, 1).split("/"))
    chosen = [trial for position, trial in enumerate(trials) if position % count == index]
    return chosen if limit is None else chosen[:limit]


def rank_shard(shard, rank, world):
    """Share of one process. Rank r of w in a job with shard i/n takes the positions i + r*n modulo n*w,
    so the ranks of a job divide the positions that equal i modulo n."""
    match = re.fullmatch(r"(\d+)/(\d+)", shard)
    if not match or not int(match[1]) < int(match[2]):
        raise ValueError("A trial shard has the form i/n with i below n")
    return f"{int(match[1]) + rank * int(match[2])}/{int(match[2]) * world}"


def epoch_components(train, trial, epoch):
    """Positions in `train` used in one epoch, in training order.

    A balanced epoch holds every materials component and a rotating window of
    ChEBI components, so that ChEBI does not supply nearly all training pairs.
    """
    materials = [position for position, component in enumerate(train) if component["group"] != "chebi"]
    chebi = [position for position, component in enumerate(train) if component["group"] == "chebi"]
    if trial["data"] == "balanced" and materials and chebi:
        random.Random(f"{trial['seed']}:chebi").shuffle(chebi)
        size = min(len(chebi), trial["chebi_ratio"] * len(materials))
        start = epoch * size
        chosen = materials + [chebi[(start + offset) % len(chebi)] for offset in range(size)]
    else:
        chosen = materials + chebi
    random.Random(f"{trial['seed']}:{epoch}").shuffle(chosen)
    return chosen


def spec_digest(trial, parent, pairs_sha256):
    """Identity of a trial: its settings, its parent, its data and this file."""
    identity = {"trial": trial, "pairs": pairs_sha256, "code": digest(__file__),
                "parent": {key: parent[key] for key in ("id", "revision", "query_prefix")}}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def latest_checkpoint(directory, spec_sha256):
    """Newest checkpoint whose receipt matches the file and the trial; older or foreign files are ignored."""
    best = None
    for receipt in sorted(Path(directory).glob("checkpoint-*.json")):
        try:
            saved = json.loads(receipt.read_text())
            path = receipt.with_suffix(".pt")
            if (saved["spec_sha256"] == spec_sha256 and path.exists() and saved["sha256"] == digest(path)
                    and (best is None or saved["step"] > best[0])):
                best = (saved["step"], path)
        except (OSError, ValueError, KeyError):
            continue
    return best


def save_checkpoint(directory, step, state, spec_sha256):
    """Write a checkpoint and its receipt, then remove the older ones.

    The caller never saves a step that is already committed: replacing a
    checkpoint file would leave no valid one until the new receipt is written.
    """
    import torch
    directory = Path(directory)
    path = directory / f"checkpoint-{step:07d}.pt"
    temporary = path.with_suffix(f".{os.getpid()}.partial")
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    atomic_json(path.with_suffix(".json"), {"step": step, "sha256": digest(path), "spec_sha256": spec_sha256})
    for other in directory.glob("checkpoint-*"):
        if other.stem != path.stem:
            other.unlink(missing_ok=True)


def attach_adapter(model, directory, sha256):
    """Merge an exported adapter into a loaded parent model for inference."""
    from peft import PeftModel
    weights = Path(directory) / "adapter_model.safetensors"
    if digest(weights) != sha256:
        raise ValueError(f"Adapter checksum mismatch: {weights}")
    parent = model[0].auto_model
    # The adapter is injected into the parent and merged in place.
    if PeftModel.from_pretrained(parent, str(directory)).merge_and_unload() is not parent or any(
            "lora_" in name for name, _parameter in model.named_parameters()):
        raise ValueError("Adapter weights were not merged into the parent model")
    return model


def collate(sequences, pad, device):
    """Pad token id lists to one tensor batch."""
    import torch
    lengths = torch.tensor([len(sequence) for sequence in sequences])
    ids = torch.full((len(sequences), int(lengths.max())), pad, dtype=torch.long)
    for row, sequence in enumerate(sequences):
        ids[row, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    mask = (torch.arange(ids.shape[1])[None, :] < lengths[:, None]).long()
    return {"input_ids": ids.to(device), "attention_mask": mask.to(device)}


def metrics(queries, documents, groups):
    """Recall and reciprocal rank where document i is the one positive of query i."""
    import torch
    ranks, positives = [], []
    for start in range(0, queries.shape[0], 1024):
        scores = queries[start:start + 1024] @ documents.T
        own = scores.gather(1, torch.arange(start, start + scores.shape[0], device=scores.device)[:, None])
        ranks.append((scores >= own).sum(dim=1) - 1)  # A tie counts against the query.
        positives.append(own[:, 0])
    ranks, positives = torch.cat(ranks).cpu(), torch.cat(positives).cpu()
    result = {}
    for name in ("all",) + GROUPS:
        mask = torch.tensor([name in ("all", group) for group in groups], dtype=torch.bool)
        chosen = ranks[mask]
        if len(chosen):
            result[name] = {"queries": len(chosen),
                            "recall@1": float((chosen < 1).float().mean()),
                            "recall@10": float((chosen < 10).float().mean()),
                            "mrr@10": float(torch.where(chosen < 10, 1.0 / (chosen + 1), torch.zeros(len(chosen))).mean()),
                            "positive_score": float(positives[mask].mean())}
    return result


def selection_score(result):
    """Mean of the group reciprocal ranks, so that ChEBI does not decide the selection alone."""
    final = result["validation"]["final"]
    present = [final[name]["mrr@10"] for name in GROUPS if name in final]
    return sum(present) / len(present)


def completed_trials(out):
    """Results of the finished trials in one output directory."""
    results = []
    for path in sorted(Path(out).glob("trials/*/result.json")):
        result = json.loads(path.read_text())
        if result.get("status") == "complete":
            results.append(result)
    return results


def best_trial(out):
    """The finished trial with the highest selection score, or None."""
    results = completed_trials(out)
    return max(results, key=lambda result: (selection_score(result), result["trial"]["id"])) if results else None


class Session:
    """Model, data and bookkeeping shared by the trials of one worker."""

    def __init__(self, rows, corpus_sha256, specs, out, device, log, available):
        self.rows, self.out, self.device, self.log, self.available = rows, Path(out), device, log, available
        self.specs = {spec["name"]: spec for spec in specs}
        self.components = build_pairs(rows)
        self.manifest = pairs_manifest(rows, self.components, corpus_sha256)
        self.train = [component for component in self.components if component["split"] == "train"]
        self.tokens = {}
        self.baselines = {}

    def split(self, name):
        return [component for component in self.components if component["split"] == name]

    def distractors(self, limit):
        """Training components whose definitions join the held-out ones as documents:
        every materials component first, then ChEBI up to the limit."""
        return sorted(self.train, key=lambda component: component["group"] == "chebi")[:limit]

    def register(self):
        """Bind the output directory to this training data."""
        self.out.mkdir(parents=True, exist_ok=True)
        path = self.out / "pairs-manifest.json"
        # Ranks of one job share the directory, so the first writer holds a lock.
        with (self.out / "pairs.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                if json.loads(path.read_text()) != self.manifest:
                    raise ValueError("Output holds trials for other training data; choose a new output directory")
                return
            listing = self.out / "pairs.jsonl"
            temporary = listing.with_suffix(f".{os.getpid()}.partial")
            with temporary.open("w") as stream:
                for component in self.components:
                    stream.write(json.dumps({"split": component["split"], "group": component["group"],
                                             "records": [self.rows[index]["id"] for index in component["members"]]}) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(listing)
            atomic_json(path, self.manifest)

    def encoded(self, model, spec, trial, indices):
        """Token ids of labels (as queries) and definitions for the given corpus rows."""
        cache = self.tokens.setdefault((spec["name"], trial["max_length"], trial["query_max_length"]),
                                       {"label": {}, "definition": {}})
        missing = [index for index in indices if index not in cache["label"]]
        tokenizer = model[0].tokenizer
        for start in range(0, len(missing), 2048):
            part = missing[start:start + 2048]
            labels = tokenizer([spec["query_prefix"] + self.rows[index]["label"] for index in part],
                               truncation=True, max_length=trial["query_max_length"], padding=False)["input_ids"]
            definitions = tokenizer([self.rows[index]["definition"] for index in part],
                                    truncation=True, max_length=trial["max_length"], padding=False)["input_ids"]
            cache["label"].update(zip(part, labels))
            cache["definition"].update(zip(part, definitions))
        return cache

    def embed(self, model, sequences, batch_size):
        """Normalized float32 embeddings of token id lists, in the given order."""
        import torch
        cuda = self.device.startswith("cuda")
        pad = model[0].tokenizer.pad_token_id
        order = sorted(range(len(sequences)), key=lambda position: len(sequences[position]))
        parts = []
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(order), batch_size):
                if not self.available():
                    raise Interrupted
                chosen = order[start:start + batch_size]
                with torch.autocast(device_type="cuda" if cuda else "cpu", dtype=torch.bfloat16,
                                    enabled=cuda and torch.cuda.is_bf16_supported()):
                    output = model(collate([sequences[position] for position in chosen], pad, self.device))
                parts.append(torch.nn.functional.normalize(output["sentence_embedding"].float(), dim=-1))
        matrix = torch.cat(parts)
        if not torch.isfinite(matrix).all():
            raise ValueError("Nonfinite embeddings")
        inverse = torch.empty(len(order), dtype=torch.long)
        inverse[torch.tensor(order)] = torch.arange(len(order))
        return matrix[inverse.to(matrix.device)]

    def evaluate(self, model, spec, trial, split="validation"):
        """Retrieve the definition of each held-out label among held-out and training definitions."""
        held = self.split(split)
        indices = [component["members"][0] for component in held + self.distractors(trial["distractors"])]
        cache = self.encoded(model, spec, trial, indices)
        batch = 4 * trial["batch_size"]
        queries = self.embed(model, [cache["label"][index] for index in indices[:len(held)]], batch)
        documents = self.embed(model, [cache["definition"][index] for index in indices], batch)
        result = metrics(queries, documents, [component["group"] for component in held])
        result["documents"] = len(indices)
        return result


def run_trial(session, trial, checkpoint_seconds=60, log_seconds=10):
    """Train one trial to its pair budget, or checkpoint and stop when work is no longer available.

    Returns "complete", "interrupted" or "skipped". After an out-of-memory error
    the trial starts again with half the batch.
    """
    import torch
    while True:
        outcome = attempt_trial(session, trial, checkpoint_seconds, log_seconds)
        if outcome != "oom":
            return outcome
        gc.collect()
        torch.cuda.empty_cache()


def attempt_trial(session, trial, checkpoint_seconds, log_seconds):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from sentence_transformers import SentenceTransformer

    spec = session.specs[trial["parent"]]
    device, log = session.device, session.log
    cuda = device.startswith("cuda")
    amp = cuda and torch.cuda.is_bf16_supported()
    directory = session.out / "trials" / trial["id"]
    identity = spec_digest(trial, spec, session.manifest["sha256"])
    result_path = directory / "result.json"
    if result_path.exists() and json.loads(result_path.read_text()).get("spec_sha256") == identity:
        log(stage="train", trial=trial["id"], status="already complete")
        return "skipped"
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("*.partial"):
        shutil.rmtree(stale) if stale.is_dir() else stale.unlink()

    torch.manual_seed(trial["seed"])
    model = SentenceTransformer(spec["id"], revision=spec["revision"], device=device,
                                local_files_only=True, trust_remote_code=False)
    model.float()
    transformer = model[0]
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    adapted = get_peft_model(transformer.auto_model, LoraConfig(
        r=trial["rank"], lora_alpha=trial["alpha"], lora_dropout=trial["dropout"], bias="none",
        target_modules=list(trial["target_modules"]), task_type=TaskType.FEATURE_EXTRACTION))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable = sum(parameter.numel() for parameter in parameters)
    if not trainable or any("lora_" not in name for name, parameter in model.named_parameters() if parameter.requires_grad):
        raise ValueError("Only the adapter weights may be trainable")
    optimizer = torch.optim.AdamW(parameters, lr=trial["learning_rate"], weight_decay=trial["weight_decay"])
    pad = transformer.tokenizer.pad_token_id

    # The batch size shrinks after an out-of-memory error and is kept across restarts.
    reduced = directory / "batch-size.json"
    size = trial["batch_size"]
    if reduced.exists() and json.loads(reduced.read_text())["spec_sha256"] == identity:
        size = json.loads(reduced.read_text())["batch_size"]
    state = {"step": 0, "epoch": 0, "batch": 0, "pairs": 0, "seconds": 0.0, "losses": [],
             "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    # Counted on disk, because an attempt can end before it writes a checkpoint.
    counter = directory / "attempts.json"
    attempts = 1
    if counter.exists() and json.loads(counter.read_text())["spec_sha256"] == identity:
        attempts += json.loads(counter.read_text())["attempts"]
    atomic_json(counter, {"attempts": attempts, "spec_sha256": identity})
    found = latest_checkpoint(directory, identity)
    if found:
        saved = torch.load(found[1], map_location="cpu", weights_only=True)
        set_peft_model_state_dict(adapted, saved["adapter"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        if cuda and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
        state.update(saved["state"])
    order = epoch_components(session.train, trial, state["epoch"])
    size = min(size, len(order))
    if size < 2:
        raise ValueError("Training needs at least two components for in-batch negatives")
    session.encoded(model, spec, trial, [index for component in session.train for index in component["members"]])
    steps = math.ceil(trial["pairs_budget"] / size)
    warmup = max(1, int(trial["warmup"] * steps))
    log(stage="train", trial=trial["id"], status="resumed" if found else "started", step=state["step"],
        steps=steps, batch_size=size, trainable_parameters=trainable, precision="bfloat16" if amp else "float32")

    # The parent is evaluated once per model and length. A fresh adapter leaves
    # the parent unchanged, and a resumed one is switched off for this pass.
    key = (spec["name"], trial["max_length"], trial["query_max_length"], trial["distractors"])
    if key not in session.baselines:
        try:
            with adapted.disable_adapter():
                session.baselines[key] = session.evaluate(model, spec, trial)
        except Interrupted:
            return "interrupted"
        log(stage="eval", trial=trial["id"], model=spec["name"], adapter=False, **session.baselines[key])

    committed = found[0] if found else None

    def checkpoint():
        nonlocal committed
        if state["step"] == committed:
            return  # This step is on disk already, and nothing has changed since.
        begin = time.perf_counter()
        weights = get_peft_model_state_dict(adapted, save_embedding_layers=False)
        save_checkpoint(directory, state["step"], {
            "adapter": {name: value.detach().cpu() for name, value in weights.items()},
            "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device) if cuda else None,
            "state": {**state, "losses": state["losses"][-50:]}}, identity)
        committed = state["step"]
        log(stage="train", trial=trial["id"], status="checkpoint", step=state["step"],
            seconds=time.perf_counter() - begin)

    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    last_checkpoint = last_log = began = time.perf_counter()
    logged_pairs = state["pairs"]
    while state["step"] < steps:
        if not session.available():
            state["seconds"] += time.perf_counter() - began
            checkpoint()
            return "interrupted"
        if (state["batch"] + 1) * size > len(order):
            state["epoch"], state["batch"] = state["epoch"] + 1, 0
            order = epoch_components(session.train, trial, state["epoch"])
        chosen = order[state["batch"] * size:(state["batch"] + 1) * size]
        indices = [session.train[position]["members"][state["epoch"] % len(session.train[position]["members"])]
                   for position in chosen]
        cache = session.encoded(model, spec, trial, indices)
        model.train()
        try:
            with torch.autocast(device_type="cuda" if cuda else "cpu", dtype=torch.bfloat16, enabled=amp):
                anchors = model(collate([cache["label"][index] for index in indices], pad, device))["sentence_embedding"]
                positives = model(collate([cache["definition"][index] for index in indices], pad, device))["sentence_embedding"]
            anchors = torch.nn.functional.normalize(anchors.float(), dim=-1)
            positives = torch.nn.functional.normalize(positives.float(), dim=-1)
            loss = torch.nn.functional.cross_entropy(anchors @ positives.T * SCALE,
                                                     torch.arange(len(indices), device=device))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            factor = min((state["step"] + 1) / warmup, max(0.0, (steps - state["step"]) / max(1, steps - warmup)))
            for group in optimizer.param_groups:
                group["lr"] = trial["learning_rate"] * factor
            optimizer.step()
        except torch.cuda.OutOfMemoryError:
            # The trial starts again with half the batch. The data order depends
            # on the batch size, so the earlier steps are not kept.
            log(stage="train", trial=trial["id"], status="oom", batch_size=size)
            if size // 2 < 2:
                raise
            for other in directory.glob("checkpoint-*"):
                other.unlink(missing_ok=True)
            atomic_json(reduced, {"batch_size": size // 2, "spec_sha256": identity})
            return "oom"
        state["step"] += 1
        state["batch"] += 1
        state["pairs"] += len(indices)
        state["losses"].append(float(loss.detach()))
        now = time.perf_counter()
        if now - last_log >= log_seconds or state["step"] == steps:
            recent = state["losses"][-20:]
            log(stage="train", trial=trial["id"], step=state["step"], steps=steps, epoch=state["epoch"],
                loss=sum(recent) / len(recent), learning_rate=optimizer.param_groups[0]["lr"],
                pairs_per_second=(state["pairs"] - logged_pairs) / max(now - last_log, 1e-9),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if cuda else 0)
            last_log, logged_pairs = now, state["pairs"]
        if now - last_checkpoint >= checkpoint_seconds and state["step"] < steps:
            state["seconds"] += now - began
            began = now
            checkpoint()
            last_checkpoint = time.perf_counter()
    state["seconds"] += time.perf_counter() - began
    try:
        final = session.evaluate(model, spec, trial)
    except Interrupted:
        checkpoint()  # At the last step, so that the restart goes straight to this evaluation.
        return "interrupted"
    log(stage="eval", trial=trial["id"], model=spec["name"], adapter=True, **final)
    export = directory / "adapter"
    staging = directory / f"adapter.{os.getpid()}.partial"
    adapted.save_pretrained(str(staging), save_embedding_layers=False)
    if export.exists():
        shutil.rmtree(export)
    staging.replace(export)
    adapter_sha256 = digest(export / "adapter_model.safetensors")
    atomic_json(directory / "model.json", [{
        **{key: spec[key] for key in ("id", "revision", "dimensions", "max_length", "batch_size", "query_prefix")},
        "name": trial["id"], "adapter": {"trial": trial["id"], "parent": spec["name"], "sha256": adapter_sha256}}])
    recent = state["losses"][-20:]
    atomic_json(result_path, {
        "format": FORMAT, "status": "complete", "trial": trial, "spec_sha256": identity,
        "parent": {key: spec[key] for key in ("name", "id", "revision", "dimensions", "query_prefix")},
        "inference": {"modules": [type(module).__name__ for module in model], "normalized": True,
                      "query_prefix": spec["query_prefix"], "index_max_length": spec["max_length"]},
        "lora": {"rank": trial["rank"], "alpha": trial["alpha"], "dropout": trial["dropout"],
                 "target_modules": list(trial["target_modules"]),
                 "trainable_parameters": trainable, "total_parameters": total_parameters},
        "data": {"pairs_sha256": session.manifest["sha256"], "corpus_sha256": session.manifest["corpus_sha256"],
                 "train_components": len(session.train), "epoch_components": len(order), "selection": trial["data"]},
        "training": {"steps": state["step"], "batch_size": size, "pairs": state["pairs"], "epochs_started": state["epoch"] + 1,
                     "seconds": state["seconds"], "pairs_per_second": state["pairs"] / max(state["seconds"], 1e-9),
                     "final_loss": sum(recent) / len(recent), "attempts": attempts,
                     "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if cuda else 0,
                     "precision": "bfloat16" if amp else "float32", "loss": "in-batch contrastive, scale 20"},
        "validation": {"baseline": session.baselines[key], "final": final},
        "adapter_sha256": adapter_sha256,
        "packages": {name: importlib.metadata.version(name)
                     for name in ["torch", "sentence-transformers", "transformers", "peft"]},
        "code": {"lora_sha256": digest(__file__), "artifacts_sha256": digest(Path(__file__).with_name("artifacts.py"))},
        "device": torch.cuda.get_device_name(device) if cuda else "cpu",
        "started": state["started"], "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "note": NOTE})
    for other in list(directory.glob("checkpoint-*")) + [reduced, counter]:
        other.unlink(missing_ok=True)
    log(stage="train", trial=trial["id"], status="complete", steps=state["step"],
        baseline_mrr=session.baselines[key]["all"]["mrr@10"], final_mrr=final["all"]["mrr@10"])
    return "complete"


def run_trials(rows, corpus_sha256, specs, trials, out, device, log, available, start_before,
               checkpoint_seconds=60, log_seconds=10):
    """Run trials in order until work is no longer available or `start_before` has passed.

    A failed trial is recorded and the next one starts. Three failures in a row
    end the stage, and the caller continues with its other work.
    """
    import torch
    session = Session(rows, corpus_sha256, specs, out, device, log, available)
    if not session.train or not session.split("validation"):
        log(stage="train", status="skipped", reason="no label and definition pairs for training and validation")
        return []
    session.register()
    done = {result["trial"]["id"] for result in completed_trials(out)}
    log(stage="train", status="queue", trials=len(trials), complete=sum(trial["id"] in done for trial in trials),
        pairs_sha256=session.manifest["sha256"],
        components={name: len(session.split(name)) for name, _bound in SPLIT_BOUNDS})
    finished, failures, reason = [], 0, "the queue has no further trial"
    for trial in trials:
        if not available():
            reason = "a signal arrived or the deadline is near"
            break
        if time.time() >= start_before:
            reason = "the time reserved before the deadline has begun"
            break
        try:
            status = run_trial(session, trial, checkpoint_seconds, log_seconds)
            failures = 0
            if status == "complete":
                finished.append(trial["id"])
            if status == "interrupted":
                reason = "a signal arrived or the deadline is near"
                break
        except Exception as error:  # A broken trial must not end the GPU load of the job.
            failures += 1
            log(stage="train", trial=trial["id"], status="failed", error=f"{type(error).__name__}: {error}"[:500])
            if failures == 3:
                reason = "three trials failed in a row"
                break
        finally:
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    log(stage="train", status="abandoned" if failures == 3 else "queue ended", reason=reason, finished=len(finished))
    return finished


def summary(outputs):
    """Finished trials of several output directories, best selection score first."""
    table = []
    for out in outputs:
        for result in completed_trials(out):
            baseline, final = result["validation"]["baseline"], result["validation"]["final"]
            table.append({"trial": result["trial"]["id"], "output": str(out), "score": selection_score(result),
                          **{f"{group} mrr@10": f"{baseline[group]['mrr@10']:.3f} -> {final[group]['mrr@10']:.3f}"
                             for group in GROUPS if group in final},
                          "pairs/s": round(result["training"]["pairs_per_second"]),
                          "seconds": round(result["training"]["seconds"])})
    return sorted(table, key=lambda row: (-row["score"], row["trial"]))


def test_split(corpus, out, trial_id, device):
    """Evaluate one finished trial and its parent on the held-out test split."""
    import torch
    from sentence_transformers import SentenceTransformer
    manifest, rows = load_corpus(corpus)
    result = json.loads((Path(out) / "trials" / trial_id / "result.json").read_text())
    trial, spec = result["trial"], {**result["parent"]}
    session = Session(rows, manifest["sha256"], [spec], out, device, lambda **_event: None, lambda: True)
    if session.manifest["sha256"] != result["data"]["pairs_sha256"]:
        raise ValueError("The corpus gives another split than the one this trial was trained on")
    report = {"trial": trial_id, "split": "test", "pairs_sha256": session.manifest["sha256"]}
    for name in ("parent", "adapted"):
        model = SentenceTransformer(spec["id"], revision=spec["revision"], device=device,
                                    local_files_only=True, trust_remote_code=False)
        model.float()
        if name == "adapted":
            attach_adapter(model, Path(out) / "trials" / trial_id / "adapter", result["adapter_sha256"])
        report[name] = session.evaluate(model, spec, trial, split="test")
        del model
    atomic_json(Path(out) / "trials" / trial_id / "test.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pairs = commands.add_parser("pairs", help="Print the counts and the identity of the data split")
    pairs.add_argument("--corpus", type=Path, required=True)
    table = commands.add_parser("summary", help="List finished trials, best selection score first")
    table.add_argument("outputs", type=Path, nargs="+")
    held = commands.add_parser("test", help="Evaluate one selected trial on the held-out test split")
    held.add_argument("--corpus", type=Path, required=True)
    held.add_argument("--out", type=Path, required=True)
    held.add_argument("--trial", required=True)
    held.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "pairs":
        manifest, rows = load_corpus(args.corpus)
        print(json.dumps(pairs_manifest(rows, build_pairs(rows), manifest["sha256"]), indent=2))
    elif args.command == "summary":
        table = [{column: f"{value:.4f}" if column == "score" else str(value) for column, value in row.items()}
                 for row in summary(args.outputs)]
        columns = list(table[0]) if table else []
        for row in [dict(zip(columns, columns))] + table:
            print("  ".join(row.get(column, "").ljust(max(len(other.get(column, "")) for other in table + [row]))
                            for column in columns))
        if not table:
            print("No finished trials")
    else:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        print(json.dumps(test_split(args.corpus, args.out, args.trial, args.device), indent=2))


if __name__ == "__main__":
    main()
