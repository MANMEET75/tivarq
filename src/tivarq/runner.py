"""Episode isolation, capability-aware execution and manifests."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import statistics

from .dataset import load_split
from .protocol import Adapter
from .scoring import report


def run(dataset: Path, split: str, track: str, command: list[str], output: Path, *, repeat: int = 1,
        model_config: str = "none", prompt_config: str = "none", seed: int = 20261001):
    if track not in ("structured", "conversation"):
        raise ValueError("track must be structured or conversation")
    episodes, probes = load_split(dataset, split)
    by_episode = {e["episode_id"]: [] for e in episodes}
    for probe in probes:
        by_episode[probe["episode_id"]].append(probe)
    output.mkdir(parents=True, exist_ok=True)
    try:
        harness_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        harness_commit = "uncommitted"
    manifest = {"schema_version": "tivarq.run.v1", "dataset_revision": os.environ.get("TIVARQ_DATASET_REVISION", "local-generated"),
                "dataset_file_hashes": json.loads((dataset / "manifest.json").read_text())["files"],
                "dataset_source": str(dataset), "harness_commit": harness_commit, "track": track,
                "split": split, "adapter_command": command, "model_config": model_config,
                "prompt_config": prompt_config, "run_seed": seed, "repeats": repeat,
                "sampling_settings": os.environ.get("TIVARQ_SAMPLING", "unspecified")}
    summaries = []
    for repetition in range(repeat):
        adapter = Adapter(command)
        if track not in adapter.capabilities.get("tracks", []):
            adapter.close()
            raise ValueError(f"adapter does not declare {track} track")
        manifest["adapter_version"] = adapter.capabilities.get("adapter_version", "unknown")
        manifest["capabilities"] = adapter.capabilities
        rows = []
        try:
            for episode in episodes:
                adapter.call({"type": "reset", "episode_id": episode["episode_id"], "seed": seed + repetition})
                pending = sorted(by_episode[episode["episode_id"]], key=lambda p: (p["after_event_index"], p["probe_id"]))
                pointer = 0
                ingest_error = None
                for idx in range(len(episode["events"]) + 1):
                    if idx:
                        event = episode["events"][idx-1]
                        public_event = {k: event[k] for k in ("event_id", "arrived_at", "speaker", "text")}
                        if track == "structured":
                            public_event["assertion"] = event["assertion"]
                        if ingest_error is None:
                            try:
                                adapter.call({"type": "ingest", "event": public_event})
                            except Exception as exc:
                                ingest_error = str(exc)
                    while pointer < len(pending) and pending[pointer]["after_event_index"] == idx:
                        probe = pending[pointer]
                        pointer += 1
                        row = {k: probe[k] for k in ("probe_id", "episode_id", "expected", "role", "transition_id", "phase", "capability", "old_value")}
                        row.update({"family": episode["family"], "repetition": repetition})
                        capability = probe["capability"]
                        if ingest_error is not None:
                            row.update({"status": "error", "error": "ingest: " + ingest_error})
                        elif capability != "basic" and not adapter.capabilities.get(capability, False):
                            row["status"] = "unsupported"
                        else:
                            try:
                                if adapter.capabilities.get("clock"):
                                    adapter.call({"type": "advance_clock", "as_of": probe["as_of"]})
                                query = {"type": "query", "question": probe["question"], "as_of": probe["as_of"]}
                                if track == "structured":
                                    query.update({"slot": probe["slot"], "scope": probe["scope"]})
                                response, ms = adapter.call(query)
                                if response.get("state") not in ("known", "unknown", "withdrawn", "needs_clarification", "expired"):
                                    raise ValueError("query response requires normalized state")
                                row.update({"status": "answered", "prediction": response,
                                            "latency_ms": round(ms, 3)})
                            except Exception as exc:
                                row.update({"status": "error", "error": str(exc)})
                        rows.append(row)
        finally:
            adapter.close()
        summary = report(rows, seed + repetition)
        suffix = f"-{repetition+1}" if repeat > 1 else ""
        (output / f"predictions{suffix}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        (output / f"report{suffix}.json").write_text(json.dumps(summary, indent=2) + "\n")
        summaries.append(summary)
    if repeat > 1:
        scores = [s["metrics"]["transition_success_rate"]["score"] for s in summaries]
        scores = [s for s in scores if s is not None]
        aggregate = {"transition_success_mean": sum(scores)/len(scores) if scores else None,
                     "transition_success_stddev": statistics.stdev(scores) if len(scores) > 1 else None,
                     "transition_success_range": [min(scores), max(scores)] if scores else None}
        (output / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return summaries
