import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from tivarq.dataset import FAMILIES, load_split, oracle, validate_dataset, write_dataset
from tivarq.model_adapter import ModelMemory
from tivarq.runner import run


def test_generation_is_reproducible():
    with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
        a, b = write_dataset(Path(left)), write_dataset(Path(right))
        assert a == b
        assert validate_dataset(Path(left))["probes"] == 1200


def test_gold_has_every_family_and_no_private_topics():
    root = Path(__file__).parents[1] / "dataset"
    episodes = [e for split in ("smoke", "dev", "test") for e in load_split(root, split)[0]]
    assert len(episodes) == 240
    assert {e["family"] for e in episodes} == set(FAMILIES)
    forbidden = ("cars24", "petrol", "diesel", "engram", "chronicle")
    assert all(not any(term in json.dumps(e).lower() for term in forbidden) for e in episodes)


def test_reference_and_naive_expected_behavior():
    root = Path(__file__).parents[1] / "dataset"
    with tempfile.TemporaryDirectory() as output:
        base = [sys.executable, "-m", "tivarq.baselines"]
        oracle_report = run(root, "smoke", "structured", base + ["oracle"], Path(output) / "oracle")[0]
        naive_report = run(root, "smoke", "structured", base + ["latest"], Path(output) / "latest")[0]
        assert oracle_report["metrics"]["transition_success_rate"]["score"] == 1
        assert oracle_report["probe_counts"]["error"] == 0
        assert naive_report["metrics"]["transition_success_rate"]["score"] < 1
        assert naive_report["probe_counts"]["unsupported"] > 0


def test_conversation_contract_and_model_styles(monkeypatch):
    root = Path(__file__).parents[1] / "dataset"
    with tempfile.TemporaryDirectory() as output:
        report = run(root, "smoke", "conversation", [sys.executable, "-m", "tivarq.baselines", "hybrid"], Path(output))[0]
        assert report["probe_counts"]["error"] == 0
        assert report["probe_counts"]["answered"] == 60
    import tivarq.model_adapter as model
    calls = []
    def fake_completion(messages):
        calls.append(messages)
        if "Extract" in messages[0]["content"]:
            return {"op": "set", "slot": "workspace_theme", "scope": "personal", "value": "dark",
                    "effective_at": "2025-01-01T00:00:00Z"}, {}
        return {"state": "known", "value": "dark"}, {"total_tokens": 7}
    monkeypatch.setattr(model, "completion", fake_completion)
    event = {"event_id": "e1", "arrived_at": "2025-01-01T00:00:00Z", "speaker": "user",
             "text": "I prefer dark as my workspace theme."}
    hybrid = ModelMemory("hybrid")
    hybrid.ingest(event)
    assert hybrid.query({"question": "What is my workspace theme for personal?", "as_of": "2025-01-02T00:00:00Z"})["value"] == "dark"
    managed = ModelMemory("managed")
    managed.ingest(event)
    assert managed.query({"question": "What is my workspace theme?", "as_of": "2025-01-02T00:00:00Z"})["state"] == "known"
    assert len(calls) == 2
