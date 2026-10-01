"""Deterministic synthetic temporal-memory episodes and their gold probes."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


VERSION = "tivarq.v1"
SEED = 20261001
FAMILIES = (
    "replacement", "reversal", "withdrawal", "add_vs_replace",
    "scoped_change", "future_effective", "expiration", "retro_correction",
    "delayed_arrival", "duplicate_event", "third_party_quote", "ambiguity",
)
TOPICS = (
    ("notification_channel", "notification channel", ("email", "chat", "push")),
    ("workspace_theme", "workspace theme", ("light", "dark", "system")),
    ("meeting_window", "meeting window", ("morning", "afternoon", "evening")),
    ("note_format", "note format", ("markdown", "plain text", "rich text")),
    ("music_genre", "music genre", ("jazz", "ambient", "classical")),
    ("coding_language", "coding language", ("Python", "Go", "Rust")),
)
CONTROL_VALUES = ("English", "Spanish", "French")


def timestamp(base: datetime, day: int) -> str:
    return (base + timedelta(days=day)).isoformat().replace("+00:00", "Z")


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def canonical(value: Any) -> Any:
    if isinstance(value, list):
        return sorted(set(value), key=str.casefold)
    return value


def oracle(events: list[dict], after_event_index: int, slot: str, scope: str, as_of: str) -> dict:
    """Resolve valid time, then arrival order. Expired values do not resurrect prior values."""
    seen: set[str] = set()
    relevant = []
    for arrival_index, event in enumerate(events[:after_event_index]):
        if event["event_id"] in seen:
            continue
        seen.add(event["event_id"])
        assertion = event.get("assertion")
        if not assertion or assertion["op"] == "noop":
            continue
        if assertion["slot"] != slot or assertion["scope"] != scope:
            continue
        if parse_time(assertion["effective_at"]) <= parse_time(as_of):
            relevant.append((parse_time(assertion["effective_at"]), arrival_index, assertion))
    relevant.sort(key=lambda item: (item[0], item[1]))
    state, value, expires_at = "unknown", None, None
    for _, _, assertion in relevant:
        op = assertion["op"]
        if op == "set":
            state, value = "known", canonical(assertion["value"])
            expires_at = assertion.get("expires_at")
        elif op == "add":
            prior = value if state == "known" else []
            prior = prior if isinstance(prior, list) else [prior]
            state, value = "known", canonical(prior + [assertion["value"]])
            expires_at = assertion.get("expires_at")
        elif op == "withdraw":
            state, value, expires_at = "withdrawn", None, None
        elif op == "uncertain":
            state, value, expires_at = "needs_clarification", None, None
    if state == "known" and expires_at and parse_time(as_of) >= parse_time(expires_at):
        return {"state": "expired", "value": None}
    return {"state": state, "value": value}


def make_episode(family: str, variant: int) -> tuple[dict, list[dict]]:
    topic = TOPICS[(variant + FAMILIES.index(family)) % len(TOPICS)]
    slot, label, choices = topic
    a, b, c = (choices[(variant + j) % 3] for j in range(3))
    control = CONTROL_VALUES[variant % 3]
    base = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=variant * 11 + FAMILIES.index(family) * 230)
    episode_id = f"{family}-{variant:02d}"
    split = "smoke" if variant == 0 else "dev" if variant < 5 else "test"
    events: list[dict] = []
    probes: list[dict] = []
    scope = "personal"

    def event(day: int, text: str, op: str, value: Any = None, *, effective: int | None = None,
              expires: int | None = None, event_id: str | None = None, target_slot: str | None = None,
              target_scope: str | None = None, speaker: str = "user") -> None:
        eid = event_id or f"{episode_id}-e{len(events)+1}"
        assertion = {"op": op, "slot": target_slot or slot, "scope": target_scope or scope,
                     "value": value, "effective_at": timestamp(base, day if effective is None else effective)}
        if expires is not None:
            assertion["expires_at"] = timestamp(base, expires)
        events.append({"event_id": eid, "arrived_at": timestamp(base, day),
                       "speaker": speaker, "text": text, "assertion": assertion})

    def probe(day: int, role: str, *, after: int | None = None, target_slot: str | None = None,
              target_scope: str | None = None, transition: str = "main", phase: str = "post",
              capability: str = "basic", old_value: Any = None) -> None:
        idx = len(events) if after is None else after
        ps, sc = target_slot or slot, target_scope or scope
        human_label = label if ps == slot else "interface language"
        question = f"As of {timestamp(base, day)[:10]}, what is my {human_label} for {sc}?"
        as_of = timestamp(base, day)
        probes.append({"probe_id": f"{episode_id}-q{len(probes)+1}", "episode_id": episode_id,
                       "after_event_index": idx, "as_of": as_of, "question": question,
                       "slot": ps, "scope": sc, "expected": oracle(events, idx, ps, sc, as_of),
                       "role": role, "transition_id": f"{episode_id}-{transition}",
                       "phase": phase, "capability": capability, "old_value": old_value})

    first_expiry = 5 if family == "expiration" else None
    event(0, f"For my personal setup, I prefer {a} as my {label}.", "set", a, expires=first_expiry)
    event(0, f"My interface language is {control} for personal use.", "set", control,
          target_slot="interface_language")
    event(1, "The shared workspace has a weekly status meeting.", "noop", speaker="assistant")
    probe(1, "target", transition="main", phase="pre")
    probe(1, "control", target_slot="interface_language", transition="main", phase="pre")

    if family == "replacement":
        event(3, f"I changed my personal {label} to {b}. Use {b} from now on.", "set", b)
        probe(4, "target", old_value=a)
    elif family == "reversal":
        event(3, f"Please switch my personal {label} to {b}.", "set", b)
        probe(3, "target", transition="first", phase="post", old_value=a)
        event(6, f"I have changed back: my personal {label} is {a} again.", "set", a)
        probe(7, "target", old_value=b)
    elif family == "withdrawal":
        event(3, f"I no longer have a personal {label} preference. Please remove it.", "withdraw")
        probe(4, "target", old_value=a)
    elif family == "add_vs_replace":
        event(3, f"Add {b} alongside {a} for my personal {label}; both are fine.", "add", b)
        probe(4, "target", transition="add", phase="post")
        event(6, f"Replace those options. My only personal {label} now is {c}.", "set", c)
        probe(7, "target", old_value=[a, b])
    elif family == "scoped_change":
        event(2, f"At work, I prefer {b} as my {label}.", "set", b, target_scope="work")
        probe(2, "target", target_scope="work", phase="pre")
        event(4, f"For work only, change my {label} from {b} to {c}.", "set", c, target_scope="work")
        probe(5, "target", target_scope="work", old_value=b)
        probe(5, "scope_control", old_value=None)
    elif family == "future_effective":
        event(2, f"Starting on {timestamp(base, 5)[:10]}, my personal {label} will be {b}.",
              "set", b, effective=5)
        probe(4, "target", transition="future", phase="pre", capability="clock")
        probe(5, "target", transition="future", phase="post", capability="clock", old_value=a)
    elif family == "expiration":
        probe(4, "target", transition="expiry", phase="pre", capability="clock")
        probe(5, "target", transition="expiry", phase="post", capability="clock", old_value=a)
    elif family == "retro_correction":
        event(4, f"From today, my personal {label} is {b}.", "set", b)
        probe(5, "target", transition="main", phase="post", old_value=a)
        probe(3, "target", transition="history", phase="pre", capability="history")
        event(6, f"Correction: on {timestamp(base, 2)[:10]} I actually preferred {c}, not {a}. My later choice still stands.",
              "set", c, effective=2)
        probe(3, "target", transition="history", phase="post", capability="history", old_value=a)
        probe(7, "target", transition="current", phase="check")
    elif family == "delayed_arrival":
        event(3, f"As of {timestamp(base, 3)[:10]}, my personal {label} is {b}.", "set", b, effective=3)
        probe(4, "target", transition="late", phase="pre")
        event(6, f"This message arrived late: on {timestamp(base, 2)[:10]} I tried {c} as my personal {label}.",
              "set", c, effective=2)
        probe(7, "target", transition="late", phase="post", old_value=c)
    elif family == "duplicate_event":
        old_id = events[0]["event_id"]
        event(3, f"Now my personal {label} is {b}.", "set", b)
        probe(3, "target", transition="duplicate", phase="pre")
        event(6, events[0]["text"], "set", a, effective=0, event_id=old_id)
        probe(7, "target", transition="duplicate", phase="post", old_value=a)
    elif family == "third_party_quote":
        event(3, f"A colleague said they prefer {b} as their {label}; that was about them, not me.",
              "noop", b)
        probe(4, "target", old_value=b)
    elif family == "ambiguity":
        event(3, f"I might prefer {b} for my personal {label}, but I am not sure yet.", "uncertain", b)
        probe(4, "target", old_value=a)
        event(6, f"I have decided: set my personal {label} to {b}.", "set", b)
        probe(7, "target", transition="resolution", phase="post", old_value=a)
    else:
        raise ValueError(family)

    probe(8, "control", target_slot="interface_language")
    episode = {"schema_version": VERSION, "episode_id": episode_id, "family": family,
               "split": split, "seed": SEED, "topic": slot, "events": events}
    return episode, probes


def iter_dataset():
    for family in FAMILIES:
        for variant in range(20):
            yield make_episode(family, variant)


def json_line(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def write_dataset(root: Path) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    buckets = {f"{kind}_{split}.jsonl": [] for kind in ("episodes", "probes")
               for split in ("smoke", "dev", "test")}
    for episode, probes in iter_dataset():
        split = episode["split"]
        buckets[f"episodes_{split}.jsonl"].append(json_line(episode))
        buckets[f"probes_{split}.jsonl"].extend(json_line(p) for p in probes)
    digests = {}
    for name, lines in buckets.items():
        content = "".join(lines).encode()
        (root / name).write_bytes(content)
        digests[name] = hashlib.sha256(content).hexdigest()
    manifest = {"schema_version": VERSION, "seed": SEED,
                "episode_count": 240, "families": list(FAMILIES), "files": digests}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return digests


def load_split(root: Path, split: str) -> tuple[list[dict], list[dict]]:
    def rows(kind: str) -> list[dict]:
        return [json.loads(line) for line in (root / f"{kind}_{split}.jsonl").read_text().splitlines()]
    return rows("episodes"), rows("probes")


def validate_dataset(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["schema_version"] == VERSION
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    counts = Counter()
    probe_count = 0
    for split in ("smoke", "dev", "test"):
        episodes, probes = load_split(root, split)
        index = {e["episode_id"]: e for e in episodes}
        assert len(index) == len(episodes)
        for episode in episodes:
            assert episode["schema_version"] == VERSION and episode["split"] == split
            counts[(episode["family"], split)] += 1
            text = json.dumps(episode).lower()
            assert not any(term in text for term in ("cars24", "car preference", "petrol", "diesel", "chronicle", "engram"))
        for probe in probes:
            ep = index[probe["episode_id"]]
            assert 0 <= probe["after_event_index"] <= len(ep["events"])
            expected = oracle(ep["events"], probe["after_event_index"], probe["slot"],
                              probe["scope"], probe["as_of"])
            assert probe["expected"] == expected, probe["probe_id"]
            probe_count += 1
    for family in FAMILIES:
        assert [counts[(family, split)] for split in ("smoke", "dev", "test")] == [1, 4, 15]
    assert sum(counts.values()) == 240
    return {"episodes": 240, "probes": probe_count, "families": len(FAMILIES),
            "split_counts": {split: sum(counts[(f, split)] for f in FAMILIES)
                             for split in ("smoke", "dev", "test")}}
