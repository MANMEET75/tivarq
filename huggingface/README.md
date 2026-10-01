---
license: cc-by-4.0
language:
- en
pretty_name: TIVARQ
task_categories:
- question-answering
tags:
- temporal-memory
- conversational-memory
- benchmark
- synthetic
configs:
- config_name: episodes
  data_files:
  - split: smoke
    path: episodes_smoke.jsonl
  - split: dev
    path: episodes_dev.jsonl
  - split: test
    path: episodes_test.jsonl
- config_name: probes
  data_files:
  - split: smoke
    path: probes_smoke.jsonl
  - split: dev
    path: probes_dev.jsonl
  - split: test
    path: probes_test.jsonl
---

# TIVARQ: Temporal Invalidation and Versioned Assertion Recall Quality

TIVARQ is a synthetic, generic benchmark for whether conversational memory handles changed, withdrawn, future, expired, scoped and uncertain facts. It contains **240 episodes and 1,200 probes**, generated deterministically with seed `20261001`. There are 12 scenario families, each with 1 smoke, 4 development and 15 test episodes. No personal conversation or employer data is included.

The [Python harness](https://github.com/MANMEET75/tivarq) provides the generator, adapter protocol, deterministic scorer, and reference baselines. The dataset is available as two viewer-compatible configurations, `episodes` and `probes`, each with `smoke`, `dev`, and `test` splits. Join probes to episodes using `episode_id`. The benchmark is licensed separately from the harness: **dataset CC BY 4.0; code Apache-2.0**.

## What it tests

For example, a synthetic user chooses email notifications, then changes to chat. At a current checkpoint the answer is chat; for a historical as-of date it is email. An unrelated interface-language setting should remain intact. An uncertain “I might switch” should produce `needs_clarification`, not a guessed replacement.

Scenario families: `replacement`, `reversal`, `withdrawal`, `add_vs_replace`, `scoped_change`, `future_effective`, `expiration`, `retro_correction`, `delayed_arrival`, `duplicate_event`, `third_party_quote`, and `ambiguity`. Topics include notification channels, workspace theme, meeting windows, note formats, music genres and coding languages.

## Schema

`episodes_*.jsonl` contains one episode per row:

```json
{"schema_version":"tivarq.v1","episode_id":"replacement-00","family":"replacement","split":"smoke","seed":20261001,"topic":"notification_channel","events":[{"event_id":"replacement-00-e1","arrived_at":"2025-01-01T00:00:00Z","speaker":"user","text":"For my personal setup, I prefer email as my notification channel.","assertion":{"op":"set","slot":"notification_channel","scope":"personal","value":"email","effective_at":"2025-01-01T00:00:00Z"}}]}
```

`probes_*.jsonl` contains one checkpoint per row with `probe_id`, `episode_id`, `after_event_index`, `as_of`, `question`, `slot`, `scope`, `expected`, `role`, `transition_id`, `phase`, `capability`, and `old_value`. `expected` is `{state, value}`; states are `known`, `unknown`, `withdrawn`, `needs_clarification`, and `expired`. `after_event_index` identifies how many turns the adapter has ingested. `arrived_at` and `effective_at` deliberately differ in some cases. The `assertion` and `expected` fields are gold annotations; **do not pass them to a conversation-track adapter**. The harness enforces this withholding.

## Load the data

```python
from datasets import load_dataset

episodes = load_dataset("MANMEET75/TIVARQ", "episodes", split="smoke")
probes = load_dataset("MANMEET75/TIVARQ", "probes", split="smoke")
print(len(episodes), len(probes))  # 12, 60
```

For reproducibility, pass `revision="<immutable Hub commit SHA>"` to both calls. The exact published commit SHA is recorded in the [release notes](https://github.com/MANMEET75/tivarq/blob/main/docs/RELEASE.md). The repository `manifest.json` gives SHA-256 hashes for all six JSONL files. The dataset can also be downloaded with `huggingface_hub.snapshot_download` and checked with `tivarq validate`.

## Scoring

Run either the **structured lifecycle** track (adapter sees normalized assertions) or the **conversation end-to-end** track (adapter sees only conversation text and natural-language questions). Do not combine their scores into one ranking. The official scorer requires normalized answer state and value and uses no model judge.

The primary metric is **transition success rate**: a transition succeeds only if its pre-change and post-change answers match gold, the invalidated value does not leak as current, and an unrelated control fact survives. Secondary metrics are current-state accuracy, stale-value leakage rate (lower is better), unaffected-fact retention, clarification precision/recall/F1, abstention precision/recall/F1, historical accuracy, expiration accuracy, and query latency. Every rate shows its denominator. Optional capabilities are scored only when declared; unsupported cases are reported. Reports include family breakdowns and an episode-level 95% bootstrap interval for transition success.

See [harness instructions](https://github.com/MANMEET75/tivarq#quickstart-no-model) for installation, all commands, the adapter contract, output files, baselines and optional OpenAI-compatible model examples. No model is required for generation or scoring.

## Generation and limitations

The generator uses fixed templates and neutral topics, with deterministic seed and ordering. Repeated generation must yield identical hashes. Gold resolution uses effective time followed by arrival order, deduplicates event IDs, and represents unknown, withdrawal, ambiguity, scope and expiry explicitly. The oracle reference is expected to pass all probes. A latest-write baseline deliberately fails nuanced cases.

The vocabulary and English templates are small; systems can overfit to wording. The structured track isolates memory logic but does not test extraction. The conversation track combines extraction and memory. The benchmark is focused on temporal invalidation, not broad conversational intelligence. Its sparse checkpoints do not establish a precise continuous update-lag measurement. Please cite the dataset revision and harness commit when reporting results.
