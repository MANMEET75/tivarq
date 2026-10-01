# TIVARQ

**Temporal Invalidation and Versioned Assertion Recall Quality** is a synthetic benchmark for conversational memory. It asks whether a system can update what is true now, retain what used to be true, preserve unrelated facts, and request clarification when a change is uncertain.

> Monday: “Use email for my notifications.” Thursday: “Switch to chat.” A good memory answers **chat** for Friday, **email** for Tuesday, and does not forget an unrelated setting. If the user says “I might switch to push,” it should ask for clarification rather than silently replace chat.

TIVARQ has **240 seeded episodes, 12 scenario families, and 1,200 probes**. All data is synthetic. The generator, runner, scorer and baselines use Python's standard library. No paid model or LLM judge is needed for official scoring.

## Two tracks

| Track | Input to adapter | What it isolates |
| --- | --- | --- |
| `structured` | Timestamped turns **plus normalized assertions** | Memory lifecycle, valid time, scope and invalidation without extraction errors. Works with deterministic systems. |
| `conversation` | Timestamped turns and natural-language questions only | Extraction and memory together. Works with deterministic, hybrid and LLM-managed systems. |

Track results are **separate**. Do not rank a structured score against a conversation score. Optional history and clock scores include capability coverage and case counts.

## Quickstart: no model

Prerequisites: Python 3.11+ and Git. From a checkout of this repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
tivarq validate --dataset dataset
tivarq run --dataset dataset --split smoke --track structured --output runs/oracle-smoke -- python -m tivarq.baselines oracle
tivarq run --dataset dataset --split smoke --track structured --output runs/latest-smoke -- python -m tivarq.baselines latest
tivarq run --dataset dataset --split smoke --track conversation --output runs/hybrid-smoke -- python -m tivarq.baselines hybrid
```

The oracle should have transition success **1.0** on 13 smoke transitions. The latest-write baseline should score lower and declare history/clock unsupported. The rule-based hybrid example illustrates conversation extraction and is not expected to match the structured oracle.

Run the development and held-out test splits:

```bash
tivarq run --dataset dataset --split dev --track structured --output runs/versioned-dev -- python -m tivarq.baselines versioned
tivarq run --dataset dataset --split test --track structured --output runs/versioned-test -- python -m tivarq.baselines versioned
tivarq run --dataset dataset --split test --track conversation --output runs/hybrid-test -- python -m tivarq.baselines hybrid
```

Every output directory has `predictions.jsonl`, `report.json`, and `run_manifest.json`. The report includes denominators and family breakdowns; the manifest records dataset file hashes, harness commit, adapter version, model/prompt settings, sampling settings, and seed. With `--repeat 3`, numbered prediction/report files and `aggregate.json` are written. Pin a prompt, model revision and sampling configuration in `--model-config`, `--prompt-config`, and `TIVARQ_SAMPLING` for model runs.

## Download the published dataset

The canonical data is [MANMEET75/TIVARQ](https://huggingface.co/datasets/MANMEET75/TIVARQ). This checkout also includes the generated files in `dataset/`. For an external run, pin the Hugging Face **commit SHA** as an immutable revision:

```bash
python -m pip install "huggingface_hub>=0.36,<2"
export TIVARQ_DATASET_REVISION="cae7ab63ff9daf1f9d167978d5496b93554ac447"
python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(repo_id="MANMEET75/TIVARQ", repo_type="dataset",
                  revision=os.environ["TIVARQ_DATASET_REVISION"], local_dir="hf-dataset")
PY
tivarq validate --dataset hf-dataset
```

The GitHub and Hugging Face copies must have identical SHA-256 hashes in `dataset/manifest.json`. The published immutable revision will be recorded in the [release notes](docs/RELEASE.md).

## Dataset design

Each family has one smoke, four development, and fifteen test episodes. Within an episode the runner keeps memory across turns; it calls `reset` before the next episode. Probe checkpoints occur before and after changes, and include unaffected-fact controls.

| Family | Challenge |
| --- | --- |
| `replacement` | New choice supersedes old |
| `reversal` | A → B → A |
| `withdrawal` | Remove a choice without inventing a replacement |
| `add_vs_replace` | Add an option, then replace the whole set |
| `scoped_change` | Work setting changes while personal setting survives |
| `future_effective` | Arrives now, becomes valid later |
| `expiration` | Choice expires without a new message |
| `retro_correction` | Later message corrects earlier valid time |
| `delayed_arrival` | Late-arriving event must not override a newer effective choice |
| `duplicate_event` | Replayed event ID has no extra effect |
| `third_party_quote` | Someone else's preference is not the user's |
| `ambiguity` | Uncertain change needs clarification |

Episode rows contain `episode_id`, `family`, `split`, `topic`, and `events`. Each event has `event_id`, `arrived_at`, `speaker`, `text`, and a gold `assertion` with `op`, `slot`, `scope`, `value`, `effective_at`, optionally `expires_at`. The distinction between **arrival time** and **valid time** is essential. Probe rows contain `after_event_index`, `as_of`, `question`, `slot`, `scope`, `expected: {state,value}`, transition IDs/phases, role, and required capability. States are `known`, `unknown`, `withdrawn`, `needs_clarification`, and `expired`.

The runner never sends `expected` to an adapter. In the conversation track it also withholds the normalized assertion and the probe slot/scope. See [the adapter contract](docs/ADAPTER_PROTOCOL.md) for full request examples and [the dataset card](huggingface/README.md) for loading the two configurations.

## Wrap your own memory system

Write an executable that reads **one JSON object per line** from stdin and writes **one JSON object per line** to stdout. It must handle `capabilities`, `reset`, `ingest`, and `query`. Declare `clock`, `history`, and `inspection` support accurately. Keep logs on stderr; stdout is protocol only. The complete contract is in [docs/ADAPTER_PROTOCOL.md](docs/ADAPTER_PROTOCOL.md). Examples:

```bash
tivarq run --dataset dataset --split smoke --track structured --output runs/my-system -- python my_adapter.py
tivarq run --dataset dataset --split smoke --track conversation --output runs/my-conversation-system -- python my_adapter.py
```

Your adapter can call an API, run a local model, use a database, or be fully deterministic. Keep it isolated from `dataset/probes_*.jsonl` and the expected answers. Declare unsupported features rather than fabricating an answer.

## Optional OpenAI-compatible model examples

`tivarq.model_adapter hybrid` asks a model to extract assertions, then uses deterministic versioned storage. `tivarq.model_adapter managed` asks a model to answer from the observed conversation. Both use the standard OpenAI-compatible `/chat/completions` wire format and JSON mode; model support for `response_format` varies. These examples are not part of official scorer dependencies.

With a local endpoint, for example Ollama's OpenAI-compatible API:

```bash
export TIVARQ_BASE_URL=http://127.0.0.1:11434/v1
export TIVARQ_MODEL=your-installed-model
export TIVARQ_TEMPERATURE=0
tivarq run --dataset dataset --split smoke --track conversation --output runs/model-hybrid \
  --model-config "${TIVARQ_MODEL}" --prompt-config tivarq-model-v1 -- python -m tivarq.model_adapter hybrid
```

For a hosted compatible endpoint, set `TIVARQ_BASE_URL`, `TIVARQ_MODEL`, and `TIVARQ_API_KEY` in your environment. Keep credentials outside the repository and results. Run `managed` instead of `hybrid` to test model-managed memory. For stochastic models use `--repeat 3` or more, record temperature and model revision, and compare mean and variation within the same track.

## Metrics

The scorer compares normalized state and value exactly; list values are treated as sets. `n` is always reported. Undefined rates are `null`, not zero.

| Metric | Formula and interpretation |
| --- | --- |
| **Transition success** (primary) | Successful paired pre/post transitions ÷ eligible transitions. Both answers must match gold, the old value must not leak as current, and the unrelated post-change control must remain correct. Higher is better. |
| Current-state accuracy | Correct non-historical checkpoints ÷ answered non-historical checkpoints. Higher is better. |
| Stale-value leakage | Invalidated old values returned as current ÷ answered stale-risk probes. Lower is better. |
| Unaffected-fact retention | Correct post-change controls ÷ answered post-change controls. Higher is better. |
| Clarification precision/recall/F1 | Standard binary classification for `needs_clarification`; catches both missed and unnecessary questions. |
| Abstention precision/recall/F1 | Binary classification for `unknown`, `withdrawn`, or `expired` versus an asserted choice. Higher is better. |
| Historical accuracy | Correct historical probes ÷ answered historical probes, only for adapters declaring history. |
| Expiration accuracy | Correct expired probes ÷ answered expiration probes, only where clock is declared. |
| Update lag | Time from first post-change query until a correct answer, with observed and censored case counts. Use `--update-polls 5 --poll-interval-ms 100` to allow asynchronous systems to catch up. Official quality still uses the first answer. |
| Query latency and operating cost | Wall-clock p50/p95 query latency, plus optional reported prompt/completion tokens and USD cost; separate from quality. |

The primary interval is an **episode-level 95% bootstrap confidence interval** with 500 seeded resamples. Reports include every denominator, capability coverage, and a family breakdown. Unsupported optional probes are counted as unsupported. Errors make a run incomplete. The update-lag clock starts at the first post-change query, not at message arrival; v1's sparse checkpoints do not establish continuous time to consistency. The polling command sends extra read-only queries only for lag measurement.

## Reproducibility and limitations

Regenerate and compare hashes:

```bash
tivarq generate --dataset /tmp/tivarq-regenerated
tivarq validate --dataset /tmp/tivarq-regenerated
diff dataset/manifest.json /tmp/tivarq-regenerated/manifest.json
```

The data is templated synthetic English, with a small value vocabulary. A parser can overfit to wording. The structured track excludes extraction quality; the conversation track combines extraction and memory quality. Exact normalized scoring avoids an LLM judge but requires adapters to map their answers to the protocol. The 12 families are deliberately focused and do not represent all real-world temporal dialogue. Reference baseline reports live in [`reports/`](reports/); they are examples, not a leaderboard. No private system, user history, or employer data is included.

## License

Harness and code: [Apache License 2.0](LICENSE). Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) as specified in the dataset card. Cite the repository and dataset revision when publishing results.
