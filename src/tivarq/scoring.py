"""Deterministic, case-counted metrics from normalized predictions."""
from collections import defaultdict
import random
import statistics

from .dataset import canonical


def same(prediction, expected):
    return prediction.get("state") == expected["state"] and canonical(prediction.get("value")) == canonical(expected["value"])


def ratio(numerator, denominator):
    return round(numerator / denominator, 6) if denominator else None


def metric(rows):
    answered = [r for r in rows if r["status"] == "answered"]
    correct = [r for r in answered if same(r["prediction"], r["expected"])]
    current = [r for r in answered if r["capability"] != "history"]
    controls = [r for r in current if r["role"] in ("control", "scope_control") and r["phase"] != "pre"]
    stale_cases = [r for r in answered if r.get("old_value") is not None and r["expected"]["state"] != "known" or
                   r.get("old_value") is not None and canonical(r["expected"].get("value")) != canonical(r["old_value"])]
    stale = [r for r in stale_cases if r["prediction"].get("state") == "known" and
             canonical(r["prediction"].get("value")) == canonical(r["old_value"])]
    def classification(states):
        tp = sum(r["prediction"].get("state") in states and r["expected"]["state"] in states for r in answered)
        fp = sum(r["prediction"].get("state") in states and r["expected"]["state"] not in states for r in answered)
        fn = sum(r["prediction"].get("state") not in states and r["expected"]["state"] in states for r in answered)
        precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
        f1 = ratio(2 * tp, 2 * tp + fp + fn)
        return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    grouped = defaultdict(list)
    for row in answered:
        if row["role"] == "target":
            grouped[(row["episode_id"], row["transition_id"])].append(row)
    transitions = []
    for key, group in grouped.items():
        pre = next((r for r in group if r["phase"] == "pre"), None)
        post = next((r for r in group if r["phase"] == "post"), None)
        if not pre or not post:
            continue
        episode_controls = [r for r in controls if r["episode_id"] == key[0]]
        control_ok = bool(episode_controls) and all(same(r["prediction"], r["expected"]) for r in episode_controls)
        old = post.get("old_value")
        leak = old is not None and post["prediction"].get("state") == "known" and canonical(post["prediction"].get("value")) == canonical(old) and canonical(post["expected"].get("value")) != canonical(old)
        transitions.append({"episode_id": key[0], "success": same(pre["prediction"], pre["expected"]) and
                            same(post["prediction"], post["expected"]) and control_ok and not leak})
    latency = sorted(r["latency_ms"] for r in answered)
    token_usage = [r["prediction"].get("usage", {}) for r in answered]
    def percentile(p):
        if not latency:
            return None
        return round(latency[min(len(latency)-1, int((len(latency)-1)*p))], 3)
    history = [r for r in answered if r["capability"] == "history"]
    expiry = [r for r in answered if r["expected"]["state"] == "expired"]
    lag_rows = [r for r in answered if "update_lag_ms" in r]
    observed_lags = sorted(r["update_lag_ms"] for r in lag_rows if not r["update_lag_censored"])
    return {
        "transition_success_rate": {"score": ratio(sum(t["success"] for t in transitions), len(transitions)), "cases": len(transitions)},
        "current_state_accuracy": {"score": ratio(sum(same(r["prediction"], r["expected"]) for r in current), len(current)), "cases": len(current)},
        "stale_value_leakage_rate": {"score": ratio(len(stale), len(stale_cases)), "cases": len(stale_cases)},
        "unaffected_fact_retention": {"score": ratio(sum(same(r["prediction"], r["expected"]) for r in controls), len(controls)), "cases": len(controls)},
        "clarification": classification({"needs_clarification"}),
        "abstention": classification({"withdrawn", "unknown", "expired"}),
        "historical_accuracy": {"score": ratio(sum(same(r["prediction"], r["expected"]) for r in history), len(history)), "cases": len(history)},
        "expiration_accuracy": {"score": ratio(sum(same(r["prediction"], r["expected"]) for r in expiry), len(expiry)), "cases": len(expiry)},
        "query_latency_ms": {"p50": percentile(.5), "p95": percentile(.95), "cases": len(latency)},
        "update_lag_ms": {"observed_mean": round(statistics.mean(observed_lags), 3) if observed_lags else None,
                          "observed_p95": observed_lags[min(len(observed_lags)-1, int(.95*(len(observed_lags)-1)))] if observed_lags else None,
                          "observed": len(observed_lags), "censored": len(lag_rows)-len(observed_lags),
                          "post_change_cases": len(lag_rows)},
        "reported_tokens": {"prompt": sum(u.get("prompt_tokens", 0) for u in token_usage),
                            "completion": sum(u.get("completion_tokens", 0) for u in token_usage),
                            "cases_with_usage": sum(bool(u) for u in token_usage)},
        "reported_cost_usd": {"total": round(sum(r["prediction"].get("cost_usd", 0) for r in answered), 8),
                              "cases_with_cost": sum("cost_usd" in r["prediction"] for r in answered)},
    }


def report(rows, seed=20261001):
    answered = [r for r in rows if r["status"] == "answered"]
    family_groups = defaultdict(list)
    for row in rows:
        family_groups[row["family"]].append(row)
    rng = random.Random(seed)
    episodes = sorted({r["episode_id"] for r in rows})
    by_episode = {e: [r for r in rows if r["episode_id"] == e] for e in episodes}
    boot = []
    if episodes:
        for _ in range(500):
            sample = [r for e in rng.choices(episodes, k=len(episodes)) for r in by_episode[e]]
            score = metric(sample)["transition_success_rate"]["score"]
            if score is not None:
                boot.append(score)
    boot.sort()
    return {"complete": all(r["status"] in ("answered", "unsupported") for r in rows),
            "probe_counts": {"total": len(rows), "answered": len(answered),
                             "unsupported": sum(r["status"] == "unsupported" for r in rows),
                             "error": sum(r["status"] == "error" for r in rows)},
            "metrics": metric(rows),
            "transition_success_95pct_bootstrap_ci": [boot[int(.025*(len(boot)-1))], boot[int(.975*(len(boot)-1))]] if boot else None,
            "by_family": {name: {"probe_counts": {"total": len(group), "answered": sum(r["status"] == "answered" for r in group)},
                                  "metrics": metric(group)} for name, group in sorted(family_groups.items())}}
