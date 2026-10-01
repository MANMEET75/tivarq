"""Optional OpenAI-compatible conversation adapters using only Python stdlib.

Modes: hybrid (model extracts assertions, deterministic versioned store queries)
and managed (model answers from the conversation). Credentials stay in environment.
"""
import json
import os
import re
import sys
import urllib.request

from .baselines import Memory, TOPIC_PATTERNS


PROMPT_VERSION = "tivarq-model-v1"
STATES = ("known", "unknown", "withdrawn", "needs_clarification", "expired")


def completion(messages):
    endpoint = os.environ.get("TIVARQ_BASE_URL", "http://127.0.0.1:11434/v1").rstrip("/")
    model = os.environ.get("TIVARQ_MODEL", "")
    if not model:
        raise ValueError("TIVARQ_MODEL is required")
    payload = json.dumps({"model": model, "messages": messages, "temperature": float(os.environ.get("TIVARQ_TEMPERATURE", "0")),
                          "response_format": {"type": "json_object"}}).encode()
    headers = {"Content-Type": "application/json"}
    if os.environ.get("TIVARQ_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["TIVARQ_API_KEY"]
    request = urllib.request.Request(endpoint + "/chat/completions", data=payload, headers=headers)
    with urllib.request.urlopen(request, timeout=90) as response:
        result = json.load(response)
    return json.loads(result["choices"][0]["message"]["content"]), result.get("usage", {})


class ModelMemory:
    def __init__(self, mode):
        self.mode = mode
        self.events = []
        self.store = Memory("versioned")

    def ingest(self, event):
        self.events.append({k: event[k] for k in ("event_id", "arrived_at", "speaker", "text")})
        if self.mode == "hybrid":
            system = ("Extract only the user's own temporal preference assertion. Return JSON with keys "
                      "op (set|add|withdraw|uncertain|noop), slot, scope, value, effective_at, expires_at. "
                      "Use noop for third-party statements. Slots: " + ", ".join(TOPIC_PATTERNS) + ". "
                      "Use ISO UTC timestamps. Do not infer unstated values.")
            result, _ = completion([{"role": "system", "content": system},
                                    {"role": "user", "content": json.dumps(event)}])
            if result.get("op") not in ("set", "add", "withdraw", "uncertain", "noop"):
                raise ValueError("invalid extracted op")
            if result.get("op") != "noop":
                assertion = {"op": result["op"], "slot": result["slot"], "scope": result["scope"],
                             "value": result.get("value"), "effective_at": result.get("effective_at") or event["arrived_at"]}
                if result.get("expires_at"):
                    assertion["expires_at"] = result["expires_at"]
                self.store.ingest({"event": {**event, "assertion": assertion}})

    def query(self, request):
        if self.mode == "hybrid":
            question = request["question"].lower()
            slot = next((s for s, phrase in TOPIC_PATTERNS.items() if phrase in question), None)
            scope = "work" if question.endswith("for work?") else "personal"
            return self.store.answer({"slot": slot, "scope": scope, "as_of": request["as_of"]})
        instruction = ("Answer the user's temporal memory question using only the supplied turns. "
                       "Respect arrival versus effective time, scope, withdrawal, uncertainty and expiration. "
                       "Return JSON: {\"state\": one of known|unknown|withdrawn|needs_clarification|expired, "
                       "\"value\": string, list or null}. Do not invent a value.")
        prompt = json.dumps({"turns": self.events, "question": request["question"], "as_of": request["as_of"]})
        result, usage = completion([{"role": "system", "content": instruction},
                                    {"role": "user", "content": prompt}])
        if result.get("state") not in STATES:
            raise ValueError("invalid model state")
        return {"state": result["state"], "value": result.get("value"), "usage": usage}


def serve(mode):
    memory = ModelMemory(mode)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            kind = request["type"]
            if kind == "capabilities":
                response = {"protocol": "tivarq.adapter.v1", "adapter_version": PROMPT_VERSION,
                            "tracks": ["conversation"], "clock": True,
                            "history": True, "inspection": False}
            elif kind == "reset":
                memory = ModelMemory(mode)
                response = {"ok": True}
            elif kind == "ingest":
                memory.ingest(request["event"])
                response = {"ok": True}
            elif kind == "advance_clock":
                response = {"ok": True}
            elif kind == "query":
                response = memory.query(request)
            else:
                response = {"error": "unsupported request"}
        except Exception as exc:
            response = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("hybrid", "managed"):
        raise SystemExit("usage: python -m tivarq.model_adapter hybrid|managed")
    serve(sys.argv[1])
