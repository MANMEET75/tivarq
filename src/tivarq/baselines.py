"""Reference adapters. Run: python -m tivarq.baselines versioned|latest|hybrid|oracle"""
import json
import re
import sys

from .dataset import canonical, oracle, parse_time


class Memory:
    def __init__(self, mode):
        self.mode = mode
        self.events = []
        self.ids = set()
        self.clock = None

    def ingest(self, request):
        event = request["event"]
        if event["event_id"] in self.ids:
            return
        self.ids.add(event["event_id"])
        if self.mode in ("hybrid", "llm"):
            assertion = parse_turn(event)
            event = {**event, "assertion": assertion}
        self.events.append(event)

    def answer(self, request):
        if self.mode == "hybrid":
            question = request["question"].lower()
            slot = next((s for s, phrase in TOPIC_PATTERNS.items() if phrase in question), None)
            scope = "work" if question.endswith("for work?") else "personal"
        else:
            slot, scope = request["slot"], request["scope"]
        as_of = request["as_of"]
        if self.mode == "latest":
            state, value = "unknown", None
            for event in self.events:
                item = event.get("assertion")
                if item and item["slot"] == slot and item["scope"] == scope and item["op"] != "noop":
                    op = item["op"]
                    if op in ("set", "add"):
                        state, value = "known", item["value"]
                    elif op == "withdraw":
                        state, value = "withdrawn", None
                    else:
                        state, value = "needs_clarification", None
            return {"state": state, "value": canonical(value)}
        return oracle(self.events, len(self.events), slot, scope, as_of)


TOPIC_PATTERNS = {
    "notification_channel": "notification channel", "workspace_theme": "workspace theme",
    "meeting_window": "meeting window", "note_format": "note format",
    "music_genre": "music genre", "coding_language": "coding language",
    "interface_language": "interface language",
}
VALUES = ("email", "chat", "push", "light", "dark", "system", "morning", "afternoon",
          "evening", "markdown", "plain text", "rich text", "jazz", "ambient", "classical",
          "Python", "Go", "Rust", "English", "Spanish", "French")


def parse_turn(event):
    """Small rule extractor to illustrate a hybrid architecture, not an official oracle."""
    text = event["text"]
    low = text.lower()
    slot = next((s for s, phrase in TOPIC_PATTERNS.items() if phrase in low), None)
    if not slot:
        return None
    scope = "work" if "at work" in low or "for work only" in low else "personal"
    op = "set"
    if event.get("speaker") != "user" or "a colleague said" in low:
        op = "noop"
    elif "might prefer" in low or "not sure yet" in low:
        op = "uncertain"
    elif "no longer have" in low:
        op = "withdraw"
    elif "add " in low and "alongside" in low:
        op = "add"
    matches = [v for v in VALUES if re.search(r"(?<!\w)" + re.escape(v.lower()) + r"(?!\w)", low)]
    value = matches[-1] if matches else None
    if op == "add":
        value = next((v for v in VALUES if low.startswith("add " + v.lower())), value)
    elif "from " in low and " to " in low:
        value = next((v for v in VALUES if low.endswith("to " + v.lower() + ".")), value)
    effective = event["arrived_at"]
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    if date_match:
        effective = date_match.group() + "T00:00:00Z"
    return {"op": op, "slot": slot, "scope": scope, "value": value,
            "effective_at": effective}


def serve(mode):
    memory = Memory(mode)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            kind = request["type"]
            if kind == "capabilities":
                response = {"protocol": "tivarq.adapter.v1", "adapter_version": "1.0.0",
                            "tracks": ["conversation"] if mode == "hybrid" else ["structured"],
                            "clock": mode != "latest", "history": mode != "latest",
                            "inspection": mode != "latest"}
            elif kind == "reset":
                memory = Memory(mode)
                response = {"ok": True}
            elif kind == "ingest":
                memory.ingest(request)
                response = {"ok": True}
            elif kind == "advance_clock":
                memory.clock = request["as_of"]
                response = {"ok": True}
            elif kind == "query":
                response = memory.answer(request)
            elif kind == "inspect":
                response = {"events": len(memory.events)}
            else:
                response = {"error": f"unsupported request: {kind}"}
        except Exception as exc:
            response = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("oracle", "versioned", "latest", "hybrid"):
        raise SystemExit("usage: python -m tivarq.baselines oracle|versioned|latest|hybrid")
    serve(sys.argv[1])
