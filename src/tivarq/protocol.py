"""Line-delimited JSON adapter client. Every request has exactly one response."""
import json
import subprocess
import time


class AdapterError(RuntimeError):
    pass


class Adapter:
    def __init__(self, command: list[str]):
        self.command = command
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True, bufsize=1)
        self.capabilities = self.call({"type": "capabilities"})[0]
        if self.capabilities.get("protocol") != "tivarq.adapter.v1":
            raise AdapterError("adapter must declare protocol tivarq.adapter.v1")

    def call(self, request: dict) -> tuple[dict, float]:
        assert self.process.stdin and self.process.stdout
        start = time.perf_counter()
        try:
            self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
        except (BrokenPipeError, OSError) as exc:
            raise AdapterError(str(exc)) from exc
        elapsed_ms = (time.perf_counter() - start) * 1000
        if not line:
            raise AdapterError(f"adapter exited before replying to {request['type']}")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AdapterError(f"invalid JSON from adapter: {line[:200]}") from exc
        if response.get("error"):
            raise AdapterError(response["error"])
        return response, elapsed_ms

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
