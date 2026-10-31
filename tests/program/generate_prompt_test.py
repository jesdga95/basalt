"""Standalone generation agrees with serving at position zero and after batched prefill."""
import argparse
import json
from pathlib import Path
import queue
import subprocess
import tempfile
import threading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expert-cache", default="auto")
    args = parser.parse_args()
    if not Path(args.model).is_file():
        print("SKIP: model pack unavailable")
        return 77
    base = [str(Path(args.engine).resolve()), "--model", str(Path(args.model).resolve()),
            "--gpu", "0", "--max-context", "4096", "--kv", "int8", "--expert-cache", args.expert_cache,
            "--adapt-swaps", "0", "--peer-adapt-swaps", "0"]
    prompts = ("1", "1,2")
    refs = {}
    with tempfile.TemporaryDirectory(prefix="basalt-generate-prompt-") as temp:
        log_path = Path(temp) / "serve.log"
        with log_path.open("w") as log:
            proc = subprocess.Popen([*base, "--serve", "--spec", "3"], stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
            lines = queue.Queue()

            def read():
                for line in proc.stdout:
                    lines.put(line.strip())
                lines.put(None)

            reader = threading.Thread(target=read, daemon=True)
            reader.start()

            def line():
                value = lines.get(timeout=180)
                if value is None or value.startswith("ERR"):
                    raise RuntimeError(f"engine reply {value!r}: {log_path.read_text()[-3000:]}")
                return value

            def send(value):
                proc.stdin.write(value + "\n")
                proc.stdin.flush()

            try:
                slots = None
                while True:
                    reply = line()
                    if reply.startswith("INFO "):
                        fields = dict(item.split("=", 1) for item in reply.split()[1:] if "=" in item)
                        if "expert_slots" in fields:
                            slots = int(fields["expert_slots"])
                    if reply.startswith("READY"):
                        break
                if slots is None or slots <= 0:
                    raise AssertionError("serving did not report its expert cache size")
                base[base.index("--expert-cache") + 1] = str(slots)
                for prompt in prompts:
                    send(f"GEN 8 temperature=0 {prompt}")
                    tokens = []
                    while True:
                        reply = line()
                        if reply.startswith("T "):
                            tokens.append(int(reply.split()[1]))
                        elif reply.startswith("DONE "):
                            break
                    if len(tokens) != 8:
                        raise AssertionError(f"serving {prompt}: expected 8 tokens, got {tokens}")
                    refs[prompt] = tokens
                send("QUIT")
                if proc.wait(timeout=60) != 0:
                    raise RuntimeError(log_path.read_text()[-3000:])
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                reader.join(timeout=5)
                proc.stdin.close()
                proc.stdout.close()
        for prompt, spec, prefill in (("1", "2", "auto"), ("1", "3", "512"), ("1,2", "3", "512")):
            result = subprocess.run([*base, "--tokens", prompt, "--max-new", "8", "--temperature", "0",
                                     "--spec", spec, "--prefill", prefill], capture_output=True, text=True, timeout=180)
            if result.returncode:
                raise RuntimeError(result.stderr[-3000:])
            output = next((line for line in result.stdout.splitlines() if line.startswith("output  :")), None)
            tokens = list(map(int, output.split(":", 1)[1].split())) if output else []
            if tokens != refs[prompt]:
                raise AssertionError(f"prompt={prompt} spec={spec} prefill={prefill}: {tokens} != {refs[prompt]}")
            print(json.dumps({"prompt": prompt, "spec": spec, "prefill": prefill, "tokens": tokens}), flush=True)
    print("PASS: standalone tokens match serving")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
