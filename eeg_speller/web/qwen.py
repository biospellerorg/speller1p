"""Optional local Qwen helper through a CPU-only llama.cpp executable."""
from pathlib import Path
import os
import re
import subprocess
import threading


class LocalQwen:
    def __init__(self, path: Path, cli: Path):
        self.path = path.expanduser().resolve()
        self.cli = cli.expanduser().resolve()
        if not self.path.is_file() or self.path.suffix != ".gguf":
            raise ValueError("--qwen-model must be an existing local GGUF model file")
        if not self.cli.is_file() or not os.access(self.cli, os.X_OK):
            raise ValueError("--qwen-cli must be an executable local llama-cli file")
        self.lock = threading.Lock()

    def status(self):
        return {"ready": True, "loaded": False,
                "model": self.path.name, "message": "local CPU Qwen available"}

    def suggest(self, text: str) -> str:
        prompt = (text[-160:] or "Привет").replace("\r", " ").replace("\n", " ")
        command = [str(self.cli), "-m", str(self.path), "-c", "512", "-t", "4",
                   "-n", "28", "--temp", "0.2", "--no-display-prompt",
                   "--single-turn", "--simple-io", "--log-disable", "-sys",
                   "Продолжи фразу по-русски. Верни только короткое продолжение, не более пяти слов.",
                   "-p", prompt]
        with self.lock:
            result = subprocess.run(command, capture_output=True, text=True, timeout=25, check=True)
        match = re.search(r"\n> [^\n]*\n(.*?)\n\n\[ Prompt:", result.stdout, re.S)
        if not match:
            raise ValueError("Qwen returned an unexpected response")
        suggestion = match.group(1).strip().splitlines()[0][:160]
        if not suggestion:
            raise ValueError("Qwen returned an empty response")
        return suggestion
