"""Optional remote correction backend; never enabled by default."""
import hashlib
from pathlib import Path


class OpenAICompatCorrection:
    """Post-factum Text Correction; NEED-F-07; MA-17/20; Q-33."""
    version = "openai-compatible"

    def __init__(self, cfg: dict, allow_remote: bool):
        # IMPROV[Q-33]
        if not allow_remote:
            raise PermissionError("remote language model requires llm.allow_remote: true")
        if not cfg.get("base_url", "").startswith("https://"):
            raise ValueError("remote base_url must use HTTPS")
        self.cfg = cfg
        self.model_id = cfg["model_id"]
        self.prompt = Path(cfg["prompt_file"]).read_text(encoding="utf-8")
        self.weights_hash = hashlib.sha256(self.prompt.encode()).hexdigest()

    def correct(self, left_context: str, window: list[str]) -> list[str]:
        import os
        import httpx
        key_name = self.cfg.get("api_key_env", "OPENAI_API_KEY")
        token = os.environ[key_name]
        url = self.cfg["base_url"].rstrip("/") + "/chat/completions"
        payload = {"model": self.model_id, "temperature": self.cfg["temperature"],
                   "messages": [{"role": "system", "content": self.prompt},
                                {"role": "user", "content": left_context + "\n" + " ".join(window)}]}
        with httpx.Client(timeout=30.0, follow_redirects=False) as client:
            response = client.post(url, headers={"Authorization": f"Bearer {token}"}, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].split()
