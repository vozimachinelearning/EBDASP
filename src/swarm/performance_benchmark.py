from __future__ import annotations

import time
from typing import Optional


class PerformanceBenchmark:
    def __init__(self, prompt: str = "benchmark", max_new_tokens: int = 64) -> None:
        self.prompt = prompt
        self.max_new_tokens = int(max_new_tokens)

    def measure(self, generator) -> Optional[float]:
        start = time.time()
        try:
            generator(self.prompt, max_new_tokens=self.max_new_tokens)
        except Exception:
            return None
        elapsed = time.time() - start
        if elapsed <= 0:
            return None
        return 1.0 / elapsed
