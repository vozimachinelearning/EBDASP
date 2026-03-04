from __future__ import annotations

from typing import Iterable, Optional


class ValidationBenchmark:
    def __init__(self, queries: Optional[Iterable[str]] = None) -> None:
        self.queries = [q for q in (queries or []) if q]

    def evaluate(self, generator) -> float:
        if not self.queries:
            return 1.0
        total = 0
        passed = 0
        for query in self.queries:
            total += 1
            try:
                result = generator(query)
            except Exception:
                result = ""
            if isinstance(result, str) and result.strip():
                passed += 1
        if total == 0:
            return 1.0
        return passed / float(total)
