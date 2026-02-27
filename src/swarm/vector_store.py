from __future__ import annotations

import json
import os
import time
import uuid
from typing import Iterable, List, Optional, Dict, Any

class VectorStore:
    def __init__(
        self,
        collection_id: str,
        model_path: Optional[str] = None,
        storage_dir: Optional[str] = None,
    ) -> None:
        self.collection_id = collection_id
        self.model_path = model_path
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.storage_dir = storage_dir or os.path.join(base_dir, "storage")
        self.memory_path = os.path.join(self.storage_dir, f"{self.collection_id}.jsonl")
        if model_path:
            print(f"VectorStore initialized with model path: {model_path}")

    def add_memory(self, text: str, source: str, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        if not text:
            return {}
        os.makedirs(self.storage_dir, exist_ok=True)
        record = {
            "memory_id": str(uuid.uuid4()),
            "text": text,
            "source": source,
            "tags": tags or [],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(self.memory_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def add_memories(self, records: Iterable[Dict[str, Any]]) -> int:
        os.makedirs(self.storage_dir, exist_ok=True)
        count = 0
        with open(self.memory_path, "a", encoding="utf-8") as f:
            for record in records:
                if not record:
                    continue
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        return count

    def query_memory(
        self,
        query: str,
        limit: int = 5,
        required_tags: Optional[List[str]] = None,
        exclude_tags: Optional[List[str]] = None,
        min_score: int = 1,
    ) -> List[Dict[str, Any]]:
        if not os.path.exists(self.memory_path):
            return []
        query_tokens = {token.lower() for token in query.split() if token.strip()}
        required = {tag for tag in (required_tags or []) if tag}
        excluded = {tag for tag in (exclude_tags or []) if tag}
        scored: List[Dict[str, Any]] = []
        with open(self.memory_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    continue
                record_tags = {str(tag) for tag in record.get("tags", []) if tag}
                if required and record_tags.isdisjoint(required):
                    continue
                if excluded and not record_tags.isdisjoint(excluded):
                    continue
                text = str(record.get("text", ""))
                tokens = {token.lower() for token in text.split() if token.strip()}
                score = len(query_tokens.intersection(tokens))
                if score >= max(1, min_score):
                    record["score"] = score
                    scored.append(record)
        scored.sort(key=lambda item: item.get("score", 0), reverse=True)
        return scored[:limit]
