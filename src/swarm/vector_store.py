from __future__ import annotations

import json
import os
import time
import uuid
from typing import Iterable, List, Optional, Dict, Any, Tuple

import torch
from transformers import AutoModel, AutoTokenizer

class VectorStore:
    def __init__(
        self,
        collection_id: str,
        model_path: Optional[str] = None,
        storage_dir: Optional[str] = None,
    ) -> None:
        self.collection_id = collection_id
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        resolved_model_path = model_path or os.getenv("SWARM_EMBEDDING_PATH") or os.getenv("SWARM_EMBED_PATH")
        if not resolved_model_path:
            default_path = os.path.join(base_dir, "models", "embeddings")
            if os.path.exists(default_path):
                resolved_model_path = default_path
        self.model_path = resolved_model_path
        self.storage_dir = storage_dir or os.path.join(base_dir, "storage")
        self.memory_path = os.path.join(self.storage_dir, f"{self.collection_id}.jsonl")
        self._tokenizer = None
        self._model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.model_path:
            print(f"VectorStore initialized with model path: {self.model_path}")

    def _ensure_embedder(self) -> None:
        if not self.model_path:
            raise RuntimeError("Embedding model path is not set.")
        if self._tokenizer is not None and self._model is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self._model = AutoModel.from_pretrained(self.model_path, local_files_only=True)
        self._model.to(self._device)
        self._model.eval()

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        self._ensure_embedder()
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        last_hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        mean_pooled = summed / counts
        normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
        return normalized.cpu().tolist()

    def _cosine_sim(self, query_vec: List[float], record_vec: List[float]) -> float:
        if not query_vec or not record_vec:
            return 0.0
        q = torch.tensor(query_vec, dtype=torch.float32)
        r = torch.tensor(record_vec, dtype=torch.float32)
        q = torch.nn.functional.normalize(q, p=2, dim=0)
        r = torch.nn.functional.normalize(r, p=2, dim=0)
        return float(torch.dot(q, r).item())

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
        embeddings = self._embed_texts([text])
        if embeddings:
            record["embedding"] = embeddings[0]
        with open(self.memory_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def add_memories(self, records: Iterable[Dict[str, Any]]) -> int:
        os.makedirs(self.storage_dir, exist_ok=True)
        count = 0
        to_store: List[Dict[str, Any]] = []
        texts_to_embed: List[Tuple[int, str]] = []
        for record in records:
            if not record:
                continue
            if "text" in record and "embedding" not in record:
                texts_to_embed.append((len(to_store), str(record.get("text", ""))))
            to_store.append(record)
        if texts_to_embed:
            embeddings = self._embed_texts([text for _, text in texts_to_embed])
            for (idx, _), embedding in zip(texts_to_embed, embeddings):
                to_store[idx]["embedding"] = embedding
        with open(self.memory_path, "a", encoding="utf-8") as f:
            for record in to_store:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        return count

    def query_memory(
        self,
        query: str,
        limit: int = 5,
        required_tags: Optional[List[str]] = None,
        exclude_tags: Optional[List[str]] = None,
        min_score: float = 0.2,
    ) -> List[Dict[str, Any]]:
        if not os.path.exists(self.memory_path):
            return []
        required = {tag for tag in (required_tags or []) if tag}
        excluded = {tag for tag in (exclude_tags or []) if tag}
        scored: List[Dict[str, Any]] = []
        candidates: List[Dict[str, Any]] = []
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
                candidates.append(record)

        if not candidates:
            return []
        query_vec = self._embed_texts([query])
        query_vec = query_vec[0] if query_vec else []
        missing_texts: List[Tuple[int, str]] = []
        for idx, record in enumerate(candidates):
            embedding = record.get("embedding")
            if not isinstance(embedding, list):
                missing_texts.append((idx, str(record.get("text", ""))))
        if missing_texts:
            embeddings = self._embed_texts([text for _, text in missing_texts])
            for (idx, _), embedding in zip(missing_texts, embeddings):
                candidates[idx]["embedding"] = embedding
        for record in candidates:
            embedding = record.get("embedding") or []
            score = self._cosine_sim(query_vec, embedding)
            if score >= min_score:
                record["score"] = score
                scored.append(record)
        scored.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return scored[:limit]
