from typing import Optional

class VectorStore:
    def __init__(self, collection_id: str, model_path: Optional[str] = None) -> None:
        self.collection_id = collection_id
        self.model_path = model_path
        if model_path:
            print(f"VectorStore initialized with model path: {model_path}")
