from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional
import threading

import torch
import torch.nn as nn


class ActivationTracker:
    def __init__(self, model: nn.Module, activation_threshold: float = 0.1, enabled: bool = True) -> None:
        self.model = model
        self.activation_threshold = float(activation_threshold)
        self.enabled = enabled
        self._active_collection: Optional[str] = None
        self._hooks = []
        self._lock = threading.Lock()
        self._counts: Dict[str, Dict[str, Dict[int, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        if self.enabled:
            self._register_hooks()

    def set_collection(self, collection_id: Optional[str]) -> None:
        self._active_collection = collection_id

    def get_stats(self, collection_id: str) -> Dict[str, Dict[int, int]]:
        with self._lock:
            return {layer: dict(neurons) for layer, neurons in self._counts.get(collection_id, {}).items()}

    def reset(self, collection_id: Optional[str] = None) -> None:
        with self._lock:
            if collection_id is None:
                self._counts.clear()
                return
            self._counts.pop(collection_id, None)

    def _register_hooks(self) -> None:
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                self._hooks.append(module.register_forward_hook(self._make_hook(name, "linear")))
            elif isinstance(module, nn.Conv2d):
                self._hooks.append(module.register_forward_hook(self._make_hook(name, "conv2d")))

    def _make_hook(self, layer_name: str, layer_type: str):
        def hook(_module, _input, output):
            if not self.enabled:
                return
            collection_id = self._active_collection or "default"
            if isinstance(output, tuple):
                output = output[0]
            if not torch.is_tensor(output):
                return
            with torch.no_grad():
                if layer_type == "conv2d" and output.dim() >= 4:
                    activations = output.detach().abs()
                    flat = activations.permute(1, 0, 2, 3).contiguous().view(activations.shape[1], -1)
                    mask = flat > self.activation_threshold
                    active = torch.nonzero(mask.any(dim=1), as_tuple=False).view(-1).tolist()
                else:
                    activations = output.detach().abs()
                    if activations.dim() < 2:
                        return
                    flat = activations.view(-1, activations.shape[-1])
                    mask = flat > self.activation_threshold
                    active = torch.nonzero(mask.any(dim=0), as_tuple=False).view(-1).tolist()
            if not active:
                return
            with self._lock:
                layer_counts = self._counts[collection_id][layer_name]
                for idx in active:
                    layer_counts[int(idx)] += 1
        return hook
