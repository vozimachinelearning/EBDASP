from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class Pruner:
    def __init__(self, base_model: nn.Module, activation_stats: Dict[str, Dict[str, Dict[int, int]]], threshold_percent: float = 5.0) -> None:
        self.base_model = base_model
        self.activation_stats = activation_stats
        self.threshold_percent = float(threshold_percent)

    def prune_for_collection(self, collection_id: str, save_dir: str) -> Optional[Tuple[nn.Module, Dict[str, Any]]]:
        stats = self.activation_stats.get(collection_id)
        if not stats:
            return None
        model_copy = copy.deepcopy(self.base_model)
        name_map = dict(model_copy.named_modules())
        pruned_layers = []
        with torch.no_grad():
            for layer_name, neuron_counts in stats.items():
                module = name_map.get(layer_name)
                if module is None:
                    continue
                if not isinstance(module, (nn.Linear, nn.Conv2d)):
                    continue
                if not neuron_counts:
                    continue
                max_count = max(neuron_counts.values())
                if max_count <= 0:
                    continue
                threshold = (self.threshold_percent / 100.0) * float(max_count)
                to_prune = [idx for idx, count in neuron_counts.items() if count < threshold]
                if not to_prune:
                    continue
                if isinstance(module, nn.Linear):
                    weight = module.weight
                    mask = torch.ones_like(weight)
                    mask[to_prune, :] = 0
                    module.weight.mul_(mask)
                    if module.bias is not None:
                        module.bias[to_prune] = 0
                else:
                    weight = module.weight
                    mask = torch.ones_like(weight)
                    mask[to_prune, :, :, :] = 0
                    module.weight.mul_(mask)
                    if module.bias is not None:
                        module.bias[to_prune] = 0
                pruned_layers.append(layer_name)
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"model_{collection_id}.pt")
        torch.save(model_copy.state_dict(), model_path)
        original_params = sum(p.numel() for p in self.base_model.parameters())
        remaining_params = sum(torch.count_nonzero(p).item() for p in model_copy.parameters())
        meta = {
            "collection": collection_id,
            "original_params": original_params,
            "remaining_params": remaining_params,
            "threshold_percent": self.threshold_percent,
            "pruned_layers": pruned_layers,
            "model_path": model_path,
        }
        meta_path = os.path.join(save_dir, f"meta_{collection_id}.json")
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, ensure_ascii=False)
        return model_copy, meta
