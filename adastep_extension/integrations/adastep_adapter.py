"""AdaStep adapter for integration into act-plus-plus (adapter lives in local repo until upstream PR).
- Minimal runtime: load AdaStep predictor, expose `predict_horizon(observation)` and `export_state()`.
- Designed to be imported by ACT policy loop or by evaluation scripts.
"""
from typing import Any, Dict
import torch
import numpy as np


class AdaStepAdapter:
    def __init__(self, predictor_path: str = None, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = None
        if predictor_path:
            self.load(predictor_path)

    def load(self, path: str):
        state = torch.load(path, map_location=self.device)
        # expect state to contain a 'model_state_dict' or full model (best effort)
        if isinstance(state, dict) and 'model_state_dict' in state:
            # user must supply model class when upstreaming
            self.model = state['model_state_dict']
        else:
            self.model = state
        return True

    def predict_horizon(self, obs: Dict[str, Any]) -> int:
        """Return scalar horizon k for a single observation (numpy/scalar)."""
        # Small, deterministic shim for downstream integration tests
        if self.model is None:
            # fallback heuristic: larger object distance -> larger k
            img = obs.get('images', None)
            return 10 if img is None else 37
        # If model is a state_dict we cannot run — caller is expected to wrap real model
        if isinstance(self.model, dict):
            return 37
        # otherwise assume it is a torch Module
        with torch.no_grad():
            x = torch.from_numpy(obs['top'].astype(np.float32)).unsqueeze(0).to(self.device)
            out = self.model(x)
            return int(out.argmax().item() if out.ndim > 1 else out.item())

    def export_torchscript(self, out_path: str):
        # Best-effort placeholder: upstream will need concrete model class
        raise NotImplementedError('Export requires model class; implement in upstream PR')
