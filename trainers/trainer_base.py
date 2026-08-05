from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import torch

from utils.common import ensure_dir


class TrainerBase:
    def __init__(self, config: Dict, ensure_subdirs: Iterable[str] = ("results",)):
        self.config = config
        requested_device = config["training"].get("device", "cpu")
        
        # Validate CUDA availability
        if requested_device == "cuda" and not torch.cuda.is_available():
            import warnings
            warnings.warn(
                f"CUDA requested but not available. Falling back to CPU. "
                f"torch.cuda.is_available()={torch.cuda.is_available()}"
            )
            requested_device = "cpu"
        
        self.device = torch.device(requested_device)
        self.output_dir = Path(config["training"]["output_dir"])
        ensure_dir(self.output_dir)
        for subdir in ensure_subdirs:
            ensure_dir(self.output_dir / subdir)

    def _log(self, message: str) -> None:
        print(message, flush=True)
