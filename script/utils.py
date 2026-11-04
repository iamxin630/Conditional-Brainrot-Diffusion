"""
Utility helpers: EMA, seeding, checkpoint management, preview grid.
"""

import copy
import json
import os
import random

import numpy as np
import torch
from torchvision.utils import make_grid, save_image


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average)
# ---------------------------------------------------------------------------

class EMA:
    """
    Maintains an exponential moving average of model parameters.

    Usage:
        ema = EMA(model, decay=0.9999)
        # after each optimizer step:
        ema.update(model)
        # to sample with EMA weights:
        ema.apply(model)      # swap in EMA weights
        # ... sample ...
        ema.restore(model)    # swap back original weights
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply(self, model: torch.nn.Module):
        """Replace model params with EMA params (saves backup)."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module):
        """Restore model params from backup."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return copy.deepcopy(self.shadow)

    def load_state_dict(self, state_dict):
        self.shadow = copy.deepcopy(state_dict)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str,
    model,
    ema: EMA,
    optimizer,
    scaler,
    step: int,
    config: dict,
):
    """Save a training checkpoint."""
    from scripts.dataset import ANIMAL_TO_ID, OBJECT_TO_ID

    if d := os.path.dirname(path):
        os.makedirs(d, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "step": step,
            "config": config,
            "animal_to_id": ANIMAL_TO_ID,
            "object_to_id": OBJECT_TO_ID,
        },
        path,
    )


def load_checkpoint(path: str, model, ema: EMA = None, optimizer=None, scaler=None):
    """Load a training checkpoint. Returns the global step."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if ema is not None and "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("step", 0)


# ---------------------------------------------------------------------------
# Preview grid
# ---------------------------------------------------------------------------

def save_preview_grid(
    images: torch.Tensor,
    path: str,
    nrow: int = 10,
):
    """
    Save a grid of generated images.

    Parameters
    ----------
    images : [N, C, H, W] in [-1, 1]
    path : output PNG path
    nrow : number of images per row
    """
    if d := os.path.dirname(path):
        os.makedirs(d, exist_ok=True)
    # Denormalize from [-1, 1] to [0, 1]
    images = (images.clamp(-1, 1) + 1) / 2
    save_image(images, path, nrow=nrow)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class LocalLogger:
    """Simple JSON-line logger."""

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, "train_log.jsonl")
        self.fp = open(self.log_path, "a")

    def log(self, data: dict):
        self.fp.write(json.dumps(data) + "\n")
        self.fp.flush()

    def close(self):
        self.fp.close()


# ---------------------------------------------------------------------------
# Learning rate scheduler (linear warmup then constant)
# ---------------------------------------------------------------------------

def get_lr_lambda(warmup_steps: int):
    """Returns a lr_lambda for torch.optim.lr_scheduler.LambdaLR."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0
    return lr_lambda
