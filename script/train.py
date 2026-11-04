"""
train.py — Training script for Conditional DDPM on Brainrot dataset.

Usage:
    conda run -n AI python scripts/train.py --config configs/default.json
    conda run -n AI python scripts/train.py --config configs/default.json --resume checkpoints/latest.pth
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch import autocast

from scripts.dataset import (
    BrainrotDataset, ANIMAL_NULL_ID, OBJECT_NULL_ID, PAIR_NULL_ID,
)
from scripts.models.unet import build_unet_from_config
from scripts.diffusion import build_diffusion_from_config
from scripts.utils import (
    set_seed, EMA, save_checkpoint, load_checkpoint,
    save_preview_grid, LocalLogger, get_lr_lambda,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Conditional DDPM")
    parser.add_argument("--config", type=str, default="configs/default.json",
                        help="Path to config JSON")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Load config ---
    with open(args.config, "r") as f:
        config = json.load(f)

    train_cfg = config["training"]
    diff_cfg = config["diffusion"]
    sample_cfg = config["sampling"]
    data_cfg = config["dataset"]

    # --- Setup ---
    device = torch.device(diff_cfg["device"])
    set_seed(diff_cfg["seed"])

    if train_cfg.get("use_tf32", False):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- Dataset ---
    dataset = BrainrotDataset(
        csv_path=data_cfg["train_csv"],
        image_dir=data_cfg["image_dir"],
        image_size=data_cfg["image_size"],
        augment=data_cfg.get("augmentation_hflip", True),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=train_cfg.get("pin_memory", True),
        persistent_workers=train_cfg.get("persistent_workers", True),
        drop_last=True,
    )

    print(f"[INFO] Dataset loaded: {len(dataset)} images")
    print(f"[INFO] Steps per epoch: {len(dataloader)}")

    # --- Model ---
    model = build_unet_from_config(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model parameters: {param_count:,} ({param_count / 1e6:.2f}M)")

    # --- Diffusion ---
    diffusion = build_diffusion_from_config(config)
    diffusion.to(device)

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        betas=tuple(train_cfg["betas"]),
        weight_decay=train_cfg["weight_decay"],
    )

    # --- LR scheduler (warmup) ---
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, get_lr_lambda(train_cfg["warmup_steps"])
    )

    # --- EMA ---
    ema = EMA(model, decay=train_cfg["ema_decay"])

    # --- Mixed precision ---
    mp_dtype_str = train_cfg.get("mixed_precision_dtype", "bf16")
    if mp_dtype_str == "bf16":
        mp_dtype = torch.bfloat16
        # bf16 doesn't need GradScaler
        scaler = None
        use_amp = True
    elif mp_dtype_str == "fp16":
        mp_dtype = torch.float16
        scaler = GradScaler()
        use_amp = True
    else:
        mp_dtype = torch.float32
        scaler = None
        use_amp = False

    print(f"[INFO] Mixed precision: {mp_dtype_str}")

    # --- Logger ---
    logger = LocalLogger(config["log_dir"])

    # --- W&B (optional) ---
    if config.get("use_wandb", False):
        import wandb
        wandb.init(project=config["wandb_project"], name=config["wandb_name"], config=config)

    # --- Resume ---
    global_step = 0
    if args.resume is not None and os.path.exists(args.resume):
        global_step = load_checkpoint(args.resume, model, ema, optimizer, scaler)
        print(f"[INFO] Resumed from step {global_step}")

    # --- Training loop ---
    max_steps = train_cfg["max_steps"]
    cond_dropout_prob = config["condition"]["cond_dropout_prob"]
    save_every = config["save_every"]
    sample_every = config["sample_every"]
    log_every = config.get("log_every", 100)
    ckpt_dir = config["checkpoint_dir"]
    sample_dir = config["sample_preview_dir"]

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    model.train()
    data_iter = iter(dataloader)
    loss_accum = 0.0
    log_count = 0
    t_start = time.time()

    print(f"[INFO] Starting training from step {global_step} to {max_steps}")

    while global_step < max_steps:
        # Get next batch (cycle through dataset)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        images, animal_ids, object_ids, pair_ids = batch
        images = images.to(device)
        animal_ids = animal_ids.to(device)
        object_ids = object_ids.to(device)
        pair_ids = pair_ids.to(device)

        # CFG dropout: randomly replace condition with null labels
        B = images.shape[0]
        drop_mask = torch.rand(B, device=device) < cond_dropout_prob
        animal_ids = torch.where(drop_mask, torch.full_like(animal_ids, ANIMAL_NULL_ID), animal_ids)
        object_ids = torch.where(drop_mask, torch.full_like(object_ids, OBJECT_NULL_ID), object_ids)
        pair_ids = torch.where(drop_mask, torch.full_like(pair_ids, PAIR_NULL_ID), pair_ids)

        # Forward + loss
        optimizer.zero_grad()

        if use_amp:
            with autocast(device_type="cuda", dtype=mp_dtype):
                loss = diffusion.training_loss(model, images, animal_ids, object_ids, pair_ids)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), train_cfg["gradient_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), train_cfg["gradient_clip"])
                optimizer.step()
        else:
            loss = diffusion.training_loss(model, images, animal_ids, object_ids, pair_ids)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg["gradient_clip"])
            optimizer.step()

        lr_scheduler.step()
        ema.update(model)

        global_step += 1
        loss_val = loss.item()
        loss_accum += loss_val
        log_count += 1

        # --- Logging ---
        if global_step % log_every == 0:
            avg_loss = loss_accum / log_count
            elapsed = time.time() - t_start
            steps_per_sec = log_every / elapsed if elapsed > 0 else 0
            lr = optimizer.param_groups[0]["lr"]

            log_data = {
                "step": global_step,
                "loss": round(avg_loss, 6),
                "lr": round(lr, 8),
                "steps_per_sec": round(steps_per_sec, 2),
            }
            logger.log(log_data)

            print(
                f"[Step {global_step:>7d}/{max_steps}] "
                f"loss={avg_loss:.5f}  lr={lr:.6f}  "
                f"speed={steps_per_sec:.1f} steps/s"
            )

            if config.get("use_wandb", False):
                import wandb
                wandb.log(log_data, step=global_step)

            loss_accum = 0.0
            log_count = 0
            t_start = time.time()

        # --- Sample preview ---
        if global_step % sample_every == 0:
            _generate_preview(
                model, ema, diffusion, config, device, mp_dtype if use_amp else torch.float32,
                global_step, sample_dir,
            )

        # --- Save checkpoint ---
        if global_step % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"step_{global_step:07d}.pth")
            save_checkpoint(ckpt_path, model, ema, optimizer, scaler, global_step, config)
            # Also save as latest
            latest_path = os.path.join(ckpt_dir, "latest.pth")
            save_checkpoint(latest_path, model, ema, optimizer, scaler, global_step, config)
            print(f"[INFO] Checkpoint saved: {ckpt_path}")

    # --- Final save ---
    final_path = os.path.join(ckpt_dir, "latest.pth")
    save_checkpoint(final_path, model, ema, optimizer, scaler, global_step, config)
    # Also save model.pth at project root for submission
    save_checkpoint("model.pth", model, ema, optimizer, scaler, global_step, config)
    print(f"[INFO] Training complete at step {global_step}. Final checkpoint saved.")

    logger.close()
    if config.get("use_wandb", False):
        import wandb
        wandb.finish()


@torch.no_grad()
def _generate_preview(model, ema, diffusion, config, device, mp_dtype, step, sample_dir):
    """Generate a 10×10 grid preview (one per pair) using EMA weights."""
    from scripts.dataset import ANIMALS, OBJECTS, ANIMAL_NULL_ID, OBJECT_NULL_ID, PAIR_NULL_ID, pair_id_from

    sample_cfg = config["sampling"]
    model.eval()
    ema.apply(model)

    # Generate one image per pair (100 total)
    animal_ids = []
    object_ids = []
    p_ids = []
    for a_idx in range(len(ANIMALS)):
        for o_idx in range(len(OBJECTS)):
            animal_ids.append(a_idx)
            object_ids.append(o_idx)
            p_ids.append(pair_id_from(a_idx, o_idx))

    animal_ids = torch.tensor(animal_ids, device=device)
    object_ids = torch.tensor(object_ids, device=device)
    p_ids = torch.tensor(p_ids, device=device)

    null_a = torch.full_like(animal_ids, ANIMAL_NULL_ID)
    null_o = torch.full_like(object_ids, OBJECT_NULL_ID)
    null_p = torch.full_like(p_ids, PAIR_NULL_ID)

    shape = (100, 3, config["unet"]["image_size"], config["unet"]["image_size"])

    with autocast(device_type="cuda", dtype=mp_dtype):
        samples = diffusion.ddim_sample(
            model, shape,
            animal_ids, object_ids, p_ids,
            null_a, null_o, null_p,
            ddim_steps=sample_cfg["ddim_steps"],
            cfg_scale=sample_cfg["cfg_scale"],
        )

    ema.restore(model)
    model.train()

    path = os.path.join(sample_dir, f"preview_step_{step:07d}.png")
    save_preview_grid(samples, path, nrow=10)
    print(f"[INFO] Preview saved: {path}")


if __name__ == "__main__":
    main()
