"""
generate.py — Generate 2,000 images for HW6 submission.

Reads generate.csv, uses DDIM + CFG with EMA weights to produce PNG images.

Usage:
    conda run -n AI python scripts/generate.py --ckpt model.pth --output_dir generated_images
    conda run -n AI python scripts/generate.py --ckpt model.pth --steps 100 --cfg_scale 2.0
"""

import argparse
import csv
import json
import os

import torch
from torch import autocast
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from scripts.dataset import (
    ANIMAL_TO_ID, OBJECT_TO_ID,
    ANIMAL_NULL_ID, OBJECT_NULL_ID, PAIR_NULL_ID,
    pair_id_from,
)
from scripts.models.unet import build_unet_from_config
from scripts.diffusion import build_diffusion_from_config
from scripts.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images for HW6 submission")
    parser.add_argument("--ckpt", type=str, default="model.pth",
                        help="Path to checkpoint (uses EMA weights)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config JSON (overrides checkpoint config)")
    parser.add_argument("--generate_csv", type=str, default=None,
                        help="Path to generate.csv (overrides config)")
    parser.add_argument("--output_dir", type=str, default="generated_images",
                        help="Output directory for generated PNGs")
    parser.add_argument("--steps", type=int, default=None,
                        help="DDIM sampling steps (overrides config)")
    parser.add_argument("--cfg_scale", type=float, default=None,
                        help="CFG guidance scale (overrides config)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Batch size for generation")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["fp32", "fp16", "bf16"],
                        help="Inference precision")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)
    set_seed(args.seed)

    # --- Load checkpoint ---
    print(f"[INFO] Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    # Get config from checkpoint or from file
    if args.config is not None:
        with open(args.config, "r") as f:
            config = json.load(f)
    else:
        config = ckpt["config"]

    # Override sampling params
    ddim_steps = args.steps if args.steps is not None else config["sampling"]["ddim_steps"]
    cfg_scale = args.cfg_scale if args.cfg_scale is not None else config["sampling"]["cfg_scale"]
    generate_csv = args.generate_csv if args.generate_csv is not None else config["dataset"]["generate_csv"]

    print(f"[INFO] DDIM steps: {ddim_steps}")
    print(f"[INFO] CFG scale:  {cfg_scale}")

    # --- Build model ---
    model = build_unet_from_config(config).to(device)

    # Load EMA weights (preferred) or model weights
    if "ema" in ckpt:
        model_state = {}
        ema_shadow = ckpt["ema"]
        # EMA shadow keys match model parameter names
        model.load_state_dict(ema_shadow, strict=False)
        # Check if loading was successful by trying direct load
        try:
            model.load_state_dict(ema_shadow)
            print("[INFO] Loaded EMA weights")
        except RuntimeError:
            # Fallback: load model weights
            model.load_state_dict(ckpt["model"])
            print("[INFO] EMA load failed, loaded model weights instead")
    else:
        model.load_state_dict(ckpt["model"])
        print("[INFO] Loaded model weights (no EMA found)")

    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model parameters: {param_count:,} ({param_count / 1e6:.2f}M)")

    # --- Build diffusion ---
    diffusion = build_diffusion_from_config(config)
    diffusion.to(device)

    # --- Read generate.csv ---
    samples = []
    with open(generate_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row["id"]
            animal = row["animal"]
            obj = row["object"]
            animal_id = ANIMAL_TO_ID[animal]
            object_id = OBJECT_TO_ID[obj]
            p_id = pair_id_from(animal_id, object_id)
            samples.append((img_id, animal_id, object_id, p_id))

    print(f"[INFO] Total images to generate: {len(samples)}")

    # --- Setup output ---
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Determine dtype ---
    if args.dtype == "bf16":
        mp_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        mp_dtype = torch.float16
    else:
        mp_dtype = torch.float32

    # --- Generate in batches ---
    image_size = config["unet"]["image_size"]
    batch_size = args.batch_size

    to_pil = transforms.ToPILImage()

    generated_count = 0
    for batch_start in tqdm(range(0, len(samples), batch_size), desc="Generating"):
        batch = samples[batch_start:batch_start + batch_size]
        B = len(batch)

        img_ids = [s[0] for s in batch]
        animal_ids = torch.tensor([s[1] for s in batch], device=device)
        object_ids = torch.tensor([s[2] for s in batch], device=device)
        pair_ids = torch.tensor([s[3] for s in batch], device=device)

        null_a = torch.full((B,), ANIMAL_NULL_ID, device=device, dtype=torch.long)
        null_o = torch.full((B,), OBJECT_NULL_ID, device=device, dtype=torch.long)
        null_p = torch.full((B,), PAIR_NULL_ID, device=device, dtype=torch.long)

        shape = (B, 3, image_size, image_size)

        with autocast(device_type="cuda", dtype=mp_dtype):
            images = diffusion.ddim_sample(
                model, shape,
                animal_ids, object_ids, pair_ids,
                null_a, null_o, null_p,
                ddim_steps=ddim_steps,
                cfg_scale=cfg_scale,
            )

        # Convert to [0, 1] and save as PNG
        images = (images.clamp(-1, 1) + 1) / 2  # [-1,1] → [0,1]

        for i, img_id in enumerate(img_ids):
            img = to_pil(images[i].cpu())
            assert img.size == (image_size, image_size), f"Wrong size: {img.size}"
            img.save(os.path.join(args.output_dir, img_id))

        generated_count += B

    print(f"[INFO] Generated {generated_count} images in {args.output_dir}/")
    print(f"[INFO] Settings: steps={ddim_steps}, cfg_scale={cfg_scale}, seed={args.seed}")

    # --- Verify ---
    files = [f for f in os.listdir(args.output_dir) if f.endswith(".png")]
    print(f"[INFO] Verification: {len(files)} PNG files found")
    if len(files) != len(samples):
        print(f"[WARNING] Expected {len(samples)} files, got {len(files)}")
    else:
        print("[INFO] ✅ File count matches generate.csv")


if __name__ == "__main__":
    main()
