"""
BrainrotDataset — loads 64×64 brainrot images with animal/object/pair labels.
"""

import os
import csv

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# Deterministic alphabetical ordering so label IDs are reproducible.
ANIMALS = [
    "bird", "capybara", "cat", "crocodile", "dog",
    "elephant", "fish", "frog", "monkey", "shark",
]
OBJECTS = [
    "airplane", "banana", "cactus", "car", "chair",
    "coffee cup", "drum", "pizza", "sneaker", "toilet",
]

ANIMAL_TO_ID = {name: idx for idx, name in enumerate(ANIMALS)}   # 0-9
OBJECT_TO_ID = {name: idx for idx, name in enumerate(OBJECTS)}   # 0-9

# Null IDs for classifier-free guidance
ANIMAL_NULL_ID = len(ANIMALS)   # 10
OBJECT_NULL_ID = len(OBJECTS)   # 10
PAIR_NULL_ID = len(ANIMALS) * len(OBJECTS)  # 100


def pair_id_from(animal_id: int, object_id: int) -> int:
    """Compute pair_id = animal_id * 10 + object_id  (0-99, or 100 for null)."""
    if animal_id == ANIMAL_NULL_ID or object_id == OBJECT_NULL_ID:
        return PAIR_NULL_ID
    return animal_id * len(OBJECTS) + object_id


class BrainrotDataset(Dataset):
    """
    Parameters
    ----------
    csv_path : str
        Path to train.csv  (columns: id, animal, object)
    image_dir : str
        Directory containing the PNG images referenced by ``id``.
    image_size : int
        Spatial resolution (both H and W).
    augment : bool
        If True apply random horizontal flip during training.
    """

    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        image_size: int = 64,
        augment: bool = True,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.image_size = image_size

        # Read CSV
        self.samples = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_id = row["id"]
                animal = row["animal"]
                obj = row["object"]
                animal_id = ANIMAL_TO_ID[animal]
                object_id = OBJECT_TO_ID[obj]
                p_id = pair_id_from(animal_id, object_id)
                self.samples.append((img_id, animal_id, object_id, p_id))

        # Transforms
        tfms = [transforms.Resize((image_size, image_size))]
        if augment:
            # 隨機水平翻轉，做資料增強以提升模型泛化能力。
            tfms.append(transforms.RandomHorizontalFlip(p=0.5))
        tfms.extend([
            transforms.ToTensor(),                               # 把像素值從 $[0, 255]$ 縮放到 $[0, 1]$ 之間
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # 再把 pixel 值從 $[0, 1]$ 放到 $[-1, 1]$，這對訓練比較穩定。
        ])
        self.transform = transforms.Compose(tfms)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, animal_id, object_id, p_id = self.samples[idx]
        img_path = os.path.join(self.image_dir, img_id)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, animal_id, object_id, p_id
