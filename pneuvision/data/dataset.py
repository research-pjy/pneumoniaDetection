"""
Dataset loading, archive extraction, and synthetic smoke-test generation for
Pediatric Chest X-Ray Pneumonia Detection.

Matches the actual extracted directory structure on Linux (`rama`):
    DATASET/
    ├── train/
    │   ├── NORMAL/             # Class 0 (4,684 total train images across NORMAL + PNEUMONIA)
    │   └── PNEUMONIA/          # Class 1
    ├── val/
    │   ├── NORMAL/             # Class 0 (585 total val images)
    │   └── PNEUMONIA/          # Class 1
    └── test/                   # 587 blind evaluation images (also supports test_unlabeled/)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import tarfile

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from pneuvision.data.enhancement import (
    apply_progressive_enhancement_numpy,
    build_multiview_3ch_numpy,
)


CLASS_TO_IDX = {"NORMAL": 0, "PNEUMONIA": 1}
IDX_TO_CLASS = {0: "NORMAL", 1: "PNEUMONIA"}
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}


def _list_images_in_dir(directory: Path, recursive: bool = False) -> List[Path]:
    """Case-insensitive image file listing for Linux (`rama`) and Windows."""
    if not directory.exists():
        return []
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    files = [
        p for p in iterator
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(files, key=lambda p: p.name)


def _resolve_existing_dataset_root(candidate: Path) -> Path:
    """
    Resolve case-sensitive dataset root on Linux (`./DATASET` vs `./dataset`).
    """
    if (candidate / "train").exists() and (candidate / "val").exists():
        return candidate

    parent = candidate.parent
    for alt_name in ("DATASET", "dataset", candidate.name.upper(), candidate.name.lower()):
        alt = parent / alt_name
        if (alt / "train").exists() and (alt / "val").exists():
            return alt

    return candidate


def _resolve_test_dir(ds_path: Path) -> Tuple[str, Path]:
    """
    Detect whether the unlabeled test split folder is named `test/` or `test_unlabeled/`.
    """
    for name in ("test", "test_unlabeled", "TEST"):
        candidate = ds_path / name
        if candidate.exists() and candidate.is_dir():
            return name, candidate
    return "test", ds_path / "test"


def ensure_dataset_extracted(
    dataset_dir: str | Path = "./DATASET",
    tar_path: Optional[str | Path] = None,
) -> Path:
    """
    Check if `dataset_dir` (or `./DATASET` / `./dataset`) contains `train/` and `val/`.
    If not, and `tar_path` is provided and exists, automatically extract `tar_path`
    and return the resolved dataset root path (`DATASET/`).
    """
    ds_path = _resolve_existing_dataset_root(Path(dataset_dir).expanduser().resolve())
    if (ds_path / "train").exists() and (ds_path / "val").exists():
        return ds_path

    if tar_path is not None:
        archive = Path(tar_path).expanduser().resolve()
        if archive.exists():
            print(f"[Data] Extracting {archive} -> {ds_path.parent} ...")
            ds_path.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive, "r:*") as tf:
                tf.extractall(path=ds_path.parent)
            ds_path = _resolve_existing_dataset_root(ds_path)

    return ds_path


def audit_dataset_directory(dataset_dir: str | Path, sample_limit: int = 300) -> Dict[str, Any]:
    """
    Audit the dataset structure (`DATASET/train`, `DATASET/val`, `DATASET/test`),
    exact image counts per split/class, class imbalance ratios, recommended positive
    class weight for BCEWithLogitsLoss, and image resolution statistics.
    """
    ds_path = _resolve_existing_dataset_root(Path(dataset_dir).expanduser().resolve())
    if not (ds_path / "train").exists():
        return {"exists": False, "dataset_dir": str(ds_path)}

    splits_info: Dict[str, Any] = {}
    for split in ("train", "val"):
        split_dir = ds_path / split
        normal_files = _list_images_in_dir(split_dir / "NORMAL")
        pneu_files = _list_images_in_dir(split_dir / "PNEUMONIA")
        n_norm = len(normal_files)
        n_pneu = len(pneu_files)
        total = n_norm + n_pneu
        splits_info[split] = {
            "NORMAL": n_norm,
            "PNEUMONIA": n_pneu,
            "total": total,
            "pneumonia_prevalence_pct": round(100.0 * n_pneu / max(total, 1), 2),
            "pos_weight_for_bce": round(n_norm / max(n_pneu, 1), 4),
        }

    test_split_name, test_dir = _resolve_test_dir(ds_path)
    test_files = _list_images_in_dir(test_dir, recursive=True)
    splits_info[test_split_name] = {
        "folder_name": test_split_name,
        "total": len(test_files),
    }

    # Sample image dimensions and channel modes
    all_train_files = (
        _list_images_in_dir(ds_path / "train" / "NORMAL")[: sample_limit // 2]
        + _list_images_in_dir(ds_path / "train" / "PNEUMONIA")[: sample_limit // 2]
    )
    widths, heights, modes = [], [], set()
    for p in all_train_files:
        with Image.open(p) as im:
            w, h = im.size
            widths.append(w)
            heights.append(h)
            modes.add(im.mode)

    return {
        "exists": True,
        "dataset_dir": str(ds_path),
        "test_folder_name": test_split_name,
        "splits": splits_info,
        "total_images": (
            splits_info["train"]["total"]
            + splits_info["val"]["total"]
            + splits_info[test_split_name]["total"]
        ),
        "resolution_sample_stats": {
            "sampled_images": len(widths),
            "modes_observed": sorted(list(modes)),
            "width_min_median_max": [
                int(np.min(widths)),
                int(np.median(widths)),
                int(np.max(widths)),
            ] if widths else [],
            "height_min_median_max": [
                int(np.min(heights)),
                int(np.median(heights)),
                int(np.max(heights)),
            ] if heights else [],
        },
    }


class ChestXRayDataset(Dataset):
    """
    PyTorch Dataset for Pediatric Chest X-Ray classification and autoencoder training.

    Supports:
    - `split="train"` or `"val"`: labeled images from `NORMAL/` (0) and `PNEUMONIA/` (1).
    - `split="test"` or `"test_unlabeled"`: unlabeled test images from `DATASET/test/`,
      returning `label=-1` and `image_id=img_path.name`.
    - `split="all_unsupervised"`: combines `train/` + (`test/` if `include_unlabeled_in_ae=True`)
      for Stage 1 unsupervised Autoencoder training.
    - Configurable preprocessing (`raw`, `fixed_clahe` with `alpha`, `multiview_3ch`) and
      paired `(raw_tensor, enhanced_tensor)` output for Enhancement-Reconstruction AE training.
    """

    def __init__(
        self,
        dataset_dir: str | Path = "./DATASET",
        split: str = "train",
        image_size: int = 256,
        preprocess_mode: str = "raw",
        alpha_strength: float = 0.0,
        clahe_clip_limit: float = 2.0,
        return_enhancement_target: bool = False,
        augment: bool = False,
        include_unlabeled_in_ae: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_dir = _resolve_existing_dataset_root(Path(dataset_dir).expanduser().resolve())
        self.split = split
        self.image_size = image_size
        self.preprocess_mode = preprocess_mode
        self.alpha_strength = alpha_strength
        self.clahe_clip_limit = clahe_clip_limit
        self.return_enhancement_target = return_enhancement_target
        self.augment = augment

        self.samples: List[Tuple[Path, int]] = []
        if split in ("train", "val"):
            for cls_name, label in CLASS_TO_IDX.items():
                cls_dir = self.dataset_dir / split / cls_name
                for p in _list_images_in_dir(cls_dir):
                    self.samples.append((p, label))
        elif split in ("test", "test_unlabeled"):
            _, test_dir = _resolve_test_dir(self.dataset_dir)
            for p in _list_images_in_dir(test_dir, recursive=True):
                self.samples.append((p, -1))
        elif split == "all_unsupervised":
            for cls_name, label in CLASS_TO_IDX.items():
                cls_dir = self.dataset_dir / "train" / cls_name
                for p in _list_images_in_dir(cls_dir):
                    self.samples.append((p, label))
            if include_unlabeled_in_ae:
                _, test_dir = _resolve_test_dir(self.dataset_dir)
                for p in _list_images_in_dir(test_dir, recursive=True):
                    self.samples.append((p, -1))
        else:
            raise ValueError(f"Unknown split '{split}'.")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_grayscale_01(self, path: Path) -> np.ndarray:
        with Image.open(path) as im:
            im_gray = im.convert("L").resize(
                (self.image_size, self.image_size), resample=Image.Resampling.BILINEAR
            )
            arr = np.asarray(im_gray, dtype=np.float32) / 255.0
        return arr

    def _apply_augmentation(self, arr: np.ndarray) -> np.ndarray:
        # Mild, clinically valid augmentations (horizontal flip + slight intensity shift/scale)
        if np.random.rand() < 0.5:
            arr = np.fliplr(arr).copy()
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.92, 1.08)
            shift = np.random.uniform(-0.04, 0.04)
            arr = np.clip(arr * scale + shift, 0.0, 1.0)
        return arr.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, label = self.samples[idx]
        raw_01 = self._load_grayscale_01(img_path)
        if self.augment:
            raw_01 = self._apply_augmentation(raw_01)

        if self.preprocess_mode == "fixed_clahe":
            proc = apply_progressive_enhancement_numpy(
                raw_01, alpha=self.alpha_strength, base_clip_limit=self.clahe_clip_limit
            )
            x_tensor = torch.from_numpy(proc)[None, :, :]
        elif self.preprocess_mode == "multiview_3ch":
            proc3 = build_multiview_3ch_numpy(raw_01, clip_limit=self.clahe_clip_limit)
            x_tensor = torch.from_numpy(proc3)
        else:
            # 'raw', 'task_guided_global', 'task_guided_adaptive'
            x_tensor = torch.from_numpy(raw_01)[None, :, :]

        item: Dict[str, Any] = {
            "image": x_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "image_id": img_path.name,
        }

        if self.return_enhancement_target:
            enh_target = apply_progressive_enhancement_numpy(
                raw_01, alpha=0.75, base_clip_limit=self.clahe_clip_limit
            )
            item["target_image"] = torch.from_numpy(enh_target)[None, :, :]
        else:
            item["target_image"] = torch.from_numpy(raw_01)[None, :, :]

        return item


def generate_synthetic_chest_xray(
    image_size: int = 256,
    is_pneumonia: bool = False,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a realistic-looking synthetic pediatric chest radiograph in [0.0, 1.0]
    (with ribcage bars, mediastinum/heart shadow, dark bilateral lung fields, and
    cloudy focal/multifocal pneumonia consolidation when `is_pneumonia=True`).
    Used for smoke testing when the real dataset archive has not yet been copied.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, image_size, dtype=np.float32),
        np.linspace(-1.0, 1.0, image_size, dtype=np.float32),
        indexing="ij",
    )

    # Torso silhouette + spine/mediastinum
    torso = np.exp(-((xx / 0.82) ** 4 + (yy / 0.95) ** 4)) * 0.58
    spine = np.exp(-((xx / 0.11) ** 2)) * 0.25 * (yy > -0.8)

    # Left & right lung radiolucent cavities
    right_lung = np.exp(-(((xx + 0.34) / 0.24) ** 2 + ((yy + 0.04) / 0.48) ** 2))
    left_lung = np.exp(-(((xx - 0.34) / 0.22) ** 2 + ((yy + 0.02) / 0.46) ** 2))
    lungs = np.clip(right_lung + left_lung, 0.0, 1.0)

    # Subtle ribcage horizontal arcs
    ribs = 0.045 * np.sin(26.0 * (yy + 0.18 * (xx ** 2))) * lungs

    img = torso + spine - 0.36 * lungs + ribs

    # If pneumonia: add focal/diffuse consolidation opacity inside lung parenchyma
    if is_pneumonia:
        cx = rng.choice([-0.32, 0.32]) + rng.uniform(-0.06, 0.06)
        cy = rng.uniform(-0.15, 0.22)
        rx = rng.uniform(0.14, 0.22)
        ry = rng.uniform(0.16, 0.26)
        opacity = np.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2))
        texture = 0.7 + 0.3 * np.sin(35.0 * xx) * np.cos(35.0 * yy)
        img = img + 0.22 * opacity * lungs * texture

    # Simulate realistic low-contrast / washed-out portable pediatric X-ray exposure
    noise = rng.normal(0.0, 0.012, size=(image_size, image_size)).astype(np.float32)
    img = 0.15 + 0.72 * np.clip(img + noise, 0.0, 1.0)
    return np.clip(img, 0.0, 1.0).astype(np.float32)
