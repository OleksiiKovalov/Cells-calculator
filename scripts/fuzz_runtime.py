"""Runtime fuzz harness for generated images and real model inference.

This is intentionally separate from pytest. Real model inference can be slow,
can hang, and may crash native libraries, so each generated case runs in a
fresh child process with a timeout. Failing cases are kept for replay.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / ".cache" / "runtime_fuzz"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

KNOWN_MODELS = {
    "cellcounter": "model.CellCounter.CellCounter",
    "cellpose": "model.CellposeSegmenter.CellposeSegmenter",
    "yolo": "model.YOLOSegmenter.YoloSegmenter",
    "instanseg": "model.InstanSegSegmenter.InstansegSegmenter",
    "stardist": "model.StardistSegmenter.StardistSegmenter",
}

COLORMAPS = [
    "gist_rainbow",
    "tab20",
    "tab20b",
    "tab20c",
    "tab10",
    "Set1",
    "Set2",
    "Set3",
    "Paired",
    "viridis",
    "plasma",
]

IMAGE_PROFILES = ("auto", "generic", "yolo", "mixed")
MODEL_STRATEGIES = ("random", "round-robin")
SANITIZERS = ("none", "python", "warnings", "numpy", "all")
WORKFLOWS = ("single", "stateful")
ORACLE_LEVELS = ("basic", "strict", "paranoid")
DETERMINISM_CHECKS = ("off", "counts", "full")
VALID_GENERATED_EXTENSIONS = {"png", "jpg", "jpeg", "tif", "tiff", "bmp", "lsm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate random images/UI-like parameters and run models."
    )
    parser.add_argument("--config", default=str(ROOT / "modelconfig.json"))
    parser.add_argument(
        "--models",
        nargs="*",
        default=[],
        help=(
            "Model names or model types. Commas are allowed. "
            "Examples: yolo,Detector or YOLO-512 Segmenter. "
            "Default: all enabled config models."
        ),
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=25,
        help="Number of cases to run. Use 0 for an endless run.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=0,
        help="Stop after this many seconds. Use 0 for no time limit.",
    )
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--image-profile",
        choices=IMAGE_PROFILES,
        default="auto",
        help=(
            "Image generator profile. 'auto' uses a model-specific profile when available; "
            "'mixed' randomly mixes generic, YOLO, and corpus mutations."
        ),
    )
    parser.add_argument(
        "--seed-corpus",
        default="",
        help="Directory or file with real images to mutate, e.g. testimages.",
    )
    parser.add_argument(
        "--corpus-probability",
        type=float,
        default=0.35,
        help="Probability of mutating a corpus image when --seed-corpus is set.",
    )
    parser.add_argument(
        "--model-strategy",
        choices=MODEL_STRATEGIES,
        default="random",
        help="How to choose models across cases. Default: random.",
    )
    parser.add_argument(
        "--workflow",
        choices=WORKFLOWS,
        default="single",
        help=(
            "single runs one calculate() call. stateful also fuzzes filtering, "
            "plotting, and repeated calculate() calls in the same child process."
        ),
    )
    parser.add_argument(
        "--max-workflow-steps",
        type=int,
        default=6,
        help="Maximum stateful workflow steps per case. Default: 6.",
    )
    parser.add_argument(
        "--scales",
        default="20",
        help="Comma-separated UI scales to fuzz, e.g. 20 or 10,20. Default: 20.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=384,
        help="Maximum generated image side length. Default: 384.",
    )
    parser.add_argument(
        "--sanitizers",
        choices=SANITIZERS,
        default="python",
        help=(
            "Runtime checks in child process. python=faulthandler/dev allocator; "
            "warnings=RuntimeWarning as error; numpy=np.seterr(all='raise'); all=all checks."
        ),
    )
    parser.add_argument(
        "--oracle-level",
        choices=ORACLE_LEVELS,
        default="strict",
        help=(
            "How aggressively to validate successful model outputs. "
            "basic=legacy crash checks, strict=result contracts, paranoid=extra geometry bounds."
        ),
    )
    parser.add_argument(
        "--determinism-check",
        choices=DETERMINISM_CHECKS,
        default="off",
        help=(
            "Rerun each case in a fresh model instance and compare output. "
            "counts compares counts and nuclei metrics; full also compares numeric summaries."
        ),
    )
    parser.add_argument(
        "--max-rss-mb",
        type=float,
        default=0,
        help="Fail a case if sampled child RSS exceeds this many MB. 0 disables.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--keep-all", action="store_true")
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Save every failure as unique instead of grouping by crash signature.",
    )
    parser.add_argument(
        "--minimize-failures",
        action="store_true",
        help="Try to shrink unique failing cases while preserving the crash signature.",
    )
    parser.add_argument(
        "--minimize-steps",
        type=int,
        default=10,
        help="Maximum accepted minimization attempts per failure. Default: 10.",
    )
    parser.add_argument(
        "--minimize-timeout",
        type=float,
        default=0,
        help="Timeout for minimization reruns. Default: use --timeout.",
    )
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--replay", default="")
    parser.add_argument("--run-case", default=argparse.SUPPRESS)
    return parser.parse_args()


def load_enabled_models(config_path: Path) -> OrderedDict[str, dict[str, Any]]:
    with config_path.open("r", encoding="utf-8") as f:
        models = json.load(f, object_pairs_hook=OrderedDict)
    return OrderedDict(
        (name, data)
        for name, data in models.items()
        if "enabled" not in data or str(data.get("enabled", "true")).lower() == "true"
    )


def selected_models(
    models: OrderedDict[str, dict[str, Any]],
    model_tokens: list[str],
) -> OrderedDict[str, dict[str, Any]]:
    if not model_tokens:
        return models

    names_text = " ".join(model_tokens)
    requested = [token.strip() for token in names_text.split(",") if token.strip()]
    selected: OrderedDict[str, dict[str, Any]] = OrderedDict()
    missing = []

    for token in requested:
        token_lower = token.lower()

        exact_matches = [
            name for name in models
            if name.lower() == token_lower
        ]
        type_matches = [
            name for name, data in models.items()
            if str(data.get("model_type", "")).lower() == token_lower
        ]

        matches = exact_matches or type_matches
        if not matches:
            missing.append(token)
            continue

        for name in matches:
            selected[name] = models[name]

    if missing:
        raise SystemExit(
            f"Unknown or disabled model(s): {', '.join(missing)}. "
            "Use --list-models to see model names and types."
        )
    return selected


def parse_scales(scales_text: str) -> list[int]:
    scales = []
    for token in scales_text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            scale = int(token)
        except ValueError as exc:
            raise SystemExit(f"Invalid scale '{token}'. Use 10, 20, or 10,20.") from exc
        if scale not in (10, 20):
            raise SystemExit(f"Invalid scale '{scale}'. Use 10, 20, or 10,20.")
        scales.append(scale)
    return scales or [20]


def clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def model_is_locally_runnable(model_data: dict[str, Any]) -> bool:
    path = str(model_data.get("path", ""))
    model_type = str(model_data.get("model_type", ""))
    if Path(path).exists() or (ROOT / path).exists():
        return True

    builtins = {
        "cellpose": {"cyto", "nuclei", "cyto2", "cyto3"},
        "instanseg": {"brightfield_nuclei", "fluorescence_nuclei_and_cells"},
        "stardist": {"2D_versatile_fluo", "2D_versatile_he", "2D_paper_dsb2018"},
    }
    return path in builtins.get(model_type, set())


def list_models(models: OrderedDict[str, dict[str, Any]]) -> None:
    for name, data in models.items():
        runnable = "yes" if model_is_locally_runnable(data) else "no"
        print(
            f"{name}: type={data.get('model_type')} "
            f"path={data.get('path')} runnable={runnable}"
        )


def discover_corpus_images(seed_corpus: str) -> list[Path]:
    if not seed_corpus:
        return []

    roots = [Path(part.strip()) for part in seed_corpus.split(",") if part.strip()]
    images: list[Path] = []
    for root in roots:
        root = root if root.is_absolute() else ROOT / root
        if root.is_file():
            candidates = [root]
        elif root.is_dir():
            candidates = [p for p in root.rglob("*") if p.is_file()]
        else:
            continue

        for path in candidates:
            if path.suffix.lower().lstrip(".") in VALID_GENERATED_EXTENSIONS:
                images.append(path)

    return sorted(set(images))


def read_corpus_image(path: Path) -> np.ndarray | None:
    try:
        if path.suffix.lower() == ".lsm":
            image = tifffile.TiffFile(str(path)).series[0].asarray()
        elif path.suffix.lower() in {".tif", ".tiff"}:
            image = tifffile.imread(str(path))
        else:
            data = np.fromfile(str(path), dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if image is None:
            return None
        return normalize_loaded_array(np.asarray(image))
    except Exception:
        return None


def normalize_loaded_array(image: np.ndarray) -> np.ndarray:
    image = np.squeeze(image)
    if image.ndim == 0:
        return np.full((1, 1), int(image), dtype=np.uint8)
    if image.ndim > 3:
        image = image.reshape((-1, image.shape[-2], image.shape[-1]))
    if image.ndim == 3 and image.shape[0] <= 8 and image.shape[1] > 8 and image.shape[2] > 8:
        image = np.transpose(image, (1, 2, 0))
    if image.dtype != np.uint8:
        image_float = image.astype(np.float32)
        finite = image_float[np.isfinite(image_float)]
        if finite.size == 0:
            image = np.zeros(image.shape, dtype=np.uint8)
        else:
            lo = float(np.min(finite))
            hi = float(np.max(finite))
            if hi > lo:
                image = ((np.clip(image_float, lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)
            else:
                image = np.zeros(image.shape, dtype=np.uint8)
    return image


def make_object_size(
    rng: random.Random,
    model_data: dict[str, Any],
    allowed_scales: list[int],
) -> dict[str, Any]:
    min_size = rng.choice([0.0, rng.random() * 0.2, rng.random()])
    max_size = rng.choice([1.0, min_size + rng.random() * max(0.0, 1.0 - min_size)])
    usable_scales = [
        scale
        for scale in allowed_scales
        if scale == 20 or (scale == 10 and "x10" in model_data)
    ]
    scale = rng.choice(usable_scales or [20])
    return {
        "min_size": min_size,
        "max_size": max_size,
        "round_parametr_slider": 10**6,
        "round_parametr_value_input": 10**4,
        "color_map": rng.choice(COLORMAPS),
        "color_map_list": COLORMAPS,
        "line_width": round(rng.uniform(0.5, 100.0), 2),
        "scale": scale,
        "alpha": rng.choice([0.25, 0.5, 0.75, 1.0]),
        "size_metric": rng.choice(["area", "diameter", "volume"]),
        "um_per_px": rng.choice([None, 0.1, 0.325, 1.0]),
    }


def make_workflow(
    rng: random.Random,
    mode: str,
    max_steps: int,
) -> dict[str, Any]:
    if mode == "single":
        return {"mode": "single", "steps": []}

    step_count = rng.randint(1, max(1, max_steps))
    steps = []
    for _ in range(step_count):
        operation = rng.choices(
            ["filter", "wide_filter", "plot_masks", "rerun"],
            weights=[4, 3, 2, 1],
            k=1,
        )[0]
        min_size = rng.choice([0.0, rng.random() * 0.15, rng.random() * 0.6])
        max_size = rng.choice([1.0, min_size, min(1.0, min_size + rng.random())])
        if rng.random() < 0.15:
            min_size, max_size = max_size, min_size
        steps.append(
            {
                "operation": operation,
                "min_size": min_size,
                "max_size": max_size,
                "margin": rng.choice([0.0, 0.01, 0.05, 0.25, 1.0]),
                "size_metric": rng.choice(["area", "diameter", "volume", "missing_metric"]),
                "alpha": rng.choice([0.0, 0.25, 0.5, 0.75, 1.0, 1.5]),
                "color_map": rng.choice(COLORMAPS),
            }
        )
    return {"mode": "stateful", "steps": steps}


def resolve_image_profile(profile: str, model_data: dict[str, Any]) -> str:
    if profile == "mixed":
        return "mixed"
    if profile != "auto":
        return profile
    if str(model_data.get("model_type", "")).lower() == "yolo":
        return "yolo"
    return "generic"


def choose_effective_profile(
    rng: random.Random,
    profile: str,
    has_corpus: bool,
) -> str:
    if profile == "mixed":
        choices = ["generic", "yolo"]
        if has_corpus:
            choices.extend(["corpus", "corpus"])
        return rng.choice(choices)
    return profile


def choose_dimensions(
    rng: random.Random,
    profile: str,
    max_side: int,
) -> tuple[int, int]:
    if profile == "yolo":
        candidates = [
            1, 2, 3, 7, 8, 15, 16, 31, 32, 47, 63, 64, 65, 95, 96,
            127, 128, 129, 191, 192, 255, 256, 320, 384, 512,
        ]
        candidates = [value for value in candidates if value <= max_side] or [max(1, max_side)]
        if rng.random() < 0.25:
            return rng.choice(candidates), rng.choice([value for value in candidates if value <= 32] or candidates)
        if rng.random() < 0.25:
            return rng.choice([value for value in candidates if value <= 32] or candidates), rng.choice(candidates)
        return rng.choice(candidates), rng.choice(candidates)

    candidates = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384]
    candidates = [value for value in candidates if value <= max_side] or [max(1, max_side)]
    return rng.choice(candidates), rng.choice(candidates)


def make_pattern_array(
    rng: random.Random,
    height: int,
    width: int,
    channels: int | None,
) -> tuple[np.ndarray, str]:
    shape = (height, width) if channels is None else (height, width, channels)
    pattern = rng.choice(["zeros", "constant", "noise", "gradient", "spots", "stripes"])

    if pattern == "zeros":
        image = np.zeros(shape, dtype=np.uint8)
    elif pattern == "constant":
        image = np.full(shape, rng.randrange(0, 256), dtype=np.uint8)
    elif pattern == "noise":
        image = rng_numpy(rng).integers(0, 256, size=shape, dtype=np.uint8)
    elif pattern == "gradient":
        x_grad = np.linspace(0, 255, width, dtype=np.uint8)
        y_grad = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
        image_2d = ((x_grad.astype(np.uint16) + y_grad.astype(np.uint16)) // 2).astype(
            np.uint8
        )
        image = image_2d if channels is None else np.repeat(image_2d[:, :, None], channels, axis=2)
    elif pattern == "spots":
        image = np.zeros(shape, dtype=np.uint8)
        draw_spots(rng, image)
    else:
        image = np.zeros(shape, dtype=np.uint8)
        stripe_width = rng.randint(1, max(1, width // 8))
        for x in range(0, width, stripe_width * 2):
            image[:, x : x + stripe_width] = rng.randrange(64, 256)

    return image, pattern


def make_yolo_pattern_array(
    rng: random.Random,
    height: int,
    width: int,
    channels: int | None,
) -> tuple[np.ndarray, str]:
    pattern = rng.choice([
        "empty",
        "low_contrast_blobs",
        "bright_blobs",
        "touching_blobs",
        "border_blobs",
        "rings",
        "speckles",
        "checkerboard",
        "textured_noise",
        "gradient_blobs",
    ])

    background = rng.randrange(0, 96)
    image_2d = np.full((height, width), background, dtype=np.uint8)

    if pattern == "empty":
        pass
    elif pattern == "checkerboard":
        tile = rng.randint(1, max(1, min(height, width, 16)))
        yy, xx = np.indices((height, width))
        image_2d = (((xx // tile + yy // tile) % 2) * rng.randrange(80, 256)).astype(np.uint8)
    elif pattern == "textured_noise":
        noise = rng_numpy(rng).normal(background, rng.uniform(4, 48), size=(height, width))
        image_2d = np.clip(noise, 0, 255).astype(np.uint8)
        if min(height, width) > 2 and rng.random() < 0.7:
            image_2d = cv2.GaussianBlur(image_2d, (3, 3), 0)
    else:
        if pattern == "gradient_blobs":
            x_grad = np.linspace(0, rng.randrange(80, 256), width, dtype=np.float32)
            y_grad = np.linspace(0, rng.randrange(80, 256), height, dtype=np.float32)[:, None]
            image_2d = np.clip((x_grad + y_grad) / 2, 0, 255).astype(np.uint8)

        if pattern == "speckles":
            count = rng.randint(10, max(10, height * width // 8))
            for _ in range(count):
                image_2d[rng.randrange(0, height), rng.randrange(0, width)] = rng.randrange(128, 256)
        else:
            blob_count = rng.randint(1, 30 if pattern == "touching_blobs" else 12)
            for index in range(blob_count):
                if pattern == "border_blobs" and rng.random() < 0.7:
                    cx = rng.choice([-rng.randint(1, max(1, width // 4)), width + rng.randint(0, max(1, width // 4))])
                    cy = rng.randrange(-max(1, height // 4), height + max(1, height // 4))
                elif pattern == "touching_blobs":
                    cx = width // 2 + rng.randint(-max(1, width // 5), max(1, width // 5))
                    cy = height // 2 + rng.randint(-max(1, height // 5), max(1, height // 5))
                else:
                    cx = rng.randrange(0, width)
                    cy = rng.randrange(0, height)

                rx = rng.randint(1, max(1, width // rng.choice([6, 8, 12, 20])))
                ry = rng.randint(1, max(1, height // rng.choice([6, 8, 12, 20])))
                value = rng.randrange(86, 120) if pattern == "low_contrast_blobs" else rng.randrange(140, 256)
                angle = rng.randrange(0, 180)
                cv2.ellipse(image_2d, (int(cx), int(cy)), (rx, ry), angle, 0, 360, int(value), -1)

                if pattern == "rings" and rx > 1 and ry > 1:
                    cv2.ellipse(
                        image_2d,
                        (int(cx), int(cy)),
                        (max(1, rx // 2), max(1, ry // 2)),
                        angle,
                        0,
                        360,
                        int(background),
                        -1,
                    )

                if rng.random() < 0.2:
                    cv2.line(
                        image_2d,
                        (rng.randrange(0, width), rng.randrange(0, height)),
                        (rng.randrange(0, width), rng.randrange(0, height)),
                        int(rng.randrange(64, 256)),
                        rng.randint(1, 3),
                    )

    if rng.random() < 0.35 and min(height, width) > 2:
        image_2d = cv2.GaussianBlur(image_2d, (3, 3), 0)

    return channelize_image(rng, image_2d, channels), pattern


def mutate_corpus_image(
    rng: random.Random,
    image: np.ndarray,
    max_side: int,
) -> tuple[np.ndarray, list[str]]:
    mutations: list[str] = []
    image = normalize_loaded_array(image)

    if image.ndim not in (2, 3):
        image = image.reshape(image.shape[-2], image.shape[-1])
        mutations.append("reshape")

    if rng.random() < 0.65:
        image = random_crop(rng, image)
        mutations.append("crop")

    if rng.random() < 0.65:
        image = resize_mutation(rng, image, max_side)
        mutations.append("resize")

    if rng.random() < 0.5:
        image = np.flip(image, axis=rng.choice([0, 1])).copy()
        mutations.append("flip")

    if rng.random() < 0.35:
        image = np.rot90(image, rng.choice([1, 2, 3])).copy()
        mutations.append("rot90")

    if rng.random() < 0.2 and image.ndim == 2:
        image = image.T.copy()
        mutations.append("transpose")

    if rng.random() < 0.7:
        alpha = rng.uniform(0.3, 2.2)
        beta = rng.uniform(-80, 80)
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        mutations.append("brightness_contrast")

    if rng.random() < 0.45:
        sigma = rng.uniform(1, 35)
        noise = rng_numpy(rng).normal(0, sigma, size=image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        mutations.append("noise")

    if rng.random() < 0.25 and min(image.shape[:2]) > 2:
        image = cv2.GaussianBlur(image, (3, 3), 0)
        mutations.append("blur")

    if rng.random() < 0.2:
        image = 255 - image
        mutations.append("invert")

    if rng.random() < 0.2:
        threshold = rng.randrange(0, 256)
        image = np.where(image > threshold, 255, 0).astype(np.uint8)
        mutations.append("threshold")

    if image.ndim == 3:
        if rng.random() < 0.35:
            order = list(range(image.shape[2]))
            rng.shuffle(order)
            image = image[:, :, order]
            mutations.append("channel_shuffle")
        if rng.random() < 0.25:
            keep = rng.randrange(0, image.shape[2])
            out = np.zeros_like(image)
            out[:, :, keep] = image[:, :, keep]
            image = out
            mutations.append("single_channel")
        if image.shape[2] > 4:
            image = image[:, :, :4]
            mutations.append("truncate_channels")

    if rng.random() < 0.15:
        pad_h = rng.randint(0, max(1, image.shape[0] // 3))
        pad_w = rng.randint(0, max(1, image.shape[1] // 3))
        pad_width = ((pad_h, pad_h), (pad_w, pad_w))
        if image.ndim == 3:
            pad_width += ((0, 0),)
        image = np.pad(image, pad_width, mode=rng.choice(["constant", "edge"]))
        mutations.append("pad")

    image = enforce_max_side(image, max_side)
    return image.astype(np.uint8, copy=False), mutations or ["identity"]


def random_crop(rng: random.Random, image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    if height <= 1 or width <= 1:
        return image
    crop_h = rng.randint(1, height)
    crop_w = rng.randint(1, width)
    y = rng.randint(0, height - crop_h)
    x = rng.randint(0, width - crop_w)
    if image.ndim == 2:
        return image[y:y + crop_h, x:x + crop_w].copy()
    return image[y:y + crop_h, x:x + crop_w, :].copy()


def resize_mutation(rng: random.Random, image: np.ndarray, max_side: int) -> np.ndarray:
    current_h, current_w = image.shape[:2]
    max_side = max(1, max_side)
    target_h = rng.choice([1, 2, 3, 7, 16, 31, 32, 64, 96, 128, max_side])
    target_w = rng.choice([1, 2, 3, 7, 16, 31, 32, 64, 96, 128, max_side])
    target_h = max(1, min(max_side, target_h))
    target_w = max(1, min(max_side, target_w))
    interpolation = rng.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA])
    if current_h == target_h and current_w == target_w:
        return image
    return cv2.resize(image, (target_w, target_h), interpolation=interpolation)


def enforce_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    max_current = max(height, width)
    if max_current <= max_side:
        return image
    scale = max_side / max_current
    target_h = max(1, int(round(height * scale)))
    target_w = max(1, int(round(width * scale)))
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)


def channelize_image(
    rng: random.Random,
    image_2d: np.ndarray,
    channels: int | None,
) -> np.ndarray:
    if channels is None:
        return image_2d

    if channels == 1:
        return image_2d[:, :, None]

    mode = rng.choice(["repeat", "single_channel", "jittered"])
    if mode == "single_channel":
        image = np.zeros((*image_2d.shape, channels), dtype=np.uint8)
        image[:, :, rng.randrange(0, channels)] = image_2d
        return image

    if mode == "jittered":
        planes = []
        for _ in range(channels):
            offset = rng.randrange(-30, 31)
            planes.append(np.clip(image_2d.astype(np.int16) + offset, 0, 255).astype(np.uint8))
        return np.stack(planes, axis=2)

    return np.repeat(image_2d[:, :, None], channels, axis=2)


def rng_numpy(rng: random.Random) -> np.random.Generator:
    return np.random.default_rng(rng.randrange(0, 2**32))


def draw_spots(rng: random.Random, image: np.ndarray) -> None:
    height, width = image.shape[:2]
    count = rng.randint(1, 12)
    yy, xx = np.ogrid[:height, :width]
    for _ in range(count):
        radius = rng.randint(1, max(1, min(height, width) // 6))
        cx = rng.randrange(0, width)
        cy = rng.randrange(0, height)
        value = rng.randrange(80, 256)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        image[mask] = value


def write_generated_image(
    rng: random.Random,
    case_dir: Path,
    model_type: str,
    profile: str,
    max_side: int,
    corpus_images: list[Path],
    corpus_probability: float,
) -> tuple[Path, dict[str, Any]]:
    if corpus_images and rng.random() < corpus_probability:
        source = rng.choice(corpus_images)
        source_image = read_corpus_image(source)
        if source_image is not None:
            return write_corpus_mutation(rng, case_dir, source, source_image, max_side)

    profile = choose_effective_profile(rng, profile, bool(corpus_images))
    if profile == "corpus":
        return write_generated_image(
            rng,
            case_dir,
            model_type,
            "generic",
            max_side,
            corpus_images,
            1.0,
        )

    height, width = choose_dimensions(rng, profile, max_side)
    extension_choices = ["png", "jpg", "tif", "lsm"] if profile == "yolo" else ["png", "tif", "lsm"]
    extension = rng.choice(extension_choices)

    if extension == "lsm":
        channels = rng.randint(2, 4)
        channel_arrays = []
        patterns = []
        for _ in range(channels):
            if profile == "yolo":
                channel, pattern = make_yolo_pattern_array(rng, height, width, None)
            else:
                channel, pattern = make_pattern_array(rng, height, width, None)
            channel_arrays.append(channel)
            patterns.append(pattern)
        image = np.stack(channel_arrays, axis=0)
        image_path = case_dir / "input.lsm"
        tifffile.imwrite(str(image_path), image, metadata={"axes": "CYX"})
        return image_path, {
            "extension": extension,
            "profile": profile,
            "shape": list(image.shape),
            "patterns": patterns,
            "cell_channel": rng.randrange(0, channels),
            "nuclei_channel": rng.randrange(0, channels),
        }

    channels = rng.choice([None, 3])
    if profile == "yolo":
        image, pattern = make_yolo_pattern_array(rng, height, width, channels)
    else:
        image, pattern = make_pattern_array(rng, height, width, channels)
    image_path = case_dir / f"input.{extension}"
    if extension in {"png", "jpg"}:
        if not cv2.imwrite(str(image_path), image):
            raise RuntimeError(f"Could not write generated image: {image_path}")
    else:
        tifffile.imwrite(str(image_path), image)

    return image_path, {
        "extension": extension,
        "profile": profile,
        "shape": list(image.shape),
        "patterns": [pattern],
        "cell_channel": 0,
        "nuclei_channel": 1,
    }


def write_corpus_mutation(
    rng: random.Random,
    case_dir: Path,
    source: Path,
    source_image: np.ndarray,
    max_side: int,
) -> tuple[Path, dict[str, Any]]:
    image, mutations = mutate_corpus_image(rng, source_image, max_side)
    extension = rng.choice(["png", "jpg", "tif", "lsm"])

    if extension == "lsm":
        if image.ndim == 2:
            channels_last = channelize_image(rng, image, rng.choice([2, 3, 4]))
        elif image.shape[2] == 1:
            channels_last = np.repeat(image, 2, axis=2)
        else:
            channels_last = image[:, :, :max(2, min(4, image.shape[2]))]
        image_to_write = np.transpose(channels_last, (2, 0, 1))
        image_path = case_dir / "input.lsm"
        tifffile.imwrite(str(image_path), image_to_write, metadata={"axes": "CYX"})
        channels = image_to_write.shape[0]
        shape = list(image_to_write.shape)
    else:
        image_path = case_dir / f"input.{extension}"
        if extension in {"png", "jpg"}:
            image_to_write = cv2_writable_image(image, extension)
            if not cv2.imwrite(str(image_path), image_to_write):
                raise RuntimeError(f"Could not write corpus mutation: {image_path}")
            image = image_to_write
        else:
            tifffile.imwrite(str(image_path), image)
        channels = image.shape[2] if image.ndim == 3 else 1
        shape = list(image.shape)

    return image_path, {
        "extension": extension,
        "profile": "corpus",
        "shape": shape,
        "patterns": mutations,
        "source": str(source),
        "cell_channel": rng.randrange(0, max(1, channels)),
        "nuclei_channel": rng.randrange(0, max(1, channels)),
    }


def cv2_writable_image(image: np.ndarray, extension: str) -> np.ndarray:
    """Return an array with a channel count accepted by cv2.imwrite."""
    image = normalize_loaded_array(np.asarray(image))

    if image.ndim == 1:
        return image.reshape(1, -1)

    if image.ndim == 2:
        return image

    if image.ndim > 3:
        image = image.reshape((*image.shape[-2:], -1))

    channels = image.shape[2]
    if channels == 1:
        return image[:, :, 0]
    if channels == 2:
        zero = np.zeros(image.shape[:2] + (1,), dtype=image.dtype)
        return np.concatenate([image, zero], axis=2)
    if extension == "jpg" and channels >= 4:
        return image[:, :, :3]
    if channels in (3, 4):
        return image
    if channels > 4:
        return image[:, :, :4]

    return image.reshape(image.shape[0], image.shape[1])


def build_case(
    rng: random.Random,
    case_id: int,
    case_dir: Path,
    model_name: str,
    model_data: dict[str, Any],
    image_profile: str,
    allowed_scales: list[int],
    max_side: int,
    corpus_images: list[Path],
    corpus_probability: float,
    workflow: str,
    max_workflow_steps: int,
) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    resolved_profile = resolve_image_profile(image_profile, model_data)
    image_path, image_meta = write_generated_image(
        rng,
        case_dir,
        str(model_data.get("model_type", "")),
        resolved_profile,
        max_side,
        corpus_images,
        corpus_probability,
    )
    return {
        "case_id": case_id,
        "model_name": model_name,
        "model_data": model_data,
        "image_path": str(image_path.resolve()),
        "image": image_meta,
        "object_size": make_object_size(rng, model_data, allowed_scales),
        "workflow": make_workflow(rng, workflow, max_workflow_steps),
    }


def child_environment(
    sanitizers: str,
    oracle_level: str,
    determinism_check: str,
) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
    env["CELLS_FUZZ_SANITIZERS"] = sanitizers
    env["CELLS_FUZZ_ORACLE_LEVEL"] = oracle_level
    env["CELLS_FUZZ_DETERMINISM_CHECK"] = determinism_check
    if sanitizers in {"python", "all"}:
        env.setdefault("PYTHONFAULTHANDLER", "1")
        env.setdefault("PYTHONMALLOC", "debug")
        env.setdefault("PYTHONDEVMODE", "1")
    return env


def run_case_subprocess(
    case_path: Path,
    timeout: float,
    sanitizers: str,
    max_rss_mb: float,
    oracle_level: str,
    determinism_check: str,
) -> subprocess.CompletedProcess:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-case",
        str(case_path),
        "--max-rss-mb",
        str(max_rss_mb),
    ]
    return subprocess.run(
        command,
        cwd=str(ROOT),
        env=child_environment(sanitizers, oracle_level, determinism_check),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
    )


def register_known_models() -> None:
    from UI.app_globals import register_model

    for model_type, model_class in KNOWN_MODELS.items():
        register_model(model_type, model_class, False)


def configure_child_sanitizers() -> None:
    sanitizer = os.environ.get("CELLS_FUZZ_SANITIZERS", "none")
    if sanitizer in {"python", "all"}:
        try:
            import faulthandler

            faulthandler.enable(all_threads=True)
        except Exception:
            pass
    if sanitizer in {"warnings", "all"}:
        warnings.simplefilter("error", RuntimeWarning)
        warnings.simplefilter("error", ResourceWarning)
    if sanitizer in {"numpy", "all"}:
        np.seterr(all="raise")


def current_oracle_level() -> str:
    level = os.environ.get("CELLS_FUZZ_ORACLE_LEVEL", "strict")
    return level if level in ORACLE_LEVELS else "strict"


def current_determinism_check() -> str:
    mode = os.environ.get("CELLS_FUZZ_DETERMINISM_CHECK", "off")
    return mode if mode in DETERMINISM_CHECKS else "off"


def start_memory_monitor(max_rss_mb: float):
    if max_rss_mb <= 0:
        return None, None
    try:
        import psutil
    except ImportError:
        return None, None

    process = psutil.Process(os.getpid())
    stop_event = threading.Event()
    state = {"peak": 0}

    def monitor() -> None:
        while not stop_event.is_set():
            try:
                state["peak"] = max(state["peak"], int(process.memory_info().rss))
            except Exception:
                pass
            stop_event.wait(0.1)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    return stop_event, state


def stop_memory_monitor(stop_event, state, max_rss_mb: float) -> float:
    if stop_event is None or state is None:
        return 0.0
    stop_event.set()
    peak_mb = state["peak"] / (1024 * 1024)
    if max_rss_mb > 0 and peak_mb > max_rss_mb:
        raise AssertionError(f"Peak RSS {peak_mb:.1f} MB exceeded limit {max_rss_mb:.1f} MB")
    return peak_mb


def count_result_cells(cells: Any) -> int:
    if cells is None:
        return 0
    if hasattr(cells, "shape"):
        return int(cells.shape[0])
    return int(cells)


def numeric_column(cells: Any, column: str) -> np.ndarray:
    try:
        return np.asarray(cells[column].tolist(), dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"Non-numeric values in '{column}'") from exc


def validate_confidence_column(cells: Any, model_type: str, oracle_level: str) -> None:
    if "confidence" not in cells.columns:
        return
    values = numeric_column(cells, "confidence")
    if values.size and not np.isfinite(values).all():
        raise AssertionError("Non-finite values in 'confidence'")
    if (
        oracle_level == "paranoid"
        and model_type != "cellpose"
        and values.size
        and ((values < -1e-6).any() or (values > 1.0 + 1e-6).any())
    ):
        raise AssertionError("Confidence values outside [0, 1]")


def validate_id_column(cells: Any) -> None:
    if "id_label" not in cells.columns:
        return
    values = numeric_column(cells, "id_label")
    if values.size and not np.isfinite(values).all():
        raise AssertionError("Non-finite values in 'id_label'")


def validate_box_geometry(
    box: Any,
    row_index: int,
    image_size: tuple[int, int] | None,
    oracle_level: str,
) -> None:
    box_array = np.asarray(box, dtype=np.float64).reshape(-1)
    if box_array.size != 4:
        raise AssertionError(f"Invalid box size at row {row_index}: {box_array.size}")
    if not np.isfinite(box_array).all():
        raise AssertionError(f"Non-finite box at row {row_index}")

    if oracle_level == "basic":
        return

    if box_array[2] < -1e-6 or box_array[3] < -1e-6:
        raise AssertionError(f"Negative box extent at row {row_index}: {box_array.tolist()}")

    if oracle_level == "paranoid" and image_size is not None:
        width, height = image_size
        loose_limit = max(width, height, 1) * 4
        if np.max(np.abs(box_array)) > loose_limit:
            raise AssertionError(
                f"Box at row {row_index} is far outside image bounds: {box_array.tolist()}"
            )


def validate_mask_geometry(
    mask: Any,
    row_index: int,
    image_size: tuple[int, int] | None,
    oracle_level: str,
) -> None:
    try:
        mask_array = np.asarray(mask, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"Non-numeric mask coordinates at row {row_index}") from exc

    if mask_array.size == 0:
        if oracle_level != "basic":
            raise AssertionError(f"Empty mask at row {row_index}")
        return
    if mask_array.size % 2 != 0:
        raise AssertionError(f"Odd number of mask coordinates at row {row_index}")

    coords = mask_array.reshape(-1, 2)
    if not np.isfinite(coords).all():
        raise AssertionError(f"Non-finite mask coordinates at row {row_index}")
    if oracle_level == "basic":
        return
    if coords.shape[0] < 3:
        raise AssertionError(f"Degenerate mask at row {row_index}")

    if oracle_level == "paranoid" and image_size is not None:
        width, height = image_size
        if coords.max() <= 1.5 and coords.min() >= -0.5:
            if (coords < -0.01).any() or (coords > 1.01).any():
                raise AssertionError(f"Normalized mask coordinates out of bounds at row {row_index}")
        else:
            x_limit = max(width, 1) * 4
            y_limit = max(height, 1) * 4
            if (
                (coords[:, 0] < -x_limit).any()
                or (coords[:, 0] > x_limit).any()
                or (coords[:, 1] < -y_limit).any()
                or (coords[:, 1] > y_limit).any()
            ):
                raise AssertionError(f"Pixel mask coordinates far outside image bounds at row {row_index}")


def validate_result_metrics(cells: Any, oracle_level: str) -> None:
    for column in ("diameter", "area", "volume"):
        if column not in cells.columns:
            continue
        values = numeric_column(cells, column)
        if values.size and not np.isfinite(values).all():
            raise AssertionError(f"Non-finite values in '{column}'")
        if values.size and (values < 0).any():
            raise AssertionError(f"Negative values in '{column}'")
        if oracle_level == "paranoid" and column == "area" and values.size and (values > 1.01).any():
            raise AssertionError("Area values exceed normalized image area")


def validate_scalar_result(key: str, value: Any, oracle_level: str) -> None:
    if value is None:
        raise AssertionError(f"Result '{key}' is None")
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(value):
            raise AssertionError(f"Result '{key}' is not finite: {value!r}")
        if oracle_level != "basic" and key == "Nuclei" and value < -100:
            raise AssertionError(f"Result '{key}' is below sentinel range: {value!r}")


def validate_model_result(
    case: dict[str, Any],
    result: dict[str, Any],
    oracle_level: str | None = None,
    image_size: tuple[int, int] | None = None,
) -> None:
    oracle_level = oracle_level or current_oracle_level()
    cells = result["Cells"]
    model_type = str(case["model_data"].get("model_type", ""))

    if hasattr(cells, "columns"):
        if model_type in {"yolo", "instanseg", "cellpose", "stardist"}:
            required_columns = {
                "id_label",
                "box",
                "mask",
                "confidence",
                "diameter",
                "area",
                "volume",
            }
            missing = required_columns - set(cells.columns)
            if missing:
                raise AssertionError(f"Missing detection columns: {sorted(missing)}")

        validate_result_metrics(cells, oracle_level)
        if oracle_level != "basic":
            validate_confidence_column(cells, model_type, oracle_level)
            validate_id_column(cells)

        if "box" in cells.columns:
            for index, box in enumerate(cells["box"].tolist()):
                validate_box_geometry(box, index, image_size, oracle_level)

        if "mask" in cells.columns:
            for index, mask in enumerate(cells["mask"].tolist()):
                validate_mask_geometry(mask, index, image_size, oracle_level)
    elif oracle_level != "basic":
        count = count_result_cells(cells)
        if count < 0:
            raise AssertionError(f"Negative cell count: {count}")

    for key in ("Nuclei", "%"):
        validate_scalar_result(key, result[key], oracle_level)


def result_dataframe(result: dict[str, Any]):
    cells = result.get("Cells")
    return cells if hasattr(cells, "columns") else None


def normalize_result_scalar(value: Any) -> int | float | str:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return "non-finite"
        return round(float(value), 6)
    if isinstance(value, int):
        return value
    return repr(value)


def result_fingerprint(result: dict[str, Any], mode: str) -> dict[str, Any]:
    cells = result.get("Cells")
    fingerprint: dict[str, Any] = {
        "cells": count_result_cells(cells),
        "nuclei": normalize_result_scalar(result.get("Nuclei")),
        "alive_percent": normalize_result_scalar(result.get("%")),
    }
    if mode != "full":
        return fingerprint

    cells_df = result_dataframe(result)
    if cells_df is None:
        return fingerprint

    numeric_summary: dict[str, Any] = {}
    for column in ("confidence", "diameter", "area", "volume"):
        if column not in cells_df.columns:
            continue
        values = numeric_column(cells_df, column)
        if values.size == 0:
            numeric_summary[column] = {"sum": 0.0, "mean": 0.0}
        else:
            numeric_summary[column] = {
                "sum": round(float(np.sum(values)), 6),
                "mean": round(float(np.mean(values)), 6),
            }
    fingerprint["numeric_summary"] = numeric_summary
    fingerprint["columns"] = sorted(str(column) for column in cells_df.columns)
    return fingerprint


def instantiate_case_model(case: dict[str, Any]):
    from model.Model import Model

    model_data = case["model_data"]
    object_size = dict(case["object_size"])
    object_size["signal"] = lambda *args, **kwargs: None
    model = Model(
        path=model_data["path"],
        object_size=object_size,
        model_type=model_data["model_type"],
        model_data=model_data,
        model_name=case["model_name"],
    )
    model.cell_counter.original_image_path = case["image_path"]
    return model, object_size


def calculate_case(model: Any, case: dict[str, Any]) -> dict[str, Any]:
    return model.calculate(
        img_path=case["image_path"],
        cell_channel=case["image"]["cell_channel"],
        nuclei_channel=case["image"]["nuclei_channel"],
    )


def assert_result_shape(case: dict[str, Any], result: Any) -> None:
    if not isinstance(result, dict):
        raise AssertionError(f"Expected result dict, got {type(result)!r}")
    if "Cells" not in result or "Nuclei" not in result or "%" not in result:
        raise AssertionError(f"Unexpected result keys: {sorted(result)}")


def run_determinism_check(
    case: dict[str, Any],
    reference_result: dict[str, Any],
    mode: str,
    oracle_level: str,
) -> None:
    if mode == "off":
        return

    rerun_model, _object_size = instantiate_case_model(case)
    rerun_result = calculate_case(rerun_model, case)
    assert_result_shape(case, rerun_result)
    validate_model_result(
        case,
        rerun_result,
        oracle_level=oracle_level,
        image_size=filter_image_size(case, rerun_model),
    )

    expected = result_fingerprint(reference_result, mode)
    actual = result_fingerprint(rerun_result, mode)
    if expected != actual:
        raise AssertionError(
            "Non-deterministic model result: "
            f"expected {json.dumps(expected, sort_keys=True)}, "
            f"got {json.dumps(actual, sort_keys=True)}"
        )


def filter_image_size(case: dict[str, Any], model: Any) -> tuple[int, int]:
    original_image = getattr(model.cell_counter, "original_image", None)
    if original_image is not None and hasattr(original_image, "shape"):
        return int(original_image.shape[1]), int(original_image.shape[0])

    shape = list(case.get("image", {}).get("shape", []))
    if len(shape) >= 3 and case.get("image", {}).get("extension") == "lsm":
        return int(shape[-1]), int(shape[-2])
    if len(shape) >= 2:
        return int(shape[1]), int(shape[0])
    return 512, 512


def apply_detection_filter(cells: Any, step: dict[str, Any], image_size: tuple[int, int]):
    from model.utils import filter_detections, filter_segmentation_detections

    min_size = float(step.get("min_size", 0.0))
    max_size = float(step.get("max_size", 1.0))

    if cells is None or not hasattr(cells, "columns"):
        return cells
    if cells.empty:
        return cells.copy()
    if {"area", "diameter", "volume"} & set(cells.columns):
        return filter_segmentation_detections(
            cells,
            min_size=min_size,
            max_size=max_size,
            size_metric=str(step.get("size_metric", "area")),
        )
    if "box" in cells.columns:
        lower = min(min_size, max_size)
        upper = max(min_size, max_size)
        return filter_detections(cells, min_size=lower, max_size=upper, img_size=image_size)
    return cells.copy()


def row_identity_set(cells: Any) -> set[str]:
    if cells is None or not hasattr(cells, "columns"):
        return set()
    return {str(index) for index in cells.index.tolist()}


def original_image_for_plot(case: dict[str, Any], model: Any) -> np.ndarray:
    from model.utils import read_img

    image = getattr(model.cell_counter, "original_image", None)
    if image is None:
        image = read_img(
            case["image_path"],
            cell_channel=case["image"]["cell_channel"],
            nuclei_channel=case["image"]["nuclei_channel"],
        )
    if image is None:
        raise AssertionError("No original image available for workflow plotting")

    image = normalize_loaded_array(np.asarray(image))
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)
    if image.ndim == 3 and image.shape[2] in (3, 4):
        return image.copy()
    return cv2_writable_image(image, "png")


def run_stateful_workflow(
    case: dict[str, Any],
    case_dir: Path,
    model: Any,
    result: dict[str, Any],
    oracle_level: str,
) -> dict[str, Any]:
    workflow = case.get("workflow", {})
    if workflow.get("mode") != "stateful":
        return result

    from model.utils import plot_predictions

    image_size = filter_image_size(case, model)
    for step_index, step in enumerate(workflow.get("steps", [])):
        operation = step.get("operation")
        cells = result_dataframe(result)

        if operation == "filter":
            filtered = apply_detection_filter(cells, step, image_size)
            if hasattr(filtered, "shape") and hasattr(cells, "shape") and filtered.shape[0] > cells.shape[0]:
                raise AssertionError("Filtered detections grew unexpectedly")
            probe_result = dict(result)
            probe_result["Cells"] = filtered
            validate_model_result(
                case,
                probe_result,
                oracle_level=oracle_level,
                image_size=image_size,
            )

        elif operation == "wide_filter":
            if cells is None or not hasattr(cells, "columns"):
                continue
            margin = float(step.get("margin", 0.0))
            narrow = apply_detection_filter(cells, step, image_size)
            wide_step = dict(step)
            wide_step["min_size"] = max(0.0, min(float(step.get("min_size", 0.0)), float(step.get("max_size", 1.0))) - margin)
            wide_step["max_size"] = min(1.0, max(float(step.get("min_size", 0.0)), float(step.get("max_size", 1.0))) + margin)
            wide = apply_detection_filter(cells, wide_step, image_size)
            if not row_identity_set(narrow).issubset(row_identity_set(wide)):
                raise AssertionError("Narrow filter result is not a subset of wide filter result")
            probe_result = dict(result)
            probe_result["Cells"] = wide
            validate_model_result(
                case,
                probe_result,
                oracle_level=oracle_level,
                image_size=image_size,
            )

        elif operation == "plot_masks":
            if cells is None or "mask" not in cells.columns:
                continue
            base_image = original_image_for_plot(case, model)
            output_path = case_dir / f"workflow_plot_{step_index:02d}.png"
            color_ids = cells["id_label"].tolist() if "id_label" in cells.columns else None
            plotted = plot_predictions(
                base_image.copy(),
                cells["mask"].tolist(),
                filename=str(output_path),
                colormap=str(step.get("color_map", "tab20")),
                alpha=float(step.get("alpha", 0.75)),
                color_ids=color_ids,
            )
            if not output_path.exists():
                raise AssertionError(f"Workflow plot was not written: {output_path}")
            plotted_array = np.asarray(plotted)
            if plotted_array.size and not np.isfinite(plotted_array.astype(np.float64)).all():
                raise AssertionError("Workflow plot contains non-finite values")

        elif operation == "rerun":
            object_size = getattr(model.cell_counter, "object_size", None)
            if isinstance(object_size, dict):
                object_size.update(
                    {
                        "min_size": step.get("min_size", object_size.get("min_size", 0.0)),
                        "max_size": step.get("max_size", object_size.get("max_size", 1.0)),
                        "size_metric": step.get("size_metric", object_size.get("size_metric", "area")),
                        "alpha": step.get("alpha", object_size.get("alpha", 0.75)),
                        "color_map": step.get("color_map", object_size.get("color_map", "tab20")),
                    }
                )
                object_size.setdefault("signal", lambda *args, **kwargs: None)
            result = model.calculate(
                img_path=case["image_path"],
                cell_channel=case["image"]["cell_channel"],
                nuclei_channel=case["image"]["nuclei_channel"],
            )
            if not isinstance(result, dict):
                raise AssertionError(f"Expected rerun result dict, got {type(result)!r}")
            if "Cells" not in result or "Nuclei" not in result or "%" not in result:
                raise AssertionError(f"Unexpected rerun result keys: {sorted(result)}")
            validate_model_result(
                case,
                result,
                oracle_level=oracle_level,
                image_size=filter_image_size(case, model),
            )

        else:
            raise AssertionError(f"Unknown workflow operation: {operation!r}")

    return result


def run_case_in_child(case_path: Path, max_rss_mb: float = 0) -> int:
    try:
        configure_child_sanitizers()
        memory_stop, memory_state = start_memory_monitor(max_rss_mb)
        case = json.loads(case_path.read_text(encoding="utf-8"))
        register_known_models()
        oracle_level = current_oracle_level()
        determinism_check = current_determinism_check()

        from model.utils import count_detected_objects

        model, object_size = instantiate_case_model(case)
        result = calculate_case(model, case)

        assert_result_shape(case, result)
        validate_model_result(
            case,
            result,
            oracle_level=oracle_level,
            image_size=filter_image_size(case, model),
        )
        run_determinism_check(case, result, determinism_check, oracle_level)
        result = run_stateful_workflow(case, case_path.parent, model, result, oracle_level)
        peak_rss_mb = stop_memory_monitor(memory_stop, memory_state, max_rss_mb)

        print(
            json.dumps(
                {
                    "case_id": case["case_id"],
                    "model": case["model_name"],
                    "profile": case["image"]["profile"],
                    "scale": object_size["scale"],
                    "cells": count_detected_objects(result["Cells"]),
                    "nuclei": result["Nuclei"],
                    "alive_percent": result["%"],
                    "peak_rss_mb": round(peak_rss_mb, 1),
                    "oracle_level": oracle_level,
                    "determinism_check": determinism_check,
                },
                ensure_ascii=True,
            )
        )
        return 0
    except Exception:
        traceback.print_exc()
        return 1


def normalize_signature_text(text: str) -> str:
    text = text.replace(str(ROOT), "<ROOT>")
    text = text.replace("\\", "/")
    text = re.sub(r'File "[^"]+", line \d+', 'File "<path>", line <n>', text)
    text = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", text)
    text = re.sub(r"\b\d+\.\d+\b", "N.N", text)
    text = re.sub(r"\b\d+\b", "N", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:500]


def extract_exception_line(output: str) -> str:
    exception_re = re.compile(
        r"^(?:[\w.]*Error|[\w.]*Exception|AssertionError|RuntimeError|"
        r"ValueError|TypeError|MemoryError|TimeoutError|cv2\.error)\b"
    )
    for line in reversed([line.strip() for line in output.splitlines() if line.strip()]):
        if exception_re.match(line):
            return line
    for line in reversed([line.strip() for line in output.splitlines() if line.strip()]):
        if "Traceback" not in line and not line.startswith("File "):
            return line
    return ""


def extract_app_frame(output: str) -> str:
    frame_re = re.compile(r'File "([^"]+)", line (\d+), in ([^\n]+)')
    frames = frame_re.findall(output)
    for path_text, _line, function in reversed(frames):
        try:
            path = Path(path_text).resolve()
            rel_path = path.relative_to(ROOT)
            return f"{rel_path.as_posix()}::{function.strip()}"
        except ValueError:
            continue
    if frames:
        path_text, _line, function = frames[-1]
        return f"{Path(path_text).name}::{function.strip()}"
    return ""


def failure_signature(
    case: dict[str, Any],
    stdout: str,
    stderr: str,
    reason: str,
) -> dict[str, Any]:
    output = "\n".join(part for part in (stdout or "", stderr or "") if part)
    exception = extract_exception_line(output) or reason
    exception_type = exception.split(":", 1)[0] if ":" in exception else exception.split(" ", 1)[0]
    kind = "timeout" if reason.startswith("timeout") else "crash"
    if "Peak RSS" in output and "exceeded limit" in output:
        kind = "memory"

    signature_basis = {
        "kind": kind,
        "exception_type": normalize_signature_text(exception_type),
        "exception": normalize_signature_text(exception),
        "frame": normalize_signature_text(extract_app_frame(output)),
        "model_type": str(case.get("model_data", {}).get("model_type", "")),
    }
    signature_hash = hashlib.sha1(
        json.dumps(signature_basis, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

    return {
        **signature_basis,
        "hash": signature_hash,
        "reason": reason,
        "model_name": case.get("model_name", ""),
        "workflow": case.get("workflow", {}).get("mode", "single"),
    }


def load_seen_signatures(signature_log: Path) -> set[str]:
    seen: set[str] = set()
    if not signature_log.exists():
        return seen
    for line in signature_log.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        signature_hash = entry.get("hash") or entry.get("signature", {}).get("hash")
        if signature_hash:
            seen.add(str(signature_hash))
    return seen


def append_signature_log(
    signature_log: Path,
    signature: dict[str, Any],
    case_dir: Path,
    duplicate: bool,
) -> None:
    signature_log.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        **signature,
        "case_dir": str(case_dir),
        "duplicate": duplicate,
        "recorded_at": int(time.time()),
    }
    with signature_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=True, sort_keys=True) + "\n")


def extract_child_result(stdout: str) -> dict[str, Any]:
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "case_id" in value and "model" in value:
            return value
    return {}


def append_pass_log(pass_log: Path, case: dict[str, Any], stdout: str) -> None:
    pass_log.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "case_id": case["case_id"],
        "model": case["model_name"],
        "model_type": case["model_data"].get("model_type", ""),
        "image": case["image"],
        "workflow": case.get("workflow", {}).get("mode", "single"),
        "child_result": extract_child_result(stdout),
    }
    with pass_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=True, sort_keys=True) + "\n")


def update_saved_case_paths(target: Path) -> dict[str, Any] | None:
    case_path = target / "case.json"
    if not case_path.exists():
        return None
    case = json.loads(case_path.read_text(encoding="utf-8"))
    old_image_path = Path(case.get("image_path", ""))
    moved_image = target / old_image_path.name
    if not moved_image.exists():
        inputs = sorted(target.glob("input.*"))
        if inputs:
            moved_image = inputs[0]
    if moved_image.exists():
        case["image_path"] = str(moved_image.resolve())
        case_path.write_text(json.dumps(case, indent=2), encoding="utf-8")
    return case


def save_failure(
    case_dir: Path,
    target_root: Path,
    stdout: str,
    stderr: str,
    reason: str,
    signature: dict[str, Any],
) -> Path:
    target_root.mkdir(parents=True, exist_ok=True)
    target = target_root / case_dir.name
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    shutil.move(str(case_dir), str(target))
    update_saved_case_paths(target)
    (target / "stdout.txt").write_text(stdout or "", encoding="utf-8")
    (target / "stderr.txt").write_text(stderr or "", encoding="utf-8")
    (target / "reason.txt").write_text(reason, encoding="utf-8")
    (target / "signature.json").write_text(
        json.dumps(signature, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target


def write_minimizer_image(path: Path, image: np.ndarray, extension: str) -> tuple[Path, dict[str, Any]]:
    image = normalize_loaded_array(np.asarray(image))
    extension = extension.lower().lstrip(".")

    if extension == "lsm":
        if image.ndim == 2:
            channels_last = np.dstack((image, np.zeros_like(image)))
        elif image.ndim == 3 and image.shape[2] == 1:
            channels_last = np.repeat(image, 2, axis=2)
        elif image.ndim == 3:
            channels_last = image[:, :, : max(2, min(4, image.shape[2]))]
        else:
            channels_last = image.reshape((*image.shape[-2:], 1))
            channels_last = np.repeat(channels_last, 2, axis=2)
        image_to_write = np.transpose(channels_last, (2, 0, 1))
        image_path = path / "input.lsm"
        tifffile.imwrite(str(image_path), image_to_write, metadata={"axes": "CYX"})
        return image_path, {
            "extension": "lsm",
            "shape": list(image_to_write.shape),
            "cell_channel": 0,
            "nuclei_channel": 1 if image_to_write.shape[0] > 1 else 0,
        }

    image_path = path / "input.png"
    writable = cv2_writable_image(image, "png")
    if not cv2.imwrite(str(image_path), writable):
        raise RuntimeError(f"Could not write minimized image: {image_path}")
    return image_path, {
        "extension": "png",
        "shape": list(writable.shape),
        "cell_channel": 0,
        "nuclei_channel": 0 if writable.ndim == 2 else min(1, writable.shape[2] - 1),
    }


def image_minimization_candidates(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    image = normalize_loaded_array(np.asarray(image))
    candidates: list[tuple[str, np.ndarray]] = []
    if image.ndim == 3 and image.shape[2] > 1:
        candidates.append(("first_channel", image[:, :, 0].copy()))

    height, width = image.shape[:2]
    if height > 1 or width > 1:
        target_h = max(1, height // 2)
        target_w = max(1, width // 2)
        candidates.append(
            (
                f"resize_{target_w}x{target_h}",
                cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA),
            )
        )

        crop_h = max(1, height // 2)
        crop_w = max(1, width // 2)
        y = max(0, (height - crop_h) // 2)
        x = max(0, (width - crop_w) // 2)
        if image.ndim == 2:
            candidates.append(("center_crop", image[y:y + crop_h, x:x + crop_w].copy()))
        else:
            candidates.append(("center_crop", image[y:y + crop_h, x:x + crop_w, :].copy()))

    candidates.append(("zeros", np.zeros_like(image)))
    if image.size:
        thresholded = np.where(image > int(np.median(image)), 255, 0).astype(np.uint8)
        candidates.append(("threshold", thresholded))
    return candidates


def case_minimization_candidates(case: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    candidates: list[tuple[str, dict[str, Any]]] = []

    simple_case = copy.deepcopy(case)
    simple_case["object_size"].update(
        {
            "min_size": 0.0,
            "max_size": 1.0,
            "line_width": 1.0,
            "alpha": 0.75,
            "color_map": "tab20",
            "size_metric": "area",
            "um_per_px": None,
        }
    )
    candidates.append(("simple_object_size", simple_case))

    workflow = case.get("workflow", {})
    if workflow.get("mode") == "stateful":
        steps = list(workflow.get("steps", []))
        if len(steps) > 1:
            shorter = copy.deepcopy(case)
            shorter["workflow"] = {"mode": "stateful", "steps": steps[: max(1, len(steps) // 2)]}
            candidates.append(("shorter_workflow", shorter))
        single_step = copy.deepcopy(case)
        single_step["workflow"] = {"mode": "stateful", "steps": steps[:1]}
        candidates.append(("one_step_workflow", single_step))

    return candidates


def materialize_minimizer_case(
    base_case: dict[str, Any],
    candidate_dir: Path,
    image: np.ndarray,
    case_variant: dict[str, Any] | None,
) -> dict[str, Any]:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    case = copy.deepcopy(case_variant or base_case)
    original_extension = str(base_case.get("image", {}).get("extension", "png"))
    image_path, image_meta = write_minimizer_image(candidate_dir, image, original_extension)
    case["image_path"] = str(image_path.resolve())
    case["image"].update(image_meta)
    case["image"]["profile"] = f"{case['image'].get('profile', 'unknown')}:minimized"
    case_path = candidate_dir / "case.json"
    case_path.write_text(json.dumps(case, indent=2), encoding="utf-8")
    return case


def candidate_preserves_signature(
    candidate_case: dict[str, Any],
    candidate_path: Path,
    expected_hash: str,
    timeout: float,
    sanitizers: str,
    max_rss_mb: float,
    oracle_level: str,
    determinism_check: str,
) -> tuple[bool, dict[str, Any], str, str, str]:
    try:
        completed = run_case_subprocess(
            candidate_path,
            timeout,
            sanitizers,
            max_rss_mb,
            oracle_level,
            determinism_check,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        reason = f"exit code {completed.returncode}"
        if completed.returncode == 0:
            return False, {}, stdout, stderr, reason
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        reason = f"timeout after {timeout} seconds"

    signature = failure_signature(candidate_case, stdout, stderr, reason)
    return signature.get("hash") == expected_hash, signature, stdout, stderr, reason


def minimize_failure(
    failure_dir: Path,
    expected_signature: dict[str, Any],
    args: argparse.Namespace,
) -> Path | None:
    case_path = failure_dir / "case.json"
    if not case_path.exists():
        return None

    base_case = json.loads(case_path.read_text(encoding="utf-8"))
    image = read_corpus_image(Path(base_case["image_path"]))
    if image is None:
        return None

    timeout = args.minimize_timeout or args.timeout
    expected_hash = str(expected_signature["hash"])
    work_dir = failure_dir / "minimize_work"
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    best_case = copy.deepcopy(base_case)
    best_image = image
    accepted: list[dict[str, Any]] = []
    attempt = 0

    while len(accepted) < max(0, args.minimize_steps):
        progress = False
        candidates: list[tuple[str, np.ndarray, dict[str, Any] | None]] = [
            (name, candidate_image, None)
            for name, candidate_image in image_minimization_candidates(best_image)
        ]
        candidates.extend(
            (name, best_image, candidate_case)
            for name, candidate_case in case_minimization_candidates(best_case)
        )

        for label, candidate_image, candidate_case_variant in candidates:
            if candidate_case_variant is None and np.array_equal(candidate_image, best_image):
                continue
            if candidate_case_variant is not None and json.dumps(
                candidate_case_variant,
                sort_keys=True,
                default=str,
            ) == json.dumps(best_case, sort_keys=True, default=str):
                continue

            attempt += 1
            candidate_dir = work_dir / f"attempt_{attempt:03d}_{label}"
            candidate_case = materialize_minimizer_case(
                best_case,
                candidate_dir,
                candidate_image,
                candidate_case_variant,
            )
            candidate_path = candidate_dir / "case.json"
            preserved, signature, stdout, stderr, reason = candidate_preserves_signature(
                candidate_case,
                candidate_path,
                expected_hash,
                timeout,
                args.sanitizers,
                args.max_rss_mb,
                args.oracle_level,
                args.determinism_check,
            )
            (candidate_dir / "stdout.txt").write_text(stdout or "", encoding="utf-8")
            (candidate_dir / "stderr.txt").write_text(stderr or "", encoding="utf-8")
            (candidate_dir / "reason.txt").write_text(reason, encoding="utf-8")
            if signature:
                (candidate_dir / "signature.json").write_text(
                    json.dumps(signature, indent=2, sort_keys=True),
                    encoding="utf-8",
                )

            if preserved:
                best_case = candidate_case
                best_image = candidate_image
                accepted.append(
                    {
                        "label": label,
                        "shape": list(np.asarray(best_image).shape),
                        "case_dir": str(candidate_dir),
                    }
                )
                progress = True
                break

        if not progress:
            break

    summary = {
        "expected_hash": expected_hash,
        "accepted": accepted,
        "attempts": attempt,
    }
    (failure_dir / "minimize_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if not accepted:
        shutil.rmtree(work_dir, ignore_errors=True)
        return None

    minimized_dir = failure_dir / "minimized"
    if minimized_dir.exists():
        shutil.rmtree(minimized_dir, ignore_errors=True)
    last_attempt_dir = Path(accepted[-1]["case_dir"])
    shutil.copytree(last_attempt_dir, minimized_dir)
    shutil.rmtree(work_dir, ignore_errors=True)
    return minimized_dir


def should_continue(case_count: int, max_cases: int, started_at: float, seconds: float) -> bool:
    if max_cases and case_count >= max_cases:
        return False
    if seconds and time.time() - started_at >= seconds:
        return False
    return True


def parent_main(args: argparse.Namespace) -> int:
    models = selected_models(load_enabled_models(Path(args.config)), args.models)
    if args.list_models:
        list_models(models)
        return 0

    runnable = OrderedDict(
        (name, data) for name, data in models.items() if model_is_locally_runnable(data)
    )
    if not runnable:
        raise SystemExit("No runnable models selected.")

    output_dir = Path(args.output_dir)
    cases_dir = output_dir / "cases"
    failures_dir = output_dir / "failures"
    duplicates_dir = output_dir / "duplicates"
    signature_log = output_dir / "signatures.jsonl"
    pass_log = output_dir / "passes.jsonl"
    run_summary_path = output_dir / "run_summary.json"
    cases_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed if args.seed is not None else int(time.time_ns() % (2**32))
    rng = random.Random(seed)
    allowed_scales = parse_scales(args.scales)
    corpus_images = discover_corpus_images(args.seed_corpus)
    corpus_probability = clamp_probability(args.corpus_probability)
    print(f"runtime fuzz seed: {seed}")
    print(f"selected models: {', '.join(runnable.keys())}")
    print(
        f"image profile: {args.image_profile}; "
        f"scales: {','.join(str(scale) for scale in allowed_scales)}; "
        f"max side: {args.max_side}; "
        f"model strategy: {args.model_strategy}; "
        f"workflow: {args.workflow}; "
        f"sanitizers: {args.sanitizers}; "
        f"oracle: {args.oracle_level}; "
        f"determinism: {args.determinism_check}"
    )
    if corpus_images:
        print(f"seed corpus images: {len(corpus_images)}; corpus probability: {corpus_probability:.2f}")

    started_at = time.time()
    case_count = 0
    passes = 0
    failures = 0
    unique_failures = 0
    duplicate_failures = 0
    seen_signatures = set() if args.no_dedupe else load_seen_signatures(signature_log)
    model_items = list(runnable.items())

    try:
        while should_continue(case_count, args.max_cases, started_at, args.seconds):
            case_count += 1
            if args.model_strategy == "round-robin":
                model_name, model_data = model_items[(case_count - 1) % len(model_items)]
            else:
                model_name, model_data = rng.choice(model_items)
            case_dir = cases_dir / f"case_{case_count:06d}_{rng.randrange(0, 2**32):08x}"
            case = build_case(
                rng,
                case_count,
                case_dir,
                model_name,
                dict(model_data),
                args.image_profile,
                allowed_scales,
                args.max_side,
                corpus_images,
                corpus_probability,
                args.workflow,
                args.max_workflow_steps,
            )
            case_path = case_dir / "case.json"
            case_path.write_text(json.dumps(case, indent=2), encoding="utf-8")

            try:
                completed = run_case_subprocess(
                    case_path,
                    args.timeout,
                    args.sanitizers,
                    args.max_rss_mb,
                    args.oracle_level,
                    args.determinism_check,
                )
                passed = completed.returncode == 0
                reason = f"exit code {completed.returncode}"
            except subprocess.TimeoutExpired as exc:
                completed = None
                passed = False
                reason = f"timeout after {args.timeout} seconds"
                stdout = exc.stdout or ""
                stderr = exc.stderr or ""
            else:
                stdout = completed.stdout
                stderr = completed.stderr

            if passed:
                passes += 1
                append_pass_log(pass_log, case, stdout)
                print(
                    f"PASS case={case_count} model={model_name} "
                    f"image={case['image']['extension']} shape={case['image']['shape']}"
                )
                if not args.keep_all:
                    shutil.rmtree(case_dir, ignore_errors=True)
            else:
                failures += 1
                signature = failure_signature(case, stdout, stderr, reason)
                is_duplicate = not args.no_dedupe and signature["hash"] in seen_signatures
                if is_duplicate:
                    duplicate_failures += 1
                    target_root = duplicates_dir / signature["hash"]
                else:
                    unique_failures += 1
                    target_root = failures_dir
                    seen_signatures.add(signature["hash"])

                failure_dir = save_failure(
                    case_dir,
                    target_root,
                    stdout,
                    stderr,
                    reason,
                    signature,
                )
                append_signature_log(signature_log, signature, failure_dir, is_duplicate)
                print(
                    f"FAIL case={case_count} model={model_name} signature={signature['hash']} "
                    f"duplicate={is_duplicate} reason={reason} saved={failure_dir}"
                )
                if args.minimize_failures and not is_duplicate:
                    minimized_dir = minimize_failure(failure_dir, signature, args)
                    if minimized_dir is not None:
                        print(f"MINIMIZED signature={signature['hash']} saved={minimized_dir}")
                if stdout:
                    print(stdout[-2000:])
                if stderr:
                    print(stderr[-2000:])
    except KeyboardInterrupt:
        print("Stopped by user.")

    print(
        f"finished cases={case_count} failures={failures} "
        f"unique={unique_failures} duplicates={duplicate_failures}"
    )
    run_summary = {
        "seed": seed,
        "selected_models": list(runnable.keys()),
        "settings": {
            "image_profile": args.image_profile,
            "model_strategy": args.model_strategy,
            "workflow": args.workflow,
            "max_workflow_steps": args.max_workflow_steps,
            "sanitizers": args.sanitizers,
            "oracle_level": args.oracle_level,
            "determinism_check": args.determinism_check,
            "max_rss_mb": args.max_rss_mb,
            "scales": allowed_scales,
            "max_side": args.max_side,
            "seed_corpus": args.seed_corpus,
            "corpus_probability": corpus_probability,
        },
        "cases": case_count,
        "passes": passes,
        "failures": failures,
        "unique_failures": unique_failures,
        "duplicate_failures": duplicate_failures,
        "duration_seconds": round(time.time() - started_at, 3),
        "passed": failures == 0,
        "signature_log": str(signature_log),
        "pass_log": str(pass_log),
    }
    run_summary_path.write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=True, sort_keys=True),
        encoding="utf-8",
    )
    print(f"run summary: {run_summary_path}")
    return 1 if failures else 0


def main() -> int:
    args = parse_args()
    if hasattr(args, "run_case"):
        return run_case_in_child(Path(args.run_case), args.max_rss_mb)
    if args.replay:
        completed = run_case_subprocess(
            Path(args.replay),
            args.timeout,
            args.sanitizers,
            args.max_rss_mb,
            args.oracle_level,
            args.determinism_check,
        )
        print(completed.stdout, end="")
        print(completed.stderr, end="", file=sys.stderr)
        return completed.returncode
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
