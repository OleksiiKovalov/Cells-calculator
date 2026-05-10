"""Runtime fuzz harness for generated images and real model inference.

This is intentionally separate from pytest. Real model inference can be slow,
can hang, and may crash native libraries, so each generated case runs in a
fresh child process with a timeout. Failing cases are kept for replay.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
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

IMAGE_PROFILES = ("auto", "generic", "yolo")


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
        help="Image generator profile. 'auto' uses a model-specific profile when available.",
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
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--keep-all", action="store_true")
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


def resolve_image_profile(profile: str, model_data: dict[str, Any]) -> str:
    if profile != "auto":
        return profile
    if str(model_data.get("model_type", "")).lower() == "yolo":
        return "yolo"
    return "generic"


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
) -> tuple[Path, dict[str, Any]]:
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


def build_case(
    rng: random.Random,
    case_id: int,
    case_dir: Path,
    model_name: str,
    model_data: dict[str, Any],
    image_profile: str,
    allowed_scales: list[int],
    max_side: int,
) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    resolved_profile = resolve_image_profile(image_profile, model_data)
    image_path, image_meta = write_generated_image(
        rng,
        case_dir,
        str(model_data.get("model_type", "")),
        resolved_profile,
        max_side,
    )
    return {
        "case_id": case_id,
        "model_name": model_name,
        "model_data": model_data,
        "image_path": str(image_path.resolve()),
        "image": image_meta,
        "object_size": make_object_size(rng, model_data, allowed_scales),
    }


def child_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
    return env


def run_case_subprocess(case_path: Path, timeout: float) -> subprocess.CompletedProcess:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-case",
        str(case_path),
    ]
    return subprocess.run(
        command,
        cwd=str(ROOT),
        env=child_environment(),
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


def validate_model_result(case: dict[str, Any], result: dict[str, Any]) -> None:
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

        for column in ("diameter", "area", "volume"):
            if column not in cells.columns:
                continue
            values = np.asarray(cells[column].tolist(), dtype=np.float64)
            if values.size and not np.isfinite(values).all():
                raise AssertionError(f"Non-finite values in '{column}'")
            if values.size and (values < 0).any():
                raise AssertionError(f"Negative values in '{column}'")

        if "box" in cells.columns:
            for index, box in enumerate(cells["box"].tolist()):
                box_array = np.asarray(box, dtype=np.float64).reshape(-1)
                if box_array.size != 4:
                    raise AssertionError(f"Invalid box size at row {index}: {box_array.size}")
                if not np.isfinite(box_array).all():
                    raise AssertionError(f"Non-finite box at row {index}")

        if "mask" in cells.columns:
            for index, mask in enumerate(cells["mask"].tolist()):
                mask_array = np.asarray(mask, dtype=np.float64).reshape(-1)
                if mask_array.size and not np.isfinite(mask_array).all():
                    raise AssertionError(f"Non-finite mask coordinates at row {index}")

    for key in ("Nuclei", "%"):
        value = result[key]
        if value is None:
            raise AssertionError(f"Result '{key}' is None")
        if isinstance(value, (int, float, np.integer, np.floating)) and not np.isfinite(value):
            raise AssertionError(f"Result '{key}' is not finite: {value!r}")


def run_case_in_child(case_path: Path) -> int:
    try:
        case = json.loads(case_path.read_text(encoding="utf-8"))
        register_known_models()

        from model.Model import Model
        from model.utils import count_detected_objects

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
        result = model.calculate(
            img_path=case["image_path"],
            cell_channel=case["image"]["cell_channel"],
            nuclei_channel=case["image"]["nuclei_channel"],
        )

        if not isinstance(result, dict):
            raise AssertionError(f"Expected result dict, got {type(result)!r}")
        if "Cells" not in result or "Nuclei" not in result or "%" not in result:
            raise AssertionError(f"Unexpected result keys: {sorted(result)}")
        validate_model_result(case, result)

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
                },
                ensure_ascii=True,
            )
        )
        return 0
    except Exception:
        traceback.print_exc()
        return 1


def save_failure(
    case_dir: Path,
    failures_dir: Path,
    stdout: str,
    stderr: str,
    reason: str,
) -> Path:
    failures_dir.mkdir(parents=True, exist_ok=True)
    target = failures_dir / case_dir.name
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    shutil.move(str(case_dir), str(target))
    (target / "stdout.txt").write_text(stdout or "", encoding="utf-8")
    (target / "stderr.txt").write_text(stderr or "", encoding="utf-8")
    (target / "reason.txt").write_text(reason, encoding="utf-8")
    return target


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
    cases_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed if args.seed is not None else int(time.time_ns() % (2**32))
    rng = random.Random(seed)
    allowed_scales = parse_scales(args.scales)
    print(f"runtime fuzz seed: {seed}")
    print(f"selected models: {', '.join(runnable.keys())}")
    print(
        f"image profile: {args.image_profile}; "
        f"scales: {','.join(str(scale) for scale in allowed_scales)}; "
        f"max side: {args.max_side}"
    )

    started_at = time.time()
    case_count = 0
    failures = 0
    model_items = list(runnable.items())

    try:
        while should_continue(case_count, args.max_cases, started_at, args.seconds):
            case_count += 1
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
            )
            case_path = case_dir / "case.json"
            case_path.write_text(json.dumps(case, indent=2), encoding="utf-8")

            try:
                completed = run_case_subprocess(case_path, args.timeout)
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
                print(
                    f"PASS case={case_count} model={model_name} "
                    f"image={case['image']['extension']} shape={case['image']['shape']}"
                )
                if not args.keep_all:
                    shutil.rmtree(case_dir, ignore_errors=True)
            else:
                failures += 1
                failure_dir = save_failure(case_dir, failures_dir, stdout, stderr, reason)
                print(
                    f"FAIL case={case_count} model={model_name} "
                    f"reason={reason} saved={failure_dir}"
                )
                if stdout:
                    print(stdout[-2000:])
                if stderr:
                    print(stderr[-2000:])
    except KeyboardInterrupt:
        print("Stopped by user.")

    print(f"finished cases={case_count} failures={failures}")
    return 1 if failures else 0


def main() -> int:
    args = parse_args()
    if hasattr(args, "run_case"):
        return run_case_in_child(Path(args.run_case))
    if args.replay:
        completed = run_case_subprocess(Path(args.replay), args.timeout)
        print(completed.stdout, end="")
        print(completed.stderr, end="", file=sys.stderr)
        return completed.returncode
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
