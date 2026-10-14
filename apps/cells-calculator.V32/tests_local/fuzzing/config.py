"""Runtime fuzz harness for generated images and real model inference.

This is intentionally separate from pytest. Real model inference can be slow,
can hang, and may crash native libraries, so each generated case runs in a
fresh child process with a timeout. Failing cases are kept for replay.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import copy
import gc
import hashlib
import inspect
import importlib.abc
import importlib.machinery
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import tracemalloc
import warnings
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parents[2]
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

IMAGE_PROFILES = ("auto", "generic", "yolo", "microscopy", "mixed")
GENERATION_ENGINES = ("legacy", "strategy", "hypothesis", "grammar-mutational")
CORPUS_MUTATION_MODES = ("stress", "cell-preserving", "mixed")
MODEL_STRATEGIES = ("random", "round-robin")
MOCK_MODES = ("none", "model", "fault-injection")
SANITIZERS = ("none", "python", "warnings", "numpy", "leaks", "all")
WORKFLOWS = ("single", "stateful")
UI_WORKFLOW_OPERATIONS = (
    "filter",
    "wide_filter",
    "plot_masks",
    "render_predictions",
    "aligned_plot",
    "publish_images",
    "channel_rerun",
    "rerun",
)
ORACLE_LEVELS = ("basic", "strict", "paranoid")
DETERMINISM_CHECKS = ("off", "counts", "full")
PYTHON_COVERAGE_MODES = ("off", "model", "mcdc")
VALID_GENERATED_EXTENSIONS = {"png", "jpg", "jpeg", "tif", "tiff", "bmp", "lsm"}
GENERIC_PATTERNS = ("zeros", "constant", "noise", "gradient", "spots", "stripes")
YOLO_PATTERNS = (
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
)
MICROSCOPY_PATTERNS = (
    "single_cells",
    "touching_cells",
    "overlapping_cells",
    "edge_cells",
    "dense_colony",
    "low_contrast_cells",
    "nuclei_cytoplasm",
    "elongated_cells",
    "debris_field",
    "saturated_cells",
    "zstack_cells",
)
GRAMMAR_MUTATIONS = (
    "expand:Scene",
    "expand:MicroscopyScene",
    "expand:CellPopulation",
    "expand:Cell",
    "expand:Artifacts",
    "expand:Optics",
    "expand:Channels",
    "expand:Encoding",
    "expand:Canvas",
    "expand:Workflow",
    "mutate:ObjectSize",
    "crossover:CellPopulation",
)
SCENE_GRAMMAR = {
    "Scene": ("MicroscopyScene", "YoloScene", "DegenerateScene"),
    "MicroscopyScene": (
        "Background CellPopulation Artifacts Optics Channels Encoding",
    ),
    "YoloScene": ("Background BlobPopulation Artifacts Optics Encoding",),
    "DegenerateScene": ("Canvas Encoding Workflow",),
    "CellPopulation": (
        "Sparse",
        "Dense",
        "Clustered",
        "Border",
        "Overlapping",
        "NucleiCytoplasm",
        "ZStack",
    ),
    "Cell": ("Ellipse", "Ring", "Elongated", "Partial", "Saturated"),
    "Workflow": ("Inference Filter* Render* Rerun?",),
}
MICROSCOPY_SCENE_RULES = {
    "single_cells": {
        "population": "Sparse",
        "cell": "Ellipse",
        "constraints": ("cell_count<=8", "non_overlapping_centers"),
    },
    "touching_cells": {
        "population": "Clustered",
        "cell": "Ellipse",
        "constraints": ("distance(center_i,center_j)<=radius_i+radius_j",),
    },
    "overlapping_cells": {
        "population": "Overlapping",
        "cell": "Ellipse",
        "constraints": ("distance(center_i,center_j)<0.75*(radius_i+radius_j)",),
    },
    "edge_cells": {
        "population": "Border",
        "cell": "Partial",
        "constraints": ("some_cells_intersect_image_border",),
    },
    "dense_colony": {
        "population": "Dense",
        "cell": "Ellipse",
        "constraints": ("cell_count>=12", "cluster_occupies_central_region"),
    },
    "low_contrast_cells": {
        "population": "Sparse",
        "cell": "Ellipse",
        "constraints": ("contrast<=34",),
    },
    "nuclei_cytoplasm": {
        "population": "NucleiCytoplasm",
        "cell": "Ellipse",
        "constraints": ("channels>=2", "nuclei_channel_present"),
    },
    "elongated_cells": {
        "population": "Sparse",
        "cell": "Elongated",
        "constraints": ("max(rx,ry)/min(rx,ry)>=2",),
    },
    "debris_field": {
        "population": "Dense",
        "cell": "Ellipse",
        "constraints": ("debris_count>0",),
    },
    "saturated_cells": {
        "population": "Sparse",
        "cell": "Saturated",
        "constraints": ("cell_intensity>=220",),
    },
    "zstack_cells": {
        "population": "ZStack",
        "cell": "Ellipse",
        "constraints": ("channels>=3", "encoding in {lsm,tif}", "z_planes>=2"),
    },
}
VALID_MOCK_VARIANTS = (
    "empty",
    "single",
    "dense",
    "border",
    "pixel_masks",
    "inference_scaled",
)
FAULT_INJECTION_VARIANTS = (
    "missing_columns",
    "nonfinite_confidence",
    "negative_area",
    "far_box",
    "degenerate_mask",
)


def canonical_minimizer_extension(extension: str) -> str:
    extension = extension.lower().lstrip(".")
    if extension == "jpeg":
        return "jpg"
    if extension == "tiff":
        return "tif"
    if extension in VALID_GENERATED_EXTENSIONS:
        return extension
    return "png"


def non_lsm_channel_count(image: np.ndarray) -> int:
    return image.shape[2] if image.ndim == 3 else 1


@dataclass
class RuntimeFuzzConfig:
    config: Path
    models: list[str]
    max_cases: int
    seconds: float
    timeout: float
    seed: int | None
    image_profile: str
    generation_engine: str
    seed_corpus: str
    corpus_probability: float
    corpus_mutation_mode: str
    model_strategy: str
    mock_mode: str
    workflow: str
    max_workflow_steps: int
    scales: str
    max_side: int
    sanitizers: str
    oracle_level: str
    determinism_check: str
    python_coverage: str
    coverage_guided: bool
    coverage_corpus_probability: float
    max_rss_mb: float
    max_tracemalloc_mb: float
    max_handle_growth: int
    max_allocated_block_growth: int
    output_dir: Path
    keep_all: bool
    no_dedupe: bool
    minimize_failures: bool
    minimize_steps: int
    minimize_timeout: float
    list_models: bool
    replay: str
    run_case: str = ""

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> "RuntimeFuzzConfig":
        return cls(
            config=Path(namespace.config),
            models=list(namespace.models),
            max_cases=namespace.max_cases,
            seconds=namespace.seconds,
            timeout=namespace.timeout,
            seed=namespace.seed,
            image_profile=namespace.image_profile,
            generation_engine=namespace.generation_engine,
            seed_corpus=namespace.seed_corpus,
            corpus_probability=namespace.corpus_probability,
            corpus_mutation_mode=namespace.corpus_mutation_mode,
            model_strategy=namespace.model_strategy,
            mock_mode=namespace.mock_mode,
            workflow=namespace.workflow,
            max_workflow_steps=namespace.max_workflow_steps,
            scales=namespace.scales,
            max_side=namespace.max_side,
            sanitizers=namespace.sanitizers,
            oracle_level=namespace.oracle_level,
            determinism_check=namespace.determinism_check,
            python_coverage=namespace.python_coverage,
            coverage_guided=namespace.coverage_guided,
            coverage_corpus_probability=namespace.coverage_corpus_probability,
            max_rss_mb=namespace.max_rss_mb,
            max_tracemalloc_mb=namespace.max_tracemalloc_mb,
            max_handle_growth=namespace.max_handle_growth,
            max_allocated_block_growth=namespace.max_allocated_block_growth,
            output_dir=Path(namespace.output_dir),
            keep_all=namespace.keep_all,
            no_dedupe=namespace.no_dedupe,
            minimize_failures=namespace.minimize_failures,
            minimize_steps=namespace.minimize_steps,
            minimize_timeout=namespace.minimize_timeout,
            list_models=namespace.list_models,
            replay=namespace.replay,
            run_case=getattr(namespace, "run_case", ""),
        )


@dataclass
class FuzzCaseSpec:
    model_name: str
    image_kind: str
    width: int
    height: int
    channels: int | None
    pattern: str
    extension: str
    object_size: dict[str, Any] = field(default_factory=dict)
    workflow: list[dict[str, Any]] = field(default_factory=list)
    image_recipe: dict[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


def parse_args() -> RuntimeFuzzConfig:
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
        "--generation-engine",
        choices=GENERATION_ENGINES,
        default="legacy",
        help=(
            "legacy uses the original random image generator. strategy uses a seed-stable "
            "case-spec generator and the same subprocess harness. grammar-mutational "
            "mutates structured image/workflow recipes. hypothesis is kept as a "
            "backward-compatible alias for strategy."
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
        "--corpus-mutation-mode",
        choices=CORPUS_MUTATION_MODES,
        default="mixed",
        help=(
            "How to mutate seed corpus images. stress keeps older destructive edge-case "
            "mutations. cell-preserving crops around foreground blobs and applies mild "
            "microscopy-like augmentation. mixed alternates between both."
        ),
    )
    parser.add_argument(
        "--model-strategy",
        choices=MODEL_STRATEGIES,
        default="random",
        help="How to choose models across cases. Default: random.",
    )
    parser.add_argument(
        "--mock-mode",
        choices=MOCK_MODES,
        default="none",
        help=(
            "none runs real model inference. model replaces model inference with a "
            "deterministic fast prediction generator so UI/rendering/filtering branches "
            "can be fuzzed deeply. fault-injection returns deliberately suspicious "
            "prediction contracts to validate oracles and UI defensive handling."
        ),
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
            "warnings=RuntimeWarning/ResourceWarning as error; "
            "numpy=np.seterr(all='raise'); leaks=tracemalloc/handle probes; "
            "all=all checks."
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
        "--python-coverage",
        choices=PYTHON_COVERAGE_MODES,
        default="off",
        help=(
            "Optional Python coverage collected in child processes. model keeps legacy "
            "line coverage for repo model/UI code; mcdc instruments boolean decisions "
            "and records condition/outcome/MC/DC signals."
        ),
    )
    parser.add_argument(
        "--coverage-guided",
        action="store_true",
        help=(
            "Enable coverage feedback. New Python model/UI coverage signals save the "
            "case under interesting/ and feed that image back into future corpus mutations."
        ),
    )
    parser.add_argument(
        "--coverage-corpus-probability",
        type=float,
        default=0.55,
        help=(
            "When --coverage-guided has found interesting cases, probability of drawing "
            "the next corpus image from interesting/ instead of the seed corpus."
        ),
    )
    parser.add_argument(
        "--max-rss-mb",
        type=float,
        default=0,
        help="Fail a case if sampled child RSS exceeds this many MB. 0 disables.",
    )
    parser.add_argument(
        "--max-tracemalloc-mb",
        type=float,
        default=0,
        help=(
            "Fail a case if Python tracemalloc peak exceeds this many MB. "
            "0 disables the tracemalloc probe because it is expensive for ML imports."
        ),
    )
    parser.add_argument(
        "--max-handle-growth",
        type=int,
        default=0,
        help=(
            "Fail a case if open handles/file descriptors grow by more than this. "
            "Requires psutil. 0 disables the failure threshold."
        ),
    )
    parser.add_argument(
        "--max-allocated-block-growth",
        type=int,
        default=0,
        help=(
            "Fail a case if CPython allocated block count grows by more than this. "
            "0 disables the failure threshold."
        ),
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
    return RuntimeFuzzConfig.from_namespace(parser.parse_args())


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
