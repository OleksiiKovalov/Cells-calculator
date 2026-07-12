"""Smoke tests for Training Studio (trainer-only).

Dataset preparation lives in the Dataset Viewer app; this app trains on a
prepared ``.pth``. These tests need neither an InstanSeg backend nor a GPU:
every step script prints --help, the config loads, the image-preview helpers
work, and the backend-kwarg gating behaves.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))


def _help(script: str):
    return subprocess.run(
        [sys.executable, str(SRC / script), "--help"],
        capture_output=True, text=True,
    )


def test_runner_help():
    r = _help("runner.py")
    assert r.returncode == 0
    assert "pipeline" in r.stdout.lower()


def test_train_help():
    assert _help("train.py").returncode == 0


def test_evaluate_help():
    assert _help("evaluate.py").returncode == 0


def test_export_model_help():
    assert _help("export_model.py").returncode == 0


def test_config_defaults_load():
    import config

    cfg = config.load_config()
    for section in ("train", "evaluate", "export"):
        assert section in cfg
    # dataset-prep config is gone (that lives in Dataset Viewer now)
    assert "preprocess" not in cfg and "dataset" not in cfg
    assert cfg["train"]["target_segmentation"] in ("C", "N")


def test_datalib_preview_helpers(tmp_path):
    import datalib

    mask = np.zeros((20, 20), dtype=np.int32)
    mask[2:8, 2:8] = 1
    mask[10:15, 10:15] = 2
    rgb = datalib.colorize_labels(mask)
    assert rgb.shape == (20, 20, 3)
    assert rgb[0, 0].sum() == 0                    # background stays black
    assert rgb[4, 4].sum() > 0                     # label 1 gets a colour

    assert datalib.is_mask_file("cell_001_masks.tif")
    assert not datalib.is_mask_file("cell_001.tif")
    assert datalib.mask_path_for(Path("cell_001.tif")).name == "cell_001_masks.tif"


def test_training_kwarg_gating(tmp_path):
    """Fork-only kwargs (max_no_improvement) are passed ONLY when the detected
    backend supports them; core kwargs always are. Backend-independent."""
    import torch
    import train

    pth = tmp_path / "x.pth"
    torch.save({"Train": [{"image": 1, "cell_masks": 1, "parent_dataset": "my_dataset"}],
                "Validation": [], "Test": []}, str(pth))

    core = {"data_path", "source_dataset", "model_folder", "model_path", "output_path",
            "experiment_str", "requested_pixel_size", "target_segmentation",
            "num_epochs", "hotstart_training"}
    tcfg = {"model_folder": "m", "source_dataset": "my_dataset", "target_segmentation": "C",
            "requested_pixel_size": 0.5, "num_epochs": 3, "max_no_improvement": 5,
            "hotstart_training": 5, "resume_weights": ""}

    captured = {}
    orig = train.resolve_training_backend
    try:
        for kind, supported, expect_marker in (
            ("mainline", set(core), False),
            ("fork", core | {"max_no_improvement"}, True),
        ):
            train.resolve_training_backend = lambda log=None, k=kind, s=supported: (
                lambda **kw: captured.update(kw) or captured, k, s)
            captured.clear()
            train.run_training(pth, tmp_path / "models", tmp_path / "models", tcfg, log=lambda *_: None)
            assert core <= set(captured), f"{kind}: missing core kwargs"
            assert ("max_no_improvement" in captured) is expect_marker
            assert "segmentation_dataset" in captured
    finally:
        train.resolve_training_backend = orig


def test_no_backend_message():
    """With no InstanSeg importable, the trainer exits cleanly with a hint."""
    import builtins
    import train

    real = builtins.__import__

    def block(name, *a, **k):
        if name == "instanseg" or name.startswith("instanseg."):
            raise ImportError("simulated: no instanseg")
        return real(name, *a, **k)

    builtins.__import__ = block
    try:
        try:
            train.resolve_training_backend(log=lambda *_: None)
            assert False, "expected SystemExit"
        except SystemExit as e:
            assert e.code == 2
    finally:
        builtins.__import__ = real
