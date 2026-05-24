"""Smoke test for all models on all test images.

This test ensures that models don't crash when running on various test images.
"""

import logging
import os
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).parent.parent
ULTRALYTICS_CONFIG_DIR = PROJECT_ROOT / ".cache" / "ultralytics"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))
ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
SMOKE_IMAGE_MAX_SIDE = int(os.environ.get("SMOKE_IMAGE_MAX_SIDE", "512"))


def load_model(model_path: str, model_type: str):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    object_size = {
        'min_size': 100,
        'max_size': 0.000,
        'signal': lambda *args, **kwargs: None,
        'round_parametr_slider': 10**6,
        'round_parametr_value_input': 10**4,
        'color_map': "viridis",
        'color_map_list': [],
        'line_width': 100.00,
        'scale': 20,
        'um_per_px': 0.325,
    }

    dummy_logger = logging.getLogger("dummy_logger")
    dummy_logger.addHandler(logging.NullHandler())

    match model_type.lower():
        case "yolo":
            pytest.importorskip("ultralytics")
            pytest.importorskip("sahi")
            from model.YOLOSegmenter import YoloSegmenter

            class SmokeYoloSegmenter(YoloSegmenter):
                def init_x10_model(self, path_to_model):
                    self.model_x10 = None

            return SmokeYoloSegmenter(model_path, object_size=object_size, model_data={})
        case "cellpose":
            pytest.importorskip("cellpose")
            from model.CellposeSegmenter import CellposeSegmenter
            return CellposeSegmenter(model_path, object_size=object_size, logger=dummy_logger, model_data={})
        case "instanseg":
            pytest.importorskip("instanseg")
            from model.InstanSegSegmenter import InstansegSegmenter
            return InstansegSegmenter(model_path, object_size=object_size, logger=dummy_logger, model_data={})
        case "stardist":
            pytest.importorskip("pkg_resources")
            pytest.importorskip("stardist")
            from model.StardistSegmenter import StardistSegmenter
            return StardistSegmenter(model_path, object_size=object_size, logger=dummy_logger, model_data={})
        case "cellcounter":
            from model.CellCounter import CellCounter
            return CellCounter(model_path, object_size=object_size, model_data={})
        case _:
            raise ValueError(f"Unsupported model type: {model_type}. "
                            f"Supported types: yolo, cellpose, instanseg, stardist, cellcounter")


def reset_model_runtime_state(model):
    for attr in (
        "detections",
        "original_image",
        "inference_image",
        "_last_original_image",
        "_last_inference_image",
        "prediction_image",
    ):
        if hasattr(model, attr):
            setattr(model, attr, None)


def run_model_on_image(model, image_path: str):
    if not Path(image_path).exists():
        return {
            'success': False,
            'results': None,
            'message': f'Image file not found: {image_path}',
            'model': None,
            'error': f'Image file not found: {image_path}'
        }

    reset_model_runtime_state(model)

    if hasattr(model, 'predict'):
        results = model.predict(image_path)
    elif hasattr(model, 'calculate'):
        results = model.calculate(image_path)
    elif hasattr(model, 'count_x20'):
        results = model.count_x20(input_image=image_path, tracking=True)
    else:
        raise AttributeError(f"Class {type(model).__name__} doesn't have methods 'predict', 'calculate' or 'count_x20'")
    return {
        'success': True,
        'results': results,
        'message': f'Inference completed successfully on {image_path}',
        'model': model,
        'error': None
    }

# ============================================================================
# Hardcoded Lists
# ============================================================================

MODELS = [
    ("YOLO-512 Segmenter", "yolo", "trainedmodels/YOLO11x-512-seg.pt"),
    ("YOLO-680 Segmenter", "yolo", "trainedmodels/YOLO11x-680-seg.pt"),
    ("YOLO-Sphero Segmenter", "yolo", "trainedmodels/YOLO11x-sphero-seg.pt"),
    ("InstanSeg V3.1", "instanseg", "trainedmodels/Instanseg-Neuroblastoma-v3.1.pt"),
    ("InstanSeg 20250605", "instanseg", "trainedmodels/instanseg_20250605.pt"),
    ("CellPose SAM", "cellpose", "trainedmodels/cpsam_finetuned.pth"),
    ("StarDist", "stardist", "trainedmodels/stardist0602"),
]

TEST_IMAGES = [
    "testimages/Neuroblastoma Cells Images/SK-N-DZ.jpg",
    "testimages/Plain Cells Images/TYPE_13_10.jpg",
    "testimages/Stained Nuclei Images/A1_1.TIF",
]


def model_id(model_case):
    return model_case[0]


def prepare_smoke_image(image_path: Path, output_dir: Path) -> Path:
    if SMOKE_IMAGE_MAX_SIDE <= 0:
        return image_path

    from PIL import Image

    with Image.open(image_path) as image:
        image_format = image.format
        if max(image.size) <= SMOKE_IMAGE_MAX_SIDE:
            return image_path

        smoke_image = image.copy()
        smoke_image.thumbnail(
            (SMOKE_IMAGE_MAX_SIDE, SMOKE_IMAGE_MAX_SIDE),
            Image.Resampling.LANCZOS,
        )
        if image_format == "JPEG" and smoke_image.mode not in ("RGB", "L"):
            smoke_image = smoke_image.convert("RGB")

        output_path = output_dir / image_path.name
        smoke_image.save(output_path, format=image_format)
        return output_path


@pytest.fixture(scope="module")
def smoke_images(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("smoke_images")
    return {
        image_path: prepare_smoke_image(PROJECT_ROOT / image_path, output_dir)
        for image_path in TEST_IMAGES
    }


@pytest.fixture(scope="module", params=MODELS, ids=model_id)
def loaded_model(request):
    model_name, model_type, model_path = request.param
    full_model_path = PROJECT_ROOT / model_path
    if not full_model_path.exists():
        pytest.skip(
            f"{model_name} weights are missing at {full_model_path}. "
            "Copy the model into trainedmodels/ to enable this smoke case."
        )

    return model_name, load_model(str(full_model_path), model_type)


# ============================================================================
# Tests
# ============================================================================

@pytest.mark.parametrize("test_image", TEST_IMAGES)
def test_smoke(loaded_model, smoke_images, test_image):
    """
    Smoke test: Run all models on all test images.
    
    Ensures that models don't crash with errors during inference.
    This test doesn't validate output correctness, only that the models
    run without raising exceptions.
    
    Args:
        loaded_model: Fixture with the model name and loaded model instance.
        smoke_images: Prepared image paths for fast smoke inference.
        test_image (str): Relative path to the test image.
    """
    model_name, model = loaded_model

    # Test model with each test image
    full_image_path = smoke_images[test_image]

    # Run model on image - should not raise exceptions
    result = run_model_on_image(
        model=model,
        image_path=str(full_image_path),
    )
    
    # Verify that inference returned a result
    assert result is not None, \
        f"Result is None for {model_name} on {test_image}"
    assert isinstance(result, dict), \
        f"Result should be dict for {model_name} on {test_image}"
    
    # Check if there was an error
    if not result.get("success"):
        error_msg = result.get("error", "Unknown error")
        pytest.fail(
            f"Model {model_name} failed on {test_image}: {error_msg}"
        )
