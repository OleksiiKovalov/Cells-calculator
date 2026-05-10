"""Smoke test for all models on all test images.

This test ensures that models don't crash when running on various test images.
"""

import pytest
from pathlib import Path

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
    match model_type.lower():
        case "yolo":
            pytest.importorskip("ultralytics")
            pytest.importorskip("sahi")
            from model.YOLOSegmenter import YoloSegmenter

            return YoloSegmenter(model_path, object_size=object_size, model_data={})
        case "cellpose":
            pytest.importorskip("cellpose")
            from model.CellposeSegmenter import CellposeSegmenter

            return CellposeSegmenter(model_path, object_size=object_size, model_data={})
        case "instanseg":
            pytest.importorskip("instanseg")
            from model.InstanSegSegmenter import InstansegSegmenter

            return InstansegSegmenter(model_path, object_size=object_size, model_data={})
        case "stardist":
            pytest.importorskip("pkg_resources")
            pytest.importorskip("stardist")
            from model.StardistSegmenter import StardistSegmenter

            return StardistSegmenter(model_path, object_size=object_size, model_data={})
        case "cellcounter":
            from model.CellCounter import CellCounter

            return CellCounter(model_path, object_size=object_size, model_data={})
        case _:
            raise ValueError(f"Unsupported model type: {model_type}. "
                            f"Supported types: yolo, cellpose, instanseg, stardist, cellcounter")

def run_model_on_image(model_path: str, image_path: str, model_type: str = "yolo"):
    if not Path(image_path).exists():
        return {
            'success': False,
            'results': None,
            'message': f'Image file not found: {image_path}',
            'model': None,
            'error': f'Image file not found: {image_path}'
        }

    model = load_model(model_path, model_type)

    if hasattr(model, 'predict'):
        results = model.predict(image_path)
    elif hasattr(model, 'calculate'):
        results = model.calculate(image_path)
    elif hasattr(model, 'count_x20'):
        results = model.count_x20(input_image=image_path, tracking=True, plot=False)
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


# ============================================================================
# Tests
# ============================================================================

@pytest.mark.parametrize("model_name,model_type,model_path", MODELS)
@pytest.mark.parametrize("test_image", TEST_IMAGES)
def test_smoke(model_name, model_type, model_path, test_image):
    """
    Smoke test: Run all models on all test images.
    
    Ensures that models don't crash with errors during inference.
    This test doesn't validate output correctness, only that the models
    run without raising exceptions.
    
    Args:
        model_name (str): Friendly name of the model
        model_type (str): Type of model (yolo, instanseg, cellpose, stardist)
        model_path (str): Relative path to the model file
    """
    project_root = Path(__file__).parent.parent
    full_model_path = project_root / model_path

    # Test model with each test image
    full_image_path = project_root / test_image  

    # Run model on image - should not raise exceptions
    result = run_model_on_image(
        model_path=str(full_model_path),
        image_path=str(full_image_path),
        model_type=model_type
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
