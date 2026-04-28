"""Smoke test for all models on all test images.

This test ensures that models don't crash when running on various test images.
"""

import sys
import pytest
from pathlib import Path

# Redirect output before importing pytest
if "pytest" in sys.modules:
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Initialize PyQt5 before importing UI modules
from PyQt5.QtWidgets import QApplication

app = QApplication.instance()
if app is None:
    sys.argv.extend(["--platform", "offscreen"])
    app = QApplication(sys.argv)

# Import model factory from main_2
from main_2 import run_model_on_image


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
    
    # Skip test if model doesn't exist
    if not full_model_path.exists():
        pytest.skip(f"Model not found: {model_path}")
    
    # Test model with each test image
    full_image_path = project_root / test_image  
        
        # Skip if test image doesn't exist
    if not full_image_path.exists():
        pytest.skip(f"Test image not found: {test_image}")
    
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
