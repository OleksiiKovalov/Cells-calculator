"""Regression tests for runtime crashes found by fuzzing and review."""

import importlib
import json
import sys
import types

import cv2
import numpy as np
import pandas as pd
import tiffile

from model.NucleiCounter import NucleiCounter
import model.utils as model_utils


class _RecordingCellCounter:
    def __init__(self):
        self.image = None

    def count_cells(self, img_path):
        self.image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        return pd.DataFrame({"box": [np.array([0, 0, 1, 1])]})


class _RecordingNucleiCounter:
    def __init__(self):
        self.channel = None

    def countNuclei(self, img_channel):
        self.channel = img_channel.copy()
        return 1


def _import_with_fakes(request, monkeypatch, module_name, fake_modules):
    previous_module = sys.modules.pop(module_name, None)
    for name, module in fake_modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    imported = importlib.import_module(module_name)

    def restore_module():
        sys.modules.pop(module_name, None)
        if previous_module is not None:
            sys.modules[module_name] = previous_module

    request.addfinalizer(restore_module)
    return imported


def _fake_instanseg_modules():
    fake_instanseg = types.ModuleType("instanseg")
    fake_instanseg.InstanSeg = object

    fake_utils_pkg = types.ModuleType("instanseg.utils")
    fake_utils = types.ModuleType("instanseg.utils.utils")
    fake_utils.labels_to_features = lambda labels: {"features": []}

    return {
        "instanseg": fake_instanseg,
        "instanseg.utils": fake_utils_pkg,
        "instanseg.utils.utils": fake_utils,
    }


def _fake_stardist_modules():
    fake_tf = types.ModuleType("tensorflow")
    fake_tf.config = types.SimpleNamespace(
        list_physical_devices=lambda device_name: []
    )

    class _FakeStarDist2D:
        def __init__(self, *args, **kwargs):
            pass

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    fake_stardist = types.ModuleType("stardist")
    fake_models = types.ModuleType("stardist.models")
    fake_models.StarDist2D = _FakeStarDist2D

    return {
        "tensorflow": fake_tf,
        "stardist": fake_stardist,
        "stardist.models": fake_models,
    }


def _fake_yolo_modules():
    fake_ultralytics = types.ModuleType("ultralytics")
    fake_ultralytics.YOLO = object

    fake_sahi = types.ModuleType("sahi")
    fake_auto_model = types.ModuleType("sahi.auto_model")
    fake_auto_model.AutoDetectionModel = types.SimpleNamespace(
        from_pretrained=lambda **kwargs: object()
    )
    fake_predict = types.ModuleType("sahi.predict")
    fake_predict.get_sliced_prediction = None
    fake_utils_pkg = types.ModuleType("sahi.utils")
    fake_cv = types.ModuleType("sahi.utils.cv")
    fake_cv.read_image = None

    return {
        "ultralytics": fake_ultralytics,
        "sahi": fake_sahi,
        "sahi.auto_model": fake_auto_model,
        "sahi.predict": fake_predict,
        "sahi.utils": fake_utils_pkg,
        "sahi.utils.cv": fake_cv,
    }


def test_plot_mask_none_returns_empty_mask_instead_of_crashing():
    mask, morphology = model_utils.plot_mask(None, image_size=(16, 16))

    assert mask.shape == (16, 16)
    assert mask.dtype == bool
    assert not mask.any()
    assert morphology == {"diameter": 0.0, "area": 0.0, "volume": 0.0}


def test_safe_image_write_reports_false_for_unsupported_extension(tmp_path):
    path = tmp_path / "image.unsupported"

    assert model_utils.safe_image_write(
        np.zeros((5, 5, 3), dtype=np.uint8),
        str(path),
    ) is False
    assert not path.exists()


def test_safegray2rgb_drops_alpha_channel_for_model_input():
    rgba = np.zeros((32, 31, 4), dtype=np.uint8)
    rgba[:, :, 0] = 10
    rgba[:, :, 3] = 255

    rgb = model_utils.safegray2rgb(rgba)

    assert rgb.shape == (32, 31, 3)
    assert rgb[:, :, 0].max() == 10


def test_prediction_alignment_restores_original_image_size(tmp_path):
    from UI.prediction_rendering import plot_predictions_with_alignment

    output_path = tmp_path / "detections.png"
    original = np.zeros((509, 512, 3), dtype=np.uint8)
    inference = np.zeros((512, 512, 3), dtype=np.uint8)
    masks = [
        np.array(
            [[10, 10], [100, 10], [100, 100], [10, 100]],
            dtype=np.float32,
        )
    ]

    rendered = plot_predictions_with_alignment(
        original,
        inference,
        masks,
        filename=str(output_path),
        mask_coordinate_space="inference",
    )

    assert rendered.shape == original.shape
    assert output_path.exists()


def test_countnuclei_blank_channel_is_empty():
    assert NucleiCounter().countNuclei(np.zeros((32, 32), dtype=np.uint8)) == 0


def test_groupnuclei_result_is_json_serializable_python_int():
    points = pd.DataFrame(
        {
            "x": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "y": [0, 1, 2, 0, 1, 2, 0, 1, 2],
        }
    )

    nuclei_count = NucleiCounter(eps=2, min_samples=5).groupNuclei(points)

    assert type(nuclei_count) is int
    assert json.dumps({"Nuclei": nuclei_count}) == '{"Nuclei": 1}'


def test_calculate_lsm_accepts_high_valid_channel_index(tmp_path, monkeypatch):
    image_path = tmp_path / "four_channel.lsm"
    image = np.zeros((4, 96, 320), dtype=np.uint8)
    image[2, :, :] = 60
    image[3, :, :] = 200
    tiffile.imwrite(
        str(image_path),
        image,
        metadata={"axes": "CYX"},
        photometric="minisblack",
    )
    monkeypatch.setattr(
        model_utils,
        "IMAGE_FILE_NAME_TMP",
        str(tmp_path / "cell_tmp_img.png"),
    )
    cell_counter = _RecordingCellCounter()
    nuclei_counter = _RecordingNucleiCounter()

    result = model_utils.calculate_lsm(
        cell_counter,
        nuclei_counter,
        str(image_path),
        cell_channel=3,
        nuclei_channel=2,
    )

    assert result["Cells"].shape[0] == 1
    assert cell_counter.image.shape == (96, 320, 3)
    assert cell_counter.image.max() == 200
    assert nuclei_counter.channel.shape == (96, 320)
    assert nuclei_counter.channel.max() == 60


def test_instanseg_pads_narrow_eval_medium_image_window(request, monkeypatch):
    instanseg_module = _import_with_fakes(
        request,
        monkeypatch,
        "model.InstanSegSegmenter",
        _fake_instanseg_modules(),
    )
    segmenter = instanseg_module.InstansegSegmenter.__new__(
        instanseg_module.InstansegSegmenter
    )

    image = np.zeros((512, 85, 3), dtype=np.uint8)
    padded = segmenter._ensure_eval_window_size(
        image,
        method_name="eval_medium_image",
        tile_size=512,
    )

    assert padded.shape[:2] == (512, 512)


def test_instanseg_preprocess_keeps_rgba_input_to_three_channels():
    rgba = np.zeros((32, 31, 4), dtype=np.uint8)

    processed = model_utils.process_loaded_image(
        rgba,
        [{"gray2rgb": ""}, {"resize": "512:512"}],
    )

    assert processed.shape[2] == 3


def test_stardist_skips_instance_when_no_usable_contour_exists(
    request,
    monkeypatch,
    tmp_path,
):
    stardist_module = _import_with_fakes(
        request,
        monkeypatch,
        "model.StardistSegmenter",
        _fake_stardist_modules(),
    )
    monkeypatch.setattr(
        stardist_module,
        "IMAGE_FILE_NAME_INSTANCES",
        str(tmp_path / "instances.png"),
    )
    segmenter = stardist_module.StardistSegmenter.__new__(
        stardist_module.StardistSegmenter
    )
    instances = np.zeros((8, 8), dtype=np.int32)
    instances[3, 3] = 1

    result = segmenter.stardist_results_to_pandas(
        instances,
        scores=np.array([0.9]),
        original_shape=instances.shape,
        inference_shape=instances.shape,
    )

    assert list(result.columns) == [
        "id_label",
        "box",
        "mask",
        "confidence",
        "diameter",
        "area",
        "volume",
    ]
    assert result.empty


def test_yolo_x10_uses_sahi_nms_postprocess(request, monkeypatch, tmp_path):
    yolo_module = _import_with_fakes(
        request,
        monkeypatch,
        "model.YOLOSegmenter",
        _fake_yolo_modules(),
    )

    calls = []
    source_image = np.zeros((384, 34, 3), dtype=np.uint8)

    class _FakeSlicedPrediction:
        def to_coco_predictions(self):
            return [
                {
                    "bbox": [0, 0, 10, 10],
                    "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]],
                    "score": 0.9,
                }
            ]

    def fake_get_sliced_prediction(*args, **kwargs):
        postprocess_type = kwargs.get("postprocess_type")
        calls.append(postprocess_type)
        return _FakeSlicedPrediction()

    monkeypatch.setattr(
        yolo_module,
        "get_sliced_prediction",
        fake_get_sliced_prediction,
    )
    monkeypatch.setattr(
        yolo_module,
        "read_image",
        lambda path: source_image,
    )
    segmenter = yolo_module.YoloSegmenter.__new__(yolo_module.YoloSegmenter)
    segmenter.object_size = {
        "color_map": "tab20",
        "signal": lambda *args, **kwargs: None,
    }
    segmenter.detections = None
    segmenter.original_image = None
    segmenter.model_x10 = object()

    output_path = tmp_path / "detections.png"
    result = segmenter.count_x10("narrow.tif")

    assert calls == ["NMS"]
    assert result.shape[0] == 1
    assert np.array_equal(result.original_image, source_image)
    assert np.array_equal(result.inference_image, source_image)
    assert not output_path.exists()
    assert getattr(segmenter, "inference_image", None) is None
