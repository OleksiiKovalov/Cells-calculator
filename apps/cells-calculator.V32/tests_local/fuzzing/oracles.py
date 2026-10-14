from .config import *
from .generation import *
from .instrumentation import *

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
        if oracle_level == "paranoid":
            raise AssertionError(f"Degenerate mask at row {row_index}")
        return

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


def case_mock_mode(case: dict[str, Any]) -> str:
    mode = str(case.get("mock_mode") or current_mock_mode())
    return mode if mode in MOCK_MODES else "none"


def make_mock_payload(rng: random.Random, mode: str) -> dict[str, Any]:
    if mode == "none":
        return {}
    variants = VALID_MOCK_VARIANTS if mode == "model" else VALID_MOCK_VARIANTS + FAULT_INJECTION_VARIANTS
    return {
        "mode": mode,
        "variant": rng.choice(variants),
        "mask_space": rng.choice(["normalized", "pixel"]),
        "count_hint": rng.choice([0, 1, 2, 5, 12, 25]),
    }


def polygon_mask_from_box(
    x: float,
    y: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
    mask_space: str,
) -> np.ndarray:
    points = np.array(
        [
            [x, y],
            [x + width, y],
            [x + width, y + height],
            [x, y + height],
        ],
        dtype=np.float32,
    )
    if mask_space == "normalized":
        points[:, 0] /= max(1, image_width)
        points[:, 1] /= max(1, image_height)
    return points


def mock_components_for_case(
    case: dict[str, Any],
    image: np.ndarray,
    variant: str,
) -> list[tuple[int, int, int, int, int]]:
    height, width = image.shape[:2]
    if variant == "empty":
        return []
    components = detect_cell_like_components(image)
    if components and variant not in {"dense", "border"}:
        return components[:25]

    recipe = case.get("image", {}).get("recipe", {})
    count_hint = int(case.get("mock", {}).get("count_hint") or recipe.get("cell_count") or 5)
    if variant == "single":
        count = 1
    elif variant == "dense":
        count = max(12, min(60, count_hint * 3))
    elif variant == "border":
        count = max(4, min(16, count_hint))
    else:
        count = max(1, min(30, count_hint))

    radius = max(1, min(width, height) // max(6, int(np.sqrt(max(1, count))) * 4))
    components = []
    for index in range(count):
        if variant == "border":
            side = index % 4
            if side == 0:
                x, y = 0, int((index + 1) * height / (count + 1))
            elif side == 1:
                x, y = max(0, width - radius * 2), int((index + 1) * height / (count + 1))
            elif side == 2:
                x, y = int((index + 1) * width / (count + 1)), 0
            else:
                x, y = int((index + 1) * width / (count + 1)), max(0, height - radius * 2)
        else:
            columns = max(1, int(np.ceil(np.sqrt(count))))
            col = index % columns
            row = index // columns
            x = int((col + 0.35) * width / columns)
            y = int((row + 0.35) * height / max(1, int(np.ceil(count / columns))))
        box_w = max(1, min(width, radius * 2))
        box_h = max(1, min(height, radius * 2))
        components.append(
            (
                int(np.clip(x, 0, max(0, width - 1))),
                int(np.clip(y, 0, max(0, height - 1))),
                box_w,
                box_h,
                box_w * box_h,
            )
        )
    return components


def build_mock_result(
    case: dict[str, Any],
    image: np.ndarray,
    variant: str,
) -> dict[str, Any]:
    import pandas as pd

    image = ui_writable_image(image)
    height, width = image.shape[:2]
    mock = case.get("mock", {})
    mask_space = str(mock.get("mask_space", "normalized"))
    if variant == "pixel_masks":
        mask_space = "pixel"
    components = mock_components_for_case(case, image, variant)

    rows: dict[str, list[Any]] = {
        "id_label": [],
        "class_id": [],
        "box": [],
        "mask": [],
        "confidence": [],
        "diameter": [],
        "area": [],
        "volume": [],
        "scale": [],
    }
    image_area = max(1, width * height)
    for index, (x, y, box_w, box_h, area_px) in enumerate(components):
        x = min(max(0, x), max(0, width - 1))
        y = min(max(0, y), max(0, height - 1))
        box_w = max(1, min(box_w, max(1, width - x)))
        box_h = max(1, min(box_h, max(1, height - y)))
        area = max(0.0, min(1.0, float(area_px) / image_area))
        diameter = max(0.0, min(1.0, (max(box_w, box_h) / max(1.0, np.sqrt(image_area)))))
        rows["id_label"].append(index)
        rows["class_id"].append(0)
        rows["box"].append(np.array([x, y, box_w, box_h], dtype=np.float32))
        rows["mask"].append(
            polygon_mask_from_box(x, y, box_w, box_h, width, height, mask_space)
        )
        rows["confidence"].append(round(0.55 + 0.4 * ((index % 7) / 6), 4))
        rows["diameter"].append(diameter)
        rows["area"].append(area)
        rows["volume"].append(area * diameter)
        rows["scale"].append(1.0)

    dataframe = pd.DataFrame(rows)
    dataframe.attrs["image_size"] = (width, height)

    if variant == "missing_columns" and not dataframe.empty:
        dataframe = dataframe.drop(columns=["mask", "area"])
    elif variant == "nonfinite_confidence" and "confidence" in dataframe:
        if dataframe.empty:
            dataframe.loc[0, ["id_label", "box", "mask", "confidence", "diameter", "area", "volume", "scale"]] = [
                0,
                np.array([0, 0, 1, 1], dtype=np.float32),
                np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32),
                np.inf,
                0.1,
                0.01,
                0.001,
                1.0,
            ]
        else:
            dataframe.loc[dataframe.index[0], "confidence"] = np.inf
    elif variant == "negative_area" and "area" in dataframe:
        if dataframe.empty:
            dataframe.loc[0, ["id_label", "box", "mask", "confidence", "diameter", "area", "volume", "scale"]] = [
                0,
                np.array([0, 0, 1, 1], dtype=np.float32),
                np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32),
                0.9,
                0.1,
                -0.1,
                0.001,
                1.0,
            ]
        else:
            dataframe.loc[dataframe.index[0], "area"] = -0.1
    elif variant == "far_box" and "box" in dataframe:
        if dataframe.empty:
            dataframe.loc[0, ["id_label", "box", "mask", "confidence", "diameter", "area", "volume", "scale"]] = [
                0,
                np.array([width * 50, height * 50, width * 8, height * 8], dtype=np.float32),
                np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32),
                0.9,
                0.1,
                0.01,
                0.001,
                1.0,
            ]
        else:
            dataframe.loc[dataframe.index[0], "box"] = np.array(
                [width * 50, height * 50, width * 8, height * 8],
                dtype=np.float32,
            )
    elif variant == "degenerate_mask" and "mask" in dataframe:
        if dataframe.empty:
            dataframe.loc[0, ["id_label", "box", "mask", "confidence", "diameter", "area", "volume", "scale"]] = [
                0,
                np.array([0, 0, 1, 1], dtype=np.float32),
                np.array([[0.1, 0.1], [0.2, 0.2]], dtype=np.float32),
                0.9,
                0.1,
                0.01,
                0.001,
                1.0,
            ]
        else:
            dataframe.loc[dataframe.index[0], "mask"] = np.array(
                [[0.1, 0.1], [0.2, 0.2]],
                dtype=np.float32,
            )

    inference_image = image
    if variant == "inference_scaled" and min(height, width) > 0:
        target_w = max(8, min(512, width * 2))
        target_h = max(8, min(512, height * 2))
        inference_image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    nuclei = max(0, int(round(len(dataframe) * 0.35))) if len(dataframe) else 0
    return {
        "Cells": dataframe,
        "Nuclei": nuclei,
        "%": -100 if len(dataframe) == 0 else round((1 - nuclei / max(1, len(dataframe))) * 100, 3),
        "original_image": image,
        "inference_image": inference_image,
    }


class FuzzMockCellCounter:
    def __init__(self, image_path: str, object_size: dict[str, Any]) -> None:
        self.original_image_path = image_path
        self.object_size = object_size
        self.original_image = None
        self.inference_image = None
        self.inference = None
        self.detections = None


class FuzzMockModel:
    def __init__(
        self,
        case: dict[str, Any],
        object_size: dict[str, Any],
        variant: str,
    ) -> None:
        self.case = case
        self.variant = variant
        self.cell_counter = FuzzMockCellCounter(case["image_path"], object_size)

    def calculate(self, img_path: str, cell_channel: int = 0, nuclei_channel: int = 1) -> dict[str, Any]:
        image = read_corpus_image(Path(img_path))
        if image is None:
            image = np.zeros((1, 1), dtype=np.uint8)
        image = ui_writable_image(image)
        self.cell_counter.original_image = image
        result = build_mock_result(self.case, image, self.variant)
        self.cell_counter.inference_image = result.get("inference_image")
        self.cell_counter.inference = result.get("inference_image")
        self.cell_counter.detections = result.get("Cells")
        return result


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
    model_data = case["model_data"]
    object_size = dict(case["object_size"])
    object_size["signal"] = lambda *args, **kwargs: None
    mock_mode = case_mock_mode(case)
    if mock_mode != "none":
        mock = case.get("mock") if isinstance(case.get("mock"), dict) else {}
        variant = str(mock.get("variant") or make_mock_payload(random.Random(case.get("case_id", 0)), mock_mode).get("variant"))
        return FuzzMockModel(case, object_size, variant), object_size

    from model.Model import Model
    from UI.app_globals import get_registered_model

    model_kwargs = {
        "path": model_data["path"],
        "object_size": object_size,
        "model_type": model_data["model_type"],
        "model_data": model_data,
        "model_name": case["model_name"],
    }
    model_parameters = inspect.signature(Model).parameters
    if "logger" in model_parameters:
        model_kwargs["logger"] = logging.getLogger("runtime_fuzz")
    if "get_registered_model" in model_parameters:
        model_kwargs["get_registered_model"] = get_registered_model
    model = Model(**model_kwargs)
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
    try:
        from model.utils import filter_detections, filter_segmentation_detections
    except ImportError:
        from model.utils import filter_detections

        filter_segmentation_detections = None

    min_size = float(step.get("min_size", 0.0))
    max_size = float(step.get("max_size", 1.0))

    if cells is None or not hasattr(cells, "columns"):
        return cells
    if cells.empty:
        return cells.copy()
    if {"area", "diameter", "volume"} & set(cells.columns):
        if filter_segmentation_detections is None:
            metric = str(step.get("size_metric", "area"))
            metric = metric if metric in cells.columns else "area"
            lower = min(min_size, max_size)
            upper = max(min_size, max_size)
            try:
                values = np.asarray(cells[metric].tolist(), dtype=np.float64)
            except (TypeError, ValueError):
                return cells.iloc[0:0].copy()
            keep = np.isfinite(values) & (values >= lower) & (values <= upper)
            return cells.loc[keep].copy()
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


def ui_writable_image(image: Any) -> np.ndarray:
    image = normalize_loaded_array(np.asarray(image))
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)
    if image.ndim == 3 and image.shape[2] in (3, 4):
        return image.copy()
    return cv2_writable_image(image, "png")


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

    return ui_writable_image(image)


def workflow_image_channel_count(case: dict[str, Any]) -> int:
    image = case.get("image", {})
    shape = list(image.get("shape", []))
    if image.get("extension") == "lsm" and shape:
        return max(1, int(shape[0]))
    if len(shape) >= 3:
        return max(1, int(shape[2]))
    return 1


def workflow_channels_for_step(case: dict[str, Any], step: dict[str, Any]) -> tuple[int, int]:
    image = case.get("image", {})
    channel_count = workflow_image_channel_count(case)
    base_cell = int(image.get("cell_channel", 0))
    base_nuclei = int(image.get("nuclei_channel", 0))
    mode = str(step.get("channel_mode", "same"))
    if mode == "swap":
        return base_nuclei, base_cell
    if mode == "first":
        return 0, 0
    if mode == "last":
        last = max(0, channel_count - 1)
        return last, last
    if mode == "negative":
        return -1, -1
    if mode == "too_high":
        return channel_count + 3, channel_count + 5
    return base_cell, base_nuclei


def validate_workflow_image(path: Path, image: Any, label: str) -> None:
    if not path.exists():
        raise AssertionError(f"{label} was not written: {path}")
    array = np.asarray(image)
    if array.size and not np.isfinite(array.astype(np.float64)).all():
        raise AssertionError(f"{label} contains non-finite values")


def inference_image_for_workflow(result: dict[str, Any], model: Any) -> np.ndarray | None:
    try:
        from model.PredictionResult import get_prediction_images
    except ImportError:
        inference_image = result.get("inference_image") if isinstance(result, dict) else None
    else:
        _original_image, inference_image = get_prediction_images(result)
    if inference_image is None:
        inference_image = getattr(model.cell_counter, "inference_image", None)
    if inference_image is None:
        inference_image = getattr(model.cell_counter, "inference", None)
    if inference_image is None:
        return None
    return ui_writable_image(inference_image)


def load_prediction_rendering_helpers():
    try:
        from UI.prediction_rendering import (
            plot_predictions,
            plot_predictions_with_alignment,
            publish_inference_image,
            render_predictions,
        )
    except ImportError:
        from model import utils as legacy_utils

        def plot_predictions(
            image,
            pred_masks,
            filename=None,
            alpha=0.75,
            colormap="tab20",
            color_ids=None,
        ):
            _ = color_ids
            rendered = legacy_utils.plot_predictions(
                image,
                pred_masks,
                filename=filename,
                alpha=alpha,
                colormap=colormap,
            )
            return image if rendered is None else rendered

        def render_predictions(
            image,
            detections,
            filename=None,
            colormap="tab20",
            alpha=0.75,
        ):
            rendered = image.copy()
            if detections is not None and hasattr(detections, "columns") and "mask" in detections:
                return plot_predictions(
                    rendered,
                    detections["mask"].tolist(),
                    filename=filename,
                    alpha=alpha,
                    colormap=colormap,
                )
            if detections is not None and hasattr(detections, "iterrows"):
                for _index, row in detections.iterrows():
                    if "box" not in row:
                        continue
                    box = np.asarray(row["box"], dtype=np.float64).reshape(-1)
                    if box.size < 4 or not np.isfinite(box[:4]).all():
                        continue
                    scale = float(row["scale"]) if "scale" in row and np.isfinite(row["scale"]) else 1.0
                    x = round(float(box[0]) * scale)
                    y = round(float(box[1]) * scale)
                    x_plus_w = round(float(box[0] + box[2]) * scale)
                    y_plus_h = round(float(box[1] + box[3]) * scale)
                    legacy_utils.draw_bounding_box(
                        rendered,
                        int(row["class_id"]) if "class_id" in row else 0,
                        float(row["confidence"]) if "confidence" in row else 1.0,
                        x,
                        y,
                        x_plus_w,
                        y_plus_h,
                    )
            if filename is not None:
                legacy_utils.safe_image_write(rendered, filename)
            return rendered

        def plot_predictions_with_alignment(
            original_image,
            img_inference,
            pred_masks,
            filename=None,
            colormap="tab20",
            alpha=0.75,
            color_ids=None,
            mask_coordinate_space="auto",
        ):
            _ = img_inference, mask_coordinate_space
            return plot_predictions(
                original_image,
                pred_masks,
                filename=filename,
                colormap=colormap,
                alpha=alpha,
                color_ids=color_ids,
            )

        def publish_inference_image(
            model,
            result,
            filename=None,
            preserve_dtype=False,
        ):
            inference_image = inference_image_for_workflow(result, model)
            if inference_image is None:
                return None
            if filename is not None:
                legacy_utils.safe_image_write(
                    inference_image,
                    filename,
                    preserve_dtype=preserve_dtype,
                )
            return inference_image

    return (
        plot_predictions,
        plot_predictions_with_alignment,
        publish_inference_image,
        render_predictions,
    )


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

    (
        plot_predictions,
        plot_predictions_with_alignment,
        publish_inference_image,
        render_predictions,
    ) = load_prediction_rendering_helpers()

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
            validate_workflow_image(output_path, plotted, "Workflow mask plot")

        elif operation == "render_predictions":
            base_image = original_image_for_plot(case, model)
            output_path = case_dir / f"workflow_render_{step_index:02d}.png"
            rendered = render_predictions(
                base_image.copy(),
                cells,
                filename=str(output_path),
                colormap=str(step.get("color_map", "tab20")),
                alpha=float(step.get("alpha", 0.75)),
            )
            validate_workflow_image(output_path, rendered, "Workflow prediction render")

        elif operation == "aligned_plot":
            if cells is None or "mask" not in cells.columns:
                continue
            inference_image = inference_image_for_workflow(result, model)
            if inference_image is None:
                continue
            original_image = original_image_for_plot(case, model)
            output_path = case_dir / f"workflow_aligned_{step_index:02d}.png"
            color_ids = cells["id_label"].tolist() if "id_label" in cells.columns else None
            rendered = plot_predictions_with_alignment(
                original_image.copy(),
                inference_image.copy(),
                cells["mask"].tolist(),
                filename=str(output_path),
                colormap=str(step.get("color_map", "tab20")),
                alpha=float(step.get("alpha", 0.75)),
                color_ids=color_ids,
                mask_coordinate_space=str(step.get("mask_coordinate_space", "auto")),
            )
            validate_workflow_image(output_path, rendered, "Workflow aligned plot")

        elif operation == "publish_images":
            output_path = case_dir / f"workflow_inference_{step_index:02d}.png"
            published = publish_inference_image(
                model,
                result,
                filename=str(output_path),
                preserve_dtype=bool(step.get("preserve_dtype", False)),
            )
            if published is not None:
                validate_workflow_image(output_path, published, "Workflow inference image")

        elif operation == "channel_rerun":
            if step.get("reset_cache", False) and hasattr(model.cell_counter, "detections"):
                model.cell_counter.detections = None
            cell_channel, nuclei_channel = workflow_channels_for_step(case, step)
            result = model.calculate(
                img_path=case["image_path"],
                cell_channel=cell_channel,
                nuclei_channel=nuclei_channel,
            )
            assert_result_shape(case, result)
            validate_model_result(
                case,
                result,
                oracle_level=oracle_level,
                image_size=filter_image_size(case, model),
            )

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
