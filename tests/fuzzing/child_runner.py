from .config import *
from .instrumentation import *
from .oracles import *
from .signatures import result_shape_bucket

def run_case_in_child(
    case_path: Path,
    max_rss_mb: float = 0,
    max_tracemalloc_mb: float = 0,
    max_handle_growth: int = 0,
    max_allocated_block_growth: int = 0,
) -> int:
    python_coverage_state: dict[str, Any] = {"enabled": False}
    try:
        configure_child_sanitizers()
        memory_stop, memory_state = start_memory_monitor(max_rss_mb)
        case = json.loads(case_path.read_text(encoding="utf-8"))
        python_coverage_state = start_python_coverage(
            current_python_coverage(),
            keep_runtime_helpers=True,
        )
        register_known_models()
        oracle_level = current_oracle_level()
        determinism_check = current_determinism_check()

        try:
            from model.utils import count_detected_objects as app_count_detected_objects
        except ImportError:
            app_count_detected_objects = count_result_cells

        model, object_size = instantiate_case_model(case)
        result = calculate_case(model, case)
        assert_result_shape(case, result)
        validate_model_result(
            case,
            result,
            oracle_level=oracle_level,
            image_size=filter_image_size(case, model),
        )
        reference_result = result

        sanitizer_probe = start_sanitizer_probe(
            max_tracemalloc_mb,
            max_handle_growth,
            max_allocated_block_growth,
        )
        if sanitizer_probe.get("enabled"):
            result = calculate_case(model, case)
            assert_result_shape(case, result)
            validate_model_result(
                case,
                result,
                oracle_level=oracle_level,
                image_size=filter_image_size(case, model),
            )
        result = run_stateful_workflow(case, case_path.parent, model, result, oracle_level)
        sanitizer_metrics = stop_sanitizer_probe(
            sanitizer_probe,
            max_tracemalloc_mb,
            max_handle_growth,
            max_allocated_block_growth,
        )
        run_determinism_check(case, reference_result, determinism_check, oracle_level)
        peak_rss_mb = stop_memory_monitor(memory_stop, memory_state, max_rss_mb)

        child_result = {
            "case_id": case["case_id"],
            "model": case["model_name"],
            "profile": case["image"]["profile"],
            "scale": object_size["scale"],
            "cells": app_count_detected_objects(result["Cells"]),
            "nuclei": result["Nuclei"],
            "alive_percent": result["%"],
            "result_shape": result_shape_bucket(result.get("Cells")),
            "peak_rss_mb": round(peak_rss_mb, 1),
            "oracle_level": oracle_level,
            "determinism_check": determinism_check,
        }
        python_coverage_metrics = stop_python_coverage(python_coverage_state)
        child_result.update(sanitizer_metrics)
        child_result.update(python_coverage_metrics)
        print(json.dumps(child_result, ensure_ascii=True))
        return 0
    except Exception:
        stop_python_coverage(python_coverage_state)
        traceback.print_exc()
        return 1
