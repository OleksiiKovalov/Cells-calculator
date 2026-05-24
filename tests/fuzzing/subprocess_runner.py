from .config import *

def child_environment(
    sanitizers: str,
    oracle_level: str,
    determinism_check: str,
    python_coverage: str,
    mock_mode: str = "none",
) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
    env["CELLS_FUZZ_SANITIZERS"] = sanitizers
    env["CELLS_FUZZ_ORACLE_LEVEL"] = oracle_level
    env["CELLS_FUZZ_DETERMINISM_CHECK"] = determinism_check
    env["CELLS_FUZZ_PYTHON_COVERAGE"] = python_coverage
    env["CELLS_FUZZ_MOCK_MODE"] = mock_mode
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
    max_tracemalloc_mb: float,
    max_handle_growth: int,
    max_allocated_block_growth: int,
    oracle_level: str,
    determinism_check: str,
    python_coverage: str,
    mock_mode: str = "none",
) -> subprocess.CompletedProcess:
    command = [
        sys.executable,
        "-m",
        "tests.fuzzing.runtime_fuzz",
        "--run-case",
        str(case_path),
        "--max-rss-mb",
        str(max_rss_mb),
        "--max-tracemalloc-mb",
        str(max_tracemalloc_mb),
        "--max-handle-growth",
        str(max_handle_growth),
        "--max-allocated-block-growth",
        str(max_allocated_block_growth),
    ]
    return subprocess.run(
        command,
        cwd=str(ROOT),
        env=child_environment(
            sanitizers,
            oracle_level,
            determinism_check,
            python_coverage,
            mock_mode,
        ),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
    )
