from .config import *
from .generation import image_meta_dimensions
from .oracles import count_result_cells

def normalize_signature_text(text: str) -> str:
    text = text.replace(str(ROOT), "<ROOT>")
    text = text.replace("\\", "/")
    text = re.sub(r'File "[^"]+", line \d+', 'File "<path>", line <n>', text)
    text = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", text)
    text = re.sub(r"\b\d+\.\d+\b", "N.N", text)
    text = re.sub(r"\b\d+\b", "N", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:500]


def extract_exception_line(output: str) -> str:
    exception_re = re.compile(
        r"^(?:[\w.]*Error|[\w.]*Exception|AssertionError|RuntimeError|"
        r"ValueError|TypeError|MemoryError|TimeoutError|cv2\.error)\b"
    )
    for line in reversed([line.strip() for line in output.splitlines() if line.strip()]):
        if exception_re.match(line):
            return line
    for line in reversed([line.strip() for line in output.splitlines() if line.strip()]):
        if "Traceback" not in line and not line.startswith("File "):
            return line
    return ""


def extract_app_frame(output: str) -> str:
    frame_re = re.compile(r'File "([^"]+)", line (\d+), in ([^\n]+)')
    frames = frame_re.findall(output)
    for path_text, _line, function in reversed(frames):
        try:
            path = Path(path_text).resolve()
            rel_path = path.relative_to(ROOT)
            return f"{rel_path.as_posix()}::{function.strip()}"
        except ValueError:
            continue
    if frames:
        path_text, _line, function = frames[-1]
        return f"{Path(path_text).name}::{function.strip()}"
    return ""


def failure_signature(
    case: dict[str, Any],
    stdout: str,
    stderr: str,
    reason: str,
) -> dict[str, Any]:
    output = "\n".join(part for part in (stdout or "", stderr or "") if part)
    exception = extract_exception_line(output) or reason
    exception_type = exception.split(":", 1)[0] if ":" in exception else exception.split(" ", 1)[0]
    kind = "timeout" if reason.startswith("timeout") else "crash"
    if reason.startswith("generation error"):
        kind = "generation"
    elif any(marker in output for marker in ("Handle/file-descriptor", "Allocated block", "Tracemalloc peak")):
        kind = "sanitizer"
    elif "Peak RSS" in output and "exceeded limit" in output:
        kind = "memory"
    elif exception_type == "AssertionError":
        kind = "oracle"

    signature_basis = {
        "kind": kind,
        "exception_type": normalize_signature_text(exception_type),
        "exception": normalize_signature_text(exception),
        "frame": normalize_signature_text(extract_app_frame(output)),
        "model_type": str(case.get("model_data", {}).get("model_type", "")),
    }
    signature_hash = hashlib.sha1(
        json.dumps(signature_basis, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

    return {
        **signature_basis,
        "hash": signature_hash,
        "reason": reason,
        "model_name": case.get("model_name", ""),
        "workflow": case.get("workflow", {}).get("mode", "single"),
    }


def load_seen_signatures(signature_log: Path) -> set[str]:
    seen: set[str] = set()
    if not signature_log.exists():
        return seen
    for line in signature_log.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        signature_hash = entry.get("hash") or entry.get("signature", {}).get("hash")
        if signature_hash:
            seen.add(str(signature_hash))
    return seen


def append_signature_log(
    signature_log: Path,
    signature: dict[str, Any],
    case_dir: Path,
    duplicate: bool,
) -> None:
    signature_log.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        **signature,
        "case_dir": str(case_dir),
        "duplicate": duplicate,
        "recorded_at": int(time.time()),
    }
    with signature_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=True, sort_keys=True) + "\n")


def extract_child_result(stdout: str) -> dict[str, Any]:
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "case_id" in value and "model" in value:
            return value
    return {}


def empty_coverage_stats() -> dict[str, Counter[str]]:
    return {
        "outcomes": Counter(),
        "failure_kinds": Counter(),
        "exception_types": Counter(),
        "oracle_failures": Counter(),
        "result_shapes": Counter(),
        "python_coverage_modes": Counter(),
        "python_coverage_files": Counter(),
        "python_coverage_lines": Counter(),
        "python_coverage_line_counts": Counter(),
        "mcdc_decisions": Counter(),
        "mcdc_outcomes": Counter(),
        "mcdc_satisfied_conditions": Counter(),
        "mcdc_signals": Counter(),
        "mcdc_signal_counts": Counter(),
        "mcdc_decision_counts": Counter(),
        "mcdc_condition_counts": Counter(),
        "mcdc_satisfied_condition_counts": Counter(),
        "mcdc_full_decision_counts": Counter(),
        "coverage_source_pools": Counter(),
        "models": Counter(),
        "model_types": Counter(),
        "mock_modes": Counter(),
        "mock_variants": Counter(),
        "profiles": Counter(),
        "image_recipes": Counter(),
        "grammar_mutations": Counter(),
        "grammar_productions": Counter(),
        "grammar_constraints": Counter(),
        "extensions": Counter(),
        "patterns": Counter(),
        "size_buckets": Counter(),
        "aspect_buckets": Counter(),
        "channels": Counter(),
        "workflow_modes": Counter(),
        "workflow_ops": Counter(),
        "cell_counts": Counter(),
    }


def dimension_bucket(width: int, height: int) -> str:
    longest = max(width, height)
    if longest <= 1:
        return "1"
    if longest <= 8:
        return "2-8"
    if longest <= 32:
        return "9-32"
    if longest <= 128:
        return "33-128"
    if longest <= 384:
        return "129-384"
    return "385+"


def aspect_bucket(width: int, height: int) -> str:
    width = max(1, width)
    height = max(1, height)
    ratio = max(width / height, height / width)
    if ratio >= 16:
        return "ultra-thin"
    if ratio >= 4:
        return "thin"
    if ratio >= 1.5:
        return "rect"
    return "square-ish"


def cell_count_bucket(value: Any) -> str:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return "unknown"
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 5:
        return "2-5"
    if count <= 20:
        return "6-20"
    return "21+"


def line_count_bucket(value: Any) -> str:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return "unknown"
    if count <= 0:
        return "0"
    if count <= 50:
        return "1-50"
    if count <= 200:
        return "51-200"
    if count <= 1000:
        return "201-1000"
    return "1001+"


def result_shape_bucket(cells: Any) -> str:
    if cells is None:
        return "none"
    if hasattr(cells, "columns"):
        return "dataframe"
    if isinstance(cells, (int, float, str, np.integer, np.floating)):
        return "scalar"
    return "malformed"


def classify_oracle_failure(output: str) -> str:
    text = (output or "").replace("\\", "/").lower()
    if not text:
        return "unknown"
    if "missing detection columns" in text:
        return "missing columns"
    if "expected result dict" in text or "unexpected result keys" in text:
        return "malformed result"
    if "non-finite" in text or "not finite" in text:
        return "non-finite"
    if "negative" in text:
        return "negative metrics"
    if "confidence values outside" in text:
        return "bad confidence"
    if "box" in text:
        return "bad box"
    if "mask" in text:
        return "bad mask"
    if "non-deterministic" in text:
        return "non-deterministic"
    if "filtered detections grew" in text or "subset of wide filter" in text:
        return "filter invariant"
    if "workflow plot" in text:
        return "plot invariant"
    if "handle/file-descriptor" in text or "allocated block" in text:
        return "sanitizer"
    if "peak rss" in text or "memory" in text:
        return "memory"
    if "generation error" in text or "could not write" in text:
        return "generation"
    return "other"


def classify_failure_kind(
    reason: str,
    output: str,
    signature: dict[str, Any] | None,
) -> str:
    if signature and signature.get("kind"):
        return str(signature["kind"])
    if reason.startswith("generation error"):
        return "generation"
    if reason.startswith("timeout"):
        return "timeout"
    if any(marker in output for marker in ("Handle/file-descriptor", "Allocated block", "Tracemalloc peak")):
        return "sanitizer"
    if "Peak RSS" in output:
        return "memory"
    if "AssertionError" in output:
        return "oracle"
    return "crash"


def classify_result_shape_from_failure(output: str) -> str:
    text = output.lower()
    if "expected result dict" in text or "unexpected result keys" in text:
        return "malformed"
    return "none"


def update_coverage_stats(
    stats: dict[str, Counter[str]],
    case: dict[str, Any],
    stdout: str,
    stderr: str,
    reason: str,
    passed: bool,
    signature: dict[str, Any] | None = None,
) -> None:
    image = case.get("image", {})
    workflow = case.get("workflow", {})
    child_result = extract_child_result(stdout)
    output = "\n".join(part for part in (stdout or "", stderr or "") if part)
    width, height, channels = image_meta_dimensions(image)

    stats["outcomes"]["pass" if passed else "fail"] += 1
    stats["models"][str(case.get("model_name", "unknown"))] += 1
    stats["model_types"][str(case.get("model_data", {}).get("model_type", "unknown"))] += 1
    stats["mock_modes"][str(case.get("mock_mode", "none"))] += 1
    mock = case.get("mock", {})
    if isinstance(mock, dict) and mock.get("variant"):
        stats["mock_variants"][str(mock.get("variant"))] += 1
    stats["profiles"][str(image.get("profile", "unknown"))] += 1
    recipe = image.get("recipe", {})
    if isinstance(recipe, dict):
        if recipe.get("scene"):
            stats["image_recipes"][str(recipe.get("scene"))] += 1
        for production in recipe.get("productions", []) or []:
            stats["grammar_productions"][str(production)] += 1
        for constraint in recipe.get("constraints", []) or []:
            stats["grammar_constraints"][str(constraint)] += 1
        for mutation in recipe.get("grammar_mutations", []) or []:
            stats["grammar_mutations"][str(mutation)] += 1
    stats["extensions"][str(image.get("extension", "unknown"))] += 1
    stats["size_buckets"][dimension_bucket(width, height)] += 1
    stats["aspect_buckets"][aspect_bucket(width, height)] += 1
    stats["channels"][str(channels if channels is not None else 1)] += 1
    stats["workflow_modes"][str(workflow.get("mode", "single"))] += 1
    guided = case.get("coverage_guided", {})
    if isinstance(guided, dict):
        stats["coverage_source_pools"][str(guided.get("source_pool", "unknown"))] += 1
    for pattern in image.get("patterns", []) or ["unknown"]:
        stats["patterns"][str(pattern)] += 1
    for step in workflow.get("steps", []):
        stats["workflow_ops"][str(step.get("operation", "unknown"))] += 1
    if child_result:
        stats["cell_counts"][cell_count_bucket(child_result.get("cells"))] += 1
        stats["result_shapes"][str(child_result.get("result_shape", "unknown"))] += 1
        python_coverage = child_result.get("python_coverage")
        if isinstance(python_coverage, dict):
            mode = str(python_coverage.get("mode", "model"))
            stats["python_coverage_modes"][mode] += 1
            line_count = python_coverage.get("line_count")
            if line_count is not None:
                stats["python_coverage_line_counts"][line_count_bucket(line_count)] += 1
            files = python_coverage.get("files", {})
            if isinstance(files, dict):
                for file_name, count in files.items():
                    stats["python_coverage_files"][str(file_name)] += int(count)
            lines = python_coverage.get("lines", [])
            if isinstance(lines, list):
                stats["python_coverage_lines"].update(str(line) for line in lines)
            signals = python_coverage.get("signals", [])
            if isinstance(signals, list):
                stats["mcdc_signals"].update(str(signal) for signal in signals)
            decisions = python_coverage.get("decisions", {})
            if isinstance(decisions, dict):
                for decision_id, decision in decisions.items():
                    stats["mcdc_decisions"][str(decision_id)] += 1
                    if isinstance(decision, dict):
                        for outcome in decision.get("outcomes", []):
                            stats["mcdc_outcomes"][f"{decision_id}:{outcome}"] += 1
                        for condition_index in decision.get("mcdc_conditions", []):
                            stats["mcdc_satisfied_conditions"][
                                f"{decision_id}:{condition_index}"
                            ] += 1
            signal_count = python_coverage.get("signal_count")
            if signal_count is not None:
                stats["mcdc_signal_counts"][line_count_bucket(signal_count)] += 1
            decision_count = python_coverage.get("decision_count")
            if decision_count is not None:
                stats["mcdc_decision_counts"][line_count_bucket(decision_count)] += 1
            condition_count = python_coverage.get("condition_count")
            if condition_count is not None:
                stats["mcdc_condition_counts"][line_count_bucket(condition_count)] += 1
            satisfied_count = python_coverage.get("satisfied_condition_count")
            if satisfied_count is not None:
                stats["mcdc_satisfied_condition_counts"][
                    line_count_bucket(satisfied_count)
                ] += 1
            full_decision_count = python_coverage.get("full_mcdc_decision_count")
            if full_decision_count is not None:
                stats["mcdc_full_decision_counts"][
                    line_count_bucket(full_decision_count)
                ] += 1
    elif passed:
        stats["result_shapes"]["unknown"] += 1
    else:
        stats["failure_kinds"][classify_failure_kind(reason, output, signature)] += 1
        exception_type = (
            signature.get("exception_type")
            if signature
            else extract_exception_line(output).split(":", 1)[0]
        )
        stats["exception_types"][str(exception_type or "unknown")] += 1
        stats["oracle_failures"][classify_oracle_failure(output)] += 1
        stats["result_shapes"][classify_result_shape_from_failure(output)] += 1


def serialize_coverage_stats(
    stats: dict[str, Counter[str]],
    include_python_lines: bool = False,
    include_mcdc_signals: bool = False,
) -> dict[str, dict[str, int]]:
    serialized: dict[str, dict[str, int]] = {}
    for name, counter in sorted(stats.items()):
        if name == "python_coverage_lines" and not include_python_lines:
            serialized["python_coverage_unique_lines"] = {"total": len(counter)}
            continue
        if name == "mcdc_signals" and not include_mcdc_signals:
            serialized["mcdc_unique_signals"] = {"total": len(counter)}
            continue
        serialized[name] = dict(counter.most_common())
    return serialized


def compact_child_result_for_log(child_result: dict[str, Any]) -> dict[str, Any]:
    compact = copy.deepcopy(child_result)
    python_coverage = compact.get("python_coverage")
    if isinstance(python_coverage, dict):
        python_coverage.pop("lines", None)
        python_coverage.pop("signals", None)
        python_coverage.pop("decisions", None)
    return compact


def child_python_coverage_lines(stdout: str) -> set[str]:
    child_result = extract_child_result(stdout)
    python_coverage = child_result.get("python_coverage")
    if not isinstance(python_coverage, dict):
        return set()
    lines = python_coverage.get("lines", [])
    if not isinstance(lines, list):
        return set()
    return {str(line) for line in lines}


def child_python_coverage_signals(stdout: str) -> set[str]:
    child_result = extract_child_result(stdout)
    python_coverage = child_result.get("python_coverage")
    if not isinstance(python_coverage, dict):
        return set()
    signals = python_coverage.get("signals")
    if isinstance(signals, list):
        return {str(signal) for signal in signals}
    lines = python_coverage.get("lines")
    if isinstance(lines, list):
        return {f"line:{line}" for line in lines}
    return set()


def load_existing_python_coverage_lines(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    lines = data.get("python_coverage_lines", {})
    if isinstance(lines, dict):
        return {str(line) for line in lines}
    return set()


def load_existing_mcdc_coverage_signals(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    signals = data.get("mcdc_signals", {})
    if isinstance(signals, dict):
        return {str(signal) for signal in signals}
    if isinstance(signals, list):
        return {str(signal) for signal in signals}
    return set()


def load_existing_python_coverage_signals(path: Path, mode: str) -> set[str]:
    if mode == "mcdc":
        return load_existing_mcdc_coverage_signals(path)
    return {f"line:{line}" for line in load_existing_python_coverage_lines(path)}


def input_images_in_case_dir(case_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in case_dir.glob("input.*")
        if path.is_file() and path.suffix.lower().lstrip(".") in VALID_GENERATED_EXTENSIONS
    )


def save_interesting_case(
    case_dir: Path,
    interesting_dir: Path,
    case: dict[str, Any],
    new_signals: set[str],
    child_result: dict[str, Any],
    copy_only: bool = False,
) -> tuple[Path, dict[str, Any], list[Path]]:
    interesting_dir.mkdir(parents=True, exist_ok=True)
    target = interesting_dir / case_dir.name
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)

    if copy_only:
        shutil.copytree(case_dir, target)
    else:
        shutil.move(str(case_dir), str(target))

    updated_case = update_saved_case_paths(target) or case
    metadata = {
        "case_id": updated_case.get("case_id"),
        "model": updated_case.get("model_name"),
        "model_type": updated_case.get("model_data", {}).get("model_type", ""),
        "new_signal_count": len(new_signals),
        "new_signals": sorted(new_signals),
        "new_line_count": len([signal for signal in new_signals if signal.startswith("line:")]),
        "new_lines": sorted(
            signal.removeprefix("line:")
            for signal in new_signals
            if signal.startswith("line:")
        ),
        "coverage": compact_child_result_for_log(child_result).get("python_coverage", {}),
        "recorded_at": int(time.time()),
    }
    (target / "new_coverage.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True, sort_keys=True),
        encoding="utf-8",
    )
    return target, updated_case, input_images_in_case_dir(target)


def append_pass_log(pass_log: Path, case: dict[str, Any], stdout: str) -> None:
    pass_log.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "case_id": case["case_id"],
        "model": case["model_name"],
        "model_type": case["model_data"].get("model_type", ""),
        "image": case["image"],
        "workflow": case.get("workflow", {}).get("mode", "single"),
        "child_result": compact_child_result_for_log(extract_child_result(stdout)),
    }
    with pass_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=True, sort_keys=True) + "\n")


def console_safe_text(text: str, stream: Any = None) -> str:
    if not text:
        return ""
    stream = stream or sys.stdout
    encoding = getattr(stream, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def update_saved_case_paths(target: Path) -> dict[str, Any] | None:
    case_path = target / "case.json"
    if not case_path.exists():
        return None
    case = json.loads(case_path.read_text(encoding="utf-8"))
    old_image_path = Path(case.get("image_path", ""))
    moved_image = target / old_image_path.name if old_image_path.name else None
    if moved_image is None or not moved_image.is_file():
        inputs = sorted(path for path in target.glob("input.*") if path.is_file())
        moved_image = inputs[0] if inputs else None
    if moved_image is not None and moved_image.is_file():
        case["image_path"] = str(moved_image.resolve())
        case_path.write_text(json.dumps(case, indent=2), encoding="utf-8")
    return case


def save_failure(
    case_dir: Path,
    target_root: Path,
    stdout: str,
    stderr: str,
    reason: str,
    signature: dict[str, Any],
) -> Path:
    target_root.mkdir(parents=True, exist_ok=True)
    target = target_root / case_dir.name
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    shutil.move(str(case_dir), str(target))
    update_saved_case_paths(target)
    (target / "stdout.txt").write_text(stdout or "", encoding="utf-8")
    (target / "stderr.txt").write_text(stderr or "", encoding="utf-8")
    (target / "reason.txt").write_text(reason, encoding="utf-8")
    (target / "signature.json").write_text(
        json.dumps(signature, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target

