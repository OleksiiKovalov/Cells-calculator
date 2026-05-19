from .config import *
from .generation import *
from .signatures import *
from .subprocess_runner import run_case_subprocess


def write_minimizer_image(path: Path, image: np.ndarray, extension: str) -> tuple[Path, dict[str, Any]]:
    image = normalize_loaded_array(np.asarray(image))
    extension = canonical_minimizer_extension(extension)

    if extension == "lsm":
        if image.ndim == 2:
            channels_last = np.dstack((image, np.zeros_like(image)))
        elif image.ndim == 3 and image.shape[2] == 1:
            channels_last = np.repeat(image, 2, axis=2)
        elif image.ndim == 3:
            channels_last = image[:, :, :min(4, image.shape[2])]
        else:
            channels_last = image.reshape((*image.shape[-2:], 1))
            channels_last = np.repeat(channels_last, 2, axis=2)
        image_to_write = np.transpose(channels_last, (2, 0, 1))
        image_path = path / "input.lsm"
        tifffile.imwrite(str(image_path), image_to_write, metadata={"axes": "CYX"})
        return image_path, {
            "extension": "lsm",
            "shape": list(image_to_write.shape),
            "cell_channel": 0,
            "nuclei_channel": 1 if image_to_write.shape[0] > 1 else 0,
        }

    image_path = path / f"input.{extension}"
    if extension in {"png", "jpg", "bmp"}:
        writable = cv2_writable_image(image, extension)
        if not cv2.imwrite(str(image_path), writable):
            raise RuntimeError(f"Could not write minimized image: {image_path}")
        image = writable
    elif extension == "tif":
        tifffile.imwrite(str(image_path), image)
    else:
        extension = "png"
        image_path = path / "input.png"
        writable = cv2_writable_image(image, extension)
        if not cv2.imwrite(str(image_path), writable):
            raise RuntimeError(f"Could not write minimized image: {image_path}")
        image = writable

    channels = non_lsm_channel_count(image)
    return image_path, {
        "extension": extension,
        "shape": list(image.shape),
        "cell_channel": 0,
        "nuclei_channel": 0 if channels == 1 else min(1, channels - 1),
    }


def image_minimization_candidates(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    image = normalize_loaded_array(np.asarray(image))
    candidates: list[tuple[str, np.ndarray]] = []
    if image.ndim == 3 and image.shape[2] > 1:
        candidates.append(("first_channel", image[:, :, 0].copy()))

    height, width = image.shape[:2]
    if height > 1 or width > 1:
        target_h = max(1, height // 2)
        target_w = max(1, width // 2)
        candidates.append(
            (
                f"resize_{target_w}x{target_h}",
                cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA),
            )
        )

        crop_h = max(1, height // 2)
        crop_w = max(1, width // 2)
        y = max(0, (height - crop_h) // 2)
        x = max(0, (width - crop_w) // 2)
        if image.ndim == 2:
            candidates.append(("center_crop", image[y:y + crop_h, x:x + crop_w].copy()))
        else:
            candidates.append(("center_crop", image[y:y + crop_h, x:x + crop_w, :].copy()))

    candidates.append(("zeros", np.zeros_like(image)))
    if image.size:
        thresholded = np.where(image > int(np.median(image)), 255, 0).astype(np.uint8)
        candidates.append(("threshold", thresholded))
    return candidates


def case_minimization_candidates(case: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    candidates: list[tuple[str, dict[str, Any]]] = []

    simple_case = copy.deepcopy(case)
    simple_case["object_size"].update(
        {
            "min_size": 0.0,
            "max_size": 1.0,
            "line_width": 1.0,
            "alpha": 0.75,
            "color_map": "tab20",
            "size_metric": "area",
            "um_per_px": None,
        }
    )
    candidates.append(("simple_object_size", simple_case))

    workflow = case.get("workflow", {})
    if workflow.get("mode") == "stateful":
        steps = list(workflow.get("steps", []))
        if len(steps) > 1:
            shorter = copy.deepcopy(case)
            shorter["workflow"] = {"mode": "stateful", "steps": steps[: max(1, len(steps) // 2)]}
            candidates.append(("shorter_workflow", shorter))
        single_step = copy.deepcopy(case)
        single_step["workflow"] = {"mode": "stateful", "steps": steps[:1]}
        candidates.append(("one_step_workflow", single_step))

    return candidates


def materialize_minimizer_case(
    base_case: dict[str, Any],
    candidate_dir: Path,
    image: np.ndarray,
    case_variant: dict[str, Any] | None,
    extension_override: str | None = None,
) -> dict[str, Any]:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    case = copy.deepcopy(case_variant or base_case)
    original_extension = str(
        extension_override or base_case.get("image", {}).get("extension", "png")
    )
    image_path, image_meta = write_minimizer_image(candidate_dir, image, original_extension)
    case["image_path"] = str(image_path.resolve())
    case["image"].update(image_meta)
    case["image"]["profile"] = f"{case['image'].get('profile', 'unknown')}:minimized"
    case_path = candidate_dir / "case.json"
    case_path.write_text(json.dumps(case, indent=2), encoding="utf-8")
    return case


def candidate_preserves_signature(
    candidate_case: dict[str, Any],
    candidate_path: Path,
    expected_hash: str,
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
) -> tuple[bool, dict[str, Any], str, str, str]:
    try:
        completed = run_case_subprocess(
            candidate_path,
            timeout,
            sanitizers,
            max_rss_mb,
            max_tracemalloc_mb,
            max_handle_growth,
            max_allocated_block_growth,
            oracle_level,
            determinism_check,
            python_coverage,
            mock_mode,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        reason = f"exit code {completed.returncode}"
        if completed.returncode == 0:
            return False, {}, stdout, stderr, reason
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        reason = f"timeout after {timeout} seconds"

    signature = failure_signature(candidate_case, stdout, stderr, reason)
    return signature.get("hash") == expected_hash, signature, stdout, stderr, reason


def minimize_failure(
    failure_dir: Path,
    expected_signature: dict[str, Any],
    args: argparse.Namespace,
) -> Path | None:
    case_path = failure_dir / "case.json"
    if not case_path.exists():
        return None

    base_case = json.loads(case_path.read_text(encoding="utf-8"))
    image = read_corpus_image(Path(base_case["image_path"]))
    if image is None:
        return None

    timeout = args.minimize_timeout or args.timeout
    expected_hash = str(expected_signature["hash"])
    work_dir = failure_dir / "minimize_work"
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    best_case = copy.deepcopy(base_case)
    best_image = image
    accepted: list[dict[str, Any]] = []
    attempt = 0

    while len(accepted) < max(0, args.minimize_steps):
        progress = False
        candidates: list[tuple[str, np.ndarray, dict[str, Any] | None, str | None]] = [
            (name, candidate_image, None, None)
            for name, candidate_image in image_minimization_candidates(best_image)
        ]
        candidates.extend(
            (name, best_image, candidate_case, None)
            for name, candidate_case in case_minimization_candidates(best_case)
        )
        if canonical_minimizer_extension(
            str(best_case.get("image", {}).get("extension", "png"))
        ) != "png":
            candidates.append(("format_png", best_image, None, "png"))

        for label, candidate_image, candidate_case_variant, extension_override in candidates:
            if (
                candidate_case_variant is None
                and extension_override is None
                and np.array_equal(candidate_image, best_image)
            ):
                continue
            if (
                candidate_case_variant is not None
                and extension_override is None
                and json.dumps(
                    candidate_case_variant,
                    sort_keys=True,
                    default=str,
                ) == json.dumps(best_case, sort_keys=True, default=str)
            ):
                continue

            attempt += 1
            candidate_dir = work_dir / f"attempt_{attempt:03d}_{label}"
            candidate_case = materialize_minimizer_case(
                best_case,
                candidate_dir,
                candidate_image,
                candidate_case_variant,
                extension_override,
            )
            candidate_path = candidate_dir / "case.json"
            preserved, signature, stdout, stderr, reason = candidate_preserves_signature(
                candidate_case,
                candidate_path,
                expected_hash,
                timeout,
                args.sanitizers,
                args.max_rss_mb,
                args.max_tracemalloc_mb,
                args.max_handle_growth,
                args.max_allocated_block_growth,
                args.oracle_level,
                args.determinism_check,
                args.python_coverage,
                args.mock_mode,
            )
            (candidate_dir / "stdout.txt").write_text(stdout or "", encoding="utf-8")
            (candidate_dir / "stderr.txt").write_text(stderr or "", encoding="utf-8")
            (candidate_dir / "reason.txt").write_text(reason, encoding="utf-8")
            if signature:
                (candidate_dir / "signature.json").write_text(
                    json.dumps(signature, indent=2, sort_keys=True),
                    encoding="utf-8",
                )

            if preserved:
                best_case = candidate_case
                best_image = candidate_image
                accepted.append(
                    {
                        "label": label,
                        "shape": list(np.asarray(best_image).shape),
                        "case_dir": str(candidate_dir),
                    }
                )
                progress = True
                break

        if not progress:
            break

    summary = {
        "expected_hash": expected_hash,
        "accepted": accepted,
        "attempts": attempt,
    }
    (failure_dir / "minimize_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if not accepted:
        shutil.rmtree(work_dir, ignore_errors=True)
        return None

    minimized_dir = failure_dir / "minimized"
    if minimized_dir.exists():
        shutil.rmtree(minimized_dir, ignore_errors=True)
    last_attempt_dir = Path(accepted[-1]["case_dir"])
    shutil.copytree(last_attempt_dir, minimized_dir)
    update_saved_case_paths(minimized_dir)
    shutil.rmtree(work_dir, ignore_errors=True)
    return minimized_dir

