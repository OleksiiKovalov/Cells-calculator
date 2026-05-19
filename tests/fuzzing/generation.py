from .config import *

def discover_corpus_images(seed_corpus: str) -> list[Path]:
    if not seed_corpus:
        return []

    roots = [Path(part.strip()) for part in seed_corpus.split(",") if part.strip()]
    images: list[Path] = []
    for root in roots:
        root = root if root.is_absolute() else ROOT / root
        if root.is_file():
            candidates = [root]
        elif root.is_dir():
            candidates = [p for p in root.rglob("*") if p.is_file()]
        else:
            continue

        for path in candidates:
            if path.suffix.lower().lstrip(".") in VALID_GENERATED_EXTENSIONS:
                images.append(path)

    return sorted(set(images))


def fuzz_case_spec_from_dict(data: dict[str, Any]) -> FuzzCaseSpec | None:
    try:
        return FuzzCaseSpec(
            model_name=str(data.get("model_name", "")),
            image_kind=str(data.get("image_kind", "generic")),
            width=max(1, int(data.get("width", 1))),
            height=max(1, int(data.get("height", 1))),
            channels=(
                None
                if data.get("channels") is None
                else max(1, min(4, int(data.get("channels"))))
            ),
            pattern=str(data.get("pattern", "noise")),
            extension=str(data.get("extension", "png")),
            object_size=copy.deepcopy(dict(data.get("object_size", {}))),
            workflow=copy.deepcopy(list(data.get("workflow", []))),
            image_recipe=copy.deepcopy(dict(data.get("image_recipe", {}))),
        )
    except (TypeError, ValueError):
        return None


def discover_case_specs(root: str | Path) -> list[FuzzCaseSpec]:
    root = Path(root)
    if not root.exists():
        return []
    candidates = [root] if root.is_file() else sorted(root.rglob("case.json"))
    specs: list[FuzzCaseSpec] = []
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        raw_spec = data.get("case_spec")
        if not isinstance(raw_spec, dict):
            continue
        spec = fuzz_case_spec_from_dict(raw_spec)
        if spec is not None:
            specs.append(spec)
    return specs


def should_sample_tiff_series(series: Any) -> bool:
    axes = getattr(series, "axes", None)
    shape = tuple(getattr(series, "shape", ()) or ())
    if not axes or len(axes) != len(shape) or len(shape) <= 3:
        return False
    non_spatial = 1
    for axis, size in zip(axes, shape):
        if axis not in {"Y", "X", "S"}:
            non_spatial *= max(1, int(size))
    return non_spatial > 16


def read_sampled_tiff_pages(tif: tifffile.TiffFile) -> tuple[np.ndarray, str | None]:
    pages = list(tif.pages[:4])
    if not pages:
        raise ValueError("TIFF contains no readable pages")

    arrays = [np.asarray(page.asarray()) for page in pages]
    arrays = [array for array in arrays if array.size]
    if not arrays:
        raise ValueError("TIFF pages are empty")

    first = arrays[0]
    if first.ndim == 2:
        if len(arrays) == 1:
            return first, "YX"
        same_shape = [array for array in arrays if array.shape == first.shape]
        return np.stack(same_shape, axis=2), "YXC"
    return first, None


def read_corpus_image_with_metadata(path: Path) -> tuple[np.ndarray, dict[str, Any]] | None:
    try:
        axes = None
        sampled_series = False
        if path.suffix.lower() == ".lsm":
            with tifffile.TiffFile(str(path)) as tif:
                series = tif.series[0]
                axes = getattr(series, "axes", None)
                image = series.asarray()
        elif path.suffix.lower() in {".tif", ".tiff"}:
            with tifffile.TiffFile(str(path), _multifile=False) as tif:
                series = tif.series[0]
                axes = getattr(series, "axes", None)
                if should_sample_tiff_series(series):
                    image, axes = read_sampled_tiff_pages(tif)
                    sampled_series = True
                else:
                    image = series.asarray()
        else:
            data = np.fromfile(str(path), dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if image is None:
            return None
        raw_image = np.asarray(image)
        normalized = normalize_loaded_array(raw_image, axes=axes)
        return normalized, {
            "source": str(path),
            "source_extension": path.suffix.lower().lstrip("."),
            "source_shape": list(raw_image.shape),
            "source_axes": axes,
            "normalized_shape": list(normalized.shape),
            "sampled_series": sampled_series,
        }
    except Exception:
        return None


def read_corpus_image(path: Path) -> np.ndarray | None:
    loaded = read_corpus_image_with_metadata(path)
    return None if loaded is None else loaded[0]


def squeeze_loaded_array(
    image: np.ndarray,
    axes: str | None = None,
) -> tuple[np.ndarray, str | None]:
    if axes and len(axes) == image.ndim:
        squeeze_axes = tuple(
            index
            for index, size in enumerate(image.shape)
            if size == 1 and axes[index] not in {"Y", "X"}
        )
        if squeeze_axes:
            image = np.squeeze(image, axis=squeeze_axes)
            axes = "".join(
                axis for index, axis in enumerate(axes) if index not in squeeze_axes
            )
        return image, axes

    return np.squeeze(image), None


def aggressively_flatten_microscopy_image_for_fuzzing(
    image: np.ndarray,
    axes: str | None = None,
) -> np.ndarray:
    """Collapse non-spatial microscopy axes into channels for fuzzing."""
    if axes and len(axes) == image.ndim and "Y" in axes and "X" in axes:
        y_axis = axes.index("Y")
        x_axis = axes.index("X")
        other_axes = [index for index in range(image.ndim) if index not in (y_axis, x_axis)]
        transposed = np.transpose(image, [y_axis, x_axis, *other_axes])
        if not other_axes:
            return transposed
        return transposed.reshape(transposed.shape[0], transposed.shape[1], -1)

    flattened = image.reshape((-1, image.shape[-2], image.shape[-1]))
    return np.transpose(flattened, (1, 2, 0))


def normalize_loaded_array(
    image: np.ndarray,
    axes: str | None = None,
) -> np.ndarray:
    image, axes = squeeze_loaded_array(np.asarray(image), axes)
    if image.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    if image.ndim == 0:
        return np.full((1, 1), int(image), dtype=np.uint8)
    if image.ndim == 1:
        image = image.reshape(1, -1)
    if image.ndim >= 2 and (image.shape[-1] <= 0 or image.shape[-2] <= 0):
        return np.zeros((1, 1), dtype=np.uint8)
    if image.ndim > 3:
        image = aggressively_flatten_microscopy_image_for_fuzzing(image, axes)
        axes = "YXC" if image.ndim == 3 else None
    if image.ndim == 3 and axes and len(axes) == 3 and "Y" in axes and "X" in axes:
        other_axes = [index for index, axis in enumerate(axes) if axis not in {"Y", "X"}]
        if other_axes and axes != "YXC":
            image = np.transpose(image, [axes.index("Y"), axes.index("X"), other_axes[0]])
            axes = "YXC"
    if image.ndim == 3 and image.shape[0] <= 8 and image.shape[1] > 8 and image.shape[2] > 8:
        image = np.transpose(image, (1, 2, 0))
    if image.dtype != np.uint8:
        image_float = image.astype(np.float32)
        finite = image_float[np.isfinite(image_float)]
        if finite.size == 0:
            image = np.zeros(image.shape, dtype=np.uint8)
        else:
            lo = float(np.min(finite))
            hi = float(np.max(finite))
            if hi > lo:
                image = ((np.clip(image_float, lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)
            else:
                image = np.zeros(image.shape, dtype=np.uint8)
    return image


def limit_channels_for_cv2(
    rng: random.Random,
    image: np.ndarray,
    mutations: list[str],
) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] <= 4:
        return image

    source_channels = image.shape[2]
    mode = rng.choice(["first", "random_one", "random_rgb", "random_rgba"])
    if mode == "first":
        mutations.append("truncate_channels")
        return image[:, :, :4]
    if mode == "random_one":
        channel = rng.randrange(0, source_channels)
        mutations.append("select_channel")
        return image[:, :, channel]

    target_channels = 3 if mode == "random_rgb" else 4
    target_channels = min(target_channels, source_channels)
    indices = sorted(rng.sample(range(source_channels), target_channels))
    mutations.append("sample_channels")
    return image[:, :, indices]


def make_object_size(
    rng: random.Random,
    model_data: dict[str, Any],
    allowed_scales: list[int],
) -> dict[str, Any]:
    min_size = rng.choice([0.0, rng.random() * 0.2, rng.random()])
    max_size = rng.choice([1.0, min_size + rng.random() * max(0.0, 1.0 - min_size)])
    usable_scales = [
        scale
        for scale in allowed_scales
        if scale == 20 or (scale == 10 and "x10" in model_data)
    ]
    scale = rng.choice(usable_scales or [20])
    return {
        "min_size": min_size,
        "max_size": max_size,
        "round_parametr_slider": 10**6,
        "round_parametr_value_input": 10**4,
        "color_map": rng.choice(COLORMAPS),
        "color_map_list": COLORMAPS,
        "line_width": round(rng.uniform(0.5, 100.0), 2),
        "scale": scale,
        "alpha": rng.choice([0.25, 0.5, 0.75, 1.0]),
        "size_metric": rng.choice(["area", "diameter", "volume"]),
        "um_per_px": rng.choice([None, 0.1, 0.325, 1.0]),
    }


def make_workflow(
    rng: random.Random,
    mode: str,
    max_steps: int,
) -> dict[str, Any]:
    if mode == "single":
        return {"mode": "single", "steps": []}

    step_count = rng.randint(1, max(1, max_steps))
    steps = []
    for _ in range(step_count):
        operation = rng.choices(
            UI_WORKFLOW_OPERATIONS,
            weights=[4, 3, 2, 2, 2, 1, 2, 1],
            k=1,
        )[0]
        min_size = rng.choice([0.0, rng.random() * 0.15, rng.random() * 0.6])
        max_size = rng.choice([1.0, min_size, min(1.0, min_size + rng.random())])
        if rng.random() < 0.15:
            min_size, max_size = max_size, min_size
        steps.append(
            {
                "operation": operation,
                "min_size": min_size,
                "max_size": max_size,
                "margin": rng.choice([0.0, 0.01, 0.05, 0.25, 1.0]),
                "size_metric": rng.choice(["area", "diameter", "volume", "missing_metric"]),
                "alpha": rng.choice([0.0, 0.25, 0.5, 0.75, 1.0, 1.5]),
                "color_map": rng.choice(COLORMAPS),
                "draw_mode": rng.choice([0, 1]),
                "mask_coordinate_space": rng.choice(["auto", "original", "inference"]),
                "preserve_dtype": rng.choice([False, True]),
                "channel_mode": rng.choice(["same", "swap", "first", "last", "negative", "too_high"]),
                "reset_cache": rng.choice([False, True]),
            }
        )
    return {"mode": "stateful", "steps": steps}


def workflow_from_steps(mode: str, steps: list[dict[str, Any]]) -> dict[str, Any]:
    if mode == "single":
        return {"mode": "single", "steps": []}
    return {"mode": "stateful", "steps": steps or []}


def make_strategy_object_size(
    rng: random.Random,
    allowed_scales: list[int],
) -> dict[str, Any]:
    usable_scales = [scale for scale in allowed_scales if scale in (10, 20)] or [20]
    min_size = rng.choice([0.0, rng.random() * 0.2, rng.random()])
    max_size = rng.choice([1.0, min_size, rng.random()])
    return {
        "min_size": min_size,
        "max_size": max_size,
        "round_parametr_slider": 10**6,
        "round_parametr_value_input": 10**4,
        "color_map": rng.choice(COLORMAPS),
        "color_map_list": COLORMAPS,
        "line_width": round(rng.uniform(0.5, 100.0), 2),
        "scale": rng.choice(usable_scales),
        "alpha": rng.choice([0.0, 0.25, 0.5, 0.75, 1.0, 1.5]),
        "size_metric": rng.choice(["area", "diameter", "volume"]),
        "um_per_px": rng.choice([None, 0.1, 0.325, 1.0, round(rng.uniform(0.001, 5.0), 4)]),
    }


def make_strategy_workflow_steps(
    rng: random.Random,
    workflow: str,
    max_workflow_steps: int,
) -> list[dict[str, Any]]:
    if workflow == "single":
        return []

    steps = []
    for _ in range(rng.randint(1, max(1, max_workflow_steps))):
        steps.append(
            {
                "operation": rng.choice(UI_WORKFLOW_OPERATIONS),
                "min_size": rng.choice([0.0, rng.random() * 0.2, rng.random()]),
                "max_size": rng.choice([1.0, rng.random(), rng.random() * 0.2]),
                "margin": rng.choice([0.0, 0.01, 0.05, 0.25, 1.0]),
                "size_metric": rng.choice(["area", "diameter", "volume", "missing_metric"]),
                "alpha": rng.choice([0.0, 0.25, 0.5, 0.75, 1.0, 1.5]),
                "color_map": rng.choice(COLORMAPS),
                "draw_mode": rng.choice([0, 1]),
                "mask_coordinate_space": rng.choice(["auto", "original", "inference"]),
                "preserve_dtype": rng.choice([False, True]),
                "channel_mode": rng.choice(["same", "swap", "first", "last", "negative", "too_high"]),
                "reset_cache": rng.choice([False, True]),
            }
        )
    return steps


def choose_strategy_image_kind(
    rng: random.Random,
    image_profile: str,
    model_data: dict[str, Any],
) -> str:
    resolved_profile = resolve_image_profile(image_profile, model_data)
    if resolved_profile == "mixed":
        model_type = str(model_data.get("model_type", "")).lower()
        if model_type in {"yolo", "instanseg", "cellpose", "stardist"}:
            return rng.choices(
                ["microscopy", "yolo", "generic"],
                weights=[5, 4, 1],
                k=1,
            )[0]
        return rng.choices(
            ["microscopy", "generic", "yolo"],
            weights=[4, 3, 1],
            k=1,
        )[0]
    if resolved_profile == "microscopy":
        return "microscopy"
    if resolved_profile == "yolo":
        return "yolo"
    return "generic"


def choose_strategy_dimensions(
    rng: random.Random,
    max_side: int,
) -> tuple[int, int]:
    max_side = max(1, max_side)
    buckets = [
        ([1, 2, 3], 1),
        ([7, 8, 15, 16, 31, 32], 3),
        ([47, 64, 96, 127, 128], 4),
        ([192, 256, 320, 384, max_side], 3),
    ]
    buckets = [
        (sorted(set(side for side in sides if side <= max_side)), weight)
        for sides, weight in buckets
    ]
    buckets = [(sides, weight) for sides, weight in buckets if sides]
    if not buckets:
        return max_side, max_side

    tiny_sides = [side for side in [1, 2, 3, 7, 8, 16, 32] if side <= max_side]
    broad_sides = [
        side
        for side in [32, 47, 64, 96, 127, 128, 192, 256, 320, 384, max_side]
        if side <= max_side
    ] or tiny_sides

    if tiny_sides and broad_sides and rng.random() < 0.18:
        if rng.random() < 0.5:
            return rng.choice(broad_sides), rng.choice(tiny_sides)
        return rng.choice(tiny_sides), rng.choice(broad_sides)

    sides = rng.choices(
        [bucket_sides for bucket_sides, _weight in buckets],
        weights=[weight for _bucket_sides, weight in buckets],
        k=1,
    )[0]
    return rng.choice(sides), rng.choice(sides)


def draw_strategy_case_spec(
    rng: random.Random,
    model_name: str,
    image_profile: str,
    model_data: dict[str, Any],
    allowed_scales: list[int],
    workflow: str,
    max_workflow_steps: int,
    max_side: int,
) -> FuzzCaseSpec:
    image_kind = choose_strategy_image_kind(rng, image_profile, model_data)
    width, height = choose_strategy_dimensions(rng, max_side)
    if image_kind == "yolo":
        patterns = YOLO_PATTERNS
    elif image_kind == "microscopy":
        patterns = MICROSCOPY_PATTERNS
    else:
        patterns = GENERIC_PATTERNS
    return FuzzCaseSpec(
        model_name,
        image_kind=image_kind,
        width=width,
        height=height,
        channels=rng.choice([None, 1, 2, 3, 4]),
        pattern=rng.choice(patterns),
        extension=rng.choice(["png", "jpg", "tif", "lsm"]),
        object_size=make_strategy_object_size(rng, allowed_scales),
        workflow=make_strategy_workflow_steps(rng, workflow, max_workflow_steps),
        image_recipe={},
    )


def grammar_production(left: str, right: str) -> str:
    return f"{left} -> {right}"


def microscopy_rule_for_scene(scene: str) -> dict[str, Any]:
    return MICROSCOPY_SCENE_RULES.get(scene, MICROSCOPY_SCENE_RULES["single_cells"])


def minimum_channels_for_scene(scene: str, extension: str | None = None) -> int:
    minimum = 1
    if scene in {"nuclei_cytoplasm", "zstack_cells"}:
        minimum = 2
    if scene == "zstack_cells":
        minimum = 3
    if extension == "lsm":
        minimum = max(minimum, 2)
    return minimum


def choose_grammar_extension(rng: random.Random, scene: str, image_kind: str) -> str:
    if image_kind == "generic":
        return rng.choice(["png", "tif", "lsm"])
    if scene == "zstack_cells":
        return rng.choice(["lsm", "tif"])
    if scene in {"nuclei_cytoplasm", "edge_cells", "dense_colony"}:
        return rng.choices(["png", "tif", "lsm", "jpg"], weights=[3, 3, 3, 1], k=1)[0]
    return rng.choice(["png", "jpg", "tif", "lsm"])


def choose_grammar_channels(
    rng: random.Random,
    scene: str,
    extension: str,
) -> int | None:
    minimum = minimum_channels_for_scene(scene, extension)
    if extension == "jpg":
        return rng.choice([None, 3])
    if minimum >= 3:
        return rng.choice([minimum, 4])
    if minimum == 2:
        return rng.choice([2, 3, 4])
    return rng.choice([None, 1, 2, 3, 4])


def grammar_base_cell_count(
    rng: random.Random,
    scene: str,
    width: int,
    height: int,
) -> int:
    area = max(1, width * height)
    base_count = max(1, min(90, area // rng.choice([650, 900, 1400, 2200])))
    rule = microscopy_rule_for_scene(scene)["population"]
    if rule == "Sparse":
        return rng.randint(1, max(1, min(8, base_count)))
    if rule in {"Dense", "ZStack"}:
        return rng.randint(max(12 if scene == "dense_colony" else 4, base_count), max(12, base_count * 3))
    if rule in {"Clustered", "Overlapping", "NucleiCytoplasm"}:
        return rng.randint(3, max(5, min(35, base_count * 2)))
    if rule == "Border":
        return rng.randint(2, max(3, min(24, base_count * 2)))
    return rng.randint(1, max(3, base_count))


def grammar_cell_radii(
    rng: random.Random,
    scene: str,
    min_side: int,
) -> tuple[int, int]:
    radius_min = max(1, min_side // rng.choice([48, 36, 28, 20]))
    radius_max = max(radius_min, min_side // rng.choice([7, 9, 12, 16]))
    rx = rng.randint(radius_min, radius_max)
    ry = rng.randint(radius_min, radius_max)
    cell_rule = microscopy_rule_for_scene(scene)["cell"]
    if cell_rule == "Elongated":
        if rng.random() < 0.5:
            rx = max(rx, ry * rng.randint(2, 5))
        else:
            ry = max(ry, rx * rng.randint(2, 5))
    return max(1, rx), max(1, ry)


def build_grammar_cell_population(
    rng: random.Random,
    scene: str,
    width: int,
    height: int,
    cell_count: int,
) -> list[dict[str, Any]]:
    rule = microscopy_rule_for_scene(scene)
    population = rule["population"]
    cell_rule = rule["cell"]
    min_side = max(1, min(width, height))
    cells: list[dict[str, Any]] = []
    cluster_x = rng.randrange(0, width)
    cluster_y = rng.randrange(0, height)

    for index in range(cell_count):
        rx, ry = grammar_cell_radii(rng, scene, min_side)
        angle = rng.randrange(0, 180)
        partial = False
        touches_border = False
        overlaps_previous = False

        if population == "Border":
            side = index % 4
            if side == 0:
                cx = rng.randint(-rx, max(0, rx // 2))
                cy = rng.randrange(0, height)
            elif side == 1:
                cx = rng.randint(max(0, width - max(1, rx // 2)), width + rx)
                cy = rng.randrange(0, height)
            elif side == 2:
                cx = rng.randrange(0, width)
                cy = rng.randint(-ry, max(0, ry // 2))
            else:
                cx = rng.randrange(0, width)
                cy = rng.randint(max(0, height - max(1, ry // 2)), height + ry)
            partial = True
            touches_border = True
        elif population in {"Clustered", "Overlapping", "NucleiCytoplasm", "ZStack"}:
            if index == 0:
                cx = cluster_x
                cy = cluster_y
            else:
                previous = cells[-1]
                prev_rx, prev_ry = previous["radius"]
                distance = (max(prev_rx, prev_ry) + max(rx, ry)) * (
                    0.55 if population == "Overlapping" else 0.9
                )
                theta = rng.uniform(0, 2 * np.pi)
                cx = int(previous["center"][0] + np.cos(theta) * distance)
                cy = int(previous["center"][1] + np.sin(theta) * distance)
                cx = int(np.clip(cx, 0, max(0, width - 1)))
                cy = int(np.clip(cy, 0, max(0, height - 1)))
                overlaps_previous = population == "Overlapping"
        elif population == "Dense":
            columns = max(1, int(np.ceil(np.sqrt(cell_count))))
            rows = max(1, int(np.ceil(cell_count / columns)))
            col = index % columns
            row = index // columns
            cx = int((col + rng.uniform(0.25, 0.75)) * width / columns)
            cy = int((row + rng.uniform(0.25, 0.75)) * height / rows)
        else:
            cx = rng.randrange(0, width)
            cy = rng.randrange(0, height)

        cells.append(
            {
                "production": grammar_production("Cell", cell_rule),
                "kind": cell_rule,
                "center": [int(cx), int(cy)],
                "radius": [int(rx), int(ry)],
                "angle": int(angle),
                "partial": partial,
                "touches_border": touches_border,
                "overlaps_previous": overlaps_previous,
                "has_nucleus": scene in {"nuclei_cytoplasm", "zstack_cells"} or population == "NucleiCytoplasm",
                "saturated": cell_rule == "Saturated",
            }
        )
    return cells


def microscopy_recipe_constraint_violations(
    recipe: dict[str, Any],
    width: int,
    height: int,
    channels: int | None,
    extension: str | None,
) -> list[str]:
    scene = str(recipe.get("scene", "single_cells"))
    cells = recipe.get("cells") if isinstance(recipe.get("cells"), list) else []
    violations: list[str] = []
    if channels is not None and channels < minimum_channels_for_scene(scene, extension):
        violations.append("channels below scene minimum")
    if scene == "zstack_cells":
        if extension not in {"lsm", "tif"}:
            violations.append("zstack encoding is not microscopy-friendly")
        if int(recipe.get("z_planes", 1)) < 2:
            violations.append("zstack has fewer than two planes")
    if scene == "edge_cells" and not any(cell.get("touches_border") for cell in cells):
        violations.append("edge scene has no border-intersecting cell")
    if scene == "overlapping_cells" and len(cells) > 1 and not any(
        cell.get("overlaps_previous") for cell in cells[1:]
    ):
        violations.append("overlapping scene has no overlapping pair")
    if scene == "elongated_cells":
        for cell in cells:
            rx, ry = [max(1, int(value)) for value in cell.get("radius", [1, 1])]
            if max(rx, ry) < 2 * min(rx, ry):
                violations.append("elongated cell ratio below 2")
                break
    if scene == "nuclei_cytoplasm" and not all(cell.get("has_nucleus") for cell in cells):
        violations.append("nuclei_cytoplasm scene has cells without nuclei")
    if scene == "dense_colony" and int(recipe.get("cell_count", 0)) < 12:
        violations.append("dense colony has too few cells")
    return violations


def enforce_microscopy_recipe_constraints(
    recipe: dict[str, Any],
    width: int,
    height: int,
    channels: int | None,
    extension: str | None = None,
) -> tuple[dict[str, Any], int | None, str | None]:
    recipe = copy.deepcopy(recipe)
    scene = str(recipe.get("scene", "single_cells"))
    if scene not in MICROSCOPY_SCENE_RULES:
        scene = "single_cells"
        recipe["scene"] = scene
    rule = microscopy_rule_for_scene(scene)
    extension = canonical_minimizer_extension(extension or str(recipe.get("extension", "png")))
    if scene == "zstack_cells" and extension not in {"lsm", "tif"}:
        extension = "lsm"

    minimum_channels = minimum_channels_for_scene(scene, extension)
    if channels is None and minimum_channels > 1:
        channels = minimum_channels
    elif channels is not None:
        channels = max(minimum_channels, min(4, int(channels)))

    cell_count = max(0, int(recipe.get("cell_count", 1)))
    if rule["population"] == "Dense":
        cell_count = max(12, cell_count)
    if rule["population"] == "Sparse":
        cell_count = min(max(1, cell_count), 8)
    if rule["population"] in {"Clustered", "Overlapping", "NucleiCytoplasm", "ZStack"}:
        cell_count = max(3, cell_count)
    if rule["population"] == "Border":
        cell_count = max(2, cell_count)
    recipe["cell_count"] = cell_count
    recipe["population"] = rule["population"]
    recipe["cell_rule"] = rule["cell"]
    productions = list(recipe.get("productions", []) or [])
    required_lhs = {"Scene", "MicroscopyScene", "CellPopulation", "Cell", "Channels", "Encoding"}
    productions = [
        production
        for production in productions
        if str(production).split(" -> ", 1)[0] not in required_lhs
    ]
    required_productions = [
        grammar_production("Scene", "MicroscopyScene"),
        grammar_production(
            "MicroscopyScene",
            "Background CellPopulation Artifacts Optics Channels Encoding",
        ),
        grammar_production("CellPopulation", rule["population"]),
        grammar_production("Cell", rule["cell"]),
        grammar_production("Channels", f"{max(1, int(channels or 1))} channel(s)"),
        grammar_production("Encoding", extension),
    ]
    for production in required_productions:
        if production not in productions:
            productions.append(production)
    recipe["productions"] = productions
    recipe["grammar_version"] = 2
    recipe["start_symbol"] = "Scene"
    recipe["nonterminals"] = {
        "Scene": "MicroscopyScene",
        "CellPopulation": rule["population"],
        "Cell": rule["cell"],
    }
    recipe["constraints"] = list(rule["constraints"])
    recipe["nuclei_channel"] = scene in {"nuclei_cytoplasm", "zstack_cells"} or (channels or 1) >= 2
    recipe["z_planes"] = max(2, int(recipe.get("z_planes", 2))) if scene == "zstack_cells" else 1
    recipe["extension"] = extension

    cells = recipe.get("cells")
    if not isinstance(cells, list) or len(cells) != cell_count:
        rng = random.Random(stable_recipe_seed(recipe, width, height, channels, extension))
        recipe["cells"] = build_grammar_cell_population(
            rng,
            scene,
            max(1, width),
            max(1, height),
            cell_count,
        )

    if scene == "edge_cells":
        if not any(cell.get("touches_border") for cell in recipe["cells"]):
            recipe["cells"][0]["touches_border"] = True
            recipe["cells"][0]["partial"] = True
            rx, ry = recipe["cells"][0]["radius"]
            recipe["cells"][0]["center"] = [-max(1, int(rx) // 2), max(0, height // 2)]
    if scene == "overlapping_cells" and len(recipe["cells"]) > 1:
        recipe["cells"][1]["overlaps_previous"] = True
    if scene == "elongated_cells":
        for cell in recipe["cells"]:
            rx, ry = cell["radius"]
            if max(rx, ry) < 2 * max(1, min(rx, ry)):
                cell["radius"] = [max(rx, ry) * 2, max(1, min(rx, ry))]

    violations = microscopy_recipe_constraint_violations(
        recipe,
        width,
        height,
        channels,
        extension,
    )
    recipe["semantic_valid"] = not violations
    recipe["semantic_violations"] = violations

    return recipe, channels, extension


def stable_recipe_seed(
    recipe: dict[str, Any],
    width: int,
    height: int,
    channels: int | None,
    extension: str | None,
) -> int:
    payload = json.dumps(
        {
            "scene": recipe.get("scene"),
            "cell_count": recipe.get("cell_count"),
            "width": width,
            "height": height,
            "channels": channels,
            "extension": extension,
        },
        sort_keys=True,
        default=str,
    )
    return int(hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8], 16)


def draw_microscopy_recipe(
    rng: random.Random,
    pattern: str,
    width: int,
    height: int,
    channels: int | None,
    extension: str | None = None,
) -> dict[str, Any]:
    area = max(1, width * height)
    pattern = pattern if pattern in MICROSCOPY_SCENE_RULES else "single_cells"
    rule = microscopy_rule_for_scene(pattern)
    cell_count = grammar_base_cell_count(rng, pattern, width, height)
    min_side = max(1, min(width, height))
    radius_min = max(1, min_side // rng.choice([48, 36, 28, 20]))
    radius_max = max(radius_min, min_side // rng.choice([7, 9, 12, 16]))
    contrast = {
        "low_contrast_cells": rng.uniform(10, 34),
        "saturated_cells": rng.uniform(150, 230),
    }.get(pattern, rng.uniform(55, 170))

    extension = choose_grammar_extension(rng, pattern, "microscopy") if extension is None else extension
    channels = choose_grammar_channels(rng, pattern, extension) if channels is None else channels
    recipe = {
        "grammar_version": 2,
        "start_symbol": "Scene",
        "nonterminals": {
            "Scene": "MicroscopyScene",
            "CellPopulation": rule["population"],
            "Cell": rule["cell"],
        },
        "productions": [
            grammar_production("Scene", "MicroscopyScene"),
            grammar_production(
                "MicroscopyScene",
                "Background CellPopulation Artifacts Optics Channels Encoding",
            ),
            grammar_production("CellPopulation", rule["population"]),
            grammar_production("Cell", rule["cell"]),
            grammar_production("Channels", f"{max(1, int(channels or 1))} channel(s)"),
            grammar_production("Encoding", canonical_minimizer_extension(extension)),
        ],
        "scene": pattern,
        "population": rule["population"],
        "cell_rule": rule["cell"],
        "cell_count": cell_count,
        "radius_min": radius_min,
        "radius_max": radius_max,
        "background": rng.randint(8, 70),
        "contrast": round(float(contrast), 3),
        "noise_sigma": round(rng.uniform(1.0, 12.0), 3),
        "blur": rng.choice([0, 0, 3, 5]),
        "debris": pattern == "debris_field" or rng.random() < 0.25,
        "edge_bias": pattern == "edge_cells",
        "clustered": pattern in {"touching_cells", "overlapping_cells", "dense_colony"},
        "overlap": pattern == "overlapping_cells",
        "elongated": pattern == "elongated_cells",
        "nuclei_channel": pattern in {"nuclei_cytoplasm", "zstack_cells"} or (channels or 1) >= 2,
        "z_planes": rng.choice([2, 3, 4]) if pattern == "zstack_cells" else 1,
        "constraints": list(rule["constraints"]),
    }
    recipe["cells"] = build_grammar_cell_population(
        rng,
        pattern,
        max(1, width),
        max(1, height),
        cell_count,
    )
    recipe, _channels, _extension = enforce_microscopy_recipe_constraints(
        recipe,
        width,
        height,
        channels,
        extension,
    )
    return recipe


def draw_grammar_seed_spec(
    rng: random.Random,
    model_name: str,
    image_profile: str,
    model_data: dict[str, Any],
    allowed_scales: list[int],
    workflow: str,
    max_workflow_steps: int,
    max_side: int,
) -> FuzzCaseSpec:
    image_kind = choose_strategy_image_kind(rng, image_profile, model_data)
    if rng.random() < 0.65:
        image_kind = "microscopy"
    width, height = choose_strategy_dimensions(rng, max_side)
    if image_kind == "microscopy" and max_side >= 64 and min(width, height) < 16:
        width = rng.choice([64, 96, 128, 192, min(max_side, 256)])
        height = rng.choice([64, 96, 128, 192, min(max_side, 256)])
        width = min(width, max_side)
        height = min(height, max_side)

    if image_kind == "yolo":
        pattern = rng.choice(YOLO_PATTERNS)
    elif image_kind == "microscopy":
        pattern = rng.choice(MICROSCOPY_PATTERNS)
    else:
        pattern = rng.choice(GENERIC_PATTERNS)

    extension = choose_grammar_extension(rng, pattern, image_kind)
    channels = choose_grammar_channels(rng, pattern, extension)
    recipe = (
        draw_microscopy_recipe(rng, pattern, width, height, channels, extension)
        if image_kind == "microscopy"
        else {
            "grammar_version": 2,
            "start_symbol": "Scene",
            "productions": [
                grammar_production(
                    "Scene",
                    "YoloScene" if image_kind == "yolo" else "DegenerateScene",
                ),
                grammar_production("Encoding", extension),
            ],
            "scene": pattern,
            "constraints": [],
            "semantic_valid": True,
        }
    )
    recipe["grammar_mutations"] = []
    return FuzzCaseSpec(
        model_name=model_name,
        image_kind=image_kind,
        width=width,
        height=height,
        channels=channels,
        pattern=pattern,
        extension=extension,
        object_size=make_strategy_object_size(rng, allowed_scales),
        workflow=make_strategy_workflow_steps(rng, workflow, max_workflow_steps),
        image_recipe=recipe,
    )


def mutate_case_spec_object_size(
    rng: random.Random,
    spec: FuzzCaseSpec,
    allowed_scales: list[int],
) -> None:
    object_size = copy.deepcopy(spec.object_size)
    mutation = rng.choice(["wide_open", "tiny_only", "large_only", "scale", "style"])
    if mutation == "wide_open":
        object_size["min_size"] = 0.0
        object_size["max_size"] = 1.0
    elif mutation == "tiny_only":
        object_size["min_size"] = 0.0
        object_size["max_size"] = rng.uniform(0.001, 0.08)
    elif mutation == "large_only":
        object_size["min_size"] = rng.uniform(0.25, 0.9)
        object_size["max_size"] = 1.0
    elif mutation == "scale":
        object_size["scale"] = rng.choice([10, 20] if 10 in allowed_scales else [20])
        object_size["um_per_px"] = rng.choice([None, 0.05, 0.1, 0.325, 1.0, 5.0])
    else:
        object_size["alpha"] = rng.choice([0.0, 0.15, 0.5, 0.9, 1.25])
        object_size["line_width"] = rng.choice([0.5, 1.0, 3.0, 25.0, 100.0])
        object_size["color_map"] = rng.choice(COLORMAPS)
    spec.object_size = object_size


def mutate_case_spec_workflow(
    rng: random.Random,
    spec: FuzzCaseSpec,
    workflow: str,
    max_workflow_steps: int,
) -> None:
    if workflow == "single":
        spec.workflow = []
        return
    steps = copy.deepcopy(spec.workflow)
    if not steps or rng.random() < 0.35:
        steps.extend(make_strategy_workflow_steps(rng, "stateful", 1))
    elif rng.random() < 0.25 and len(steps) > 1:
        del steps[rng.randrange(0, len(steps))]
    else:
        step = steps[rng.randrange(0, len(steps))]
        field_name = rng.choice(
            [
                "operation",
                "min_size",
                "max_size",
                "margin",
                "size_metric",
                "alpha",
                "mask_coordinate_space",
                "channel_mode",
                "reset_cache",
            ]
        )
        replacements = {
            "operation": rng.choice(UI_WORKFLOW_OPERATIONS),
            "min_size": rng.choice([0.0, rng.random() * 0.05, rng.random()]),
            "max_size": rng.choice([1.0, rng.random() * 0.2, rng.random()]),
            "margin": rng.choice([0.0, 0.001, 0.05, 0.5, 1.0]),
            "size_metric": rng.choice(["area", "diameter", "volume", "missing_metric"]),
            "alpha": rng.choice([0.0, 0.05, 0.5, 1.0, 1.5]),
            "mask_coordinate_space": rng.choice(["auto", "original", "inference"]),
            "channel_mode": rng.choice(["same", "swap", "first", "last", "negative", "too_high"]),
            "reset_cache": rng.choice([False, True]),
        }
        step[field_name] = replacements[field_name]
    spec.workflow = steps[: max(1, max_workflow_steps)]


def repair_grammar_case_spec(
    rng: random.Random,
    spec: FuzzCaseSpec,
    max_side: int,
) -> None:
    spec.width = max(1, min(max_side, int(spec.width)))
    spec.height = max(1, min(max_side, int(spec.height)))
    if spec.image_kind == "microscopy":
        scene = spec.pattern if spec.pattern in MICROSCOPY_SCENE_RULES else str(
            spec.image_recipe.get("scene", "single_cells")
        )
        if scene not in MICROSCOPY_SCENE_RULES:
            scene = "single_cells"
        recipe = copy.deepcopy(spec.image_recipe or {})
        recipe.setdefault("scene", scene)
        recipe["scene"] = scene
        recipe, channels, extension = enforce_microscopy_recipe_constraints(
            recipe,
            spec.width,
            spec.height,
            spec.channels,
            spec.extension,
        )
        spec.pattern = scene
        spec.channels = channels
        spec.extension = extension or spec.extension
        spec.image_recipe = recipe
        return

    if spec.image_kind == "yolo":
        spec.pattern = spec.pattern if spec.pattern in YOLO_PATTERNS else rng.choice(YOLO_PATTERNS)
    else:
        spec.image_kind = "generic"
        spec.pattern = spec.pattern if spec.pattern in GENERIC_PATTERNS else rng.choice(GENERIC_PATTERNS)
    spec.extension = canonical_minimizer_extension(spec.extension)
    spec.image_recipe = {
        "grammar_version": 2,
        "start_symbol": "Scene",
        "productions": [
            grammar_production(
                "Scene",
                "YoloScene" if spec.image_kind == "yolo" else "DegenerateScene",
            ),
            grammar_production("Encoding", spec.extension),
        ],
        "scene": spec.pattern,
        "constraints": [],
        "semantic_valid": True,
        **{
            key: value
            for key, value in spec.image_recipe.items()
            if key in {"grammar_mutations"}
        },
    }


def mutate_case_spec_recipe(
    rng: random.Random,
    spec: FuzzCaseSpec,
    max_side: int,
) -> str:
    mutation = rng.choice(GRAMMAR_MUTATIONS)
    if mutation in {"expand:Scene", "expand:MicroscopyScene"}:
        spec.image_kind = rng.choices(
            ["microscopy", "yolo", "generic"],
            weights=[6, 3, 1],
            k=1,
        )[0]
        if spec.image_kind == "microscopy":
            spec.pattern = rng.choice(MICROSCOPY_PATTERNS)
            spec.extension = choose_grammar_extension(rng, spec.pattern, spec.image_kind)
            spec.channels = choose_grammar_channels(rng, spec.pattern, spec.extension)
            spec.image_recipe = draw_microscopy_recipe(
                rng,
                spec.pattern,
                spec.width,
                spec.height,
                spec.channels,
                spec.extension,
            )
        elif spec.image_kind == "yolo":
            spec.pattern = rng.choice(YOLO_PATTERNS)
            spec.extension = choose_grammar_extension(rng, spec.pattern, spec.image_kind)
            spec.image_recipe = {
                "grammar_version": 2,
                "start_symbol": "Scene",
                "productions": [
                    grammar_production("Scene", "YoloScene"),
                    grammar_production("Encoding", spec.extension),
                ],
                "scene": spec.pattern,
                "semantic_valid": True,
            }
        else:
            spec.pattern = rng.choice(GENERIC_PATTERNS)
            spec.extension = choose_grammar_extension(rng, spec.pattern, spec.image_kind)
            spec.image_recipe = {
                "grammar_version": 2,
                "start_symbol": "Scene",
                "productions": [
                    grammar_production("Scene", "DegenerateScene"),
                    grammar_production("Encoding", spec.extension),
                ],
                "scene": spec.pattern,
                "semantic_valid": True,
            }
    elif mutation == "expand:CellPopulation":
        spec.image_kind = "microscopy"
        spec.pattern = rng.choice(MICROSCOPY_PATTERNS)
        spec.image_recipe = draw_microscopy_recipe(
            rng,
            spec.pattern,
            spec.width,
            spec.height,
            spec.channels,
            spec.extension,
        )
    elif mutation == "expand:Cell":
        if spec.image_kind != "microscopy":
            spec.image_kind = "microscopy"
        spec.pattern = rng.choice(["elongated_cells", "saturated_cells", "edge_cells"])
        spec.image_recipe = draw_microscopy_recipe(
            rng,
            spec.pattern,
            spec.width,
            spec.height,
            spec.channels,
            spec.extension,
        )
    elif mutation == "expand:Artifacts":
        if spec.image_kind == "microscopy":
            recipe = copy.deepcopy(spec.image_recipe or {})
            recipe["debris"] = True
            recipe["noise_sigma"] = round(rng.uniform(6.0, 18.0), 3)
            recipe.setdefault("productions", []).append(grammar_production("Artifacts", "Debris Speckles"))
            spec.image_recipe = recipe
        elif spec.image_kind == "yolo":
            spec.pattern = rng.choice(["speckles", "checkerboard", "textured_noise"])
    elif mutation == "expand:Optics":
        if spec.image_kind == "microscopy":
            recipe = copy.deepcopy(spec.image_recipe or {})
            recipe["blur"] = rng.choice([0, 3, 5])
            recipe["contrast"] = round(
                rng.uniform(10, 34) if spec.pattern == "low_contrast_cells" else rng.uniform(45, 210),
                3,
            )
            recipe.setdefault("productions", []).append(grammar_production("Optics", "Blur Contrast Noise"))
            spec.image_recipe = recipe
    elif mutation == "expand:Encoding":
        spec.extension = choose_grammar_extension(rng, spec.pattern, spec.image_kind)
        spec.channels = choose_grammar_channels(rng, spec.pattern, spec.extension)
    elif mutation == "expand:Canvas":
        spec.width, spec.height = choose_strategy_dimensions(rng, max_side)
    elif mutation == "expand:Channels":
        spec.channels = choose_grammar_channels(rng, spec.pattern, spec.extension)
    elif mutation == "expand:Workflow":
        return "expand:Workflow"
    elif mutation == "crossover:CellPopulation":
        donor_pattern = rng.choice(MICROSCOPY_PATTERNS)
        donor = draw_microscopy_recipe(
            rng,
            donor_pattern,
            spec.width,
            spec.height,
            spec.channels,
            spec.extension,
        )
        if spec.image_kind != "microscopy":
            spec.image_kind = "microscopy"
        merged = copy.deepcopy(spec.image_recipe or {})
        for key in ("scene", "population", "cell_rule", "cell_count", "cells", "constraints"):
            merged[key] = donor[key]
        merged.setdefault("productions", []).append(grammar_production("CellPopulation", donor["population"]))
        spec.pattern = donor_pattern
        spec.image_recipe = merged
    elif mutation == "mutate:ObjectSize":
        return "mutate:ObjectSize"
    else:
        long_side = rng.choice([64, 96, 128, 192, 256, max_side])
        short_side = rng.choice([1, 2, 3, 7, 16, 31, 32])
        spec.width, spec.height = (
            (min(max_side, long_side), min(max_side, short_side))
            if rng.random() < 0.5
            else (min(max_side, short_side), min(max_side, long_side))
        )
    repair_grammar_case_spec(rng, spec, max_side)
    return mutation


def draw_grammar_case_spec(
    rng: random.Random,
    model_name: str,
    image_profile: str,
    model_data: dict[str, Any],
    allowed_scales: list[int],
    workflow: str,
    max_workflow_steps: int,
    max_side: int,
    spec_corpus: list[FuzzCaseSpec] | None = None,
) -> FuzzCaseSpec:
    if spec_corpus and rng.random() < 0.65:
        spec = copy.deepcopy(rng.choice(spec_corpus))
        spec.model_name = model_name
        spec.width = max(1, min(max_side, int(spec.width)))
        spec.height = max(1, min(max_side, int(spec.height)))
        if not spec.object_size:
            spec.object_size = make_strategy_object_size(rng, allowed_scales)
        if workflow == "single":
            spec.workflow = []
        elif not spec.workflow:
            spec.workflow = make_strategy_workflow_steps(rng, workflow, max_workflow_steps)
    else:
        spec = draw_grammar_seed_spec(
            rng,
            model_name,
            image_profile,
            model_data,
            allowed_scales,
            workflow,
            max_workflow_steps,
            max_side,
        )
    mutation_count = rng.randint(2, 6)
    applied: list[str] = []
    if spec_corpus:
        applied.append("parent_case_spec")
    for _ in range(mutation_count):
        mutation = rng.choice(["recipe", "object_size", "workflow"])
        if mutation == "object_size":
            mutate_case_spec_object_size(rng, spec, allowed_scales)
            applied.append("tune_object_size")
        elif mutation == "workflow":
            mutate_case_spec_workflow(rng, spec, workflow, max_workflow_steps)
            applied.append("expand:Workflow")
        else:
            recipe_mutation = mutate_case_spec_recipe(rng, spec, max_side)
            if recipe_mutation == "mutate:ObjectSize":
                mutate_case_spec_object_size(rng, spec, allowed_scales)
            elif recipe_mutation == "expand:Workflow":
                mutate_case_spec_workflow(rng, spec, workflow, max_workflow_steps)
            applied.append(recipe_mutation)
    recipe = copy.deepcopy(spec.image_recipe or {})
    recipe["grammar_mutations"] = applied
    spec.image_recipe = recipe
    repair_grammar_case_spec(rng, spec, max_side)
    return spec


def effective_generation_engine(engine: str) -> str:
    if engine == "hypothesis":
        return "strategy"
    return engine


def resolve_image_profile(profile: str, model_data: dict[str, Any]) -> str:
    if profile == "mixed":
        return "mixed"
    if profile != "auto":
        return profile
    if str(model_data.get("model_type", "")).lower() == "yolo":
        return "yolo"
    return "generic"


def choose_effective_profile(
    rng: random.Random,
    profile: str,
    has_corpus: bool,
) -> str:
    if profile == "mixed":
        choices = ["generic", "yolo", "microscopy", "microscopy"]
        if has_corpus:
            choices.extend(["corpus", "corpus"])
        return rng.choice(choices)
    return profile


def choose_dimensions(
    rng: random.Random,
    profile: str,
    max_side: int,
) -> tuple[int, int]:
    if profile == "microscopy":
        candidates = [32, 48, 64, 96, 128, 192, 256, 320, 384, max_side]
        candidates = [value for value in candidates if value <= max_side] or [max(1, max_side)]
        if rng.random() < 0.15:
            thin = [value for value in [7, 16, 31, 32] if value <= max_side] or candidates
            return rng.choice(candidates), rng.choice(thin)
        return rng.choice(candidates), rng.choice(candidates)

    if profile == "yolo":
        candidates = [
            1, 2, 3, 7, 8, 15, 16, 31, 32, 47, 63, 64, 65, 95, 96,
            127, 128, 129, 191, 192, 255, 256, 320, 384, 512,
        ]
        candidates = [value for value in candidates if value <= max_side] or [max(1, max_side)]
        if rng.random() < 0.25:
            return rng.choice(candidates), rng.choice([value for value in candidates if value <= 32] or candidates)
        if rng.random() < 0.25:
            return rng.choice([value for value in candidates if value <= 32] or candidates), rng.choice(candidates)
        return rng.choice(candidates), rng.choice(candidates)

    candidates = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384]
    candidates = [value for value in candidates if value <= max_side] or [max(1, max_side)]
    return rng.choice(candidates), rng.choice(candidates)


def make_pattern_array(
    rng: random.Random,
    height: int,
    width: int,
    channels: int | None,
    pattern: str | None = None,
) -> tuple[np.ndarray, str]:
    shape = (height, width) if channels is None else (height, width, channels)
    pattern = pattern if pattern in GENERIC_PATTERNS else rng.choice(GENERIC_PATTERNS)

    if pattern == "zeros":
        image = np.zeros(shape, dtype=np.uint8)
    elif pattern == "constant":
        image = np.full(shape, rng.randrange(0, 256), dtype=np.uint8)
    elif pattern == "noise":
        image = rng_numpy(rng).integers(0, 256, size=shape, dtype=np.uint8)
    elif pattern == "gradient":
        x_grad = np.linspace(0, 255, width, dtype=np.uint8)
        y_grad = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
        image_2d = ((x_grad.astype(np.uint16) + y_grad.astype(np.uint16)) // 2).astype(
            np.uint8
        )
        image = image_2d if channels is None else np.repeat(image_2d[:, :, None], channels, axis=2)
    elif pattern == "spots":
        image = np.zeros(shape, dtype=np.uint8)
        draw_spots(rng, image)
    else:
        image = np.zeros(shape, dtype=np.uint8)
        stripe_width = rng.randint(1, max(1, width // 8))
        for x in range(0, width, stripe_width * 2):
            image[:, x : x + stripe_width] = rng.randrange(64, 256)

    return image, pattern


def make_yolo_pattern_array(
    rng: random.Random,
    height: int,
    width: int,
    channels: int | None,
    pattern: str | None = None,
) -> tuple[np.ndarray, str]:
    pattern = pattern if pattern in YOLO_PATTERNS else rng.choice(YOLO_PATTERNS)

    background = rng.randrange(0, 96)
    image_2d = np.full((height, width), background, dtype=np.uint8)

    if pattern == "empty":
        pass
    elif pattern == "checkerboard":
        tile = rng.randint(1, max(1, min(height, width, 16)))
        yy, xx = np.indices((height, width))
        image_2d = (((xx // tile + yy // tile) % 2) * rng.randrange(80, 256)).astype(np.uint8)
    elif pattern == "textured_noise":
        noise = rng_numpy(rng).normal(background, rng.uniform(4, 48), size=(height, width))
        image_2d = np.clip(noise, 0, 255).astype(np.uint8)
        if min(height, width) > 2 and rng.random() < 0.7:
            image_2d = cv2.GaussianBlur(image_2d, (3, 3), 0)
    else:
        if pattern == "gradient_blobs":
            x_grad = np.linspace(0, rng.randrange(80, 256), width, dtype=np.float32)
            y_grad = np.linspace(0, rng.randrange(80, 256), height, dtype=np.float32)[:, None]
            image_2d = np.clip((x_grad + y_grad) / 2, 0, 255).astype(np.uint8)

        if pattern == "speckles":
            count = rng.randint(10, max(10, height * width // 8))
            for _ in range(count):
                image_2d[rng.randrange(0, height), rng.randrange(0, width)] = rng.randrange(128, 256)
        else:
            blob_count = rng.randint(1, 30 if pattern == "touching_blobs" else 12)
            for index in range(blob_count):
                if pattern == "border_blobs" and rng.random() < 0.7:
                    cx = rng.choice([-rng.randint(1, max(1, width // 4)), width + rng.randint(0, max(1, width // 4))])
                    cy = rng.randrange(-max(1, height // 4), height + max(1, height // 4))
                elif pattern == "touching_blobs":
                    cx = width // 2 + rng.randint(-max(1, width // 5), max(1, width // 5))
                    cy = height // 2 + rng.randint(-max(1, height // 5), max(1, height // 5))
                else:
                    cx = rng.randrange(0, width)
                    cy = rng.randrange(0, height)

                rx = rng.randint(1, max(1, width // rng.choice([6, 8, 12, 20])))
                ry = rng.randint(1, max(1, height // rng.choice([6, 8, 12, 20])))
                value = rng.randrange(86, 120) if pattern == "low_contrast_blobs" else rng.randrange(140, 256)
                angle = rng.randrange(0, 180)
                cv2.ellipse(image_2d, (int(cx), int(cy)), (rx, ry), angle, 0, 360, int(value), -1)

                if pattern == "rings" and rx > 1 and ry > 1:
                    cv2.ellipse(
                        image_2d,
                        (int(cx), int(cy)),
                        (max(1, rx // 2), max(1, ry // 2)),
                        angle,
                        0,
                        360,
                        int(background),
                        -1,
                    )

                if rng.random() < 0.2:
                    cv2.line(
                        image_2d,
                        (rng.randrange(0, width), rng.randrange(0, height)),
                        (rng.randrange(0, width), rng.randrange(0, height)),
                        int(rng.randrange(64, 256)),
                        rng.randint(1, 3),
                    )

    if rng.random() < 0.35 and min(height, width) > 2:
        image_2d = cv2.GaussianBlur(image_2d, (3, 3), 0)

    return channelize_image(rng, image_2d, channels), pattern


def microscopy_position(
    rng: random.Random,
    width: int,
    height: int,
    recipe: dict[str, Any],
    index: int,
    count: int,
) -> tuple[int, int]:
    if recipe.get("edge_bias"):
        side = rng.choice(["left", "right", "top", "bottom"])
        if side == "left":
            return rng.randint(-max(1, width // 10), max(0, width // 12)), rng.randrange(0, height)
        if side == "right":
            return rng.randint(max(0, width - max(1, width // 12)), width + max(1, width // 10)), rng.randrange(0, height)
        if side == "top":
            return rng.randrange(0, width), rng.randint(-max(1, height // 10), max(0, height // 12))
        return rng.randrange(0, width), rng.randint(max(0, height - max(1, height // 12)), height + max(1, height // 10))

    if recipe.get("clustered"):
        center_x = int(recipe.get("cluster_x", width // 2))
        center_y = int(recipe.get("cluster_y", height // 2))
        spread_x = max(1, width // (5 if recipe.get("overlap") else 3))
        spread_y = max(1, height // (5 if recipe.get("overlap") else 3))
        return (
            int(np.clip(center_x + rng.randint(-spread_x, spread_x), 0, max(0, width - 1))),
            int(np.clip(center_y + rng.randint(-spread_y, spread_y), 0, max(0, height - 1))),
        )

    if count > 1 and rng.random() < 0.25:
        columns = max(1, int(np.ceil(np.sqrt(count))))
        rows = max(1, int(np.ceil(count / columns)))
        col = index % columns
        row = index // columns
        return (
            int((col + rng.uniform(0.25, 0.75)) * width / columns),
            int((row + rng.uniform(0.25, 0.75)) * height / rows),
        )

    return rng.randrange(0, width), rng.randrange(0, height)


def add_microscopy_debris(
    rng: random.Random,
    planes: list[np.ndarray],
    count: int,
    bright_value: float,
) -> None:
    height, width = planes[0].shape
    for _ in range(count):
        x = rng.randrange(0, width)
        y = rng.randrange(0, height)
        radius = rng.randint(1, max(1, min(width, height) // 80))
        value = rng.uniform(bright_value * 0.35, bright_value)
        channel = rng.randrange(0, len(planes))
        cv2.circle(planes[channel], (int(x), int(y)), int(radius), float(value), -1)


def make_microscopy_pattern_array(
    rng: random.Random,
    height: int,
    width: int,
    channels: int | None,
    pattern: str | None = None,
    recipe: dict[str, Any] | None = None,
) -> tuple[np.ndarray, str]:
    if recipe and recipe.get("scene") in MICROSCOPY_PATTERNS:
        pattern = str(recipe["scene"])
    pattern = pattern if pattern in MICROSCOPY_PATTERNS else rng.choice(MICROSCOPY_PATTERNS)
    recipe = copy.deepcopy(recipe) if recipe else draw_microscopy_recipe(
        rng,
        pattern,
        width,
        height,
        channels,
    )
    recipe.setdefault("scene", pattern)
    recipe, channels, _extension = enforce_microscopy_recipe_constraints(
        recipe,
        width,
        height,
        channels,
        str(recipe.get("extension", "png")),
    )
    pattern = str(recipe.get("scene", pattern))
    channel_count = max(1, min(4, int(channels or (2 if recipe.get("nuclei_channel") else 1))))

    background = float(recipe.get("background", rng.randint(8, 70)))
    noise_sigma = float(recipe.get("noise_sigma", rng.uniform(1.0, 12.0)))
    np_rng = rng_numpy(rng)
    planes = [
        np_rng.normal(background + index * rng.uniform(-4, 6), noise_sigma, size=(height, width)).astype(np.float32)
        for index in range(channel_count)
    ]

    if width > 1 and height > 1 and rng.random() < 0.55:
        x_grad = np.linspace(-rng.uniform(0, 18), rng.uniform(0, 18), width, dtype=np.float32)
        y_grad = np.linspace(-rng.uniform(0, 18), rng.uniform(0, 18), height, dtype=np.float32)[:, None]
        gradient = x_grad + y_grad
        planes = [plane + gradient for plane in planes]

    cell_count = max(0, int(recipe.get("cell_count", 1)))
    radius_min = max(1, int(recipe.get("radius_min", 1)))
    radius_max = max(radius_min, int(recipe.get("radius_max", max(1, min(width, height) // 8))))
    contrast = float(recipe.get("contrast", 100.0))
    if recipe.get("clustered"):
        recipe.setdefault("cluster_x", rng.randrange(0, width))
        recipe.setdefault("cluster_y", rng.randrange(0, height))

    explicit_cells = recipe.get("cells") if isinstance(recipe.get("cells"), list) else None
    for index in range(cell_count):
        if explicit_cells and index < len(explicit_cells):
            cell = explicit_cells[index]
            cx, cy = [int(value) for value in cell.get("center", [width // 2, height // 2])]
            rx, ry = [max(1, int(value)) for value in cell.get("radius", [radius_min, radius_min])]
            angle = int(cell.get("angle", 0))
            saturated = bool(cell.get("saturated", False))
        else:
            cx, cy = microscopy_position(rng, width, height, recipe, index, cell_count)
            rx = rng.randint(radius_min, radius_max)
            ry = rng.randint(radius_min, radius_max)
            if recipe.get("elongated"):
                if rng.random() < 0.5:
                    rx = max(rx, ry * rng.randint(2, 5))
                else:
                    ry = max(ry, rx * rng.randint(2, 5))
            if recipe.get("overlap"):
                rx = int(rx * rng.uniform(1.2, 1.9))
                ry = int(ry * rng.uniform(1.2, 1.9))
            angle = rng.randrange(0, 180)
            saturated = pattern == "saturated_cells"
        cytoplasm_value = 245 if saturated else background + contrast * rng.uniform(0.65, 1.25)
        nucleus_value = 255 if saturated else background + contrast * rng.uniform(0.9, 1.5)
        cv2.ellipse(
            planes[0],
            (int(cx), int(cy)),
            (max(1, int(rx)), max(1, int(ry))),
            float(angle),
            0,
            360,
            float(cytoplasm_value),
            -1,
        )
        if channel_count > 1:
            n_rx = max(1, int(rx * rng.uniform(0.28, 0.55)))
            n_ry = max(1, int(ry * rng.uniform(0.28, 0.55)))
            offset_x = rng.randint(-max(1, rx // 4), max(1, rx // 4))
            offset_y = rng.randint(-max(1, ry // 4), max(1, ry // 4))
            cv2.ellipse(
                planes[1],
                (int(cx + offset_x), int(cy + offset_y)),
                (n_rx, n_ry),
                float(angle + rng.randint(-25, 25)),
                0,
                360,
                float(nucleus_value),
                -1,
            )
        if pattern == "zstack_cells" and channel_count > 2:
            for channel_index in range(2, channel_count):
                shift = channel_index - 1
                cv2.ellipse(
                    planes[channel_index],
                    (int(cx + shift), int(cy - shift)),
                    (max(1, int(rx * 0.8)), max(1, int(ry * 0.8))),
                    float(angle),
                    0,
                    360,
                    float(cytoplasm_value * (0.7 + channel_index * 0.08)),
                    -1,
                )

    if recipe.get("debris"):
        debris_count = rng.randint(6, max(8, min(400, width * height // 70)))
        add_microscopy_debris(rng, planes, debris_count, background + contrast)

    blur = int(recipe.get("blur", 0) or 0)
    if blur >= 3 and blur % 2 == 1 and min(height, width) > 2:
        planes = [cv2.GaussianBlur(plane, (blur, blur), 0) for plane in planes]

    planes_u8 = [np.clip(plane, 0, 255).astype(np.uint8) for plane in planes]
    if channels is None:
        return planes_u8[0], pattern
    if channels == 1:
        return planes_u8[0][:, :, None], pattern
    return np.stack(planes_u8[:channel_count], axis=2), pattern


def mutate_stress_corpus_image(
    rng: random.Random,
    image: np.ndarray,
    max_side: int,
) -> tuple[np.ndarray, list[str]]:
    mutations: list[str] = []
    image = normalize_loaded_array(image)
    image = limit_channels_for_cv2(rng, image, mutations)

    if image.ndim not in (2, 3):
        image = image.reshape(image.shape[-2], image.shape[-1])
        mutations.append("reshape")

    if rng.random() < 0.65:
        image = random_crop(rng, image)
        mutations.append("crop")

    if rng.random() < 0.65:
        image = resize_mutation(rng, image, max_side)
        mutations.append("resize")

    if rng.random() < 0.5:
        image = np.flip(image, axis=rng.choice([0, 1])).copy()
        mutations.append("flip")

    if rng.random() < 0.35:
        image = np.rot90(image, rng.choice([1, 2, 3])).copy()
        mutations.append("rot90")

    if rng.random() < 0.2 and image.ndim == 2:
        image = image.T.copy()
        mutations.append("transpose")

    if rng.random() < 0.7:
        alpha = rng.uniform(0.3, 2.2)
        beta = rng.uniform(-80, 80)
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        mutations.append("brightness_contrast")

    if rng.random() < 0.45:
        sigma = rng.uniform(1, 35)
        noise = rng_numpy(rng).normal(0, sigma, size=image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        mutations.append("noise")

    if rng.random() < 0.25 and min(image.shape[:2]) > 2:
        image = cv2.GaussianBlur(image, (3, 3), 0)
        mutations.append("blur")

    if rng.random() < 0.2:
        image = 255 - image
        mutations.append("invert")

    if rng.random() < 0.2:
        threshold = rng.randrange(0, 256)
        image = np.where(image > threshold, 255, 0).astype(np.uint8)
        mutations.append("threshold")

    if image.ndim == 3:
        if rng.random() < 0.35:
            order = list(range(image.shape[2]))
            rng.shuffle(order)
            image = image[:, :, order]
            mutations.append("channel_shuffle")
        if rng.random() < 0.25:
            keep = rng.randrange(0, image.shape[2])
            out = np.zeros_like(image)
            out[:, :, keep] = image[:, :, keep]
            image = out
            mutations.append("single_channel")
        if image.shape[2] > 4:
            image = image[:, :, :4]
            mutations.append("truncate_channels")

    if rng.random() < 0.15:
        pad_h = rng.randint(0, max(1, image.shape[0] // 3))
        pad_w = rng.randint(0, max(1, image.shape[1] // 3))
        pad_width = ((pad_h, pad_h), (pad_w, pad_w))
        if image.ndim == 3:
            pad_width += ((0, 0),)
        image = np.pad(image, pad_width, mode=rng.choice(["constant", "edge"]))
        mutations.append("pad")

    image = enforce_max_side(image, max_side)
    return image.astype(np.uint8, copy=False), mutations or ["identity"]


def grayscale_projection(image: np.ndarray) -> np.ndarray:
    image = normalize_loaded_array(image)
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    if image.ndim == 3:
        return np.max(image[:, :, : min(4, image.shape[2])], axis=2).astype(np.uint8)
    return cv2_writable_image(image, "png")


def detect_cell_like_components(image: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    gray = grayscale_projection(image)
    if gray.size == 0 or min(gray.shape[:2]) <= 1:
        return []
    if int(gray.max()) == int(gray.min()):
        return []

    blurred = cv2.GaussianBlur(gray, (3, 3), 0) if min(gray.shape[:2]) > 2 else gray
    _threshold, mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    if mask.mean() > 160:
        mask = 255 - mask
    kernel = np.ones((3, 3), dtype=np.uint8)
    if min(mask.shape[:2]) > 4:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    image_area = int(gray.shape[0] * gray.shape[1])
    min_area = max(4, image_area // 20000)
    max_area = max(min_area + 1, image_area // 3)
    components: list[tuple[int, int, int, int, int]] = []
    for index in range(1, count):
        x, y, width, height, area = [int(value) for value in stats[index]]
        if area < min_area or area > max_area:
            continue
        if width <= 1 or height <= 1:
            continue
        aspect = max(width / max(1, height), height / max(1, width))
        if aspect > 8:
            continue
        fill = area / max(1, width * height)
        if fill < 0.08:
            continue
        components.append((x, y, width, height, area))
    return components


def crop_around_cell_like_component(
    rng: random.Random,
    image: np.ndarray,
    components: list[tuple[int, int, int, int, int]],
) -> np.ndarray:
    if not components:
        return image

    component = rng.choices(
        components,
        weights=[max(1, component[4]) for component in components],
        k=1,
    )[0]
    x, y, width, height, _area = component
    image_h, image_w = image.shape[:2]
    margin = int(round(max(width, height) * rng.uniform(0.8, 2.5)))
    crop_x0 = max(0, x - margin)
    crop_y0 = max(0, y - margin)
    crop_x1 = min(image_w, x + width + margin)
    crop_y1 = min(image_h, y + height + margin)

    if crop_x1 - crop_x0 < 8 and image_w >= 8:
        extra = (8 - (crop_x1 - crop_x0) + 1) // 2
        crop_x0 = max(0, crop_x0 - extra)
        crop_x1 = min(image_w, crop_x1 + extra)
    if crop_y1 - crop_y0 < 8 and image_h >= 8:
        extra = (8 - (crop_y1 - crop_y0) + 1) // 2
        crop_y0 = max(0, crop_y0 - extra)
        crop_y1 = min(image_h, crop_y1 + extra)

    if image.ndim == 2:
        return image[crop_y0:crop_y1, crop_x0:crop_x1].copy()
    return image[crop_y0:crop_y1, crop_x0:crop_x1, :].copy()


def resize_preserving_objects(
    rng: random.Random,
    image: np.ndarray,
    max_side: int,
) -> np.ndarray:
    height, width = image.shape[:2]
    if image.size == 0 or height <= 0 or width <= 0:
        return np.zeros((1, 1), dtype=np.uint8)
    max_side = max(1, max_side)
    target_long_choices = [
        value for value in [64, 96, 128, 192, 256, 320, max_side] if value <= max_side
    ] or [max_side]
    target_long = rng.choice(target_long_choices)
    scale = target_long / max(1, max(height, width))
    target_h = max(8 if height >= 8 else 1, min(max_side, int(round(height * scale))))
    target_w = max(8 if width >= 8 else 1, min(max_side, int(round(width * scale))))
    if target_h == height and target_w == width:
        return image
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(cv2_writable_image(image, "png"), (target_w, target_h), interpolation=interpolation)


def mutate_cell_preserving_corpus_image(
    rng: random.Random,
    image: np.ndarray,
    max_side: int,
) -> tuple[np.ndarray, list[str]]:
    mutations: list[str] = ["cell_preserving"]
    image = normalize_loaded_array(image)
    image = limit_channels_for_cv2(rng, image, mutations)
    components = detect_cell_like_components(image)

    if components and rng.random() < 0.9:
        image = crop_around_cell_like_component(rng, image, components)
        mutations.append("cell_crop")
    elif rng.random() < 0.25:
        image = random_crop(rng, image)
        mutations.append("crop")

    if rng.random() < 0.8:
        image = resize_preserving_objects(rng, image, max_side)
        mutations.append("object_preserving_resize")

    if rng.random() < 0.5:
        image = np.flip(image, axis=rng.choice([0, 1])).copy()
        mutations.append("flip")
    if rng.random() < 0.25:
        image = np.rot90(image, rng.choice([1, 2, 3])).copy()
        mutations.append("rot90")
    if rng.random() < 0.75:
        alpha = rng.uniform(0.75, 1.45)
        beta = rng.uniform(-28, 28)
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        mutations.append("mild_brightness_contrast")
    if rng.random() < 0.35:
        sigma = rng.uniform(1, 8)
        noise = rng_numpy(rng).normal(0, sigma, size=image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        mutations.append("mild_noise")
    if rng.random() < 0.15 and min(image.shape[:2]) > 2:
        image = cv2.GaussianBlur(image, (3, 3), 0)
        mutations.append("mild_blur")
    if rng.random() < 0.08:
        image = 255 - image
        mutations.append("invert")

    image = enforce_max_side(image, max_side)
    return image.astype(np.uint8, copy=False), mutations


def mutate_corpus_image(
    rng: random.Random,
    image: np.ndarray,
    max_side: int,
    mode: str = "mixed",
) -> tuple[np.ndarray, list[str]]:
    mode = mode if mode in CORPUS_MUTATION_MODES else "mixed"
    if mode == "stress":
        return mutate_stress_corpus_image(rng, image, max_side)
    if mode == "cell-preserving":
        return mutate_cell_preserving_corpus_image(rng, image, max_side)
    if rng.random() < 0.65 and detect_cell_like_components(normalize_loaded_array(image)):
        return mutate_cell_preserving_corpus_image(rng, image, max_side)
    return mutate_stress_corpus_image(rng, image, max_side)


def random_crop(rng: random.Random, image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    if height <= 1 or width <= 1:
        return image
    crop_h = rng.randint(1, height)
    crop_w = rng.randint(1, width)
    y = rng.randint(0, height - crop_h)
    x = rng.randint(0, width - crop_w)
    if image.ndim == 2:
        return image[y:y + crop_h, x:x + crop_w].copy()
    return image[y:y + crop_h, x:x + crop_w, :].copy()


def resize_mutation(rng: random.Random, image: np.ndarray, max_side: int) -> np.ndarray:
    current_h, current_w = image.shape[:2]
    if image.size == 0 or current_h <= 0 or current_w <= 0:
        return np.zeros((1, 1), dtype=np.uint8)
    max_side = max(1, max_side)
    target_h = rng.choice([1, 2, 3, 7, 16, 31, 32, 64, 96, 128, max_side])
    target_w = rng.choice([1, 2, 3, 7, 16, 31, 32, 64, 96, 128, max_side])
    target_h = max(1, min(max_side, target_h))
    target_w = max(1, min(max_side, target_w))
    interpolation = rng.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA])
    if current_h == target_h and current_w == target_w:
        return image
    try:
        return cv2.resize(image, (target_w, target_h), interpolation=interpolation)
    except cv2.error:
        writable = cv2_writable_image(image, "png")
        if writable.size == 0:
            return np.zeros((target_h, target_w), dtype=np.uint8)
        return cv2.resize(writable, (target_w, target_h), interpolation=interpolation)


def enforce_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    if image.size == 0 or height <= 0 or width <= 0:
        return np.zeros((1, 1), dtype=np.uint8)
    max_current = max(height, width)
    if max_current <= max_side:
        return image
    scale = max_side / max_current
    target_h = max(1, int(round(height * scale)))
    target_w = max(1, int(round(width * scale)))
    try:
        return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    except cv2.error:
        writable = cv2_writable_image(image, "png")
        if writable.size == 0:
            return np.zeros((target_h, target_w), dtype=np.uint8)
        return cv2.resize(writable, (target_w, target_h), interpolation=cv2.INTER_AREA)


def channelize_image(
    rng: random.Random,
    image_2d: np.ndarray,
    channels: int | None,
) -> np.ndarray:
    if channels is None:
        return image_2d

    if channels == 1:
        return image_2d[:, :, None]

    mode = rng.choice(["repeat", "single_channel", "jittered"])
    if mode == "single_channel":
        image = np.zeros((*image_2d.shape, channels), dtype=np.uint8)
        image[:, :, rng.randrange(0, channels)] = image_2d
        return image

    if mode == "jittered":
        planes = []
        for _ in range(channels):
            offset = rng.randrange(-30, 31)
            planes.append(np.clip(image_2d.astype(np.int16) + offset, 0, 255).astype(np.uint8))
        return np.stack(planes, axis=2)

    return np.repeat(image_2d[:, :, None], channels, axis=2)


def rng_numpy(rng: random.Random) -> np.random.Generator:
    return np.random.default_rng(rng.randrange(0, 2**32))


def draw_spots(rng: random.Random, image: np.ndarray) -> None:
    height, width = image.shape[:2]
    count = rng.randint(1, 12)
    yy, xx = np.ogrid[:height, :width]
    for _ in range(count):
        radius = rng.randint(1, max(1, min(height, width) // 6))
        cx = rng.randrange(0, width)
        cy = rng.randrange(0, height)
        value = rng.randrange(80, 256)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        image[mask] = value


def write_generated_image(
    rng: random.Random,
    case_dir: Path,
    model_type: str,
    profile: str,
    max_side: int,
    corpus_images: list[Path],
    corpus_probability: float,
    corpus_mutation_mode: str,
) -> tuple[Path, dict[str, Any]]:
    if corpus_images and rng.random() < corpus_probability:
        source = rng.choice(corpus_images)
        loaded_source = read_corpus_image_with_metadata(source)
        if loaded_source is not None:
            source_image, source_meta = loaded_source
            return write_corpus_mutation(
                rng,
                case_dir,
                source,
                source_image,
                max_side,
                corpus_mutation_mode,
                source_meta,
            )

    profile = choose_effective_profile(rng, profile, bool(corpus_images))
    if profile == "corpus":
        return write_generated_image(
            rng,
            case_dir,
            model_type,
            "generic",
            max_side,
            corpus_images,
            1.0,
            corpus_mutation_mode,
        )

    height, width = choose_dimensions(rng, profile, max_side)
    extension_choices = (
        ["png", "jpg", "tif", "lsm"]
        if profile in {"yolo", "microscopy"}
        else ["png", "tif", "lsm"]
    )
    extension = rng.choice(extension_choices)

    if extension == "lsm":
        channels = rng.randint(2, 4)
        if profile == "microscopy":
            pattern = rng.choice(MICROSCOPY_PATTERNS)
            recipe = draw_microscopy_recipe(rng, pattern, width, height, channels)
            channels_last, pattern = make_microscopy_pattern_array(
                rng,
                height,
                width,
                channels,
                pattern=pattern,
                recipe=recipe,
            )
            image = np.transpose(channels_last, (2, 0, 1))
            patterns = [pattern]
        else:
            channel_arrays = []
            patterns = []
            for _ in range(channels):
                if profile == "yolo":
                    channel, pattern = make_yolo_pattern_array(rng, height, width, None)
                else:
                    channel, pattern = make_pattern_array(rng, height, width, None)
                channel_arrays.append(channel)
                patterns.append(pattern)
            image = np.stack(channel_arrays, axis=0)
            recipe = {}
        image_path = case_dir / "input.lsm"
        tifffile.imwrite(str(image_path), image, metadata={"axes": "CYX"})
        meta = {
            "extension": extension,
            "profile": profile,
            "shape": list(image.shape),
            "patterns": patterns,
            "cell_channel": rng.randrange(0, channels),
            "nuclei_channel": rng.randrange(0, channels),
        }
        if recipe:
            meta["recipe"] = recipe
        return image_path, meta

    channels = rng.choice([None, 3])
    if profile == "yolo":
        image, pattern = make_yolo_pattern_array(rng, height, width, channels)
        recipe = {}
    elif profile == "microscopy":
        pattern = rng.choice(MICROSCOPY_PATTERNS)
        recipe = draw_microscopy_recipe(rng, pattern, width, height, channels)
        image, pattern = make_microscopy_pattern_array(
            rng,
            height,
            width,
            channels,
            pattern=pattern,
            recipe=recipe,
        )
    else:
        image, pattern = make_pattern_array(rng, height, width, channels)
        recipe = {}
    image_path = case_dir / f"input.{extension}"
    if extension in {"png", "jpg"}:
        if not cv2.imwrite(str(image_path), image):
            raise RuntimeError(f"Could not write generated image: {image_path}")
    else:
        tifffile.imwrite(str(image_path), image)

    meta = {
        "extension": extension,
        "profile": profile,
        "shape": list(image.shape),
        "patterns": [pattern],
        "cell_channel": 0,
        "nuclei_channel": 1,
    }
    if recipe:
        meta["recipe"] = recipe
        channels_written = non_lsm_channel_count(image)
        meta["nuclei_channel"] = 0 if channels_written == 1 else min(1, channels_written - 1)
    return image_path, meta


def write_corpus_mutation(
    rng: random.Random,
    case_dir: Path,
    source: Path,
    source_image: np.ndarray,
    max_side: int,
    corpus_mutation_mode: str = "mixed",
    source_meta: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    image, mutations = mutate_corpus_image(
        rng,
        source_image,
        max_side,
        mode=corpus_mutation_mode,
    )
    extension = rng.choice(["png", "jpg", "tif", "lsm"])

    if extension == "lsm":
        if image.ndim == 2:
            channels_last = channelize_image(rng, image, rng.choice([2, 3, 4]))
        elif image.shape[2] == 1:
            channels_last = np.repeat(image, 2, axis=2)
        else:
            channels_last = image[:, :, :min(4, image.shape[2])]
        image_to_write = np.transpose(channels_last, (2, 0, 1))
        image_path = case_dir / "input.lsm"
        tifffile.imwrite(str(image_path), image_to_write, metadata={"axes": "CYX"})
        channels = image_to_write.shape[0]
        shape = list(image_to_write.shape)
    else:
        image_path = case_dir / f"input.{extension}"
        if extension in {"png", "jpg"}:
            image_to_write = cv2_writable_image(image, extension)
            if not cv2.imwrite(str(image_path), image_to_write):
                raise RuntimeError(f"Could not write corpus mutation: {image_path}")
            image = image_to_write
        else:
            tifffile.imwrite(str(image_path), image)
        channels = image.shape[2] if image.ndim == 3 else 1
        shape = list(image.shape)

    return image_path, {
        "extension": extension,
        "profile": "corpus",
        "shape": shape,
        "patterns": mutations,
        "source": str(source),
        "source_meta": source_meta or {},
        "cell_channel": rng.randrange(0, max(1, channels)),
        "nuclei_channel": rng.randrange(0, max(1, channels)),
    }


def write_spec_image(
    rng: random.Random,
    case_dir: Path,
    spec: FuzzCaseSpec,
) -> tuple[Path, dict[str, Any]]:
    if spec.image_kind == "yolo":
        profile = "yolo"
    elif spec.image_kind == "microscopy":
        profile = "microscopy"
    else:
        profile = "generic"
    extension = canonical_minimizer_extension(spec.extension)
    height = max(1, int(spec.height))
    width = max(1, int(spec.width))

    if extension == "lsm":
        channels = max(2, min(4, int(spec.channels or 2)))
        if profile == "microscopy":
            channels_last, pattern = make_microscopy_pattern_array(
                rng,
                height,
                width,
                channels,
                pattern=spec.pattern,
                recipe=spec.image_recipe,
            )
            image = np.transpose(channels_last, (2, 0, 1))
        else:
            channel_arrays = []
            for _ in range(channels):
                if profile == "yolo":
                    channel, pattern = make_yolo_pattern_array(
                        rng,
                        height,
                        width,
                        None,
                        pattern=spec.pattern,
                    )
                else:
                    channel, pattern = make_pattern_array(
                        rng,
                        height,
                        width,
                        None,
                        pattern=spec.pattern,
                    )
                channel_arrays.append(channel)
            image = np.stack(channel_arrays, axis=0)
        image_path = case_dir / "input.lsm"
        tifffile.imwrite(str(image_path), image, metadata={"axes": "CYX"})
        meta = {
            "extension": "lsm",
            "profile": f"strategy:{profile}",
            "shape": list(image.shape),
            "patterns": [pattern],
            "cell_channel": rng.randrange(0, channels),
            "nuclei_channel": rng.randrange(0, channels),
        }
        if spec.image_recipe:
            meta["recipe"] = copy.deepcopy(spec.image_recipe)
        return image_path, meta

    channels = spec.channels
    if profile == "yolo":
        image, pattern = make_yolo_pattern_array(
            rng,
            height,
            width,
            channels,
            pattern=spec.pattern,
        )
    elif profile == "microscopy":
        image, pattern = make_microscopy_pattern_array(
            rng,
            height,
            width,
            channels,
            pattern=spec.pattern,
            recipe=spec.image_recipe,
        )
    else:
        image, pattern = make_pattern_array(
            rng,
            height,
            width,
            channels,
            pattern=spec.pattern,
        )

    image_path = case_dir / f"input.{extension}"
    if extension in {"png", "jpg", "bmp"}:
        image_to_write = cv2_writable_image(image, extension)
        if not cv2.imwrite(str(image_path), image_to_write):
            raise RuntimeError(f"Could not write spec image: {image_path}")
        image = image_to_write
    elif extension == "tif":
        tifffile.imwrite(str(image_path), image)
    else:
        image_path = case_dir / "input.png"
        image = cv2_writable_image(image, "png")
        if not cv2.imwrite(str(image_path), image):
            raise RuntimeError(f"Could not write spec image: {image_path}")
        extension = "png"

    channels_written = non_lsm_channel_count(image)
    return image_path, {
        "extension": extension,
        "profile": f"strategy:{profile}",
        "shape": list(image.shape),
        "patterns": [pattern],
        "recipe": copy.deepcopy(spec.image_recipe),
        "cell_channel": 0,
        "nuclei_channel": 0 if channels_written == 1 else min(1, channels_written - 1),
    }


def image_meta_dimensions(image_meta: dict[str, Any]) -> tuple[int, int, int | None]:
    shape = list(image_meta.get("shape", []))
    if len(shape) >= 3 and image_meta.get("extension") == "lsm":
        return int(shape[-1]), int(shape[-2]), int(shape[0])
    if len(shape) >= 3:
        return int(shape[1]), int(shape[0]), int(shape[2])
    if len(shape) >= 2:
        return int(shape[1]), int(shape[0]), None
    if len(shape) == 1:
        return int(shape[0]), 1, None
    return 1, 1, None


def case_spec_from_case(
    model_name: str,
    image_meta: dict[str, Any],
    object_size: dict[str, Any],
    workflow: dict[str, Any],
) -> FuzzCaseSpec:
    width, height, channels = image_meta_dimensions(image_meta)
    patterns = list(image_meta.get("patterns", []))
    return FuzzCaseSpec(
        model_name=model_name,
        image_kind=str(image_meta.get("profile", "unknown")),
        width=width,
        height=height,
        channels=channels,
        pattern="+".join(str(pattern) for pattern in patterns) or "unknown",
        extension=str(image_meta.get("extension", "unknown")),
        object_size=copy.deepcopy(object_size),
        workflow=copy.deepcopy(list(workflow.get("steps", []))),
        image_recipe=copy.deepcopy(dict(image_meta.get("recipe", {}))),
    )


def cv2_writable_image(image: np.ndarray, extension: str) -> np.ndarray:
    """Return an array with a channel count accepted by cv2.imwrite."""
    image = normalize_loaded_array(np.asarray(image))

    if image.ndim == 1:
        return image.reshape(1, -1)

    if image.ndim == 2:
        return image

    if image.ndim > 3:
        image = image.reshape((*image.shape[-2:], -1))

    channels = image.shape[2]
    if channels == 1:
        return image[:, :, 0]
    if channels == 2:
        zero = np.zeros(image.shape[:2] + (1,), dtype=image.dtype)
        return np.concatenate([image, zero], axis=2)
    if extension == "jpg" and channels >= 4:
        return image[:, :, :3]
    if channels in (3, 4):
        return image
    if channels > 4:
        return image[:, :, :4]

    return image.reshape(image.shape[0], image.shape[1])


def build_case(
    rng: random.Random,
    case_id: int,
    case_dir: Path,
    model_name: str,
    model_data: dict[str, Any],
    image_profile: str,
    generation_engine: str,
    allowed_scales: list[int],
    max_side: int,
    corpus_images: list[Path],
    corpus_probability: float,
    corpus_mutation_mode: str,
    workflow: str,
    max_workflow_steps: int,
    grammar_spec_corpus: list[FuzzCaseSpec] | None = None,
) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    generation_engine = effective_generation_engine(generation_engine)
    use_corpus = bool(corpus_images) and rng.random() < corpus_probability
    if generation_engine in {"strategy", "grammar-mutational"} and not use_corpus:
        if generation_engine == "grammar-mutational":
            spec = draw_grammar_case_spec(
                rng,
                model_name,
                image_profile,
                model_data,
                allowed_scales,
                workflow,
                max_workflow_steps,
                max_side,
                grammar_spec_corpus,
            )
        else:
            spec = draw_strategy_case_spec(
                rng,
                model_name,
                image_profile,
                model_data,
                allowed_scales,
                workflow,
                max_workflow_steps,
                max_side,
            )
        image_path, image_meta = write_spec_image(rng, case_dir, spec)
        if generation_engine == "grammar-mutational":
            image_meta["profile"] = image_meta.get("profile", "").replace(
                "strategy:",
                "grammar:",
                1,
            )
        object_size = spec.object_size
        workflow_data = workflow_from_steps(workflow, spec.workflow)
    else:
        resolved_profile = "corpus" if use_corpus else resolve_image_profile(image_profile, model_data)
        image_path, image_meta = write_generated_image(
            rng,
            case_dir,
            str(model_data.get("model_type", "")),
            resolved_profile,
            max_side,
            corpus_images,
            1.0 if use_corpus else 0.0,
            corpus_mutation_mode,
        )
        object_size = make_object_size(rng, model_data, allowed_scales)
        workflow_data = make_workflow(rng, workflow, max_workflow_steps)
        spec = case_spec_from_case(model_name, image_meta, object_size, workflow_data)

    return {
        "case_id": case_id,
        "model_name": model_name,
        "model_data": model_data,
        "image_path": str(image_path.resolve()),
        "image": image_meta,
        "object_size": object_size,
        "workflow": workflow_data,
        "case_spec": spec.to_jsonable(),
    }


def build_generation_error_case(
    case_id: int,
    model_name: str,
    model_data: dict[str, Any],
    workflow: str,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "model_name": model_name,
        "model_data": model_data,
        "image_path": "",
        "image": {
            "extension": "none",
            "profile": "generation-error",
            "shape": [],
            "patterns": ["generation_error"],
        },
        "object_size": {},
        "workflow": {"mode": workflow, "steps": []},
        "case_spec": {
            "model_name": model_name,
            "image_kind": "generation-error",
            "width": 0,
            "height": 0,
            "channels": None,
            "pattern": "generation_error",
            "extension": "none",
            "object_size": {},
            "workflow": [],
            "image_recipe": {},
        },
    }


