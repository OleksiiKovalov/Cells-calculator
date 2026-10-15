from .config import *

def register_known_models() -> None:
    from UI.app_globals import register_model

    for model_type, model_class in KNOWN_MODELS.items():
        register_model(model_type, model_class, False)


def sanitizer_includes(sanitizer: str, feature: str) -> bool:
    return sanitizer == "all" or sanitizer == feature


def configure_child_sanitizers() -> None:
    sanitizer = os.environ.get("CELLS_FUZZ_SANITIZERS", "none")
    if sanitizer_includes(sanitizer, "python"):
        try:
            import faulthandler

            faulthandler.enable(all_threads=True)
        except Exception:
            pass
    if sanitizer_includes(sanitizer, "warnings"):
        warnings.simplefilter("error", RuntimeWarning)
        warnings.simplefilter("error", ResourceWarning)
        encoding_warning = getattr(__builtins__, "EncodingWarning", None)
        if encoding_warning is not None:
            warnings.simplefilter("error", encoding_warning)
    if sanitizer_includes(sanitizer, "numpy"):
        np.seterr(all="raise")


def current_oracle_level() -> str:
    level = os.environ.get("CELLS_FUZZ_ORACLE_LEVEL", "strict")
    return level if level in ORACLE_LEVELS else "strict"


def current_determinism_check() -> str:
    mode = os.environ.get("CELLS_FUZZ_DETERMINISM_CHECK", "off")
    return mode if mode in DETERMINISM_CHECKS else "off"


def current_python_coverage() -> str:
    mode = os.environ.get("CELLS_FUZZ_PYTHON_COVERAGE", "off")
    return mode if mode in PYTHON_COVERAGE_MODES else "off"


def current_mock_mode() -> str:
    mode = os.environ.get("CELLS_FUZZ_MOCK_MODE", "none")
    return mode if mode in MOCK_MODES else "none"


MCDC_CONDITION_HELPER = "__cells_mcdc_cond__"
MCDC_DECISION_HELPER = "__cells_mcdc_decision__"
MCDC_MODULE_PREFIXES = ("model.", "UI.")
MCDC_OBSERVATION_LIMIT = 512
_MCDC_THREAD_STATE = threading.local()
_ACTIVE_MCDC_STATE: dict[str, Any] | None = None


def _mcdc_bool_label(value: bool) -> str:
    return "T" if value else "F"


def _mcdc_pending_conditions() -> dict[str, list[tuple[int, str, bool]]]:
    pending = getattr(_MCDC_THREAD_STATE, "pending_conditions", None)
    if pending is None:
        pending = {}
        _MCDC_THREAD_STATE.pending_conditions = pending
    return pending


def _mcdc_condition_label(node: ast.AST) -> str:
    try:
        label = ast.unparse(node)
    except Exception:
        label = type(node).__name__
    label = " ".join(label.split())
    return label[:160] if label else type(node).__name__


def _mcdc_relative_path(filename: str) -> str:
    try:
        return Path(filename).resolve().relative_to(ROOT).as_posix()
    except (OSError, ValueError):
        return str(filename).replace("\\", "/")


def _mcdc_record_observation(
    state: dict[str, Any],
    decision_id: str,
    outcome: bool,
    conditions: list[tuple[int, str, bool]],
) -> None:
    decisions: dict[str, dict[str, Any]] = state["decisions"]
    signals: set[str] = state["signals"]
    decision = decisions.setdefault(
        decision_id,
        {
            "conditions": {},
            "condition_values": {},
            "mcdc": set(),
            "observation_keys": set(),
            "observations": [],
            "outcomes": set(),
        },
    )
    decision["outcomes"].add(outcome)
    signals.add(f"mcdc:D:{decision_id}:{_mcdc_bool_label(outcome)}")

    evaluated: dict[int, bool] = {}
    for condition_index, label, value in conditions:
        evaluated[condition_index] = value
        decision["conditions"].setdefault(condition_index, label)
        condition_values = decision["condition_values"].setdefault(condition_index, set())
        condition_values.add(value)
        signals.add(
            f"mcdc:C:{decision_id}:{condition_index}:{_mcdc_bool_label(value)}"
        )

    values_key = tuple(sorted(evaluated.items()))
    evaluated_key = ",".join(
        f"{index}={_mcdc_bool_label(value)}" for index, value in values_key
    )
    signals.add(f"mcdc:E:{decision_id}:{evaluated_key}:{_mcdc_bool_label(outcome)}")

    observation_key = (values_key, outcome)
    if observation_key not in decision["observation_keys"]:
        decision["observation_keys"].add(observation_key)
        if len(decision["observations"]) < MCDC_OBSERVATION_LIMIT:
            decision["observations"].append({"values": values_key, "outcome": outcome})

    for previous in decision["observations"]:
        previous_values = dict(previous["values"])
        common_conditions = set(previous_values) & set(evaluated)
        if not common_conditions:
            continue
        if previous["outcome"] == outcome:
            continue
        differing = [
            index for index in common_conditions if previous_values[index] != evaluated[index]
        ]
        if len(differing) == 1:
            condition_index = differing[0]
            decision["mcdc"].add(condition_index)
            signals.add(f"mcdc:M:{decision_id}:{condition_index}")


def __cells_mcdc_cond__(
    decision_id: str,
    condition_index: int,
    label: str,
    value: Any,
) -> Any:
    bool_value = bool(value)
    pending = _mcdc_pending_conditions()
    pending.setdefault(decision_id, []).append(
        (int(condition_index), str(label), bool_value)
    )
    return value


def __cells_mcdc_decision__(decision_id: str, value: Any) -> bool:
    outcome = bool(value)
    pending = _mcdc_pending_conditions()
    conditions = pending.pop(decision_id, [])
    state = _ACTIVE_MCDC_STATE
    if state is not None:
        lock = state.get("lock")
        if lock is None:
            _mcdc_record_observation(state, decision_id, outcome, conditions)
        else:
            with lock:
                _mcdc_record_observation(state, decision_id, outcome, conditions)
    return outcome


class MCDCConditionInstrumenter:
    def __init__(self, decision_id: str) -> None:
        self.decision_id = decision_id
        self.condition_index = 0

    def instrument(self, node: ast.AST) -> ast.AST:
        if isinstance(node, ast.BoolOp):
            node.values = [self.instrument(value) for value in node.values]
            return node
        return self._wrap_condition(node)

    def _wrap_condition(self, node: ast.AST) -> ast.AST:
        condition_index = self.condition_index
        self.condition_index += 1
        wrapped = ast.Call(
            func=ast.Name(id=MCDC_CONDITION_HELPER, ctx=ast.Load()),
            args=[
                ast.Constant(self.decision_id),
                ast.Constant(condition_index),
                ast.Constant(_mcdc_condition_label(node)),
                node,
            ],
            keywords=[],
        )
        return ast.copy_location(wrapped, node)


class MCDCTransformer(ast.NodeTransformer):
    def __init__(self, filename: str) -> None:
        self.filename = _mcdc_relative_path(filename)

    def visit_If(self, node: ast.If) -> ast.AST:
        node.body = [self.visit(child) for child in node.body]
        node.orelse = [self.visit(child) for child in node.orelse]
        node.test = self._instrument_decision(node.test, "If")
        return node

    def visit_While(self, node: ast.While) -> ast.AST:
        node.body = [self.visit(child) for child in node.body]
        node.orelse = [self.visit(child) for child in node.orelse]
        node.test = self._instrument_decision(node.test, "While")
        return node

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        node.body = self.visit(node.body)
        node.orelse = self.visit(node.orelse)
        node.test = self._instrument_decision(node.test, "IfExp")
        return node

    def visit_Assert(self, node: ast.Assert) -> ast.AST:
        if node.msg is not None:
            node.msg = self.visit(node.msg)
        node.test = self._instrument_decision(node.test, "Assert")
        return node

    def _instrument_decision(self, test: ast.AST, kind: str) -> ast.AST:
        decision_id = (
            f"{self.filename}:{getattr(test, 'lineno', 0)}:"
            f"{getattr(test, 'col_offset', 0)}:{kind}"
        )
        instrumented_test = MCDCConditionInstrumenter(decision_id).instrument(test)
        wrapped = ast.Call(
            func=ast.Name(id=MCDC_DECISION_HELPER, ctx=ast.Load()),
            args=[ast.Constant(decision_id), instrumented_test],
            keywords=[],
        )
        return ast.copy_location(wrapped, test)


class MCDCInstrumentingLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname: str) -> Any:
        source_path = self.get_filename(fullname)
        source_bytes = self.get_data(source_path)
        return self.source_to_code(source_bytes, source_path)

    def source_to_code(
        self,
        data: bytes | str,
        path: str,
        *,
        _optimize: int = -1,
    ) -> Any:
        source = data.decode("utf-8") if isinstance(data, bytes) else data
        tree = ast.parse(source, filename=path)
        tree = MCDCTransformer(path).visit(tree)
        ast.fix_missing_locations(tree)
        return compile(tree, path, "exec", dont_inherit=True, optimize=_optimize)


class MCDCInstrumentingFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: Any = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if not fullname.startswith(MCDC_MODULE_PREFIXES):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not spec.origin or not spec.origin.endswith(".py"):
            return None
        try:
            origin = Path(spec.origin).resolve()
        except OSError:
            return None
        include_roots = (ROOT / "model", ROOT / "UI")
        if not any(origin == root or root in origin.parents for root in include_roots):
            return None
        spec.loader = MCDCInstrumentingLoader(fullname, str(origin))
        return spec


def start_python_line_coverage(mode: str) -> dict[str, Any]:
    state: dict[str, Any] = {"enabled": False}
    if mode == "off":
        return state

    include_roots = (ROOT / "model", ROOT / "UI")
    hits: set[str] = set()
    filename_cache: dict[str, str | None] = {}
    previous_trace = sys.gettrace()
    previous_thread_trace = threading.gettrace()

    def relative_traced_file(filename: str) -> str | None:
        if filename in filename_cache:
            return filename_cache[filename]
        try:
            path = Path(filename).resolve()
        except OSError:
            filename_cache[filename] = None
            return None
        if not any(path == root or root in path.parents for root in include_roots):
            filename_cache[filename] = None
            return None
        try:
            rel = path.relative_to(ROOT).as_posix()
        except ValueError:
            filename_cache[filename] = None
            return None
        filename_cache[filename] = rel
        return rel

    def tracer(frame, event, arg):
        if event == "line":
            rel = relative_traced_file(frame.f_code.co_filename)
            if rel is not None:
                hits.add(f"{rel}:{frame.f_lineno}")
        if previous_trace is not None:
            previous_trace(frame, event, arg)
        return tracer

    sys.settrace(tracer)
    threading.settrace(tracer)
    state.update(
        {
            "enabled": True,
            "hits": hits,
            "previous_trace": previous_trace,
            "previous_thread_trace": previous_thread_trace,
        }
    )
    return state


def stop_python_line_coverage(state: dict[str, Any]) -> dict[str, Any]:
    if not state.get("enabled"):
        return {}
    sys.settrace(state.get("previous_trace"))
    threading.settrace(state.get("previous_thread_trace"))
    lines = sorted(state.get("hits", set()))
    files = Counter(line.rsplit(":", 1)[0] for line in lines)
    return {
        "python_coverage": {
            "mode": "model",
            "line_count": len(lines),
            "file_count": len(files),
            "files": dict(files.most_common()),
            "lines": lines,
        }
    }


def start_python_mcdc_coverage(
    mode: str,
    keep_runtime_helpers: bool = False,
) -> dict[str, Any]:
    global _ACTIVE_MCDC_STATE
    state: dict[str, Any] = {"enabled": False}
    if mode != "mcdc":
        return state

    finder = MCDCInstrumentingFinder()
    previous_builtins = {
        MCDC_CONDITION_HELPER: (
            hasattr(builtins, MCDC_CONDITION_HELPER),
            getattr(builtins, MCDC_CONDITION_HELPER, None),
        ),
        MCDC_DECISION_HELPER: (
            hasattr(builtins, MCDC_DECISION_HELPER),
            getattr(builtins, MCDC_DECISION_HELPER, None),
        ),
    }
    setattr(builtins, MCDC_CONDITION_HELPER, __cells_mcdc_cond__)
    setattr(builtins, MCDC_DECISION_HELPER, __cells_mcdc_decision__)

    state.update(
        {
            "enabled": True,
            "mode": "mcdc",
            "finder": finder,
            "previous_builtins": previous_builtins,
            "decisions": {},
            "signals": set(),
            "lock": threading.Lock(),
            "keep_runtime_helpers": keep_runtime_helpers,
        }
    )
    _ACTIVE_MCDC_STATE = state
    sys.meta_path.insert(0, finder)
    return state


def stop_python_mcdc_coverage(state: dict[str, Any]) -> dict[str, Any]:
    global _ACTIVE_MCDC_STATE
    if not state.get("enabled"):
        return {}

    finder = state.get("finder")
    if finder in sys.meta_path:
        sys.meta_path.remove(finder)

    if not state.get("keep_runtime_helpers"):
        for name, (had_value, previous_value) in state.get("previous_builtins", {}).items():
            if had_value:
                setattr(builtins, name, previous_value)
            elif hasattr(builtins, name):
                delattr(builtins, name)
    if _ACTIVE_MCDC_STATE is state:
        _ACTIVE_MCDC_STATE = None

    decisions = state.get("decisions", {})
    serialized_decisions: dict[str, dict[str, Any]] = {}
    total_conditions = 0
    satisfied_conditions = 0
    full_mcdc_decisions = 0
    for decision_id, decision in sorted(decisions.items()):
        conditions = decision.get("conditions", {})
        mcdc_conditions = decision.get("mcdc", set())
        condition_count = len(conditions)
        total_conditions += condition_count
        satisfied_conditions += len(mcdc_conditions)
        if condition_count and len(mcdc_conditions) >= condition_count:
            full_mcdc_decisions += 1
        serialized_decisions[decision_id] = {
            "condition_count": condition_count,
            "conditions": {
                str(index): str(label)
                for index, label in sorted(conditions.items())
            },
            "mcdc_conditions": sorted(int(index) for index in mcdc_conditions),
            "mcdc_condition_count": len(mcdc_conditions),
            "observation_count": len(decision.get("observation_keys", set())),
            "outcomes": [
                _mcdc_bool_label(value)
                for value in sorted(decision.get("outcomes", set()))
            ],
        }

    signals = sorted(str(signal) for signal in state.get("signals", set()))
    return {
        "python_coverage": {
            "mode": "mcdc",
            "signal_count": len(signals),
            "decision_count": len(serialized_decisions),
            "condition_count": total_conditions,
            "satisfied_condition_count": satisfied_conditions,
            "full_mcdc_decision_count": full_mcdc_decisions,
            "decisions": serialized_decisions,
            "signals": signals,
        }
    }


def start_python_coverage(
    mode: str,
    keep_runtime_helpers: bool = False,
) -> dict[str, Any]:
    if mode == "mcdc":
        return start_python_mcdc_coverage(mode, keep_runtime_helpers=keep_runtime_helpers)
    return start_python_line_coverage(mode)


def stop_python_coverage(state: dict[str, Any]) -> dict[str, Any]:
    if state.get("mode") == "mcdc":
        return stop_python_mcdc_coverage(state)
    return stop_python_line_coverage(state)


def current_handle_count() -> int | None:
    try:
        import psutil
    except ImportError:
        return None

    process = psutil.Process(os.getpid())
    if hasattr(process, "num_handles"):
        try:
            return int(process.num_handles())
        except Exception:
            return None
    if hasattr(process, "num_fds"):
        try:
            return int(process.num_fds())
        except Exception:
            return None
    return None


def current_allocated_blocks() -> int | None:
    getallocatedblocks = getattr(sys, "getallocatedblocks", None)
    if getallocatedblocks is None:
        return None
    try:
        return int(getallocatedblocks())
    except Exception:
        return None


def leak_probe_enabled(
    max_tracemalloc_mb: float,
    max_handle_growth: int,
    max_allocated_block_growth: int,
) -> bool:
    sanitizer = os.environ.get("CELLS_FUZZ_SANITIZERS", "none")
    return (
        sanitizer_includes(sanitizer, "leaks")
        or max_tracemalloc_mb > 0
        or max_handle_growth > 0
        or max_allocated_block_growth > 0
    )


def start_sanitizer_probe(
    max_tracemalloc_mb: float,
    max_handle_growth: int,
    max_allocated_block_growth: int,
) -> dict[str, Any]:
    state: dict[str, Any] = {"enabled": False}
    if not leak_probe_enabled(
        max_tracemalloc_mb,
        max_handle_growth,
        max_allocated_block_growth,
    ):
        return state

    state["enabled"] = True
    state["tracemalloc_started_here"] = False
    state["tracemalloc_enabled"] = max_tracemalloc_mb > 0
    if state["tracemalloc_enabled"]:
        if not tracemalloc.is_tracing():
            tracemalloc.start(25)
            state["tracemalloc_started_here"] = True
        tracemalloc.reset_peak()
    state["handles_start"] = current_handle_count()
    state["blocks_start"] = current_allocated_blocks()
    return state


def stop_sanitizer_probe(
    state: dict[str, Any],
    max_tracemalloc_mb: float,
    max_handle_growth: int,
    max_allocated_block_growth: int,
) -> dict[str, Any]:
    if not state.get("enabled"):
        return {}

    gc.collect()
    metrics: dict[str, Any] = {}
    try:
        if state.get("tracemalloc_enabled") and tracemalloc.is_tracing():
            _current_bytes, peak_bytes = tracemalloc.get_traced_memory()
            peak_mb = peak_bytes / (1024 * 1024)
            metrics["tracemalloc_peak_mb"] = round(peak_mb, 3)
            if max_tracemalloc_mb > 0 and peak_mb > max_tracemalloc_mb:
                raise AssertionError(
                    f"Tracemalloc peak {peak_mb:.1f} MB exceeded limit "
                    f"{max_tracemalloc_mb:.1f} MB"
                )

        handles_start = state.get("handles_start")
        handles_end = current_handle_count()
        if handles_start is not None and handles_end is not None:
            growth = handles_end - handles_start
            metrics["handle_growth"] = int(growth)
            metrics["handles_end"] = int(handles_end)
            if max_handle_growth > 0 and growth > max_handle_growth:
                raise AssertionError(
                    f"Handle/file-descriptor growth {growth} exceeded limit "
                    f"{max_handle_growth}"
                )
        elif max_handle_growth > 0:
            raise AssertionError("Handle growth sanitizer requires psutil handle/fd support")

        blocks_start = state.get("blocks_start")
        blocks_end = current_allocated_blocks()
        if blocks_start is not None and blocks_end is not None:
            block_growth = blocks_end - blocks_start
            metrics["allocated_block_growth"] = int(block_growth)
            metrics["allocated_blocks_end"] = int(blocks_end)
            if (
                max_allocated_block_growth > 0
                and block_growth > max_allocated_block_growth
            ):
                raise AssertionError(
                    f"Allocated block growth {block_growth} exceeded limit "
                    f"{max_allocated_block_growth}"
                )
        elif max_allocated_block_growth > 0:
            raise AssertionError("Allocated block sanitizer is unavailable on this Python")

        return metrics
    finally:
        if state.get("tracemalloc_started_here") and tracemalloc.is_tracing():
            tracemalloc.stop()


def start_memory_monitor(max_rss_mb: float):
    if max_rss_mb <= 0:
        return None, None
    try:
        import psutil
    except ImportError:
        return None, None

    process = psutil.Process(os.getpid())
    try:
        initial_rss = int(process.memory_info().rss)
    except Exception:
        initial_rss = 0
    stop_event = threading.Event()
    state = {"initial": initial_rss, "peak": initial_rss}

    def monitor() -> None:
        while not stop_event.is_set():
            try:
                state["peak"] = max(state["peak"], int(process.memory_info().rss))
            except Exception:
                pass
            stop_event.wait(0.1)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    return stop_event, state


def stop_memory_monitor(stop_event, state, max_rss_mb: float) -> float:
    if stop_event is None or state is None:
        return 0.0
    stop_event.set()
    peak_mb = state["peak"] / (1024 * 1024)
    if max_rss_mb > 0 and peak_mb > max_rss_mb:
        raise AssertionError(f"Peak RSS {peak_mb:.1f} MB exceeded limit {max_rss_mb:.1f} MB")
    return peak_mb
