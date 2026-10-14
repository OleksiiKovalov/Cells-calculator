from .config import *
from .generation import *
from .subprocess_runner import *
from .oracles import make_mock_payload
from .child_runner import run_case_in_child
from .signatures import *
from .minimizer import minimize_failure

def should_continue(case_count: int, max_cases: int, started_at: float, seconds: float) -> bool:
    if max_cases and case_count >= max_cases:
        return False
    if seconds and time.time() - started_at >= seconds:
        return False
    return True


def parent_main(args: RuntimeFuzzConfig) -> int:
    models = selected_models(load_enabled_models(Path(args.config)), args.models)
    if args.list_models:
        list_models(models)
        return 0

    if args.mock_mode == "none":
        runnable = OrderedDict(
            (name, data) for name, data in models.items() if model_is_locally_runnable(data)
        )
    else:
        runnable = OrderedDict((name, data) for name, data in models.items())
    if not runnable:
        raise SystemExit("No runnable models selected.")

    output_dir = Path(args.output_dir)
    cases_dir = output_dir / "cases"
    failures_dir = output_dir / "failures"
    duplicates_dir = output_dir / "duplicates"
    signature_log = output_dir / "signatures.jsonl"
    pass_log = output_dir / "passes.jsonl"
    run_summary_path = output_dir / "run_summary.json"
    coverage_summary_path = output_dir / "coverage_summary.json"
    line_coverage_path = output_dir / "python_line_coverage.json"
    mcdc_coverage_path = output_dir / "python_mcdc_coverage.json"
    interesting_dir = output_dir / "interesting"
    cases_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed if args.seed is not None else int(time.time_ns() % (2**32))
    rng = random.Random(seed)
    allowed_scales = parse_scales(args.scales)
    corpus_images = discover_corpus_images(args.seed_corpus)
    corpus_probability = clamp_probability(args.corpus_probability)
    coverage_corpus_probability = clamp_probability(args.coverage_corpus_probability)
    if args.coverage_guided and args.python_coverage == "off":
        args.python_coverage = "mcdc"
    requested_generation_engine = args.generation_engine
    generation_engine = effective_generation_engine(args.generation_engine)
    feedback_corpus_images: list[Path] = []
    feedback_case_specs: list[FuzzCaseSpec] = []
    seen_coverage_signals: set[str] = set()
    if args.coverage_guided:
        interesting_dir.mkdir(parents=True, exist_ok=True)
        feedback_corpus_images = discover_corpus_images(str(interesting_dir))
        feedback_case_specs = discover_case_specs(interesting_dir)
        coverage_state_path = (
            mcdc_coverage_path if args.python_coverage == "mcdc" else line_coverage_path
        )
        seen_coverage_signals = load_existing_python_coverage_signals(
            coverage_state_path,
            args.python_coverage,
        )
    print(f"runtime fuzz seed: {seed}")
    print(f"selected models: {', '.join(runnable.keys())}")
    if requested_generation_engine == "hypothesis":
        print(
            "warning: --generation-engine hypothesis is a compatibility alias for "
            "seed-stable --generation-engine strategy; Hypothesis shrinking is not used."
        )
    print(
        f"image profile: {args.image_profile}; "
        f"generation engine: {generation_engine}; "
        f"corpus mutation: {args.corpus_mutation_mode}; "
        f"scales: {','.join(str(scale) for scale in allowed_scales)}; "
        f"max side: {args.max_side}; "
        f"model strategy: {args.model_strategy}; "
        f"mock mode: {args.mock_mode}; "
        f"workflow: {args.workflow}; "
        f"sanitizers: {args.sanitizers}; "
        f"rss limit: {args.max_rss_mb}; "
        f"trace limit: {args.max_tracemalloc_mb}; "
        f"handle growth limit: {args.max_handle_growth}; "
        f"oracle: {args.oracle_level}; "
        f"determinism: {args.determinism_check}; "
        f"python coverage: {args.python_coverage}"
    )
    if corpus_images:
        print(f"seed corpus images: {len(corpus_images)}; corpus probability: {corpus_probability:.2f}")
    if args.coverage_guided:
        print(
            "coverage-guided: on; "
            f"interesting corpus={len(feedback_corpus_images)}; "
            f"interesting specs={len(feedback_case_specs)}; "
            f"feedback probability={coverage_corpus_probability:.2f}; "
            f"known signals={len(seen_coverage_signals)}"
        )

    started_at = time.time()
    case_count = 0
    passes = 0
    failures = 0
    unique_failures = 0
    duplicate_failures = 0
    new_coverage_cases = 0
    new_coverage_signals_total = 0
    seen_signatures = set() if args.no_dedupe else load_seen_signatures(signature_log)
    model_items = list(runnable.items())
    coverage_stats = empty_coverage_stats()

    try:
        while should_continue(case_count, args.max_cases, started_at, args.seconds):
            case_count += 1
            if args.model_strategy == "round-robin":
                model_name, model_data = model_items[(case_count - 1) % len(model_items)]
            else:
                model_name, model_data = rng.choice(model_items)
            case_dir = cases_dir / f"case_{case_count:06d}_{rng.randrange(0, 2**32):08x}"
            case_corpus_images = corpus_images
            case_corpus_probability = corpus_probability
            case_grammar_specs: list[FuzzCaseSpec] = []
            source_pool = "seed"
            if (
                args.coverage_guided
                and feedback_corpus_images
                and generation_engine != "grammar-mutational"
                and rng.random() < coverage_corpus_probability
            ):
                case_corpus_images = feedback_corpus_images
                case_corpus_probability = 1.0
                source_pool = "interesting"
            elif (
                args.coverage_guided
                and generation_engine == "grammar-mutational"
                and feedback_case_specs
                and rng.random() < coverage_corpus_probability
            ):
                case_grammar_specs = feedback_case_specs
                case_corpus_probability = 0.0
                source_pool = "interesting-spec"
            try:
                case = build_case(
                    rng,
                    case_count,
                    case_dir,
                    model_name,
                    dict(model_data),
                    args.image_profile,
                    generation_engine,
                    allowed_scales,
                    args.max_side,
                    case_corpus_images,
                    case_corpus_probability,
                    args.corpus_mutation_mode,
                    args.workflow,
                    args.max_workflow_steps,
                    case_grammar_specs,
                )
                case["coverage_guided"] = {
                    "enabled": bool(args.coverage_guided),
                    "source_pool": source_pool,
                }
                case["mock_mode"] = args.mock_mode
                if args.mock_mode != "none":
                    case["mock"] = make_mock_payload(rng, args.mock_mode)
                build_error = False
            except Exception:
                case_dir.mkdir(parents=True, exist_ok=True)
                case = build_generation_error_case(
                    case_count,
                    model_name,
                    dict(model_data),
                    args.workflow,
                )
                build_error = True
                stdout = ""
                stderr = traceback.format_exc()
                reason = "generation error"
            case_path = case_dir / "case.json"
            case_path.write_text(json.dumps(case, indent=2), encoding="utf-8")

            if build_error:
                passed = False
            else:
                try:
                    completed = run_case_subprocess(
                        case_path,
                        args.timeout,
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
                    passed = completed.returncode == 0
                    reason = f"exit code {completed.returncode}"
                except subprocess.TimeoutExpired as exc:
                    completed = None
                    passed = False
                    reason = f"timeout after {args.timeout} seconds"
                    stdout = exc.stdout or ""
                    stderr = exc.stderr or ""
                else:
                    stdout = completed.stdout
                    stderr = completed.stderr

            signature = None
            if not passed:
                signature = failure_signature(case, stdout, stderr, reason)

            child_result = extract_child_result(stdout)
            new_coverage_signals: set[str] = set()
            if args.coverage_guided and passed:
                coverage_signals = child_python_coverage_signals(stdout)
                new_coverage_signals = coverage_signals - seen_coverage_signals
                seen_coverage_signals.update(coverage_signals)

            update_coverage_stats(
                coverage_stats,
                case,
                stdout,
                stderr,
                reason,
                passed,
                signature,
            )
            if passed:
                passes += 1
                interesting_path = None
                if args.coverage_guided and new_coverage_signals:
                    interesting_path, case, new_images = save_interesting_case(
                        case_dir,
                        interesting_dir,
                        case,
                        new_coverage_signals,
                        child_result,
                        copy_only=args.keep_all,
                    )
                    known_feedback = {path.resolve() for path in feedback_corpus_images if path.exists()}
                    for image_path in new_images:
                        resolved_image = image_path.resolve()
                        if resolved_image not in known_feedback:
                            feedback_corpus_images.append(resolved_image)
                            known_feedback.add(resolved_image)
                    raw_spec = case.get("case_spec")
                    if isinstance(raw_spec, dict):
                        spec = fuzz_case_spec_from_dict(raw_spec)
                        if spec is not None:
                            feedback_case_specs.append(spec)
                    new_coverage_cases += 1
                    new_coverage_signals_total += len(new_coverage_signals)

                append_pass_log(pass_log, case, stdout)
                print(
                    f"PASS case={case_count} model={model_name} "
                    f"image={case['image']['extension']} shape={case['image']['shape']}"
                    + (
                        f" new_coverage={len(new_coverage_signals)} saved={interesting_path}"
                        if interesting_path is not None
                        else ""
                    )
                )
                if not args.keep_all and interesting_path is None:
                    shutil.rmtree(case_dir, ignore_errors=True)
            else:
                failures += 1
                assert signature is not None
                is_duplicate = not args.no_dedupe and signature["hash"] in seen_signatures
                if is_duplicate:
                    duplicate_failures += 1
                    target_root = duplicates_dir / signature["hash"]
                else:
                    unique_failures += 1
                    target_root = failures_dir
                    seen_signatures.add(signature["hash"])

                failure_dir = save_failure(
                    case_dir,
                    target_root,
                    stdout,
                    stderr,
                    reason,
                    signature,
                )
                append_signature_log(signature_log, signature, failure_dir, is_duplicate)
                print(
                    f"FAIL case={case_count} model={model_name} signature={signature['hash']} "
                    f"duplicate={is_duplicate} reason={reason} saved={failure_dir}"
                )
                if args.minimize_failures and not is_duplicate:
                    minimized_dir = minimize_failure(failure_dir, signature, args)
                    if minimized_dir is not None:
                        print(f"MINIMIZED signature={signature['hash']} saved={minimized_dir}")
                if stdout:
                    print(console_safe_text(stdout[-2000:]))
                if stderr:
                    print(console_safe_text(stderr[-2000:], sys.stderr))
    except KeyboardInterrupt:
        print("Stopped by user.")

    print(
        f"finished cases={case_count} failures={failures} "
        f"unique={unique_failures} duplicates={duplicate_failures}"
    )
    run_summary = {
        "seed": seed,
        "selected_models": list(runnable.keys()),
        "settings": {
            "image_profile": args.image_profile,
            "generation_engine": generation_engine,
            "requested_generation_engine": requested_generation_engine,
            "seed_stable_generation": True,
            "model_strategy": args.model_strategy,
            "mock_mode": args.mock_mode,
            "workflow": args.workflow,
            "max_workflow_steps": args.max_workflow_steps,
            "sanitizers": args.sanitizers,
            "oracle_level": args.oracle_level,
            "determinism_check": args.determinism_check,
            "max_rss_mb": args.max_rss_mb,
            "max_tracemalloc_mb": args.max_tracemalloc_mb,
            "max_handle_growth": args.max_handle_growth,
            "max_allocated_block_growth": args.max_allocated_block_growth,
            "scales": allowed_scales,
            "max_side": args.max_side,
            "seed_corpus": args.seed_corpus,
            "corpus_probability": corpus_probability,
            "corpus_mutation_mode": args.corpus_mutation_mode,
            "python_coverage": args.python_coverage,
            "coverage_guided": args.coverage_guided,
            "coverage_corpus_probability": coverage_corpus_probability,
            "interesting_case_specs": len(feedback_case_specs),
        },
        "cases": case_count,
        "passes": passes,
        "failures": failures,
        "unique_failures": unique_failures,
        "duplicate_failures": duplicate_failures,
        "duration_seconds": round(time.time() - started_at, 3),
        "passed": failures == 0,
        "coverage_guided": {
            "enabled": args.coverage_guided,
            "new_coverage_cases": new_coverage_cases,
            "new_coverage_signals": new_coverage_signals_total,
            "total_unique_signals": len(seen_coverage_signals),
            "signal_mode": args.python_coverage,
            "interesting_corpus_size": len(feedback_corpus_images),
            "interesting_case_spec_size": len(feedback_case_specs),
            "interesting_dir": str(interesting_dir),
        },
        "signature_log": str(signature_log),
        "pass_log": str(pass_log),
        "coverage_summary": str(coverage_summary_path),
        "python_line_coverage": str(line_coverage_path),
        "python_mcdc_coverage": str(mcdc_coverage_path),
        "coverage": serialize_coverage_stats(coverage_stats),
    }
    run_summary_path.write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=True, sort_keys=True),
        encoding="utf-8",
    )
    coverage_summary_path.write_text(
        json.dumps(serialize_coverage_stats(coverage_stats), indent=2, ensure_ascii=True, sort_keys=True),
        encoding="utf-8",
    )
    line_coverage_path.write_text(
        json.dumps(
            serialize_coverage_stats({"python_coverage_lines": coverage_stats["python_coverage_lines"]}, include_python_lines=True),
            indent=2,
            ensure_ascii=True,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    mcdc_coverage_path.write_text(
        json.dumps(
            serialize_coverage_stats(
                {
                    "mcdc_signals": coverage_stats["mcdc_signals"],
                    "mcdc_decisions": coverage_stats["mcdc_decisions"],
                    "mcdc_outcomes": coverage_stats["mcdc_outcomes"],
                    "mcdc_satisfied_conditions": coverage_stats["mcdc_satisfied_conditions"],
                },
                include_mcdc_signals=True,
            ),
            indent=2,
            ensure_ascii=True,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"run summary: {run_summary_path}")
    print(f"coverage summary: {coverage_summary_path}")
    if coverage_stats["python_coverage_lines"]:
        print(f"python line coverage: {line_coverage_path}")
    if coverage_stats["mcdc_signals"]:
        print(f"python MC/DC coverage: {mcdc_coverage_path}")
    return 1 if failures else 0


def main() -> int:
    args = parse_args()
    if args.run_case:
        os.environ["CELLS_FUZZ_MOCK_MODE"] = args.mock_mode
        return run_case_in_child(
            Path(args.run_case),
            args.max_rss_mb,
            args.max_tracemalloc_mb,
            args.max_handle_growth,
            args.max_allocated_block_growth,
        )
    if args.replay:
        completed = run_case_subprocess(
            Path(args.replay),
            args.timeout,
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
        print(completed.stdout, end="")
        print(completed.stderr, end="", file=sys.stderr)
        return completed.returncode
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
