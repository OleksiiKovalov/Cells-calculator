"""Parent fuzzing loop: generate cases, run the model, check oracles, report."""
import argparse
import json
import sys
import traceback
from pathlib import Path

from tests._models import KNOWN_MODELS, enabled_models as _enabled_models
from .generation import PROFILES, generate_case
from .oracles import check_detections

ROOT = Path(__file__).resolve().parents[2]

# Make the app package importable when run as a standalone CLI
# (`python -m tests.fuzzing`) without an editable install.
_SRC = ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

DEFAULT_CORPUS = ROOT / "testimages"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".lsm"}


def discover_models():
    return list(_enabled_models().keys())


def load_model(name):
    data = _enabled_models()[name]
    from ui.app_globals import register_model
    from model.Model import Model
    register_model(data["model_type"], KNOWN_MODELS[data["model_type"]], False)
    return Model(path=data["path"], model_type=data["model_type"],
                 model_data=data, model_name=name)


def corpus_paths(corpus_dir):
    base = Path(corpus_dir) if corpus_dir else DEFAULT_CORPUS
    if not base.exists():
        return []
    return sorted(p for p in base.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def run(model_name, max_cases=200, seed=0, profile="mixed",
        corpus_dir=None, corpus_prob=0.6, out_dir=None, verbose=False):
    import cv2  # noqa: F401  (ensure decoder is available before generation)
    model = load_model(model_name)
    corpus = corpus_paths(corpus_dir)
    out = Path(out_dir) if out_dir else (ROOT / ".cache" / "fuzz_failures")

    failures = []
    for i in range(max_cases):
        img, kind = generate_case(seed, i, profile, corpus, corpus_prob)
        if img is None:
            continue
        shape = getattr(img, "shape", None)
        try:
            det = model.inference(img)
        except Exception as exc:
            failures.append({"index": i, "kind": kind, "shape": shape,
                             "type": "crash", "detail": repr(exc),
                             "traceback": traceback.format_exc()})
            if verbose:
                print(f"[crash] case {i} ({kind} {shape}): {exc!r}")
            continue
        violations = check_detections(det)
        if violations:
            failures.append({"index": i, "kind": kind, "shape": shape,
                             "type": "oracle", "detail": "; ".join(violations)})
            if verbose:
                print(f"[oracle] case {i} ({kind} {shape}): {violations}")
        elif verbose:
            print(f"[ok] case {i} ({kind} {shape}) -> {len(det)} detections")

    print(f"\nFuzzed {model_name}: {max_cases} cases, seed={seed}, "
          f"profile={profile}, corpus={len(corpus)} images -> "
          f"{len(failures)} failures")
    if failures:
        out.mkdir(parents=True, exist_ok=True)
        report = out / f"failures_seed{seed}.json"
        report.write_text(json.dumps(failures, indent=2, default=str), encoding="utf-8")
        print(f"  wrote {report}")
        for f in failures[:10]:
            print(f"  - case {f['index']} [{f['kind']} {f['shape']}] {f['type']}: {f['detail']}")
    return 1 if failures else 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="tests.fuzzing", description="Runtime image fuzzer")
    p.add_argument("--list-models", action="store_true")
    p.add_argument("--model", default=None, help="model name from modelconfig.json")
    p.add_argument("--max-cases", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--profile", choices=PROFILES, default="mixed")
    p.add_argument("--corpus", default=None, help="directory of seed images")
    p.add_argument("--corpus-probability", type=float, default=0.6)
    p.add_argument("--out-dir", default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    if args.list_models:
        for name in discover_models():
            print(name)
        return 0
    model_name = args.model or (discover_models()[0] if discover_models() else None)
    if not model_name:
        p.error("no enabled models found")
    # StarDist's TensorFlow backend must initialize before torch/cv2 on Windows.
    if _enabled_models().get(model_name, {}).get("model_type") == "stardist":
        try:
            import tensorflow  # noqa: F401
        except Exception:
            pass
    return run(model_name, max_cases=args.max_cases, seed=args.seed, profile=args.profile,
               corpus_dir=args.corpus, corpus_prob=args.corpus_probability,
               out_dir=args.out_dir, verbose=args.verbose)
