"""Training step — fine-tune an InstanSeg model on a ``.pth`` dataset.

Thin wrapper around ``instanseg_training`` (cell 23 of
``instanseg-datasets-training-testing.ipynb``).  Works with EITHER InstanSeg
backend:

  * the stock ``instanseg-torch`` (mainline), whose ``instanseg_training`` takes
    ``(segmentation_dataset, **kwargs)`` and maps kwargs onto its argparse args;
  * the cryobiology fork ``sonyalytv/instanseg_cryobiology``, which additionally
    exposes early stopping via ``max_no_improvement`` (and a custom test.py).

The backend is auto-detected: :func:`resolve_training_backend` introspects the
installed ``instanseg_training`` to learn which arguments it accepts, and
:func:`run_training` passes fork-only extras (currently ``max_no_improvement``)
ONLY when supported — omitting them with a logged note on mainline (so training
still runs, just without early stopping).

The heavy import is deferred inside these functions so ``--help`` and the smoke
tests stay cheap and the app imports cleanly with no backend installed — the
same philosophy as ``morphology/src/segment.py``'s deferred instanseg import.

CLI:
    python src/train.py --dataset data/datasets/my_dataset.pth --model-folder my_first_instanseg \
        --target C --epochs 500 --source-dataset my_dataset
"""

import argparse
import sys
from pathlib import Path

from config import load_config

FORK_HINT = (
    "No InstanSeg backend is installed.\n"
    "Train / Evaluate / Export need InstanSeg (either backend works):\n"
    "    pip install instanseg-torch monai            # mainline\n"
    "  OR the cryobiology fork (adds early stopping + custom bits):\n"
    "    pip install git+https://github.com/sonyalytv/instanseg_cryobiology.git\n"
    "    pip install monai stardist edt\n"
    "and a working PyTorch install (GPU strongly recommended)."
)

# A parameter the FORK's instanseg_training exposes but stock mainline does not;
# used both as a fork "marker" and as the one kwarg we gate on backend support.
_FORK_ONLY_PARAMS = ("max_no_improvement",)


def resolve_training_backend(log=print):
    """Locate ``instanseg_training`` and learn which arguments it accepts.

    Returns ``(instanseg_training, kind, supported)`` where ``kind`` is
    ``"fork"`` or ``"mainline"`` and ``supported`` is the set of accepted keyword
    names (or ``None`` when it could not be determined — callers then omit any
    optional/fork-only kwargs to stay safe).  Raises ``SystemExit(2)`` with an
    actionable message when no InstanSeg backend is importable at all.
    """
    import inspect

    try:
        from instanseg.scripts.train import instanseg_training
    except ImportError:
        log(FORK_HINT)
        raise SystemExit(2)

    params = inspect.signature(instanseg_training).parameters
    has_var_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())
    explicit = {name for name, p in params.items()
                if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
                and name != "segmentation_dataset"}

    # Fork: exposes the fork-only params explicitly in its signature.
    if any(p in explicit for p in _FORK_ONLY_PARAMS):
        return instanseg_training, "fork", explicit

    # Mainline: kwargs are forwarded onto the module's argparse; the accepted
    # names are that parser's destinations.
    supported = set(explicit)
    if has_var_kw:
        try:
            from instanseg.scripts import train as _train_mod
            supported |= {a.dest for a in _train_mod.parser._actions}
        except Exception:  # noqa: BLE001 — introspection is best-effort
            supported = None
    return instanseg_training, "mainline", supported


def ensure_instanseg(log=print):
    """Verify SOME InstanSeg backend is importable (used by evaluate/export).

    Neither evaluation nor export depends on fork-only features, so any backend
    is fine.  Raises ``SystemExit(2)`` with the install hint when none is present.
    """
    try:
        import instanseg  # noqa: F401
    except ImportError:
        log(FORK_HINT)
        raise SystemExit(2)


def run_training(dataset_path, models_dir, output_dir, tcfg, log=print):
    """Load a .pth dataset and run ``instanseg_training`` (cell 23 params)."""
    import torch
    instanseg_training, kind, supported = resolve_training_backend(log)
    log(f"InstanSeg backend detected: {kind}.")

    dataset_path = Path(dataset_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset .pth not found: {dataset_path}")

    log(f"Loading dataset: {dataset_path}")
    segmentation_dataset = torch.load(str(dataset_path), weights_only=False)

    models_dir = Path(models_dir)
    output_dir = Path(output_dir)
    model_folder = tcfg["model_folder"]
    model_dir = models_dir / model_folder
    model_dir.mkdir(parents=True, exist_ok=True)

    # Optional resume-from-checkpoint: copy the weights into the model folder so
    # training continues from them (cell 21).
    resume = str(tcfg.get("resume_weights", "") or "").strip()
    if resume:
        import shutil
        src = Path(resume)
        if not src.is_file():
            raise FileNotFoundError(f"Resume weights not found: {src}")
        shutil.copy(str(src), str(model_dir / "model_weights_best.pth"))
        log(f"Resuming from weights: {src.name}")

    # Core kwargs accepted by BOTH backends (mainline argparse dests == fork
    # signature params).
    call_kwargs = {
        "data_path": str(dataset_path),
        "segmentation_dataset": segmentation_dataset,
        "source_dataset": tcfg["source_dataset"],
        "model_folder": model_folder,
        "model_path": str(models_dir) + "/",
        "output_path": str(output_dir) + "/",
        "experiment_str": model_folder,
        "requested_pixel_size": float(tcfg["requested_pixel_size"]),
        "target_segmentation": tcfg["target_segmentation"],
        "num_epochs": int(tcfg["num_epochs"]),
        "hotstart_training": int(tcfg["hotstart_training"]),
    }
    # Fork-only extras: pass ONLY when the detected backend accepts them.
    optional = {"max_no_improvement": int(tcfg["max_no_improvement"])}
    for name, value in optional.items():
        if supported is not None and name in supported:
            call_kwargs[name] = value
        else:
            log(f"note: '{name}' is not supported by the {kind} backend; omitting it "
                f"(training will run the full {tcfg['num_epochs']} epochs without "
                f"early stopping, saving the best checkpoint as it improves).")

    log(f"Starting training: model_folder='{model_folder}', "
        f"target='{tcfg['target_segmentation']}', epochs={tcfg['num_epochs']}")
    instanseg_training(**call_kwargs)
    log("Training complete!")
    return str(model_dir)


def main():
    ap = argparse.ArgumentParser(description="Fine-tune an InstanSeg model on a .pth dataset.")
    ap.add_argument("--dataset", required=True, help="Path to the .pth dataset.")
    ap.add_argument("--models-dir", default="data/models", help="Directory holding model folders.")
    ap.add_argument("--output-dir", default="data/models", help="Training output directory.")
    ap.add_argument("--model-folder", default=None, help="Model folder name (default from config).")
    ap.add_argument("--source-dataset", default=None, help="parent_dataset tag to train on.")
    ap.add_argument("--target", dest="target_segmentation", choices=["C", "N"], default=None,
                    help="C = Cells, N = Nuclei.")
    ap.add_argument("--pixel-size", dest="requested_pixel_size", type=float, default=None)
    ap.add_argument("--epochs", dest="num_epochs", type=int, default=None)
    ap.add_argument("--max-no-improvement", type=int, default=None)
    ap.add_argument("--hotstart", dest="hotstart_training", type=int, default=None)
    ap.add_argument("--resume-weights", default=None, help="model_weights_best.pth to resume from.")
    ap.add_argument("--config", default=None, help="Alternative config.json path.")
    args = ap.parse_args()

    tcfg = dict(load_config(args.config)["train"])
    for key in ("model_folder", "source_dataset", "target_segmentation",
                "requested_pixel_size", "num_epochs", "max_no_improvement",
                "hotstart_training", "resume_weights"):
        val = getattr(args, key, None)
        if val is not None:
            tcfg[key] = val

    try:
        run_training(args.dataset, args.models_dir, args.output_dir, tcfg)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"Training failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
