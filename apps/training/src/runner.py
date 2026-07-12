"""Trainer pipeline orchestrator for Training Studio.

Runs train -> evaluate -> export on an already-prepared ``.pth`` dataset.
Dataset preparation/conversion is out of scope for this app — produce the
``.pth`` in the sibling Dataset Viewer app first.

Like the sibling apps' runners, this invokes each step as a child process and
streams its output, so a heavy import (torch / instanseg) in one step can't take
down the orchestrator.

    train     ->  data/models/<model_folder>/...      (trained weights)
    evaluate  ->  data/test_results/...               (metrics + overlays)
    export    ->  data/exported/<...>.pt              (TorchScript model)

Usage:
    python src/runner.py --dataset data/datasets/my_dataset.pth --steps train
    python src/runner.py --dataset data/datasets/my_dataset.pth --steps train,evaluate,export
"""

import argparse
import subprocess
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent

ALL_STEPS = ["train", "evaluate", "export"]


def run(cmd):
    """Run a child script, streaming its output. Aborts the pipeline on failure."""
    printable = " ".join(str(c) for c in cmd)
    print("\n" + "=" * 70)
    print(f">> {printable}")
    print("=" * 70, flush=True)
    result = subprocess.run([sys.executable] + cmd)
    if result.returncode != 0:
        print(f"\n!! Step failed (exit {result.returncode}): {printable}", file=sys.stderr)
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Run the InstanSeg trainer pipeline (train -> evaluate -> export) "
                    "on a prepared .pth dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, help="Path to the prepared .pth dataset.")
    parser.add_argument("--config", default=None, help="Alternative config.json path.")
    parser.add_argument("--steps", default=",".join(ALL_STEPS),
                        help=f"Comma list of steps to run. Available: {','.join(ALL_STEPS)}")
    parser.add_argument("--models-dir", default="data/models", help="Model folders dir (train output).")
    parser.add_argument("--results-dir", default="data/test_results", help="Evaluation output dir.")
    parser.add_argument("--exported-dir", default="data/exported", help="TorchScript export dir.")
    parser.add_argument("--model-folder", default=None, help="Model folder name for train/export.")
    parser.add_argument("--model-path", default=None, help="Trained model dir for evaluate/export.")
    parser.add_argument("--version", default=None, help="Version subfolder name for export.")
    args = parser.parse_args()

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    bad = [s for s in steps if s not in ALL_STEPS]
    if bad:
        print(f"Unknown step(s): {bad}. Valid: {ALL_STEPS}", file=sys.stderr)
        sys.exit(2)

    dataset = Path(args.dataset)
    datasets_dir = dataset.parent
    name = dataset.stem
    print(f"Trainer pipeline for dataset='{dataset}' | steps: {steps}")

    def default_model_path():
        return args.model_path or str(Path(args.models_dir) / (args.model_folder or name))

    if "train" in steps:
        cmd = [str(_SRC / "train.py"), "--dataset", str(dataset),
               "--models-dir", args.models_dir, "--output-dir", args.models_dir]
        if args.model_folder:
            cmd += ["--model-folder", args.model_folder]
        if args.config:
            cmd += ["--config", args.config]
        run(cmd)

    if "evaluate" in steps:
        cmd = [str(_SRC / "evaluate.py"), "--dataset-dir", str(datasets_dir),
               "--data", dataset.name, "--model-path", default_model_path(),
               "--out", args.results_dir]
        if args.config:
            cmd += ["--config", args.config]
        run(cmd)

    if "export" in steps:
        cmd = [str(_SRC / "export_model.py"), "--model-path", default_model_path(),
               "--out", args.exported_dir]
        if args.version:
            cmd += ["--version", args.version]
        if args.config:
            cmd += ["--config", args.config]
        run(cmd)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
