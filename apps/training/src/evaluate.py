"""Evaluation step — score a trained InstanSeg model on a ``.pth`` dataset split.

Wraps InstanSeg's ``scripts/test.py`` (cell 28 of the notebook), which computes
segmentation metrics and can save prediction overlays.  Because InstanSeg ships
the evaluation as a CLI script, we locate it inside the installed ``instanseg``
package (either backend) and invoke it as a child process, forwarding the same
flags the notebook used (-d_p / -m_p / -o_f / -data / -target / -set / -save_ims
/ -cpu_and_ram).

If no InstanSeg backend is installed, a clear message is printed and the step
exits non-zero (same contract as ``train.py``).

CLI:
    python src/evaluate.py --dataset-dir data/datasets --data my_dataset.pth \
        --model-path data/models/my_first_instanseg --out data/test_results --set Test --target C
"""

import argparse
import subprocess
import sys
from pathlib import Path

from config import load_config
from train import ensure_instanseg


def _find_test_script():
    """Return the path to InstanSeg's ``scripts/test.py``, or None if absent."""
    try:
        import instanseg
    except ImportError:
        return None
    pkg_dir = Path(instanseg.__file__).resolve().parent
    candidate = pkg_dir / "scripts" / "test.py"
    return candidate if candidate.is_file() else None


def run_evaluation(dataset_dir, data_file, model_path, out_dir, ecfg, log=print):
    """Invoke InstanSeg's test.py with the notebook's flags. Returns exit code."""
    ensure_instanseg(log)                  # clean message if no backend is present
    test_script = _find_test_script()
    if test_script is None:
        raise RuntimeError("Could not locate instanseg/scripts/test.py in the install.")

    # test.py's augmentation import needs monai; surface a friendly hint early.
    try:
        import monai  # noqa: F401
    except ImportError:
        log("note: 'monai' is not installed; InstanSeg's test.py imports it. "
            "Install it with:  pip install monai")

    dataset_dir = Path(dataset_dir)
    model_path = Path(model_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(test_script),
        "-d_p", str(dataset_dir),
        "-m_f", "",
        "-o_f", str(out_dir),
        "-m_p", str(model_path),
        "-data", data_file,
        "-target", ecfg["target_segmentation"],
        "-set", ecfg["set"],
        "-save_ims", str(bool(ecfg["save_images"])),
    ]
    if ecfg.get("cpu_and_ram"):
        cmd += ["-cpu_and_ram", "True"]

    log("Running InstanSeg evaluation:")
    log("  " + " ".join(cmd))
    # Run from the script's own directory (the notebook %cd's into scripts/).
    result = subprocess.run(cmd, cwd=str(test_script.parent))
    if result.returncode != 0:
        raise RuntimeError(f"test.py exited with code {result.returncode}")
    log(f"Evaluation complete. Results in {out_dir}")
    return result.returncode


def main():
    ap = argparse.ArgumentParser(description="Evaluate a trained InstanSeg model on a dataset split.")
    ap.add_argument("--dataset-dir", required=True, help="Directory containing the .pth dataset file.")
    ap.add_argument("--data", dest="data_file", required=True, help="Dataset .pth file name (inside --dataset-dir).")
    ap.add_argument("--model-path", required=True, help="Trained model directory (with the version subfolder).")
    ap.add_argument("--out", dest="out_dir", default="data/test_results", help="Output folder for metrics/overlays.")
    ap.add_argument("--set", dest="set", choices=["Train", "Validation", "Test"], default=None)
    ap.add_argument("--target", dest="target_segmentation", choices=["C", "N"], default=None)
    ap.add_argument("--save-images", dest="save_images", action="store_true", default=None)
    ap.add_argument("--no-save-images", dest="save_images", action="store_false")
    ap.add_argument("--cpu-and-ram", dest="cpu_and_ram", action="store_true", default=None)
    ap.add_argument("--config", default=None, help="Alternative config.json path.")
    args = ap.parse_args()

    ecfg = dict(load_config(args.config)["evaluate"])
    for key in ("set", "target_segmentation", "save_images", "cpu_and_ram"):
        val = getattr(args, key, None)
        if val is not None:
            ecfg[key] = val

    try:
        run_evaluation(args.dataset_dir, args.data_file, args.model_path, args.out_dir, ecfg)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"Evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
