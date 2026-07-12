"""Export step — convert a trained InstanSeg model to TorchScript.

Wraps ``export_to_torchscript`` from ``instanseg.utils.utils`` (cell 26 of the
notebook; present in both InstanSeg backends).  TorchScript optimizes the model
and makes it deployable in the sibling ``spheroid`` / ``morphology`` /
``cells-calculator`` apps (which load a ``.pt`` via ``torch.jit.load``).

The function is driven by two environment variables the fork reads:
    INSTANSEG_MODEL_PATH        model dir *excluding* the version subfolder
    INSTANSEG_TORCHSCRIPT_PATH  where the exported .pt is written
and a positional ``version`` = the final (version) folder name.

Deferred import + clean "no InstanSeg backend" message, same contract as train.py.

CLI:
    python src/export_model.py --model-path data/models/my_first_instanseg --version 1 --out data/exported
"""

import argparse
import os
import sys
from pathlib import Path

from config import load_config
from train import ensure_instanseg


def run_export(model_path, version, out_dir, show_example=False, log=print):
    """Export ``<model_path>/<version>`` to TorchScript under ``out_dir``."""
    ensure_instanseg(log)                  # clean message if no backend is present
    from instanseg.utils.utils import export_to_torchscript

    model_path = Path(model_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # export_to_torchscript reads these env vars (cell 26). MODEL_PATH excludes
    # the version folder; ``version`` is that last folder name.
    os.environ["INSTANSEG_MODEL_PATH"] = str(model_path.resolve())
    os.environ["INSTANSEG_TORCHSCRIPT_PATH"] = str(out_dir.resolve())

    log(f"Exporting '{model_path}/{version}' -> TorchScript in {out_dir}")
    export_to_torchscript(str(version), show_example=bool(show_example))
    log("Export complete.")
    return str(out_dir)


def main():
    ap = argparse.ArgumentParser(description="Export a trained InstanSeg model to TorchScript (.pt).")
    ap.add_argument("--model-path", required=True,
                    help="Model directory EXCLUDING the version subfolder.")
    ap.add_argument("--version", default=None, help="Version subfolder name (default from config).")
    ap.add_argument("--out", dest="out_dir", default="data/exported", help="Output dir for the .pt.")
    ap.add_argument("--show-example", action="store_true", default=None)
    ap.add_argument("--config", default=None, help="Alternative config.json path.")
    args = ap.parse_args()

    xcfg = dict(load_config(args.config)["export"])
    if args.version is not None:
        xcfg["version"] = args.version
    if args.show_example is not None:
        xcfg["show_example"] = args.show_example

    try:
        run_export(args.model_path, xcfg["version"], args.out_dir, xcfg["show_example"])
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"Export failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
