"""Headless prepare & convert — the non-GUI counterpart of Dataset Viewer's
*File → Prepare & Export*.

Detects a dataset's format, optionally preprocesses (standardize / resize /
contrast) and re-splits it, then exports to another format — all without opening
the window. Handy for scripting and batch jobs; it drives the exact same engine
(``PreparedLoader`` + the format exporters) the GUI uses.

Loaders read image dimensions via ``QImageReader``, which needs a Qt
application; this module creates an off-screen one on demand, so it runs in a
plain terminal / CI with no display.

Library use:
    from convert import convert_dataset
    convert_dataset("path/to/coco", "out", "yolo",
                    prepare_spec={"resize": True, "resize_target": 512,
                                  "split_mode": "ratio"})

CLI:
    python src/convert.py --src path/to/coco --out out --to yolo \\
        --resize --resize-target 512 --split ratio --train 0.7 --val 0.1 --test 0.2
"""

import argparse
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_EXPORTERS = {
    "YOLO": ("datasets.yolo_exporter", "YOLOExporter"),
    "COCO": ("datasets.coco_exporter", "COCOExporter"),
    "Pascal VOC": ("datasets.voc_exporter", "VOCExporter"),
    "InstanSeg PTH": ("datasets.pth_exporter", "PTHExporter"),
}

_FORMAT_ALIASES = {
    "yolo": "YOLO", "coco": "COCO", "voc": "Pascal VOC", "pascal voc": "Pascal VOC",
    "pth": "InstanSeg PTH", "instanseg": "InstanSeg PTH", "instanseg pth": "InstanSeg PTH",
}


def _ensure_qapp():
    """A QApplication is needed because loaders read image sizes via QImageReader."""
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def resolve_format(name: str) -> str:
    key = _FORMAT_ALIASES.get(name.strip().lower())
    if key is None and name in _EXPORTERS:
        key = name
    if key is None:
        raise ValueError(f"Unknown target format {name!r}. "
                         f"Choose one of: {', '.join(sorted(_FORMAT_ALIASES))}.")
    return key


def convert_dataset(src: str, out: str, target_format: str,
                    prepare_spec: dict | None = None,
                    pth_options: dict | None = None, progress_cb=None) -> str:
    """Prepare + convert the dataset at ``src`` into ``target_format`` under ``out``.

    ``prepare_spec`` is forwarded to :class:`PreparedLoader` (preprocess + split);
    ``pth_options`` to :class:`PTHExporter` (ignored for other targets). Returns
    the resolved ``"<detected> -> <target>"`` label.
    """
    from datasets.format_detector import detect_format
    from datasets.prepared_loader import PreparedLoader

    _ensure_qapp()
    fmt = resolve_format(target_format)

    loader, detected = detect_format(src)
    if loader is None:
        raise ValueError(f"Could not detect a supported dataset at: {src}")

    source = PreparedLoader(loader, prepare_spec or {})

    module_name, cls_name = _EXPORTERS[fmt]
    module = __import__(module_name, fromlist=[cls_name])
    exporter_cls = getattr(module, cls_name)
    exporter = exporter_cls(**(pth_options or {})) if fmt == "InstanSeg PTH" else exporter_cls()

    os.makedirs(out, exist_ok=True)
    exporter.export(source, out, progress_cb)
    return f"{detected} -> {fmt}"


def _prepare_spec_from_args(args) -> dict:
    return {
        "standardize": args.standardize,
        "resize": args.resize,
        "resize_target": args.resize_target,
        "resample": args.resample,
        "contrast": args.contrast,
        "contrast_factor": args.contrast_factor,
        "split_mode": args.split,
        "ratios": (args.train, args.val, args.test),
        "seed": args.seed,
    }


def main():
    ap = argparse.ArgumentParser(description="Prepare & convert a dataset between formats (headless).")
    ap.add_argument("--src", required=True, help="Source dataset folder (or .pth file).")
    ap.add_argument("--out", required=True, help="Output folder.")
    ap.add_argument("--to", dest="target", required=True, help="Target format: yolo | coco | voc | pth.")
    # Preprocess (all formats)
    ap.add_argument("--standardize", action="store_true", default=True, help="Standardize to RGB/uint8 (default on).")
    ap.add_argument("--no-standardize", dest="standardize", action="store_false")
    ap.add_argument("--resize", action="store_true", help="Downscale toward --resize-target.")
    ap.add_argument("--resize-target", type=int, default=512)
    ap.add_argument("--resample", choices=["LANCZOS", "NEAREST"], default="LANCZOS")
    ap.add_argument("--contrast", action="store_true", help="Enhance contrast by --contrast-factor.")
    ap.add_argument("--contrast-factor", type=float, default=2.0)
    # Split (all formats)
    ap.add_argument("--split", choices=["keep", "ratio"], default="keep")
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    # InstanSeg PTH-only
    ap.add_argument("--name", default=None, help="Output .pth file name (PTH only).")
    ap.add_argument("--modality", default="Brightfield", help="image_modality tag (PTH only).")
    ap.add_argument("--mask-key", dest="mask_key", default=None,
                    choices=["cell_masks", "nucleus_masks"], help="Force the mask key (PTH only).")
    args = ap.parse_args()

    pth_options = None
    if resolve_format(args.target) == "InstanSeg PTH":
        pth_options = {"modality": args.modality}
        if args.name:
            pth_options["filename"] = args.name
        if args.mask_key:
            pth_options["mask_key"] = args.mask_key

    def on_progress(done, total):
        pct = int(done * 100 / total) if total else 100
        print(f"\r  {done}/{total} ({pct}%)", end="", flush=True)
        return True

    try:
        result = convert_dataset(args.src, args.out, args.target,
                                 prepare_spec=_prepare_spec_from_args(args),
                                 pth_options=pth_options, progress_cb=on_progress)
    except Exception as exc:  # noqa: BLE001
        print(f"\nConversion failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"\nDone: {result}. Output in {args.out}")


if __name__ == "__main__":
    main()
