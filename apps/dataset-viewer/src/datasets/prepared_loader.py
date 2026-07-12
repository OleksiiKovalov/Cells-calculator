"""A loader that presents a *prepared* view of another dataset.

Preparation = image preprocessing (standardize to RGB, resize toward a target
edge, contrast) + train/val/test split (keep the source's, or re-split by ratio).
It wraps any :class:`BaseDatasetLoader` and exposes the same interface, so the
existing exporters (YOLO / COCO / VOC / PTH) consume it **unchanged** and thereby
write prepared datasets in any format.

How it stays transparent to exporters (which read images by *path* and take
pixel-absolute annotations):

  * When an image transform is active, each image is **materialized** to a temp
    directory (resized / contrast-adjusted / RGB) and its path is handed to the
    exporter; otherwise the original path is used.
  * ``get_annotations`` scales the base annotations by the per-image resize
    factor so bboxes / polygons / RLE masks line up with the transformed image.
    RLE survives via ``rle.decode_mask`` -> nearest resize -> ``rle.encode_mask``.
  * ``get_splits`` / ``get_images`` reflect the chosen split.

Temp images are cleaned up via ``atexit`` and ``__del__`` (mirrors
``pth_loader.py``).
"""

import atexit
import os
import shutil
import tempfile

import numpy as np

from .base_loader import BaseDatasetLoader
from .rle import decode_mask, encode_mask

# Canonical split order for re-split output.
_RATIO_SPLITS = ("train", "val", "test")


class PreparedLoader(BaseDatasetLoader):
    def __init__(self, base: BaseDatasetLoader, spec: dict | None = None):
        super().__init__(base.folder)
        self.base = base
        self.class_names = list(base.class_names)
        self.spec = dict(spec or {})
        self._tmp: str | None = None
        self._records: list[dict] = []
        self._by_path: dict[str, dict] = {}
        self._build()
        atexit.register(self._cleanup)

    # ------------------------------------------------------------------
    def _image_transform_active(self) -> bool:
        s = self.spec
        return bool(s.get("standardize") or s.get("resize") or s.get("contrast"))

    def _build(self):
        base = self.base
        splits = base.get_splits()
        pairs = ([(sp, info) for sp in splits for info in base.get_images(sp)]
                 if splits else [(None, info) for info in base.get_images()])

        transform = self._image_transform_active()
        if transform:
            self._tmp = tempfile.mkdtemp(prefix="dv_prep_")

        records = []
        for i, (sp, info) in enumerate(pairs):
            orig = info["path"]
            if transform:
                path, sx, sy = self._materialize(orig, i)
            else:
                path, sx, sy = orig, 1.0, 1.0
            records.append({"orig": orig, "path": path, "split": sp,
                            "sx": sx, "sy": sy, "name": os.path.basename(orig)})

        if self.spec.get("split_mode", "keep") == "ratio":
            self._assign_ratio(records)

        self._records = records
        self._by_path = {r["path"]: r for r in records}

    def _materialize(self, orig: str, i: int):
        """Write a transformed copy of ``orig`` to the temp dir; return (path, sx, sy)."""
        from PIL import Image, ImageEnhance

        assert self._tmp is not None
        s = self.spec
        opened = Image.open(orig)
        if s.get("standardize", True) or opened.mode not in ("RGB", "L"):
            img = opened.convert("RGB")
        else:
            img = opened.copy()

        ow, oh = img.width, img.height
        sx = sy = 1.0
        if s.get("resize"):
            target = max(1, int(s.get("resize_target", 512)))
            scale = max(1, ow // target, oh // target)
            if scale > 1:
                nw, nh = ow // scale, oh // scale
                want_lanczos = str(s.get("resample", "LANCZOS")).upper() == "LANCZOS"
                flt = Image.Resampling.LANCZOS if want_lanczos else Image.Resampling.NEAREST
                img = img.resize((nw, nh), flt)
                sx, sy = nw / ow, nh / oh

        if s.get("contrast"):
            img = ImageEnhance.Contrast(img).enhance(float(s.get("contrast_factor", 2.0)))

        # Per-index subdir keeps original basenames (and avoids train/val name clashes).
        sub = os.path.join(self._tmp, f"{i:04d}")
        os.makedirs(sub, exist_ok=True)
        stem = os.path.splitext(os.path.basename(orig))[0]
        path = os.path.join(sub, f"{stem}.png")
        img.save(path)
        return path, sx, sy

    def _assign_ratio(self, records: list[dict]):
        tr, va, te = self.spec.get("ratios", (0.7, 0.1, 0.2))
        total = tr + va + te
        if total <= 0:
            tr, va, te, total = 0.7, 0.1, 0.2, 1.0
        seed = int(self.spec.get("seed", 42))
        order = np.random.default_rng(seed).permutation(len(records))
        n = len(records)
        n_tr = int(n * tr / total)
        n_va = int(n * (tr + va) / total)
        for rank, idx in enumerate(order):
            records[idx]["split"] = "train" if rank < n_tr else "val" if rank < n_va else "test"

    # ------------------------------------------------------------------
    # BaseDatasetLoader interface
    # ------------------------------------------------------------------
    def get_splits(self) -> list[str]:
        present = [r["split"] for r in self._records if r["split"]]
        if not present:
            return []
        uniq = [s for s in _RATIO_SPLITS if s in present]
        uniq += [s for s in dict.fromkeys(present) if s not in uniq]
        return uniq

    def get_images(self, split: str | None = None) -> list[dict]:
        recs = self._records if split is None else [r for r in self._records if r["split"] == split]
        return [{"path": r["path"], "name": r["name"]} for r in recs]

    def get_annotations(self, image_path: str) -> list[dict]:
        rec = self._by_path.get(image_path) or next(
            (r for r in self._records if r["orig"] == image_path), None)
        if rec is None:
            return self.base.get_annotations(image_path)
        anns = self.base.get_annotations(rec["orig"])
        sx, sy = rec["sx"], rec["sy"]
        if sx == 1.0 and sy == 1.0:
            return anns
        return [_scale_annotation(a, sx, sy) for a in anns]

    # ------------------------------------------------------------------
    def _cleanup(self):
        if self._tmp and os.path.isdir(self._tmp):
            shutil.rmtree(self._tmp, ignore_errors=True)
        self._tmp = None

    def __del__(self):
        try:
            self._cleanup()
        except Exception:
            pass


def _scale_annotation(ann: dict, sx: float, sy: float) -> dict:
    """Return a copy of ``ann`` with coordinates scaled by (sx, sy)."""
    out = dict(ann)
    for key, s in (("x", sx), ("y", sy), ("w", sx), ("h", sy)):
        if key in out:
            out[key] = out[key] * s
    if out.get("points"):
        out["points"] = [(px * sx, py * sy) for px, py in out["points"]]
    if out.get("polygons"):
        out["polygons"] = [[(px * sx, py * sy) for px, py in poly] for poly in out["polygons"]]
    if out.get("type") == "mask" and out.get("rle_counts") and out.get("rle_size"):
        from PIL import Image

        mask = decode_mask(out["rle_counts"], out["rle_size"])
        oh, ow = mask.shape
        nh, nw = max(1, int(round(oh * sy))), max(1, int(round(ow * sx)))
        resized = np.asarray(
            Image.fromarray((mask > 0).astype(np.uint8) * 255).resize((nw, nh), Image.Resampling.NEAREST)
        ) > 0
        out["rle_counts"] = encode_mask(resized)
        out["rle_size"] = [nh, nw]
    return out
