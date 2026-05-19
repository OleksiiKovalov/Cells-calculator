"""Download the default trained model archive for CI."""

from __future__ import annotations

import base64
import json
import re
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, cast


ROOT = Path(__file__).resolve().parents[1]
TRAINED_MODELS = ROOT / "trainedmodels"
CACHE_DIR = ROOT / ".cache" / "model_downloads"
ARCHIVE_NAME = "TrainedModels.v3.1.zip"
MODELS_README = TRAINED_MODELS / "models.readme"
REQUIRED_FILES = [
    "trainedmodels/yolov8m-det.onnx",
    "trainedmodels/YOLO11x-512-seg.pt",
    "trainedmodels/instanseg_20250605.pt",
    "trainedmodels/Instanseg-Neuroblastoma-v3.1.pt",
]


def _request(url: str):
    return urllib.request.Request(
        url,
        headers={"User-Agent": "CellsCalculatorCI/1.0"},
    )


def _share_id(url: str) -> str:
    encoded = base64.urlsafe_b64encode(url.encode("utf-8")).decode("ascii")
    return "u!" + encoded.rstrip("=")


def _read_onedrive_folder_url() -> str:
    text = MODELS_README.read_text(encoding="utf-8")
    match = re.search(r"https?://\S+", text)
    if match is None:
        raise RuntimeError(f"No OneDrive URL found in {MODELS_README}")
    return match.group(0)


def _discover_onedrive_archive_url() -> str:
    folder_url = _read_onedrive_folder_url()
    endpoint = (
        "https://api.onedrive.com/v1.0/shares/"
        f"{_share_id(folder_url)}/root/children"
    )
    with urllib.request.urlopen(_request(endpoint), timeout=60) as response:
        data = cast(dict[str, Any], json.loads(response.read().decode("utf-8")))

    for item in data.get("value", []):
        if item.get("name") == ARCHIVE_NAME:
            download_url = item.get("@content.downloadUrl") or item.get(
                "@microsoft.graph.downloadUrl"
            )
            if download_url:
                return str(download_url)

    raise RuntimeError(f"{ARCHIVE_NAME} was not found in the OneDrive folder")


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(_request(url), timeout=60) as response:
        with target.open("wb") as output:
            shutil.copyfileobj(response, output)


def _extract(zip_path: Path) -> None:
    TRAINED_MODELS.mkdir(parents=True, exist_ok=True)
    root = TRAINED_MODELS.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (TRAINED_MODELS / member.filename).resolve()
            if not target.is_relative_to(root):
                raise RuntimeError(f"Refusing to extract outside {TRAINED_MODELS}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _models_are_present() -> bool:
    return all((ROOT / path).exists() for path in REQUIRED_FILES)


def main() -> int:
    if _models_are_present():
        print("Trained models are already present.")
        return 0

    try:
        archive_url = _discover_onedrive_archive_url()
        zip_path = CACHE_DIR / ARCHIVE_NAME
        print(f"Downloading {ARCHIVE_NAME}...")
        _download(archive_url, zip_path)
        print(f"Extracting {ARCHIVE_NAME}...")
        _extract(zip_path)
        zip_path.unlink(missing_ok=True)
    except Exception as error:
        print(f"Failed to download trained models: {error}", file=sys.stderr)
        print(
            "Check the public OneDrive URL in trainedmodels/models.readme.",
            file=sys.stderr,
        )
        return 1

    if not _models_are_present():
        print("Downloaded archive did not contain all required model files.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
