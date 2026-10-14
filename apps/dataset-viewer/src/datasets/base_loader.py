from abc import ABC, abstractmethod


class BaseDatasetLoader(ABC):
    def __init__(self, folder: str):
        self.folder = folder
        self.class_names: list[str] = []

    @abstractmethod
    def get_splits(self) -> list[str]:
        """Return list of split names (train/val/test) or [] if unsplit."""
        ...

    @abstractmethod
    def get_images(self, split: str | None = None) -> list[dict]:
        """Return list of {'path': str, 'name': str} dicts."""
        ...

    @abstractmethod
    def get_annotations(self, image_path: str) -> list[dict]:
        """Return list of annotation dicts with pixel coordinates:
        {'class_id': int, 'label': str, 'x': float, 'y': float, 'w': float, 'h': float}
        """
        ...
