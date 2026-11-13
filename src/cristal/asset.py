"""
Class representing assets and their real (or generated)
"""
import os
import pickle
from enum import Enum
from typing import Any

from cristal.env import PROJECT_ROOT


class AssetCategory(Enum):
    HIGH_GROWTH = "high_growth"
    STABLE = "stable"
    TIME_BOMB = "time_bomb"


ALLOWED_DATA_FORMATS = [
    ".pkl",
]


class Asset:
    def __init__(
        self,
        name: str,
        category: AssetCategory,
        data: str | None = None,
    ):
        """
        Initialize an asset

        :param name:        Unique name
        :param category:    One of AssetCategory
        :param data:        Optional path to asset data
        """
        self.name = name
        self.category = category
        self.data = self.load_data(data) if data else None

    def load_data(self, path: str) -> None:
        """
        Load asset data

        :param path:
        """
        # First check if extension is allowed
        if not any([path.endswith(fmt) for fmt in ALLOWED_DATA_FORMATS]):
            raise ValueError(f"Invalid data format: {path}. Must be one of {ALLOWED_DATA_FORMATS}")

        # Handle if data path is absolute, project relative, or runtime relative
        if os.path.isabs(path):
            data_path = path
        else:
            data_path = os.path.join(PROJECT_ROOT, path)
            if not os.path.exists(data_path):
                # Try as cwd relative
                data_path = os.path.join(os.getcwd(), path)

        # Check if valid
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Asset data not found: {data_path}")

        self._load_pkl_data(data_path)

    def _load_pkl_data(self, data_path) -> Any:
        """
        Load asset data from a pickle file

        :param data_path:    Path to pickle file
        """
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)

    def __str__(self):
        return f"{self.name} [Asset|{self.category}]"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.name.lower() == other.name.lower()

    def __hash__(self):
        return hash(self.name)
