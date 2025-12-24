"""Configuration management for the project."""

import random
from dataclasses import dataclass

import numpy as np

from src.utils.paths import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
)


@dataclass(frozen=True)
class Config:
    """Project configuration with paths, seeds, and parameters."""

    # Random seed for reproducibility
    seed: int = 42

    # Data paths
    raw_data_path: str = str(RAW_DATA_DIR)
    processed_data_path: str = str(PROCESSED_DATA_DIR)
    reports_path: str = str(REPORTS_DIR)

    def set_seed(self) -> None:
        """Set random seeds for reproducibility."""
        random.seed(self.seed)
        np.random.seed(self.seed)

    def __post_init__(self) -> None:
        """Initialize seed after object creation."""
        self.set_seed()


# Global configuration instance
config = Config()

