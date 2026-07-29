from pathlib import Path
from typing import List

import numpy as np

from src.carpet_detector import CarpetDetector
from src.interface import Wave


class LubricationModel:
    def __init__(self):
        self.carpet_detector = CarpetDetector()

    def fit(self, data: List[Wave]) -> None:
        """Optional: fit on reference (healthy) data. No-op by default."""
        pass

    def predict_sample(self, wave: Wave) -> float:
        """Score a single waveform in [0, 1] (0 = healthy, 1 = starved lubrication)."""
        raise NotImplementedError

    def predict(self, data: List[Wave]) -> bool:
        """Aggregate per-sample scores; alert if 75th percentile > 0.75."""
        scores = [self.predict_sample(wave) for wave in data]
        return float(np.percentile(scores, 75)) > 0.75

    def plot_results(self, data: List[Wave], output_dir: Path) -> None:
        """Save PNG plots about the asset's condition to ``output_dir``."""
        raise NotImplementedError
