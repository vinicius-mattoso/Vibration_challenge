from pathlib import Path
from typing import List

from src.interface import CarpetRegion, Wave


class CarpetDetector:
    def predict(self, wave: Wave) -> List[CarpetRegion]:
        """Return detected carpet regions, or an empty list if none."""
        raise NotImplementedError

    def plot_results(
        self,
        wave: Wave,
        regions: List[CarpetRegion],
        output_dir: Path,
        sample_id: str = "sample",
    ) -> None:
        """Plot frequency spectrum with regions highlighted; save to ``output_dir / f"{sample_id}.png"``."""
        raise NotImplementedError
