import json
from collections.abc import Mapping
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src.interface import AssetData, CarpetRegion, Wave

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _PROJECT_ROOT / "data"


def _load_npz_waves(path: Path) -> List[Wave]:
    npz = np.load(path, allow_pickle=False)
    data = npz["data"].astype(np.float32)
    dts = npz["dt"]
    sample_ids = npz["sample_ids"]
    L = data.shape[1]
    waves = []
    for i in range(len(data)):
        t = (np.arange(L) * dts[i]).tolist()
        waves.append(
            Wave(sample_id=str(sample_ids[i]), time=t, signal=data[i].tolist())
        )
    return waves


def load_part1() -> tuple[dict[str, Wave], dict[str, List[CarpetRegion]]]:
    """Load Part 1 waves and ground-truth carpet regions, keyed by sample_id."""
    data_dir = DATA_DIR / "part1" / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Part 1 data directory not found: {data_dir}\n")
    labels_path = DATA_DIR / "part1" / "labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Part 1 labels not found: {labels_path}\n")
    waves: dict[str, Wave] = {}
    for csv_file in sorted(data_dir.glob("*.csv")):
        sample_id = csv_file.stem
        df = pd.read_csv(csv_file)
        waves[sample_id] = Wave(
            sample_id=sample_id,
            time=df["t"].tolist(),
            signal=df["data"].tolist(),
        )

    labels_df = pd.read_csv(DATA_DIR / "part1" / "labels.csv")

    wave_ids = set(waves)
    label_ids = set(labels_df["sample_id"].astype(str))
    if wave_ids != label_ids:
        raise ValueError(
            f"Mismatch between data files and labels.csv.\n"
            f"  In data/ but not labels.csv: {wave_ids - label_ids}\n"
            f"  In labels.csv but not data/: {label_ids - wave_ids}"
        )

    labels: dict[str, List[CarpetRegion]] = {}
    for _, row in labels_df.iterrows():
        raw = json.loads(row["regions"])
        labels[str(row["sample_id"])] = [
            CarpetRegion(start_hz=r[0], end_hz=r[1]) for r in raw
        ]

    return waves, labels


class _LazyAssetDict(Mapping):
    def __init__(self, asset_dirs: dict):
        self._dirs = asset_dirs

    def __iter__(self):
        return iter(self._dirs)

    def __len__(self):
        return len(self._dirs)

    def __getitem__(self, key):
        asset_dir = self._dirs[key]
        return AssetData(
            fit=_load_npz_waves(asset_dir / "fit.npz"),
            predict=_load_npz_waves(asset_dir / "predict.npz"),
        )

    def __contains__(self, key):
        return key in self._dirs


def load_part2() -> tuple[
    dict[str, AssetData],
    dict[str, bool],
    dict[str, AssetData],
]:
    """Load Part 2 assets, labels and unlabeled test assets."""
    part2_dir = DATA_DIR / "part2"
    for subpath in ["assets", "labels.csv", "test_data"]:
        if not (part2_dir / subpath).exists():
            raise FileNotFoundError(f"Part 2 data not found: {part2_dir / subpath}\n")

    asset_dirs = {
        asset_dir.name: asset_dir
        for asset_dir in sorted((part2_dir / "assets").iterdir())
        if asset_dir.is_dir()
    }
    assets = _LazyAssetDict(asset_dirs)

    labels_df = pd.read_csv(part2_dir / "labels.csv")
    labels: dict[str, bool] = {
        str(row["asset_id"]): "starved" in str(row["label"]).strip().lower()
        for _, row in labels_df.iterrows()
    }

    test_dirs = {
        asset_dir.name: asset_dir
        for asset_dir in sorted((part2_dir / "test_data").iterdir())
        if asset_dir.is_dir()
    }
    test_data = _LazyAssetDict(test_dirs)

    return assets, labels, test_data
