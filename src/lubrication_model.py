from pathlib import Path
from typing import List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from src.carpet_detector import CarpetDetector
from src.interface import Wave

EPS = 1e-18


class LubricationModel:
    def __init__(self, config_path: Path | None = None):
        project_root = Path(__file__).resolve().parent.parent
        self.config_path = config_path or (
            project_root / "models" / "part2" / "lubrication_model_config.pkl"
        )
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Part 2 model config not found: {self.config_path}. "
                "Run notebooks/part2_modeling.ipynb first."
            )

        config = joblib.load(self.config_path)
        self.features_usadas = list(config["features_usadas"])
        self.pesos_features = dict(config["pesos_features"])
        self.centro_score = float(config["centro_score"])
        self.escala_score = float(config["escala_score"])
        self.max_z = float(config["max_z"])

        self.carpet_detector: CarpetDetector | None = None
        self.baseline_mediana_: pd.Series | None = None
        self.baseline_escala_: pd.Series | None = None
        self.features_fit_: pd.DataFrame | None = None

    def fit(self, data: List[Wave]) -> None:
        """Optional: fit on reference (healthy) data. No-op by default."""
        if not data:
            raise ValueError("LubricationModel.fit received no reference signals.")

        features_fit = pd.DataFrame([self._extract_features(wave) for wave in data])
        baseline = features_fit[self.features_usadas]

        scale = baseline.quantile(0.75) - baseline.quantile(0.25)
        scale = (
            scale.replace(0, np.nan)
            .fillna(baseline.std().replace(0, np.nan))
            .fillna(1.0)
        )

        self.features_fit_ = features_fit
        self.baseline_mediana_ = baseline.median()
        self.baseline_escala_ = scale + EPS

    def predict_sample(self, wave: Wave) -> float:
        """Score a single waveform in [0, 1] (0 = healthy, 1 = starved lubrication)."""
        score, _, _, _ = self._score_details(wave)
        return score

    def predict(self, data: List[Wave]) -> bool:
        """Aggregate per-sample scores; alert if 75th percentile > 0.75."""
        scores = [self.predict_sample(wave) for wave in data]
        return float(np.percentile(scores, 75)) > 0.75

    def plot_results(self, data: List[Wave], output_dir: Path) -> None:
        """Save PNG plots about the asset's condition to ``output_dir``."""
        output_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for idx, wave in enumerate(data):
            score, raw_score, _, z_scores = self._score_details(wave)
            row = {
                "index": idx,
                "sample_id": wave.sample_id,
                "score": score,
                "raw_score": raw_score,
            }
            row.update({f"z_{name}": float(z_scores[name]) for name in self.features_usadas})
            rows.append(row)

        diagnostics = pd.DataFrame(rows)
        diagnostics.to_csv(output_dir / "diagnostics.csv", index=False)

        scores = diagnostics["score"].to_numpy(dtype=float)
        p75 = float(np.percentile(scores, 75))
        decision = p75 > 0.75

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(diagnostics["index"], diagnostics["score"], marker="o", linewidth=1.0)
        ax.axhline(0.75, color="#111827", linestyle="--", linewidth=1.0, label="threshold")
        ax.axhline(p75, color="#dc2626", linestyle="-", linewidth=1.0, label="p75")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Signal index")
        ax.set_ylabel("Score")
        ax.set_title(f"Lubrication scores - {'starved' if decision else 'healthy'}")
        ax.legend(loc="best")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "scores.png", dpi=140)
        plt.close(fig)

        z_columns = [f"z_{feature}" for feature in self.features_usadas]
        median_z = diagnostics[z_columns].median().sort_values()
        median_z.index = [name.replace("z_", "") for name in median_z.index]

        fig, ax = plt.subplots(figsize=(9, 5))
        median_z.plot.barh(ax=ax, color="#dc2626")
        ax.set_xlabel("Median robust z-score")
        ax.set_title("Feature deviation against healthy baseline")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "feature_z_scores.png", dpi=140)
        plt.close(fig)

    @staticmethod
    def _sampling_interval(wave: Wave) -> float:
        time = np.asarray(wave.time, dtype=np.float64)
        dt = np.diff(time)
        dt = dt[dt > 0]
        if len(dt) == 0:
            raise ValueError(f"Wave {wave.sample_id} has invalid time axis.")
        return float(np.median(dt))

    @staticmethod
    def _band_area(freq: np.ndarray, psd: np.ndarray, start_hz: float, end_hz: float) -> float:
        mask = (freq >= start_hz) & (freq < end_hz)
        if mask.sum() < 2:
            return 0.0
        return float(np.trapz(psd[mask], freq[mask]))

    @staticmethod
    def _spectral_entropy(psd: np.ndarray) -> float:
        energy = np.maximum(psd, EPS)
        probability = energy / (np.sum(energy) + EPS)
        if len(probability) <= 1:
            return 0.0
        return float(
            -np.sum(probability * np.log2(probability + EPS))
            / np.log2(len(probability))
        )

    def _compute_psd(self, wave: Wave, nperseg: int = 4096) -> tuple[np.ndarray, np.ndarray]:
        vibration = np.asarray(wave.signal, dtype=np.float64)
        vibration = signal.detrend(vibration - np.mean(vibration), type="linear")
        dt = self._sampling_interval(wave)
        fs = 1.0 / dt
        nperseg = min(nperseg, len(vibration))

        freq, psd = signal.welch(
            vibration,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=nperseg // 2,
            scaling="density",
        )
        return freq, psd

    def _extract_features(self, wave: Wave) -> dict[str, float]:
        vibration = np.asarray(wave.signal, dtype=np.float64)
        vibration = vibration - np.mean(vibration)

        rms = float(np.sqrt(np.mean(vibration**2)))
        freq, psd = self._compute_psd(wave)
        total_energy = self._band_area(freq, psd, 0, float(freq.max())) + EPS

        above_1k = freq >= 1000
        log_psd_above_1k = 10 * np.log10(psd[above_1k] + EPS)

        return {
            "rms": rms,
            "energia_acima_1k_frac": self._band_area(freq, psd, 1000, float(freq.max()))
            / total_energy,
            "energia_acima_3k_frac": self._band_area(freq, psd, 3000, float(freq.max()))
            / total_energy,
            "energia_3_6k_frac": self._band_area(freq, psd, 3000, 6000)
            / total_energy,
            "centroide_espectral": float(np.sum(freq * psd) / (np.sum(psd) + EPS)),
            "entropia_espectral": self._spectral_entropy(psd),
            "log_psd_mediana_acima_1k": float(np.median(log_psd_above_1k)),
            "log_psd_q90_acima_1k": float(np.percentile(log_psd_above_1k, 90)),
        }

    def _score_details(
        self, wave: Wave
    ) -> tuple[float, float, dict[str, float], pd.Series]:
        if self.baseline_mediana_ is None or self.baseline_escala_ is None:
            raise RuntimeError("LubricationModel.fit must be called before predict.")

        features = self._extract_features(wave)
        values = pd.Series(features)[self.features_usadas]
        z_scores = (values - self.baseline_mediana_) / self.baseline_escala_
        z_scores = z_scores.clip(lower=0, upper=self.max_z)

        total_weight = sum(self.pesos_features[name] for name in self.features_usadas)
        raw_score = sum(
            self.pesos_features[name] * z_scores[name] for name in self.features_usadas
        ) / total_weight

        score_input = np.clip(
            (raw_score - self.centro_score) / self.escala_score,
            -60,
            60,
        )
        score = float(1.0 / (1.0 + np.exp(-score_input)))
        return score, float(raw_score), features, z_scores
