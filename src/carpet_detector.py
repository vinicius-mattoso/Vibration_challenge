from pathlib import Path
from typing import List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from src.interface import CarpetRegion, Wave

EPS = 1e-18


class CarpetDetector:
    def __init__(self, model_path: Path | None = None) -> None:
        project_root = Path(__file__).resolve().parent.parent
        self.model_path = model_path or (
            project_root / "models" / "part1" / "Modelo_mlflow_v1s.pkl"
        )
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Part 1 model not found: {self.model_path}. "
                "Run notebooks/part1_modeling_mlflow_v1s.ipynb first."
            )

        package = joblib.load(self.model_path)
        self.model = package["modelo"]
        self.feature_columns = package["feature_columns"]
        self.threshold = float(package["threshold"])

        config = package.get("feature_config", {})
        self.min_freq_hz = float(config.get("min_freq_hz", 1000))
        self.window_hz = float(config.get("window_hz", 250))
        self.step_hz = float(config.get("step_hz", 125))

    def predict(self, wave: Wave) -> List[CarpetRegion]:
        """Return detected carpet regions, or an empty list if none."""
        features = self._build_window_features(wave)
        if features.empty:
            return []

        probabilities = self.model.predict_proba(features[self.feature_columns])[:, 1]
        positive_windows = features.loc[probabilities >= self.threshold]
        return self._windows_to_regions(positive_windows)

    def plot_results(
        self,
        wave: Wave,
        regions: List[CarpetRegion],
        output_dir: Path,
        sample_id: str = "sample",
    ) -> None:
        """Plot frequency spectrum with regions highlighted; save to ``output_dir / f"{sample_id}.png"``."""
        output_dir.mkdir(parents=True, exist_ok=True)

        time = np.asarray(wave.time, dtype=np.float64)
        vibration = np.asarray(wave.signal, dtype=np.float64)
        freq, psd = self._compute_psd(time, vibration)
        log_psd = 10 * np.log10(psd + EPS)

        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(freq, log_psd, color="#1f2937", linewidth=0.9)
        ax.axvspan(0, self.min_freq_hz, color="#94a3b8", alpha=0.14)
        ax.axvline(
            self.min_freq_hz,
            color="#111827",
            linestyle="--",
            linewidth=1.0,
            label=f"{self.min_freq_hz:.0f} Hz",
        )

        for region in regions:
            ax.axvspan(
                region.start_hz,
                region.end_hz,
                color="#ef4444",
                alpha=0.22,
                label="detected carpet",
            )

        handles, names = ax.get_legend_handles_labels()
        if names:
            unique = dict(zip(names, handles))
            ax.legend(unique.values(), unique.keys(), loc="upper right")

        ax.set_xlim(0, min(float(freq.max()), 8000))
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (dB)")
        ax.set_title(f"Carpet detection - {sample_id}")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"{sample_id}.png", dpi=140)
        plt.close(fig)

    @staticmethod
    def _sampling_rate(time: np.ndarray) -> float:
        dt = np.diff(time)
        dt = dt[dt > 0]
        if len(dt) == 0:
            return float("nan")
        return float(1.0 / np.median(dt))

    def _compute_psd(
        self, time: np.ndarray, vibration: np.ndarray, nperseg_max: int = 8192
    ) -> tuple[np.ndarray, np.ndarray]:
        vibration = signal.detrend(vibration - np.mean(vibration))
        fs = self._sampling_rate(time)
        nperseg = min(nperseg_max, len(vibration))
        freq, psd = signal.welch(
            vibration,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=nperseg // 2,
            scaling="density",
        )
        return freq, psd

    @staticmethod
    def _band_energy(
        freq: np.ndarray, psd: np.ndarray, start_hz: float, end_hz: float
    ) -> float:
        mask = (freq >= start_hz) & (freq < end_hz)
        if mask.sum() < 2:
            return 0.0
        return float(np.trapz(psd[mask], freq[mask]))

    @staticmethod
    def _spectral_flatness(power: np.ndarray) -> float:
        power = np.maximum(np.asarray(power, dtype=np.float64), EPS)
        return float(np.exp(np.mean(np.log(power))) / (np.mean(power) + EPS))

    def _build_window_features(self, wave: Wave) -> pd.DataFrame:
        time = np.asarray(wave.time, dtype=np.float64)
        vibration = np.asarray(wave.signal, dtype=np.float64)
        freq, psd = self._compute_psd(time, vibration)
        log_psd = 10 * np.log10(psd + EPS)

        freq_max = float(freq.max())
        total_high_energy = (
            self._band_energy(freq, psd, self.min_freq_hz, freq_max) + EPS
        )
        rows = []

        start_hz = self.min_freq_hz
        while start_hz + self.window_hz <= freq_max:
            end_hz = start_hz + self.window_hz
            mask = (freq >= start_hz) & (freq < end_hz)

            if mask.sum() >= 3:
                freq_window = freq[mask]
                psd_window = psd[mask]
                log_window = log_psd[mask]
                peaks, _ = signal.find_peaks(log_window, prominence=0.5)
                slope = (
                    np.polyfit(freq_window, log_window, 1)[0]
                    if len(freq_window) >= 2
                    else 0.0
                )

                rows.append(
                    {
                        "sample_id": wave.sample_id,
                        "start_hz": start_hz,
                        "end_hz": end_hz,
                        "center_hz": start_hz + self.window_hz / 2,
                        "energia_relativa_janela": self._band_energy(
                            freq, psd, start_hz, end_hz
                        )
                        / total_high_energy,
                        "log_psd_mean": float(np.mean(log_window)),
                        "log_psd_std": float(np.std(log_window)),
                        "log_psd_max": float(np.max(log_window)),
                        "log_psd_p90": float(np.quantile(log_window, 0.90)),
                        "log_psd_p10": float(np.quantile(log_window, 0.10)),
                        "log_psd_range": float(
                            np.max(log_window) - np.min(log_window)
                        ),
                        "slope_log_psd": float(slope),
                        "flatness_psd": self._spectral_flatness(psd_window),
                        "n_picos_log": int(len(peaks)),
                        "densidade_picos": float(len(peaks) / (self.window_hz / 1000)),
                        "std_sinal": float(np.std(vibration)),
                        "max_abs_sinal": float(np.max(np.abs(vibration))),
                    }
                )

            start_hz += self.step_hz

        return pd.DataFrame(rows)

    def _windows_to_regions(self, windows: pd.DataFrame) -> List[CarpetRegion]:
        if windows.empty:
            return []

        intervals = (
            windows[["start_hz", "end_hz"]]
            .sort_values(["start_hz", "end_hz"])
            .to_numpy(dtype=np.float64)
        )
        merged: list[list[float]] = []

        for start_hz, end_hz in intervals:
            start_hz = max(float(start_hz), self.min_freq_hz + 1e-6)
            end_hz = float(end_hz)
            if end_hz <= start_hz:
                continue

            if not merged or start_hz > merged[-1][1] + self.step_hz:
                merged.append([start_hz, end_hz])
            else:
                merged[-1][1] = max(merged[-1][1], end_hz)

        return [
            CarpetRegion(start_hz=start_hz, end_hz=end_hz)
            for start_hz, end_hz in merged
            if end_hz > start_hz
        ]
