#!/usr/bin/env python3
"""
Tractian DS Challenge — Condition Monitoring

Usage:
    uv run python main.py --part1
    uv run python main.py --part2
    uv run python main.py --part1 --part2
"""

import argparse
from pathlib import Path
from typing import List

from src import metrics
from src.carpet_detector import CarpetDetector
from src.data_loader import load_part1, load_part2
from src.interface import CarpetRegion
from src.lubrication_model import LubricationModel

OUTPUTS_DIR = Path("outputs")


def _write_report(
    path: Path, title: str, metrics_result: dict, extra: str = ""
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write("## Metrics\n\n")
        for k, v in metrics_result.items():
            f.write(f"- **{k}**: {v}\n")
        if extra:
            f.write(f"\n{extra}")


def run_part1(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    waves, labels = load_part1()
    detector = CarpetDetector()

    predictions: dict[str, List[CarpetRegion]] = {}
    for sample_id, wave in waves.items():
        predictions[sample_id] = detector.predict(wave)

    metrics_result = metrics.evaluate_part1(predictions, labels)

    report_path = output_dir / "report_part1.md"
    _write_report(report_path, "Part 1 — Carpet Detector Report", metrics_result)

    print(f"[Part 1] Metrics: {metrics_result}")
    print(f"[Part 1] Report  -> {report_path}")

    for sample_id, wave in waves.items():
        detector.plot_results(
            wave, predictions[sample_id], plots_dir, sample_id=sample_id
        )


def run_part2(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_plots_dir = output_dir / "plots" / "assets"
    test_plots_dir = output_dir / "plots" / "test"
    assets_plots_dir.mkdir(parents=True, exist_ok=True)
    test_plots_dir.mkdir(parents=True, exist_ok=True)

    assets, labels, test_data = load_part2()

    predictions: dict[str, bool] = {}
    for asset_id, data in assets.items():
        model = LubricationModel()
        model.fit(data.fit)
        predictions[asset_id] = model.predict(data.predict)

        asset_dir = assets_plots_dir / asset_id
        asset_dir.mkdir(exist_ok=True)
        model.plot_results(data.predict, asset_dir)

    metrics_result = metrics.evaluate_part2(predictions, labels)

    test_predictions: dict[str, bool] = {}
    for asset_id, data in test_data.items():
        model = LubricationModel()
        model.fit(data.fit)
        test_predictions[asset_id] = model.predict(data.predict)

        asset_dir = test_plots_dir / asset_id
        asset_dir.mkdir(exist_ok=True)
        model.plot_results(data.predict, asset_dir)

    header = ["## Test Data Predictions", "", "| Asset ID | Prediction |", "|---|---|"]
    body = [
        f"| {aid} | {'Starved Lubrication' if pred else 'Healthy'} |"
        for aid, pred in test_predictions.items()
    ]
    test_table = "\n".join(header + body) + "\n"

    report_path = output_dir / "report_part2.md"
    _write_report(
        report_path,
        "Part 2 — Starved Lubrication Model Report",
        metrics_result,
        extra=test_table,
    )

    print(f"[Part 2] Metrics: {metrics_result}")
    print(f"[Part 2] Report  -> {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tractian DS Challenge — Condition Monitoring"
    )
    parser.add_argument(
        "--part1", action="store_true", help="Run Part 1: Carpet Detector"
    )
    parser.add_argument(
        "--part2", action="store_true", help="Run Part 2: Lubrication Model"
    )
    args = parser.parse_args()

    if not args.part1 and not args.part2:
        parser.print_help()
        return

    OUTPUTS_DIR.mkdir(exist_ok=True)

    if args.part1:
        run_part1(OUTPUTS_DIR / "part1")

    if args.part2:
        run_part2(OUTPUTS_DIR / "part2")


if __name__ == "__main__":
    main()
