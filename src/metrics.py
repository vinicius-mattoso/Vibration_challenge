from typing import List

from src.interface import CarpetRegion


def evaluate_part1(
    predictions: dict[str, List[CarpetRegion]],
    labels: dict[str, List[CarpetRegion]],
) -> dict:
    """Evaluate carpet detector predictions against ground truth.

    Example return: ``{"my_metric": 0.85, "another_metric": 42}``.
    """
    raise NotImplementedError


def evaluate_part2(
    predictions: dict[str, bool],
    labels: dict[str, bool],
) -> dict:
    """Evaluate lubrication predictions against ground truth.

    Example return: ``{"my_metric": 0.85, "another_metric": 42}``.
    """
    raise NotImplementedError
