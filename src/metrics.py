from typing import List

from src.interface import CarpetRegion


def _regions_to_intervals(regions: List[CarpetRegion]) -> list[tuple[float, float]]:
    intervals = [
        (float(region.start_hz), float(region.end_hz))
        for region in regions
        if float(region.end_hz) > float(region.start_hz)
    ]
    intervals.sort()
    return intervals


def _merge_intervals(
    intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _interval_length(intervals: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in intervals)


def _intersection_length(
    left: list[tuple[float, float]], right: list[tuple[float, float]]
) -> float:
    i = 0
    j = 0
    total = 0.0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        total += max(0.0, end - start)

        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    intersection = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return _safe_divide(intersection, union)


def _region_match_counts(
    predictions: list[tuple[float, float]],
    labels: list[tuple[float, float]],
    iou_threshold: float,
) -> tuple[int, int, int]:
    pairs = []
    for pred_idx, pred in enumerate(predictions):
        for label_idx, label in enumerate(labels):
            iou = _iou(pred, label)
            if iou >= iou_threshold:
                pairs.append((iou, pred_idx, label_idx))

    matched_predictions = set()
    matched_labels = set()
    for _, pred_idx, label_idx in sorted(pairs, reverse=True):
        if pred_idx in matched_predictions or label_idx in matched_labels:
            continue
        matched_predictions.add(pred_idx)
        matched_labels.add(label_idx)

    true_positive = len(matched_predictions)
    false_positive = len(predictions) - true_positive
    false_negative = len(labels) - len(matched_labels)
    return true_positive, false_positive, false_negative


def evaluate_part1(
    predictions: dict[str, List[CarpetRegion]],
    labels: dict[str, List[CarpetRegion]],
) -> dict:
    """Evaluate carpet detector predictions against ground truth.

    Example return: ``{"my_metric": 0.85, "another_metric": 42}``.
    """
    all_sample_ids = sorted(set(predictions) | set(labels))

    total_predicted_hz = 0.0
    total_label_hz = 0.0
    total_intersection_hz = 0.0
    sample_ious = []
    invalid_prediction_count = 0

    total_pred_regions = 0
    total_label_regions = 0
    matched_tp = 0
    matched_fp = 0
    matched_fn = 0

    for sample_id in all_sample_ids:
        raw_pred = predictions.get(sample_id, [])
        raw_label = labels.get(sample_id, [])

        invalid_prediction_count += sum(
            1
            for region in raw_pred
            if region.end_hz <= region.start_hz or region.start_hz < 1000
        )

        pred_intervals = _merge_intervals(_regions_to_intervals(raw_pred))
        label_intervals = _merge_intervals(_regions_to_intervals(raw_label))

        pred_len = _interval_length(pred_intervals)
        label_len = _interval_length(label_intervals)
        intersection = _intersection_length(pred_intervals, label_intervals)
        union = pred_len + label_len - intersection

        total_predicted_hz += pred_len
        total_label_hz += label_len
        total_intersection_hz += intersection
        sample_ious.append(_safe_divide(intersection, union))

        total_pred_regions += len(pred_intervals)
        total_label_regions += len(label_intervals)
        tp, fp, fn = _region_match_counts(
            pred_intervals, label_intervals, iou_threshold=0.10
        )
        matched_tp += tp
        matched_fp += fp
        matched_fn += fn

    frequency_precision = _safe_divide(total_intersection_hz, total_predicted_hz)
    frequency_recall = _safe_divide(total_intersection_hz, total_label_hz)
    frequency_f1 = _safe_divide(
        2 * frequency_precision * frequency_recall,
        frequency_precision + frequency_recall,
    )

    region_precision = _safe_divide(matched_tp, matched_tp + matched_fp)
    region_recall = _safe_divide(matched_tp, matched_tp + matched_fn)
    region_f1 = _safe_divide(
        2 * region_precision * region_recall,
        region_precision + region_recall,
    )

    return {
        "samples": len(all_sample_ids),
        "label_regions": total_label_regions,
        "predicted_regions": total_pred_regions,
        "frequency_precision": round(frequency_precision, 4),
        "frequency_recall": round(frequency_recall, 4),
        "frequency_f1": round(frequency_f1, 4),
        "mean_sample_iou": round(float(sum(sample_ious) / len(sample_ious)), 4)
        if sample_ious
        else 0.0,
        "region_precision_iou_0_10": round(region_precision, 4),
        "region_recall_iou_0_10": round(region_recall, 4),
        "region_f1_iou_0_10": round(region_f1, 4),
        "invalid_prediction_count": invalid_prediction_count,
    }


def evaluate_part2(
    predictions: dict[str, bool],
    labels: dict[str, bool],
) -> dict:
    """Evaluate lubrication predictions against ground truth.

    Example return: ``{"my_metric": 0.85, "another_metric": 42}``.
    """
    raise NotImplementedError
