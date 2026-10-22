"""
src/evaluation/metrics.py
=========================
Rigorous, CPU-compatible evaluation suite for industrial visual anomaly detection.
Computes image-level AUROC, pixel-level localization AUROC, per-defect-type score
distributions, and inference latency benchmarks.
"""

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import torch

from src.models.anomaly_detector import FeatureAnomalyDetector


@dataclass
class ImageEvaluationResult:
    """Evaluation metrics for image-level anomaly classification."""
    roc_auc: float
    best_threshold: float
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    total_samples: int
    total_anomalies: int
    total_normals: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "roc_auc": round(self.roc_auc, 4),
            "best_threshold": round(self.best_threshold, 6),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "accuracy": round(self.accuracy, 4),
            "confusion_matrix": {
                "TP": self.true_positives,
                "FP": self.false_positives,
                "TN": self.true_negatives,
                "FN": self.false_negatives,
            },
            "counts": {
                "total": self.total_samples,
                "anomalies": self.total_anomalies,
                "normals": self.total_normals,
            },
        }


@dataclass
class PixelEvaluationResult:
    """Evaluation metrics for pixel-level anomaly localization."""
    roc_auc: float
    total_pixels: int
    defect_pixels: int
    normal_pixels: int
    defect_pixel_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pixel_roc_auc": round(self.roc_auc, 4),
            "total_pixels": self.total_pixels,
            "defect_pixels": self.defect_pixels,
            "normal_pixels": self.normal_pixels,
            "defect_pixel_ratio": round(self.defect_pixel_ratio, 6),
        }


@dataclass
class DefectTypeStats:
    """Statistics of anomaly scores within a specific defect category."""
    defect_name: str
    count: int
    mean_score: float
    std_score: float
    min_score: float
    max_score: float
    p50_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "defect_name": self.defect_name,
            "count": self.count,
            "mean_score": round(self.mean_score, 6),
            "std_score": round(self.std_score, 6),
            "min_score": round(self.min_score, 6),
            "max_score": round(self.max_score, 6),
            "p50_score": round(self.p50_score, 6),
        }


@dataclass
class LatencyBenchmarkResult:
    """Benchmark results for single-image CPU inference latency."""
    device: str
    warmup_runs: int
    timed_runs: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    throughput_fps: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "warmup_runs": self.warmup_runs,
            "timed_runs": self.timed_runs,
            "mean_ms": round(self.mean_ms, 2),
            "std_ms": round(self.std_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "p50_ms": round(self.p50_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "throughput_fps": round(self.throughput_fps, 2),
        }


def compute_image_auroc(
    y_true: Sequence[int],
    y_scores: Sequence[float],
) -> ImageEvaluationResult:
    """
    Compute image-level Area Under the Receiver Operating Characteristic Curve (ROC-AUC),
    along with optimal decision threshold and binary classification metrics.
    
    Args:
        y_true: Binary ground-truth labels (0 for normal, 1 for anomaly).
        y_scores: Continuous anomaly scores (higher indicates anomalous).
        
    Returns:
        ImageEvaluationResult containing ROC-AUC, threshold, F1, and confusion matrix.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_scores_arr = np.asarray(y_scores, dtype=np.float64)

    if len(y_true_arr) == 0:
        raise ValueError("Cannot compute AUROC on empty label array.")

    unique_labels = np.unique(y_true_arr)
    if len(unique_labels) < 2:
        raise ValueError(
            f"AUROC requires both positive and negative samples, but got labels: {unique_labels.tolist()}"
        )

    # Compute ROC curve and AUC
    auc = float(roc_auc_score(y_true_arr, y_scores_arr))

    fpr, tpr, thresholds = roc_curve(y_true_arr, y_scores_arr)

    # Find optimal threshold using Youden's J statistic: max(TPR - FPR)
    # Exclude infinite threshold returned by scikit-learn at index 0 if present
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_threshold = float(thresholds[best_idx])

    # If best threshold is inf (boundary edge case), fallback to median of scores
    if np.isinf(best_threshold):
        best_threshold = float(np.median(y_scores_arr))

    # Evaluate predictions at optimal threshold
    preds = (y_scores_arr >= best_threshold).astype(np.int32)
    tp = int(np.sum((preds == 1) & (y_true_arr == 1)))
    fp = int(np.sum((preds == 1) & (y_true_arr == 0)))
    tn = int(np.sum((preds == 0) & (y_true_arr == 0)))
    fn = int(np.sum((preds == 0) & (y_true_arr == 1)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = float((tp + tn) / len(y_true_arr))

    return ImageEvaluationResult(
        roc_auc=auc,
        best_threshold=best_threshold,
        precision=precision,
        recall=recall,
        f1_score=f1,
        accuracy=accuracy,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        total_samples=len(y_true_arr),
        total_anomalies=int(np.sum(y_true_arr == 1)),
        total_normals=int(np.sum(y_true_arr == 0)),
    )


def compute_pixel_auroc(
    masks: Sequence[np.ndarray],
    anomaly_maps: Sequence[np.ndarray],
) -> PixelEvaluationResult:
    """
    Compute pixel-level Area Under the Receiver Operating Characteristic Curve (ROC-AUC)
    across a collection of ground-truth masks and predicted spatial error maps.
    
    Standard MVTec AD protocol:
    All spatial error maps and binary masks across the test set are flattened and
    evaluated together to test spatial localization fidelity.

    Args:
        masks: List of 2D binary numpy arrays (H, W) where 1=defect, 0=normal.
        anomaly_maps: List of 2D float numpy arrays (H, W) of anomaly scores.

    Returns:
        PixelEvaluationResult with pixel-level AUROC and pixel distribution stats.
    """
    if len(masks) != len(anomaly_maps):
        raise ValueError(f"Mismatch in counts: {len(masks)} masks vs {len(anomaly_maps)} maps.")
    if len(masks) == 0:
        raise ValueError("Cannot compute pixel AUROC on empty input.")

    flat_masks: List[np.ndarray] = []
    flat_maps: List[np.ndarray] = []

    for idx, (m, a) in enumerate(zip(masks, anomaly_maps)):
        if m.shape != a.shape:
            raise ValueError(
                f"Sample {idx} shape mismatch: mask {m.shape} vs anomaly map {a.shape}."
            )
        flat_masks.append((m.flatten() > 0.5).astype(np.uint8))
        flat_maps.append(a.flatten().astype(np.float32))

    all_masks = np.concatenate(flat_masks)
    all_maps = np.concatenate(flat_maps)

    unique_labels = np.unique(all_masks)
    if len(unique_labels) < 2:
        raise ValueError(
            f"Pixel AUROC requires both defective and normal pixels, but got labels: {unique_labels.tolist()}"
        )

    auc = float(roc_auc_score(all_masks, all_maps))
    total_px = int(len(all_masks))
    defect_px = int(np.sum(all_masks == 1))
    normal_px = total_px - defect_px
    ratio = float(defect_px / total_px) if total_px > 0 else 0.0

    return PixelEvaluationResult(
        roc_auc=auc,
        total_pixels=total_px,
        defect_pixels=defect_px,
        normal_pixels=normal_px,
        defect_pixel_ratio=ratio,
    )


def compute_per_defect_stats(
    labels: Sequence[str],
    scores: Sequence[float],
) -> Dict[str, DefectTypeStats]:
    """
    Compute summary statistics of predicted anomaly scores segmented by defect category.

    Args:
        labels: Category/defect name string for each sample (e.g. 'good', 'broken_large').
        scores: Predicted anomaly scores for each sample.

    Returns:
        Dictionary mapping defect name to DefectTypeStats.
    """
    if len(labels) != len(scores):
        raise ValueError("Labels and scores sequences must have identical length.")

    grouped: Dict[str, List[float]] = {}
    for lbl, scr in zip(labels, scores):
        grouped.setdefault(lbl, []).append(float(scr))

    stats: Dict[str, DefectTypeStats] = {}
    for name, scrs in sorted(grouped.items()):
        arr = np.asarray(scrs, dtype=np.float64)
        stats[name] = DefectTypeStats(
            defect_name=name,
            count=len(arr),
            mean_score=float(np.mean(arr)),
            std_score=float(np.std(arr)),
            min_score=float(np.min(arr)),
            max_score=float(np.max(arr)),
            p50_score=float(np.median(arr)),
        )

    return stats


def benchmark_inference_latency(
    detector: FeatureAnomalyDetector,
    sample_image: Union[Image.Image, torch.Tensor],
    warmup_runs: int = 5,
    timed_runs: int = 30,
) -> LatencyBenchmarkResult:
    """
    Benchmark single-image inference latency on CPU using high-resolution monotonic timer.

    Args:
        detector: Initialized FeatureAnomalyDetector instance.
        sample_image: Sample image (PIL or Tensor) of representative dimensions.
        warmup_runs: Number of unmeasured passes to populate caches and JIT paths.
        timed_runs: Number of timed repetitions to compute mean and percentiles.

    Returns:
        LatencyBenchmarkResult containing detailed millisecond statistics.
    """
    # 1. Warm-up passes
    for _ in range(warmup_runs):
        _ = detector.inspect_image(sample_image)

    # 2. Timed passes
    latencies_ms: List[float] = []
    for _ in range(timed_runs):
        t0 = time.perf_counter()
        _ = detector.inspect_image(sample_image)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    arr = np.asarray(latencies_ms, dtype=np.float64)
    mean_ms = float(np.mean(arr))
    throughput = float(1000.0 / mean_ms) if mean_ms > 0 else 0.0

    return LatencyBenchmarkResult(
        device=str(detector.device),
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
        mean_ms=mean_ms,
        std_ms=float(np.std(arr)),
        min_ms=float(np.min(arr)),
        max_ms=float(np.max(arr)),
        p50_ms=float(np.median(arr)),
        p95_ms=float(np.percentile(arr, 95)),
        throughput_fps=throughput,
    )


def generate_evaluation_overlay(
    image: Image.Image,
    normalized_heatmap: np.ndarray,
    ground_truth_mask: Optional[np.ndarray] = None,
    alpha: float = 0.45,
) -> Image.Image:
    """
    Create an explainable multi-panel visual artifact:
    Panel 1: Original input image
    Panel 2: Anomaly heatmap (jet/turbo colormap in pure Pillow/NumPy)
    Panel 3: Heatmap overlaid on original image
    Panel 4 (if mask provided): Ground-truth mask overlay for visual comparison
    """
    # Ensure RGB
    base_img = image.convert("RGB")
    w, h = base_img.size

    # Ensure heatmap matches image size
    if normalized_heatmap.shape != (h, w):
        heat_img = Image.fromarray((normalized_heatmap * 255.0).astype(np.uint8))
        heat_img = heat_img.resize((w, h), Image.Resampling.BILINEAR)
        heat_arr = np.asarray(heat_img, dtype=np.float32) / 255.0
    else:
        heat_arr = np.clip(normalized_heatmap, 0.0, 1.0)

    # Pure NumPy Jet Colormap implementation
    # R: max(0, min(1, 1.5 - |4*x - 3|))
    # G: max(0, min(1, 1.5 - |4*x - 2|))
    # B: max(0, min(1, 1.5 - |4*x - 1|))
    r = np.clip(1.5 - np.abs(4.0 * heat_arr - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * heat_arr - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * heat_arr - 1.0), 0.0, 1.0)
    color_heat = np.stack([r, g, b], axis=-1)
    heat_rgb = (color_heat * 255.0).astype(np.uint8)
    heatmap_pil = Image.fromarray(heat_rgb, mode="RGB")

    # Blend overlay: base_img * (1 - alpha) + heatmap * alpha
    base_np = np.asarray(base_img, dtype=np.float32)
    overlay_np = (1.0 - alpha) * base_np + alpha * heat_rgb.astype(np.float32)
    overlay_pil = Image.fromarray(np.clip(overlay_np, 0, 255).astype(np.uint8), mode="RGB")

    # If mask is provided, create a visual mask panel
    panels = [base_img, heatmap_pil, overlay_pil]

    if ground_truth_mask is not None:
        if ground_truth_mask.shape != (h, w):
            mask_p = Image.fromarray((ground_truth_mask * 255.0).astype(np.uint8))
            mask_p = mask_p.resize((w, h), Image.Resampling.NEAREST)
            m_arr = np.asarray(mask_p, dtype=np.float32) / 255.0
        else:
            m_arr = (ground_truth_mask > 0.5).astype(np.float32)

        # Mask visual panel: black & white mask
        mask_rgb = (m_arr * 255.0).astype(np.uint8)
        mask_pil = Image.fromarray(np.stack([mask_rgb] * 3, axis=-1), mode="RGB")
        panels.append(mask_pil)

    # Stitch side-by-side
    total_w = sum(p.width for p in panels)
    max_h = max(p.height for p in panels)
    combined = Image.new("RGB", (total_w, max_h), (20, 20, 25))

    cur_x = 0
    for p in panels:
        combined.paste(p, (cur_x, 0))
        cur_x += p.width

    return combined
