"""
src/evaluation/__init__.py
==========================
Evaluation metrics for Industrial AI Inspector:
- Image-level AUROC & classification metrics
- Pixel-level localization AUROC
- Defect-type breakdown and score distributions
- Monotonic inference latency benchmarking
- Visual explainability overlay generation
"""

from src.evaluation.metrics import (
    ImageEvaluationResult,
    PixelEvaluationResult,
    DefectTypeStats,
    LatencyBenchmarkResult,
    compute_image_auroc,
    compute_pixel_auroc,
    compute_per_defect_stats,
    benchmark_inference_latency,
    generate_evaluation_overlay,
)

__all__ = [
    "ImageEvaluationResult",
    "PixelEvaluationResult",
    "DefectTypeStats",
    "LatencyBenchmarkResult",
    "compute_image_auroc",
    "compute_pixel_auroc",
    "compute_per_defect_stats",
    "benchmark_inference_latency",
    "generate_evaluation_overlay",
]
