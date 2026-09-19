"""Backward-compatible imports for PyTorch inference services."""

from pig_behavior.services.pt_inference import (
    PTDetection,
    PTModelService,
    PTPrediction,
    print_pt_prediction,
    run_pt_inference,
)

__all__ = [
    "PTDetection",
    "PTModelService",
    "PTPrediction",
    "print_pt_prediction",
    "run_pt_inference",
]
