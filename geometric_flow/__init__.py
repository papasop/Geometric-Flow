"""Geometry-first optimization tools for PyTorch.

The package exposes three layers:

* measure: build an implicit Hessian/grad-square-like curvature operator.
* navigate: solve A * step = -grad with conjugate gradients.
* scan: probe a parameter ray and emit a phase/topography map.
"""

from .curvature import CurvatureOperator, compute_curvature, hutchinson_trace
from .navigation import conjugate_gradient, geometric_step
from .optimizer import GeometricOptimizer
from .phase import (
    PhaseGridPoint,
    PhasePoint,
    phase_diagram_scanner,
    phase_diagram_scanner_2d,
    write_phase_diagram,
    write_phase_diagram_csv,
)
from .layers import GeometricRotation
from .models import ChannelGeometricRotation, GeoCNN, GeoConv2D, GeoMLP
from . import geo

__all__ = [
    "CurvatureOperator",
    "GeometricOptimizer",
    "GeometricRotation",
    "GeoMLP",
    "GeoCNN",
    "GeoConv2D",
    "ChannelGeometricRotation",
    "PhaseGridPoint",
    "PhasePoint",
    "geo",
    "compute_curvature",
    "conjugate_gradient",
    "geometric_step",
    "hutchinson_trace",
    "phase_diagram_scanner",
    "phase_diagram_scanner_2d",
    "write_phase_diagram",
    "write_phase_diagram_csv",
]
