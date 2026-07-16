"""Functional-space optimization tools for redundant parameterizations."""

from .curvature import CurvatureOperator, compute_curvature, hutchinson_trace
from .fixed_rank import FixedRankDiagnostics, FixedRankManifold
from .fixed_rank_optimizer import CapacityAdaptiveQuotientFlow, FixedRankFunctionalAdam, SubsteppedQuotientFlow
from .functional_geometry import (
    FunctionalMap,
    FunctionalGeometry,
    FunctionalJTJOperator,
    MatrixFreeFunctionalJTJOperator,
    functional_projectors,
    functional_response_operator,
    implicit_cg_response_direction,
    low_rank_response_direction,
    projected_functional_geoflow_direction,
    randomized_normal_basis,
)
from .navigation import conjugate_gradient, geometric_step
from .optimizer import GeometricOptimizer
from .product_state import ProductParameter, ProductState
from .split_metric import (
    InverseGramDiagnostics,
    InverseGramDirection,
    inverse_gram_direction,
    product_capacity,
    product_velocity,
    split_metric_norm,
)
from .trust_region import HeldOutTrustRegion, TrustRegionResult
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
    "FixedRankDiagnostics",
    "CapacityAdaptiveQuotientFlow",
    "FixedRankFunctionalAdam",
    "FixedRankManifold",
    "InverseGramDiagnostics",
    "InverseGramDirection",
    "SubsteppedQuotientFlow",
    "GeometricOptimizer",
    "GeometricRotation",
    "FunctionalMap",
    "FunctionalGeometry",
    "FunctionalJTJOperator",
    "MatrixFreeFunctionalJTJOperator",
    "HeldOutTrustRegion",
    "GeoMLP",
    "GeoCNN",
    "GeoConv2D",
    "ChannelGeometricRotation",
    "PhaseGridPoint",
    "PhasePoint",
    "ProductParameter",
    "ProductState",
    "TrustRegionResult",
    "geo",
    "compute_curvature",
    "conjugate_gradient",
    "geometric_step",
    "hutchinson_trace",
    "inverse_gram_direction",
    "product_capacity",
    "product_velocity",
    "split_metric_norm",
    "functional_projectors",
    "functional_response_operator",
    "implicit_cg_response_direction",
    "low_rank_response_direction",
    "projected_functional_geoflow_direction",
    "randomized_normal_basis",
    "phase_diagram_scanner",
    "phase_diagram_scanner_2d",
    "write_phase_diagram",
    "write_phase_diagram_csv",
]
