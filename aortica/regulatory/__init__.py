"""Regulatory document templates and auto-population utilities."""

from aortica.regulatory.populate_atd import (
    ATDPopulationResult,
    populate_atd,
    validate_atd_completeness,
)

__all__ = [
    "ATDPopulationResult",
    "populate_atd",
    "validate_atd_completeness",
]
