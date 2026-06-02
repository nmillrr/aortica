"""Regulatory document templates and auto-population utilities."""

from aortica.regulatory.populate_atd import (
    ATDPopulationResult,
    populate_atd,
    validate_atd_completeness,
)
from aortica.regulatory.reporting_checklists import (
    ChecklistResult,
    generate_reporting_checklist,
)

__all__ = [
    "ATDPopulationResult",
    "ChecklistResult",
    "generate_reporting_checklist",
    "populate_atd",
    "validate_atd_completeness",
]
