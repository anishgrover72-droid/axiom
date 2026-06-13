"""Specific exception types. Raise these at boundaries; never `except: pass`."""

from __future__ import annotations


class AxiomError(Exception):
    """Base class for all AXIOM errors."""


class ConfigError(AxiomError):
    """A config value is missing, malformed, or inconsistent."""


class ContractError(AxiomError):
    """A shared-contract invariant was violated (e.g. step-segmentation drift)."""


class LabelingError(AxiomError):
    """The label foundry could not produce a valid label for a step."""


class GateError(AxiomError):
    """A capability gate (e.g. XD-PRM validation) failed; downstream is blocked."""
