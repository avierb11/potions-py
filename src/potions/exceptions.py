"""Exception hierarchy mirroring the reference (compiled) ``potions`` package.

The compiled reference groups its errors under ``PotionsError`` and defines two
solver-specific numeric errors — :class:`HydrologyNumericalError` (a zone
storage solve failed) and :class:`RtNumericalError` (a reactive-transport mass
balance solve failed).  Downstream code (e.g. the showcase calibration
pipeline) imports these directly from ``potions`` and catches them, so we
recreate the same hierarchy here with the same constructors and the same rich
constructor-captured fields, so failure records and ``isinstance`` checks
behave identically.

Notes
-----
* :class:`OptimizationError` exists in the compiled core as the base "root
  solver failed" error; it has no body in the reference, so it is redefined
  here as a plain ``Exception``.
* Zone types are imported under :data:`TYPE_CHECKING` only to avoid a runtime
  import cycle (the runtime values are already plain Python objects).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .common_types import HydroForcing, RtForcing
    from .hydro import HydrologicZone
    from .rt_zone import RtZone

__all__: list[str] = [
    "PotionsError",
    "OptimizationError",
    "HydrologyError",
    "ReactiveTransportError",
    "HydrologyNumericalError",
    "RtNumericalError",
]


class PotionsError(Exception):
    """Base class for every error raised by the potions model."""


class OptimizationError(Exception):
    """A root-finding / optimization solve did not converge or became invalid.

    Mirrors the compiled ``potions.core.OptimizationError`` (an opaque failure
    marker carried inside :class:`RtNumericalError`).
    """


class HydrologyError(PotionsError):
    """A hydrological (water-balance) solve failed."""


class ReactiveTransportError(PotionsError):
    """A reactive-transport mass-balance solve failed."""


class HydrologyNumericalError(HydrologyError):
    """Raised when a hydrological zone storage solve fails.

    Captures the model type, the zone, and its parameters / state / forcing at
    the point of failure, mirroring the reference constructor.
    """

    def __init__(
        self,
        model_type: type,
        zone: HydrologicZone,
        parameters: np.ndarray,
        state: float,
        hydro_forcing: HydroForcing,
        *args: object,
    ) -> None:
        super().__init__(*args)
        self.model_type: type = model_type
        self.zone: HydrologicZone = zone
        self.parameters: np.ndarray = parameters
        self.state: float = state
        self.hydro_forcing: HydroForcing = hydro_forcing


class RtNumericalError(ReactiveTransportError):
    """Raised when a reactive-transport zone mass-balance solve fails.

    ``error_type`` is ``"iteration"`` (Newton/LM did not converge),
    ``"linear_system"`` (a singular Jacobian could not be solved), or
    ``"other"``.  ``math_err`` carries the underlying :class:`OptimizationError`.
    """

    def __init__(
        self,
        error_type: Literal["iteration", "linear_system", "other"],
        model_type: type,
        zone: RtZone,
        parameters: np.ndarray,
        state: np.ndarray,
        rt_forcing: RtForcing,
        math_err: OptimizationError,
        *args: object,
    ) -> None:
        super().__init__(*args)
        self.error_type: Literal["iteration", "linear_system", "other"] = error_type
        self.model_type: type = model_type
        self.zone: RtZone = zone
        self.parameters: np.ndarray = parameters
        self.state: np.ndarray = state
        self.rt_forcing: RtForcing = rt_forcing
        self.math_err: OptimizationError = math_err
