"""Commonly-used model configurations.

Faithful port of the reference (compiled) ``potions`` package's
``common_models.py``.  Each class fixes a class-level ``structure`` (a nested
list of ``(layer -> zones)`` of :mod:`~potions.hydro` zones) on top of the
:class:`~potions.model.Model` calibration engine.  The reference imports its
zones from the compiled ``potions.core``; this project imports the pure-Python
zone classes from :mod:`~potions.hydro` (the public API of the zones is
identical).

The structures are transcribed verbatim.  The ``HbvModel`` ``repr`` is kept
with one mechanical change: it reads zone fields through the
:meth:`~potions.hydro_model.HydrologicalModel.hydro_zones` name-keyed dict
instead of ``self[<name>]`` indexing, because :class:`HydrologicalModel`
subscripts by flat integer index in this implementation.
"""

from __future__ import annotations

from typing import ClassVar, cast

from .hydro import GroundZone, GroundZoneB, SnowZone, SurfaceZone
from .hydro_model import LayerT
from .model import Model

__all__: list[str] = [
    "HbvModel",
    "HbvLateralModel",
    "HbvNonlinearModel",
    "ThreeLayerModel",
    "LateralThreeLayerModel",
]


class HbvModel(Model):
    """A standard, single-column HBV-like model structure.

    Layers (top to bottom): snow -> surface -> shallow groundwater -> deep
    groundwater.  A single column, so the one river zone is the deepest layer.
    """

    structure: ClassVar[list[LayerT]] = [
        [SnowZone(name="snow")],
        [SurfaceZone(name="surface")],
        [GroundZone(name="shallow")],
        [GroundZoneB(name="deep")],
    ]

    def __repr__(self) -> str:
        """Multi-line human-readable dump of the hydrological parameters.

        Zone fields are read through the name-keyed
        :meth:`~potions.hydro_model.HydrologicalModel.hydro_zones` dict; each
        entry is cast to its concrete zone type to expose the named parameters.
        """
        snow = cast(SnowZone, self.hydro_zones["snow"])
        surface = cast(SurfaceZone, self.hydro_zones["surface"])
        shallow = cast(GroundZone, self.hydro_zones["shallow"])
        deep = cast(GroundZoneB, self.hydro_zones["deep"])
        lines: list[str] = [
            "HbvModel(",
            f"\ttt={round(snow.tt, 2)},",
            f"\tfmax={round(snow.fmax, 2)},",
            f"\tfc={round(surface.fc, 2)},",
            f"\tlp={round(surface.lp, 2)},",
            f"\tbeta={round(surface.beta, 2)},",
            f"\tk0={round(surface.k0, 4)},",
            f"\tthr={round(surface.thr, 2)},",
            f"\tk1={round(shallow.k, 4)},",
            f"\tshallow_alpha={round(shallow.alpha, 2)},",
            f"\tperc={round(shallow.perc, 2)},",
            f"\tk2={round(deep.k, 4)},",
            f"\tdeep_alpha={round(deep.alpha, 2)},",
            ")",
        ]
        return "\n".join(lines)


class HbvLateralModel(Model):
    """An HBV-like model with two lateral columns (hillslope / riparian).

    The East River showcase uses this structure.  Layers (top to bottom), each
    with a hillslope (``_hs``) and a riparian (``_rp``) column:

        snow_hs / snow_rp  (snow)
        surface_hs / surface_rp  (surface)
        shallow_hs / shallow_rp  (shallow groundwater)
        deep_hs / deep_rp  (deep groundwater, river zones)
    """

    structure: ClassVar[list[LayerT]] = [
        [SnowZone(name="snow_hs"), SnowZone(name="snow_rp")],
        [SurfaceZone(name="surface_hs"), SurfaceZone(name="surface_rp")],
        [GroundZone(name="shallow_hs"), GroundZone(name="shallow_rp")],
        [GroundZoneB(name="deep_hs"), GroundZoneB(name="deep_rp")],
    ]


class HbvNonlinearModel(Model):
    """A single-column HBV-like model with non-linear groundwater reservoirs.

    Structurally identical to :class:`HbvModel`, kept as a distinct class for
    parity with the reference (some workflows select on the class identity).
    """

    structure: ClassVar[list[LayerT]] = [
        [SnowZone(name="snow")],
        [SurfaceZone(name="surface")],
        [GroundZone(name="shallow")],
        [GroundZoneB(name="deep")],
    ]


class ThreeLayerModel(Model):
    """A simple three-layer model: Snow, Soil, and a single Groundwater zone."""

    structure: ClassVar[list[LayerT]] = [
        [SnowZone(name="snow")],
        [SurfaceZone(name="surface")],
        [GroundZoneB(name="ground")],
    ]


class LateralThreeLayerModel(Model):
    """A three-layer model with two lateral columns (hillslope / riparian).

    Snow, soil, and a single groundwater zone, each duplicated into a hillslope
    (``_hs``) and riparian (``_rp``) column.
    """

    structure: ClassVar[list[LayerT]] = [
        [SnowZone(name="snow_hs"), SnowZone(name="snow_rp")],
        [SurfaceZone(name="surface_hs"), SurfaceZone(name="surface_rp")],
        [GroundZoneB(name="ground_hs"), GroundZoneB(name="ground_rp")],
    ]
