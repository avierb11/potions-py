"""potions: pure-Python + PyTorch re-implementation of the potions hydrology /
reactive-transport ODE model.

The model supports batched parameter sets on GPU (float32 on CUDA, float64 on
CPU).  All Jacobians used by the internal root solvers are analytic (derived
from the Rust reference implementation in the ``potions`` project); batched
spatial/parameter dimensions map directly onto tensor lanes.
"""

from .common_types import (
    BatchResults,
    ChemicalState,
    ForcingData,
    HydroForcing,
    HydroModelResults,
    HydroModelStep,
    HydroStep,
    LapseRateParameters,
    MiscData,
    ModelResults,
    RtForcing,
    RtModelResults,
    RtStep,
    RtZoneConfiguration,
    Scalar,
    ZERO_CONC,
)
from .common_models import (
    HbvLateralModel,
    HbvModel,
    HbvNonlinearModel,
    LateralThreeLayerModel,
    ThreeLayerModel,
)
from .database import (
    ChemicalDatabase,
    ExchangeReaction,
    MineralKineticData,
    MineralSpecies,
    MonodReaction,
    PrimaryAqueousSpecies,
    SecondarySpecies,
    TstReaction,
)
from .device import DEFAULT_DEVICE, DEFAULT_DTYPE, get_device
from .equilibrium import EquilibriumParameters
from .exceptions import (
    HydrologyError,
    HydrologyNumericalError,
    OptimizationError,
    PotionsError,
    ReactiveTransportError,
    RtNumericalError,
)
from .model import Model
from .model_batched import BatchedRunOutput, run_hydro_rt_batched
from .hydro import (
    GroundZone,
    GroundZoneB,
    HydrologicZone,
    SnowZone,
    SurfaceZone,
)
from .hydro_model import (
    DEFAULT_OBJECTIVE_FUNCTIONS,
    HydrologicalModel,
    Layer,
)
from .kinetic_structures import (
    MineralAuxParams,
    MineralParameters,
    MonodParameters,
    RiverDimensions,
    RiverParameters,
    RtParameters,
    TstParameters,
    ZoneDimensions,
)
from .reaction_network import ReactionNetwork
from .rt_zone import RiverZone, RtZone, WaterVolumeError

__all__: list[str] = [
    "BatchedRunOutput",
    "BatchResults",
    "ChemicalDatabase",
    "ChemicalState",
    "DEFAULT_DEVICE",
    "DEFAULT_DTYPE",
    "DEFAULT_OBJECTIVE_FUNCTIONS",
    "EquilibriumParameters",
    "ExchangeReaction",
    "ForcingData",
    "GroundZone",
    "GroundZoneB",
    "HbvLateralModel",
    "HbvModel",
    "HbvNonlinearModel",
    "HydroForcing",
    "HydrologicZone",
    "HydrologyError",
    "HydrologyNumericalError",
    "HydrologicalModel",
    "HydroModelResults",
    "HydroModelStep",
    "HydroStep",
    "LapseRateParameters",
    "LateralThreeLayerModel",
    "Layer",
    "MineralAuxParams",
    "MineralKineticData",
    "MineralParameters",
    "MineralSpecies",
    "MiscData",
    "Model",
    "ModelResults",
    "MonodParameters",
    "MonodReaction",
    "OptimizationError",
    "PotionsError",
    "PrimaryAqueousSpecies",
    "RtForcing",
    "RtModelResults",
    "RtNumericalError",
    "RtParameters",
    "RtStep",
    "RtZone",
    "RtZoneConfiguration",
    "ReactiveTransportError",
    "ReactionNetwork",
    "RiverDimensions",
    "RiverParameters",
    "RiverZone",
    "Scalar",
    "SecondarySpecies",
    "SnowZone",
    "SurfaceZone",
    "ThreeLayerModel",
    "TstParameters",
    "TstReaction",
    "WaterVolumeError",
    "ZoneDimensions",
    "ZERO_CONC",
    "get_device",
    "run_hydro_rt_batched",
]

__version__: str = "0.1.0"