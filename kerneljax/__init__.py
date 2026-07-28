"""Low-level JAX interface for nonparametric kernel smoothing of mixed-type data."""

from kerneljax.bandwidth import Bandwidth, BandwidthTransform, ConditionalBandwidth, normal_reference
from kerneljax.basis import LocalPolyBasis
from kerneljax.data import ColumnSpec, Kind, MixedData
from kerneljax.estimators import DensityFit, LocalPolyFit, density, local_poly
from kerneljax.kernels import (
    AitchisonAitken,
    ContinuousKernel,
    Gaussian,
    KernelSet,
    Op,
    OrderedKernel,
    UnorderedKernel,
    WangVanRyzin,
)
from kerneljax.ksum import ksum, kweights
from kerneljax.linalg import WLS, hat_diagonal, wls
from kerneljax.tuning import (
    SelectionResult,
    aic_c_regression,
    cv_ls_density,
    cv_ls_regression,
    cv_ml_density,
    lbfgs,
    select_bandwidth,
)

__version__ = "0.0.1"

__all__ = [
    "WLS",
    "AitchisonAitken",
    "Bandwidth",
    "BandwidthTransform",
    "ColumnSpec",
    "ConditionalBandwidth",
    "ContinuousKernel",
    "DensityFit",
    "Gaussian",
    "KernelSet",
    "Kind",
    "LocalPolyBasis",
    "LocalPolyFit",
    "MixedData",
    "Op",
    "OrderedKernel",
    "SelectionResult",
    "UnorderedKernel",
    "WangVanRyzin",
    "aic_c_regression",
    "cv_ls_density",
    "cv_ls_regression",
    "cv_ml_density",
    "density",
    "hat_diagonal",
    "ksum",
    "kweights",
    "lbfgs",
    "local_poly",
    "normal_reference",
    "select_bandwidth",
    "wls",
]
