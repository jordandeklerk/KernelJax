"""Low-level JAX interface for nonparametric kernel smoothing of mixed-type data."""

from kerneljax.bandwidth import Bandwidth, ConditionalBandwidth, SelectionResult, normal_reference
from kerneljax.data import MixedData, grid, quantile_grid
from kerneljax.estimators import (
    ConditionalFit,
    QuantileFit,
    DensityFit,
    DistributionFit,
    LocalPolyFit,
    cdensity,
    cdf,
    cdist,
    cquantile,
    density,
    local_poly,
)
from kerneljax.kernels import (
    AitchisonAitken,
    ContinuousKernel,
    Gaussian,
    KernelSet,
    LiRacine,
    OrderedKernel,
    UnorderedKernel,
    WangVanRyzin,
)
from kerneljax.ksum import ksum, kweights
from kerneljax.linalg import WLS, hat_diagonal, wls
from kerneljax.selection import (
    DensityCriterion,
    DistributionCriterion,
    RegressionCriterion,
    aic_c_regression,
    cv_cdf_distribution,
    cv_ls_density,
    cv_ls_regression,
    cv_ml_density,
    lbfgs,
    select_bandwidth,
)
from kerneljax.summary import Summary, summary

__version__ = "0.0.1"

__all__ = [
    "WLS",
    "AitchisonAitken",
    "Bandwidth",
    "ConditionalBandwidth",
    "ConditionalFit",
    "ContinuousKernel",
    "DensityCriterion",
    "DensityFit",
    "DistributionCriterion",
    "DistributionFit",
    "Gaussian",
    "KernelSet",
    "LiRacine",
    "LocalPolyFit",
    "MixedData",
    "OrderedKernel",
    "QuantileFit",
    "RegressionCriterion",
    "SelectionResult",
    "Summary",
    "UnorderedKernel",
    "WangVanRyzin",
    "aic_c_regression",
    "cdensity",
    "cdf",
    "cdist",
    "cquantile",
    "cv_cdf_distribution",
    "cv_ls_density",
    "cv_ls_regression",
    "cv_ml_density",
    "density",
    "grid",
    "hat_diagonal",
    "ksum",
    "kweights",
    "lbfgs",
    "local_poly",
    "normal_reference",
    "quantile_grid",
    "select_bandwidth",
    "summary",
    "wls",
]
