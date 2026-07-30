"""Low-level JAX interface for nonparametric kernel smoothing of mixed-type data."""

from kerneljax.bandwidth import (
    Bandwidth,
    BandwidthTransform,
    ConditionalBandwidth,
    SelectionResult,
    normal_reference,
)
from kerneljax.basis import LocalPolyBasis
from kerneljax.data import ColumnSpec, Kind, MixedData, grid, quantile_grid
from kerneljax.estimators import (
    DensityFit,
    DistributionFit,
    LocalPolyFit,
    density,
    cdf,
    local_poly,
)
from kerneljax.kernels import (
    AitchisonAitken,
    ContinuousKernel,
    Gaussian,
    KernelSet,
    LiRacine,
    Op,
    OrderedKernel,
    UnorderedKernel,
    WangVanRyzin,
)
from kerneljax.ksum import ksum, kweights
from kerneljax.linalg import WLS, hat_diagonal, wls
from kerneljax.summary import Summary, summary
from kerneljax.tuning import (
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

__version__ = "0.0.1"

__all__ = [
    "WLS",
    "AitchisonAitken",
    "Bandwidth",
    "BandwidthTransform",
    "ColumnSpec",
    "ConditionalBandwidth",
    "ContinuousKernel",
    "DensityCriterion",
    "DensityFit",
    "DistributionCriterion",
    "DistributionFit",
    "Gaussian",
    "KernelSet",
    "Kind",
    "LiRacine",
    "LocalPolyBasis",
    "LocalPolyFit",
    "MixedData",
    "Op",
    "OrderedKernel",
    "RegressionCriterion",
    "SelectionResult",
    "Summary",
    "UnorderedKernel",
    "WangVanRyzin",
    "aic_c_regression",
    "cdf",
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
