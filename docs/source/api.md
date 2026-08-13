# API reference

Everything below is available from the top-level `kerneljax` namespace. For a guided introduction, see the [user guide](user-guide/index.md).

## Data

Mixed-type data containers and helpers for constructing evaluation grids.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   kerneljax.MixedData
   kerneljax.grid
   kerneljax.quantile_grid
```

## Estimators

The main estimation entry points. Each takes training data and a bandwidth rule and returns a fitted result.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   kerneljax.local_poly
   kerneljax.density
   kerneljax.cdf
   kerneljax.cdensity
   kerneljax.cdist
   kerneljax.cquantile
   kerneljax.cmode
```

## Bandwidth selection

Select bandwidths explicitly when you want to reuse a result, start from a plug-in rule, or customize the optimization.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   kerneljax.select_bandwidth
   kerneljax.normal_reference
```

### Criteria

Criterion objects configure the objective minimized by {func}`~kerneljax.select_bandwidth`.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   kerneljax.RegressionCriterion
   kerneljax.DensityCriterion
   kerneljax.DistributionCriterion
```

### Criterion functions

Each criterion is backed by an ordinary differentiable function that can also be called directly.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   kerneljax.cv_ls_regression
   kerneljax.aic_c_regression
   kerneljax.cv_ml_density
   kerneljax.cv_ls_density
   kerneljax.cv_cdf_distribution
```

### Solvers

Optimization routines used by bandwidth selection.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   kerneljax.lbfgs
```

## Kernels

Kernel families for continuous, unordered, and ordered variables. A {class}`~kerneljax.KernelSet` collects the choices used by an estimator.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   kerneljax.KernelSet
   kerneljax.Gaussian
   kerneljax.AitchisonAitken
   kerneljax.WangVanRyzin
   kerneljax.LiRacine
```

## Bandwidths

Objects that carry smoothing parameters, with the conditional form holding one bandwidth block for each sample.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   kerneljax.Bandwidth
   kerneljax.ConditionalBandwidth
```

## Results

Fit and selection objects returned by estimators and optimization routines.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   kerneljax.LocalPolyFit
   kerneljax.DensityFit
   kerneljax.DistributionFit
   kerneljax.ConditionalFit
   kerneljax.QuantileFit
   kerneljax.ModeFit
   kerneljax.SelectionResult
   kerneljax.Summary
   kerneljax.WLS
```

### Summaries

{func}`~kerneljax.summary` produces the formatted reports shown throughout the documentation.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   kerneljax.summary
```

## Primitives

Low-level operations used to build the estimators above.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   kerneljax.ksum
   kerneljax.kweights
   kerneljax.wls
   kerneljax.hat_diagonal
```

## Kernel interfaces

Base classes for implementing custom kernels.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   kerneljax.ContinuousKernel
   kerneljax.UnorderedKernel
   kerneljax.OrderedKernel
```
