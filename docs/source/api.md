# API reference

Everything below is exported from the top-level `kerneljax` namespace. The
[Quickstart](quickstart.md) walks through the first few sections, which cover most use.

## Data

Mixed-type samples and the evaluation grids built from them.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   kerneljax.MixedData
   kerneljax.grid
   kerneljax.quantile_grid
```

## Estimators

The entry points. Each takes training data and a bandwidth rule, and returns a fit object.

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

Any fit passes to {func}`~kerneljax.summary` for the report shown throughout these docs.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   kerneljax.summary
```

## Bandwidth selection

Selecting once to reuse the result, starting from a plug-in rule, or swapping the solver.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   kerneljax.select_bandwidth
   kerneljax.normal_reference
   kerneljax.lbfgs
```

### Criteria

What selection minimizes, configured and handed to {func}`~kerneljax.select_bandwidth`.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   kerneljax.RegressionCriterion
   kerneljax.DensityCriterion
   kerneljax.DistributionCriterion
```

Underneath each is an ordinary differentiable function, callable directly.

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

## Kernels

One kernel per column kind, collected into a {class}`~kerneljax.KernelSet`.

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

## Containers

Smoothing parameters, one per column, with the conditional form carrying one block per
sample.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   kerneljax.Bandwidth
   kerneljax.ConditionalBandwidth
```

The results estimators and solvers hand back, read rather than constructed.

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

## Primitives

The pieces every estimator above is built from.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   kerneljax.ksum
   kerneljax.kweights
   kerneljax.wls
   kerneljax.hat_diagonal
```

## Extension points

Subclass these to add a kernel of your own.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   kerneljax.ContinuousKernel
   kerneljax.UnorderedKernel
   kerneljax.OrderedKernel
```
