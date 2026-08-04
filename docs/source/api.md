# API reference

Everything below is exported from the top-level `kerneljax` namespace. The
[Quickstart](quickstart.md) walks through the first few sections, which cover most use.

## Estimators

The entry points. Each takes training data and a bandwidth rule, and returns a fit object.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   kerneljax.local_poly
   kerneljax.density
   kerneljax.cdf
```

Once we've ran an estimator, we can pass the fit object to {func}`~kerneljax.summary`. It reads a density or regression fit back and renders the report shown throughout these docs.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   kerneljax.summary
```

## Kernels

One kernel per column kind, collected into a {class}`~kerneljax.KernelSet` and passed to any
estimator through `kernels=`. The defaults are a second-order Gaussian for continuous
columns, Aitchison-Aitken for unordered ones and Wang-van Ryzin for ordered ones.

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

## Bandwidth selection

Passing a string to an estimator covers the common case. These are for selecting once and
reusing the result, fixing a bandwidth by hand, or swapping the solver.
{class}`~kerneljax.Bandwidth` is the container the rest of this section produces and every
estimator consumes, holding one smoothing parameter per column.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   kerneljax.Bandwidth
   kerneljax.select_bandwidth
   kerneljax.normal_reference
   kerneljax.lbfgs
```

### Criteria

What selection minimizes. The classes configure a method and are what you hand to
{func}`~kerneljax.select_bandwidth`.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   kerneljax.RegressionCriterion
   kerneljax.DensityCriterion
   kerneljax.DistributionCriterion
```

Underneath each is an ordinary differentiable function, callable directly and usable inside
a larger JAX program.

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

## Data

Mixed-type samples and the evaluation grids built from them. A column's kind determines
which kernel applies to it.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   kerneljax.MixedData
   kerneljax.grid
   kerneljax.quantile_grid
```

## Fit objects

What the estimators and solvers hand back. You read these rather than construct them, so
the pages below are mostly attribute listings. Each pairs with a callable listed elsewhere,
so {func}`~kerneljax.local_poly` returns a {class}`~kerneljax.LocalPolyFit` and
{func}`~kerneljax.wls` returns a {class}`~kerneljax.WLS`.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   kerneljax.LocalPolyFit
   kerneljax.DensityFit
   kerneljax.DistributionFit
   kerneljax.SelectionResult
   kerneljax.Summary
   kerneljax.WLS
```

## Primitives

The kernel weight matrix, the contraction underneath every estimator, and the weighted least
squares solver behind local polynomial regression. Build estimators that no entry point
above covers directly from these.

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

Subclass these to add a kernel of your own. A kernel must be elementwise and differentiable
in its smoothing parameter.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   kerneljax.ContinuousKernel
   kerneljax.UnorderedKernel
   kerneljax.OrderedKernel
```
