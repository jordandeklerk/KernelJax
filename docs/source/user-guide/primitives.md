# Building from primitives

The estimators in the earlier pages are built from lower-level operations that KernelJax exposes directly. Those primitives are useful when you want to inspect how a shipped estimator works or assemble something the package does not provide itself.

The setup is the shared wage example from [Working with data](data.md), together with the fits and evaluation grid used earlier in the guide.

```python
import jax
import numpy as np

import kerneljax as kj

rng = np.random.default_rng(0)
n = 300
exper = rng.uniform(0, 30, n)
educ = rng.integers(0, 4, n)
region = rng.integers(0, 4, n)
wage = 8 + 0.35 * exper - 0.008 * exper**2 + 1.2 * educ + rng.normal(0, 0.6, n)

data = kj.MixedData.from_blocks(
    continuous=exper,
    unordered=region,
    ordered=educ,
    unordered_levels=4,
    ordered_levels=4,
    names=("exper", "region", "educ"),
)

fit = kj.local_poly(data, wage, "cv_ls", degree=1)
quick = kj.local_poly(data, wage, "normal_reference", degree=1)
points = kj.grid(data, vary="exper", n=200)
```

The central operation is a kernel-weighted contraction. A density contracts kernel weights against a column of ones. A Nadaraya-Watson regression contracts the same weights against the response and divides by their row sums.

Both {func}`~kerneljax.kweights` and {func}`~kerneljax.ksum` start from the same weight matrix. Write $x_j$ for evaluation point $j$, $X_i$ for training point $i$, and $K_d$ for the kernel factor associated with column $d$. Then

$$
W_{ji} = \prod_d K_d(x_{jd}, X_{id}).
$$

{func}`~kerneljax.kweights` returns this matrix directly. {func}`~kerneljax.ksum` contracts it against values attached to the training observations, so for a vector $v$,

$$
\operatorname{ksum}(v)_j = \sum_{i=1}^n W_{ji} v_i.
$$

When `v` has several columns, the contraction is performed for each one. Leaving `v` out is equivalent to supplying a column of ones, so `ksum` returns the row sums of $W$.

One convention is important before using those sums. Continuous kernel factors in $W$ contain no $1/h_d$ term, and `ksum` applies no $1/n$ normalization by default. The matrix therefore contains the product-kernel values themselves rather than all of the scale factors needed by a particular estimator.

For example, a density with continuous columns $\mathcal{C}$ needs the additional factor $1/(n \prod_{d \in \mathcal{C}} h_d)$. Regression does not, because a common scale factor cancels between its numerator and denominator. That difference is deliberate. The primitive computes the reusable contraction, while the estimator decides what normalization gives that contraction its statistical meaning.

## Rebuilding local constant regression

We can reconstruct KernelJax's local-constant estimator directly from `ksum`.

```python
numerator = kj.ksum(
    data,
    fit.bandwidth,
    wage[:, None],
    at=points,
)

denominator = kj.ksum(
    data,
    fit.bandwidth,
    at=points,
)

nadaraya_watson = (numerator / denominator).ravel()

local_constant = kj.local_poly(
    data,
    wage,
    fit.bandwidth,
    at=points,
)

gap = abs(nadaraya_watson - local_constant.mean).max()
print(f"largest gap to local_poly={gap:.3e}")
```

```text
largest gap to local_poly=1.335e-05
```

Algebraically, these are the same estimator.

$$
\hat m(x) =
\frac{
\sum_i W_i(x) Y_i
}{
\sum_i W_i(x)
}.
$$

The small numerical gap comes from evaluating that expression through two different computational paths in float32. The direct construction takes a ratio of kernel sums, while `local_poly` obtains the same local constant through its weighted least-squares solve.

Notice that we passed `fit.bandwidth`, not `fit`, to the second call. A bare {class}`~kerneljax.Bandwidth` carries smoothing parameters but no polynomial degree, so {func}`~kerneljax.local_poly` falls back to its default degree of zero. Passing the complete fit would also carry `degree=1`, reproducing the local linear fit instead.

The distinction matters for other carried settings as well. A bare bandwidth does not remember the kernels that produced it. This example uses KernelJax's defaults throughout, but when working with custom kernels you should pass the corresponding `kernels=` explicitly when calling the primitives.

## Differentiating a criterion

The selection criteria are public functions too, and they remain differentiable with respect to a bandwidth.

```python
def cv_ls(bandwidth):
    return kj.cv_ls_regression(
        data,
        bandwidth,
        y=wage,
        degree=1,
    )


at_plug_in = jax.grad(cv_ls)(quick.bandwidth)
at_selected = jax.grad(cv_ls)(fit.bandwidth)

for label, grad in [("plug-in", at_plug_in), ("selected", at_selected)]:
    print(
        f"{label:9s} "
        f"d/dh={grad.h[0]:+.3e}  "
        f"d/dlam_uno={grad.lam_uno[0]:+.3e}  "
        f"d/dlam_ord={grad.lam_ord[0]:+.3e}"
    )
```

```text
plug-in   d/dh=-3.008e-02  d/dlam_uno=-3.304e+00  d/dlam_ord=+3.895e-07
selected  d/dh=+2.045e-07  d/dlam_uno=-6.394e-05  d/dlam_ord=+1.146e-06
```

The gradient has the same pytree structure as the bandwidth itself, including one derivative for each categorical smoothing parameter. These numbers are derivatives with respect to the bandwidths in their **natural constrained scale**. They are not exactly the gradients seen by the bandwidth optimizer. As described in the [introduction](intro.md#keep-the-full-path-differentiable), selection works in unconstrained coordinates and differentiates through the transformation back to $h$ and $\lambda$.

For the interior parameters, the natural-scale derivatives at the selected solution are already close to zero. Region is different. Its unordered smoothing parameter is at the Aitchison-Aitken upper bound, so a constrained optimum does not require $\partial \mathrm{CV} / \partial \lambda$ itself to vanish. In fact, a negative derivative at the upper boundary says that the criterion would prefer still more smoothing if the admissible range allowed it.

The optimizer sees an even flatter direction because the logistic transformation approaches zero slope near the boundary. If $\lambda = b,\, \sigma(z)$, then

$$
\frac{\partial \mathrm{CV}}{\partial z}
=
\frac{\partial \mathrm{CV}}{\partial \lambda}
\, b\, \sigma(z)\bigl(1-\sigma(z)\bigr),
$$

so the transformed gradient can be close to zero even when the natural-scale derivative is not exactly zero. This is the optimization view of the same statistical result seen in [Local polynomial regression](regression.md#reading-the-bandwidths), where region has been smoothed to its maximum-smoothing limit.

The lower-level interface is useful precisely because none of these pieces are hidden. Kernels, bandwidths, contractions, and criteria can be inspected or recombined into estimators beyond the ones KernelJax ships directly.

## The weight matrix and per-column operators

{func}`~kerneljax.ksum` performs the contraction, while {func}`~kerneljax.kweights` exposes the matrix being contracted. Its first axis indexes evaluation points and its second indexes training observations.

```python
weights = kj.kweights(
    data,
    fit.bandwidth,
    at=points,
)

print(weights.shape)
```

```text
(200, 300)
```

By default, every column contributes its ordinary kernel value. The `op` argument lets that interpretation change. A single string applies one operator to every column. A mapping can choose an operator by column kind, and a tuple specifies one operator per column in the sample's original column order.

For a continuous column, let

$$
u = \frac{x_{jd} - X_{id}}{h_d}.
$$

The main operators are

$$
\begin{aligned}
K_d^{\mathrm{value}}(x_{jd}, X_{id})   &= k(u), \\\\
K_d^{\mathrm{cdf}}(x_{jd}, X_{id})     &= \int_{-\infty}^u k(t)\,dt, \\\\
K_d^{\mathrm{deriv}}(x_{jd}, X_{id})   &= \frac{1}{h_d} k'(u), \\\\
K_d^{\mathrm{conv}}(x_{jd}, X_{id})    &= (k * k)(u).
\end{aligned}
$$

`value`, `cdf`, and `conv` contain no external bandwidth divisor. `deriv` is different because differentiation is with respect to the evaluation coordinate $x_{jd}$, so the chain rule contributes $1/h_d$. For the default Gaussian kernel, `cdf` is $\Phi(u)$ and `conv` is the Gaussian self-convolution, equivalently the normal density with variance two evaluated at $u$.

Categorical kernels use the same operator interface where the operation is meaningful. An unordered kernel has no natural cumulative ordering, so the default Aitchison-Aitken kernel provides `value` and `conv` but not `cdf`. An ordered kernel can provide a `cdf` by summing its mass over integer levels at or below the evaluation level. Its convolution similarly sums the overlap of two kernel functions over the integer lattice.

A per-column operator tuple combines these choices into

$$
W_{ji}^{(o)} = \prod_d K_d^{o_d}(x_{jd}, X_{id}),
$$

where $o_d$ names the operator used for column $d$.

This makes mixed constructions possible. A single product can, for example, accumulate one ordered or continuous coordinate while leaving another at its ordinary kernel value. Conditional estimators use the same operator machinery, although they apply it to separate conditioning and response blocks rather than constructing one all-purpose mixed matrix.

Here we accumulate the continuous experience coordinate while leaving region and education at their ordinary kernel values.

```python
accumulated = kj.kweights(
    data,
    fit.bandwidth,
    at=points,
    op=("cdf", "value", "value"),
)

print(f"value weights sum on the first row  {float(weights[0].sum()):8.3f}")
print(f"cdf weights sum on the first row    {float(accumulated[0].sum()):8.3f}")
```

```text
value weights sum on the first row     1.268
cdf weights sum on the first row       1.052
```

Neither row sum is an estimator by itself because neither includes the normalization appropriate to its operator.

For the ordinary value weights, restoring the sample-size and continuous-bandwidth factors gives a mixed density estimate,

$$
\hat f(x_j) = \frac{1}{n\prod_{d\in\mathcal{C}} h_d}
\sum_{i=1}^n W_{ji}.
$$

The second construction is different. Only experience has been accumulated, while region and education remain pointwise kernel factors. Dividing by $n$ gives

$$
\hat H(x_j) = \frac{1}{n}
\sum_{i=1}^n W_{ji}^{(\mathrm{cdf},\,\mathrm{value},\,\mathrm{value})}.
$$

It is useful to think of $\hat H$ as **cumulative in experience and pointwise in the two categorical coordinates**. It is not the full multivariate CDF returned by {func}`~kerneljax.cdf`, and with an unordered column in the sample there is no full joint CDF for KernelJax to construct through a cumulative operator in every coordinate. The first quantity is likewise a density estimate evaluated at the bandwidth selected for the regression, not at a bandwidth selected specifically for density estimation. The primitives do not attach a statistical interpretation on their own. That interpretation comes from the operator choices, normalization, bandwidths, and contraction assembled around them.

That separation is the point of the primitive interface. Changing an operator can change the mathematical object being estimated, while the underlying machinery for constructing and contracting kernel weights stays the same.
