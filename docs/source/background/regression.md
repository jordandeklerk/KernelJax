# Kernel regression

On the [previous page](smoothing.md) we built a kernel density estimator and derived its
bias and variance. We now collect the reward. This page turns that machinery into an
estimator of the regression function $m(x) = \mathbb{E}[Y \mid X = x]$, derives the bias of
that estimator, finds a term in it that ought not to be there, and removes it.

Throughout we assume the model $Y_i = m(X_i) + \varepsilon_i$ with
$\mathbb{E}[\varepsilon_i \mid X_i] = 0$ and
$\operatorname{Var}[\varepsilon_i \mid X_i = x] = \sigma^2(x)$, that $m$ and the design
density $f$ are twice continuously differentiable, and that $f(x) > 0$ at the point of
interest.

## From densities to regression

We begin by writing the regression function in terms of densities,

$$
m(x) = \mathbb{E}[Y \mid X = x] = \frac{\int y\, f(x, y)\, dy}{f(x)} .
$$

Both numerator and denominator are functionals of densities, and densities are something we
can now estimate. Take a product kernel estimate of the joint density,

$$
\hat f(x, y) = \frac{1}{n} \sum_{i=1}^n k_h(x - X_i)\, k_b(y - Y_i),
$$

with bandwidth $b$ in the response direction, and integrate against $y$. Only the second
factor depends on $y$, and substituting $y = Y_i + bv$ gives

$$
\int y\, k_b(y - Y_i)\, dy = \int (Y_i + bv)\, k(v)\, dv = Y_i ,
$$

since $k$ integrates to one and has zero first moment. The response bandwidth therefore
disappears entirely, which is why kernel regression carries no bandwidth for $Y$. The
numerator collapses to $\frac{1}{n}\sum_i k_h(x - X_i) Y_i$, and dividing by the kernel
density estimate of $f(x)$ leaves

$$
\hat m_{\mathrm{NW}}(x)
 = \frac{\sum_{i=1}^n k_h(x - X_i)\, Y_i}{\sum_{i=1}^n k_h(x - X_i)} .
$$

This is the *Nadaraya-Watson* estimator. It is exactly the weighted average of responses our
opening intuition suggested, and we did not postulate it; it fell out of substituting
density estimates into the definition of a conditional expectation.

## The bias of the local constant fit

Deriving the bias of a ratio requires a little care, since $\mathbb{E}[A/B] \neq
\mathbb{E}[A]/\mathbb{E}[B]$. Writing $\hat g(x) = \frac{1}{n}\sum_i k_h(x - X_i) Y_i$ for
the numerator and $\hat f(x)$ for the denominator, a first-order expansion of the ratio
about $(\mathbb{E}\hat g, \mathbb{E}\hat f)$ together with the expansions of the previous
page gives, after some algebra,

$$
\operatorname{Bias}\bigl[\hat m_{\mathrm{NW}}(x)\bigr]
 = h^2 \mu_2(k) \left\{ \frac{m''(x)}{2}
   + \frac{m'(x) f'(x)}{f(x)} \right\} + o(h^2),
$$

while the variance is

$$
\operatorname{Var}\bigl[\hat m_{\mathrm{NW}}(x)\bigr]
 = \frac{R(k)\, \sigma^2(x)}{n h f(x)} + o\!\left(\frac{1}{nh}\right).
$$

The variance is what we would expect: the noise level $\sigma^2(x)$ divided by the effective
local sample size $nhf(x)$. The bias, however, contains two terms, and the second is
troubling. The quantity $m'(x) f'(x) / f(x)$ has nothing to do with the curvature of $m$. It
is a *design bias*: even where $m$ is perfectly straight, so that $m'' = 0$ and there is
nothing to smooth away, the estimator is biased whenever the covariates are unevenly
distributed.

The reason is easy to see once stated. A window centered at $x$ collects more observations
from whichever side is denser, so the weighted average sits nearer the centroid of those
observations than to $x$ itself. If $m$ is sloping, evaluating it at the wrong place costs
us $m'(x)$ times the displacement, and the displacement is proportional to $f'(x)/f(x)$.

The same mechanism does something worse at the edge of the support. There the window is
one-sided by construction, the displacement no longer shrinks in proportion to $h^2$, and
the bias degrades to $O(h)$. For a Gaussian kernel at a left endpoint the leading term is

$$
\operatorname{Bias}\bigl[\hat m_{\mathrm{NW}}\bigr] \approx m'(x)\, h
\frac{\int_0^\infty v\, k(v)\, dv}{\int_0^\infty k(v)\, dv}
 = m'(x)\, h \, \frac{2}{\sqrt{2\pi}} \approx 0.798\, m'(x)\, h .
$$

An order of $h$ rather than $h^2$ is a serious loss, and it afflicts a neighborhood of the
boundary whose width is proportional to $h$.

## Local polynomials

The cleanest way to repair this is to notice what the Nadaraya-Watson estimator actually is.
Its weighted average is precisely the solution of

$$
\min_{\beta_0} \sum_{i=1}^n k_h(x - X_i)\, (Y_i - \beta_0)^2 ,
$$

a weighted least squares problem fitting a *constant* near $x$. A constant cannot follow a
tilted neighborhood, which is exactly the defect we diagnosed. So let us fit something that
can:

$$
\min_{\beta \in \mathbb{R}^{p+1}} \sum_{i=1}^n k_h(x - X_i)
\Bigl( Y_i - \sum_{j=0}^{p} \beta_j (X_i - x)^j \Bigr)^2 ,
$$

reporting $\hat\beta_0$ as our estimate of $m(x)$ and refitting at every evaluation point.
Collecting the design into

$$
\mathbf{X}_x = \begin{bmatrix}
1 & (X_1 - x) & \cdots & (X_1 - x)^p \\
\vdots & \vdots & & \vdots \\
1 & (X_n - x) & \cdots & (X_n - x)^p
\end{bmatrix},
$$

$$
\mathbf{W}_x = \operatorname{diag}\bigl(k_h(x - X_1), \dots, k_h(x - X_n)\bigr),
$$

the solution is the familiar weighted least squares formula, and the estimate is its first
component,

$$
\hat\beta(x) = \bigl(\mathbf{X}_x^\top \mathbf{W}_x \mathbf{X}_x\bigr)^{-1}
               \mathbf{X}_x^\top \mathbf{W}_x \mathbf{y},
\qquad
\hat m(x) = e_1^\top \hat\beta(x),
$$

with $e_1$ the first standard basis vector. Note that $\hat\beta_j(x)$ estimates
$m^{(j)}(x)/j!$, so the same fit delivers derivative estimates at no extra cost.

Degree $p = 0$ returns us to Nadaraya-Watson. Degree $p = 1$, which we call *local linear*
regression, is the usual default, and we can now say exactly why.

## Why local linear is the default

Write $s_j = \sum_i k_h(x - X_i)(X_i - x)^j$ for the weighted moments of the design. Solving
the $2 \times 2$ system explicitly, the local linear estimate is

$$
\hat m_{\mathrm{LL}}(x) = \sum_{i=1}^n \ell_i(x)\, Y_i,
\qquad
\ell_i(x) = \frac{k_h(x - X_i)\bigl[s_2 - (X_i - x)\, s_1\bigr]}{s_0 s_2 - s_1^2}.
$$

These $\ell_i(x)$ are the *equivalent kernel* weights. Unlike the Nadaraya-Watson weights,
which are always positive, they may be negative, and it is exactly this that lets the fit
tilt to accommodate an asymmetric window. Two identities follow directly from the formula
and are worth stating, since they are the whole story:

$$
\sum_{i=1}^n \ell_i(x) = 1, \qquad \sum_{i=1}^n \ell_i(x)\, (X_i - x) = 0 .
$$

The first says the weights reproduce constants; the second says they reproduce linear
functions exactly. Any $m$ that is locally linear is therefore fitted without error,
regardless of how the design is distributed. Carrying this through the expansion gives

$$
\begin{aligned}
\operatorname{Bias}\bigl[\hat m_{\mathrm{LL}}(x)\bigr]
 &= \frac{h^2}{2}\, \mu_2(k)\, m''(x) + o(h^2), \\[6pt]
\operatorname{Var}\bigl[\hat m_{\mathrm{LL}}(x)\bigr]
 &= \frac{R(k)\, \sigma^2(x)}{n h f(x)} + o\!\left(\frac{1}{nh}\right).
\end{aligned}
$$

Compare this with the local constant result. The variance is identical, so nothing has been
paid. The bias has lost its $m'f'/f$ term entirely and depends only on the curvature of $m$,
which is the irreducible cost of smoothing. Fan and Gijbels (1996) call this *design
adaptivity*: the estimator's leading bias does not depend on the design density at all.

The same argument repairs the boundary. Because the weights reproduce linear functions on
whatever data lie inside the window, a one-sided window costs nothing to leading order, and
the bias remains $O(h^2)$ right up to the edge with the same constant as in the interior. No
explicit boundary correction is required, which is the practical case Hastie and Loader
(1993) make for reaching for local linear by default.

Higher degrees continue the pattern. Odd $p$ is generally preferred to even, since moving
from $p = 2j$ to $p = 2j+1$ removes a design-dependent bias term without inflating the
variance order, exactly as we saw moving from $p = 0$ to $p = 1$.

## The smoother matrix

Since $\hat m(x)$ is linear in $\mathbf{y}$ for every $x$, stacking the fits at the
observed covariates gives $\hat{\mathbf{m}} = H \mathbf{y}$, where the $i$th row of the
*smoother matrix* $H$ is $\ell(X_i)^\top$. Explicitly,

$$
H_{ij} = e_1^\top \bigl(\mathbf{X}_{X_i}^\top \mathbf{W}_{X_i} \mathbf{X}_{X_i}\bigr)^{-1}
\mathbf{X}_{X_i}^\top \mathbf{W}_{X_i} e_j .
$$

The trace of $H$ plays the role that a parameter count plays in a parametric model, and we
call it the *effective degrees of freedom*, $\nu = \operatorname{tr}(H)$. It decreases as
$h$ grows, running from roughly $n$ at $h \to 0$, where the fit interpolates, down toward
$p+1$ as $h \to \infty$, where it becomes a global polynomial fit. The diagonal entries
$H_{ii}$ are the leverages, and the residual variance is estimated by
$\hat\sigma^2 = \mathrm{RSS} / (n - \nu)$. Both quantities reappear in
[Bandwidth selection](selection.md).

In KernelJax the moment matrices $\mathbf{X}_x^\top \mathbf{W}_x \mathbf{X}_x$ and
$\mathbf{X}_x^\top \mathbf{W}_x \mathbf{y}$ are formed directly, without ever building
$\mathbf{W}_x$, and the system is solved by Cholesky factorization. Writing
$\mathbf{X}_x^\top \mathbf{W}_x \mathbf{X}_x = LL^\top$ and solving $Lz = e_1$ by forward
substitution gives the leverage as $H_{ii} = k_h(0)\, z^\top z$, so the coefficients and the
leverages come out of a single factorization.

## Where next

We can now estimate a regression function and say precisely how well, provided every
covariate is continuous. Real datasets are rarely so obliging, and
[Mixed-type data](mixed-data.md) takes up the question of what to do about the categorical
ones.

## References

- Fan, J., & Gijbels, I. (1996). [*Local Polynomial Modelling and Its
  Applications*](https://doi.org/10.1201/9780203748725). Chapman and Hall.
- Hastie, T., & Loader, C. (1993). [Local regression: automatic kernel
  carpentry](https://doi.org/10.1214/ss/1177011002). *Statistical Science*, 8(2), 120-129.
- Ruppert, D., & Wand, M. P. (1994). [Multivariate locally weighted least squares
  regression](https://doi.org/10.1214/aos/1176325632). *The Annals of Statistics*, 22(3),
  1346-1370.
