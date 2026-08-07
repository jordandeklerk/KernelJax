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

That route assumes more than the estimator needs. Writing $m$ as a ratio of integrals
requires a joint density for $(X, Y)$ and $\mathbb{E}|Y| < \infty$, while Nadaraya-Watson
itself requires neither and applies perfectly well to a discrete response. Read the
derivation as motivation for the form rather than as its most general justification.

## The bias of the local constant fit

Having built the estimator we should ask what it costs, exactly as we did for the density.
The calculation takes more care here, because $\hat m_{\mathrm{NW}}$ is a ratio of two
random quantities rather than a single average, and the expectation of a ratio is not the
ratio of the expectations. The care is repaid, because the answer contains a term that has
no business being there and the rest of the page is about removing it.

Write $\hat g(x) = \frac{1}{n}\sum_i k_h(x - X_i) Y_i$ for the numerator and $\hat f(x)$
for the denominator, so that $\hat m_{\mathrm{NW}} = \hat g / \hat f$.

Take the numerator first. Conditioning on $X$ and using
$\mathbb{E}[Y \mid X] = m(X)$ replaces $Y_i$ by $m(X_i)$ inside the expectation, so
$\hat g$ is estimating the *product* $\varphi = m f$ rather than $m$ itself,

$$
\begin{aligned}
\mathbb{E}[\hat g(x)]
 &= \int k_h(x - u)\, m(u) f(u)\, du
  = \int k(v)\, \varphi(x + hv)\, dv \\[4pt]
 &= \varphi(x) + \frac{h^2}{2}\mu_2(k)\, \varphi''(x) + o(h^2),
\end{aligned}
$$

by exactly the substitution and Taylor expansion of the previous page. Differentiating the
product twice gives $\varphi'' = m'' f + 2 m' f' + m f''$, and the denominator we already
know,

$$
\mathbb{E}[\hat f(x)] = f(x) + \frac{h^2}{2}\mu_2(k)\, f''(x) + o(h^2).
$$

A ratio is not the ratio of the expectations, so we linearize. Subtracting $m(x)$ and
putting the two over a common denominator,

$$
\hat m_{\mathrm{NW}}(x) - m(x) = \frac{\hat g(x) - m(x)\, \hat f(x)}{\hat f(x)},
$$

and replacing $\hat f$ in the denominator by $f$. Consistency alone does not license that
step. It needs the rate $\hat f(x) - f(x) = O_p\bigl(h^2 + (nh)^{-1/2}\bigr)$ together with
$n h^3 \to \infty$, which make the remainder $o_p(h^2)$. The numerator of that expression has
expectation

$$
\begin{aligned}
\mathbb{E}\bigl[\hat g - m(x)\, \hat f\bigr]
 &= \frac{h^2}{2}\mu_2(k)\Bigl[\, m'' f + 2 m' f' + m f'' - m f'' \,\Bigr] + o(h^2) \\[4pt]
 &= \frac{h^2}{2}\mu_2(k)\bigl[\, m'' f + 2 m' f' \,\bigr] + o(h^2),
\end{aligned}
$$

where the two $m f''$ terms cancel. That cancellation is the whole point. The curvature of
the design density drops out, but the interaction between the slope of $m$ and the slope of
$f$ does not. Dividing by $f(x)$,

$$
\operatorname{Bias}\bigl[\hat m_{\mathrm{NW}}(x)\bigr]
 = h^2 \mu_2(k) \left\{ \frac{m''(x)}{2}
   + \frac{m'(x) f'(x)}{f(x)} \right\} + o(h^2).
$$

The variance follows from the same linearization. To leading order $\hat m_{\mathrm{NW}}$
is a weighted average of the $Y_i$ with weights $k_h(x - X_i) / \sum_j k_h(x - X_j)$, and
those weights are $O(1/nh)$ apiece, so

$$
\operatorname{Var}\bigl[\hat m_{\mathrm{NW}}(x)\bigr]
 = \frac{R(k)\, \sigma^2(x)}{n h f(x)} + o\!\left(\frac{1}{nh}\right),
$$

the $R(k)$ and the $nhf(x)$ arriving exactly as they did for the density.

The variance is what we would expect, the noise level $\sigma^2(x)$ divided by the effective
local sample size $nhf(x)$. The bias, however, contains two terms, and the second is
troubling. The quantity $m'(x) f'(x) / f(x)$ has nothing to do with the curvature of $m$. It
is a *design bias*. Even where $m$ is perfectly straight, so that $m'' = 0$ and there is
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
can,

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

Fitting a line rather than a level ought to help, but we have not yet shown that it does,
nor that it costs nothing. Both follow from writing the degree-one fit out explicitly,
because the weights it ends up applying have two properties the local constant weights lack.

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
and are worth stating, since they are the whole story,

$$
\sum_{i=1}^n \ell_i(x) = 1, \qquad \sum_{i=1}^n \ell_i(x)\, (X_i - x) = 0 .
$$

The first says the weights reproduce constants; the second says they reproduce linear
functions exactly. Any $m$ that is exactly linear across the window is therefore estimated
without bias, conditional on the design and however the covariates are distributed. That is
a statement about the conditional mean; the noise in the responses is untouched.

The variance also follows directly. Because $\hat m_{\mathrm{LL}}$ is linear in the
responses and the errors are conditionally uncorrelated, and since the weights depend on
the design the equality below is a conditional one,

$$
\operatorname{Var}\bigl[\hat m_{\mathrm{LL}}(x) \,\big|\, X_1, \dots, X_n\bigr]
 = \sum_{i=1}^n \ell_i(x)^2\, \sigma^2(X_i),
$$

and the same expansion that produced $R(k)$ for the density gives

$$
\sum_{i=1}^n \ell_i(x)^2 = \frac{R(k)}{n h f(x)} + o\!\left(\frac{1}{nh}\right).
$$

Carrying both through,

$$
\begin{aligned}
\operatorname{Bias}\bigl[\hat m_{\mathrm{LL}}(x)\bigr]
 &= \frac{h^2}{2}\, \mu_2(k)\, m''(x) + o(h^2), \\[6pt]
\operatorname{Var}\bigl[\hat m_{\mathrm{LL}}(x)\bigr]
 &= \frac{R(k)\, \sigma^2(x)}{n h f(x)} + o\!\left(\frac{1}{nh}\right).
\end{aligned}
$$

Compare this with the local constant result. The leading interior variance is identical, so
the improvement costs nothing at that order, though finite-sample variances differ and the
boundary constants differ as the previous paragraph describes. The bias has lost its $m'f'/f$ term entirely and depends only on the curvature of $m$,
which is the irreducible cost of smoothing. Fan and Gijbels (1996) call this *design
adaptivity*, since the estimator's leading bias does not depend on the design density at all.

The same argument repairs the boundary, though it is worth being exact about what survives.
Because the weights reproduce linear functions on whatever data lie inside the window, the
bias stays $O(h^2)$ right up to the edge, where the local constant fit degrades to $O(h)$.
The *order* is what carries over; the constant is not. At an edge the equivalent kernel is
truncated to one side, so $\mu_2(k)$ is replaced by the second moment of that truncated
kernel, which for a Gaussian at a left endpoint is about $-0.752$ rather than $1$. Local
linear is boundary adaptive in rate, not in constant. That is still enough to make an
explicit boundary correction unnecessary, which is the practical case Hastie and Loader
(1993) make for reaching for it by default.

Higher degrees continue the pattern. Odd $p$ is generally preferred to even, since moving
from $p = 2j$ to $p = 2j+1$ removes a design-dependent bias term without inflating the
variance order, exactly as we saw moving from $p = 0$ to $p = 1$.

## The smoother matrix

Since $\hat m(x)$ is linear in $\mathbf{y}$ for every $x$, stacking the fits at the
observed covariates gives $\hat{\mathbf{m}} = H \mathbf{y}$ where the $i$th row of the
*smoother matrix* $H$ is $\ell(X_i)^\top$. Explicitly,

$$
H_{ij} = e_1^\top \bigl(\mathbf{X}_{X_i}^\top \mathbf{W}_{X_i} \mathbf{X}_{X_i}\bigr)^{-1}
\mathbf{X}_{X_i}^\top \mathbf{W}_{X_i} e_j .
$$

The trace of $H$ plays the role that a parameter count plays in a parametric model, and we
call it the *effective degrees of freedom*, $\nu = \operatorname{tr}(H)$. It typically decreases as
$h$ grows, running from roughly $n$ at $h \to 0$, where the fit interpolates, down toward
$p+1$ as $h \to \infty$, where it becomes a global polynomial fit, though strict monotonicity
is not guaranteed for every design and kernel. The diagonal entries
$H_{ii}$ are the leverages.

Two conventions for the residual variance are in use. The plain one is
$\hat\sigma^2 = \mathrm{RSS}/n$. The corrected one is usually written
$\mathrm{RSS}/(n - \nu)$, but that denominator is exact only when $H$ is a symmetric
idempotent projection, which a local polynomial smoother is not. In general
$\mathbb{E}[\mathrm{RSS} \mid X] = \sigma^2 \operatorname{tr}\bigl[(I - H)^\top (I - H)\bigr]$,
so the exact residual degrees of freedom are
$n - 2\operatorname{tr}(H) + \operatorname{tr}(H^\top H)$. KernelJax takes the plain
convention and reports $\sqrt{\mathrm{RSS}/n}$ as `residual_se`, which is also what the
corrected Akaike criterion of the [next page](selection.md) is defined against. Both
quantities reappear there.

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
