# Bandwidth selection

Everything we have derived depends on $h$ and $\lambda$. On the [first page](smoothing.md)
we found the optimal bandwidth in closed form,

$$
h_{\text{opt}} = \left(\frac{R(k)}{\mu_2(k)^2 R(f'')\, n}\right)^{1/5},
$$

and immediately noticed the difficulty: it depends on $R(f'') = \int f''^2$, a functional of
the very density we are trying to estimate. Something must supply these values from the
data, and that is the last piece we need.

This page derives the criteria KernelJax minimizes, gives each in the explicit form the
library actually computes, and then makes the case that selection is doing considerably more
work here than the word *tuning* would suggest.

## Cross validation

The idea is one we can state in a sentence: hold each observation out, predict it using the
others, and choose the smoothing parameters that predict held-out observations best. This
aims at performance on new data rather than fit to the sample we have, a distinction that is
not academic. Fit to the sample is maximized by letting $h \to 0$, at which point the
estimate interpolates the observations perfectly and generalizes not at all.

Write

$$
\hat f_{-i}(x) = \frac{1}{(n-1)\prod_d h_d} \sum_{j \neq i} K_{h,\lambda}(x, X_j)
$$

for the leave-one-out density estimate, with the normalization taken over $n-1$ points.

### Likelihood cross validation

The Kullback-Leibler divergence from $f$ to $\hat f$ is
$\int f \log(f / \hat f)$, and only the term $-\int f \log \hat f$ depends on the smoothing
parameters. Estimating that expectation by its leave-one-out sample average gives the
criterion

$$
\mathrm{CV}_{ml}(h, \lambda) = -\sum_{i=1}^{n} \log \hat f_{-i}(X_i),
$$

so minimizing it is maximizing the leave-one-out log likelihood, and asymptotically it
targets Kullback-Leibler loss.

### Least squares cross validation

A different target is the integrated squared error. Expanding it,

$$
\int (\hat f - f)^2 = \int \hat f^2 - 2 \int \hat f f + \int f^2 ,
$$

the final term involves no smoothing parameters and may be discarded. The middle term is
$\mathbb{E}[\hat f(X)]$ for an independent draw $X \sim f$, which we estimate by its
leave-one-out average. What survives is

$$
\mathrm{CV}_{ls}(h, \lambda) = \int \hat f^2 - \frac{2}{n} \sum_{i=1}^{n} \hat f_{-i}(X_i),
$$

an unbiased estimate, up to the discarded constant, of the integrated squared error. A
quantity we could not compute, because it involves the unknown $f$, has been replaced by one
we can.

The first term is computable in closed form rather than by numerical integration, which
matters because it is evaluated at every step of the optimization. Expanding the square,

$$
\int \hat f^2
 = \frac{1}{n^2 \prod_d h_d} \sum_{i=1}^n \sum_{j=1}^n \bar K(X_i, X_j),
$$

where $\bar K$ is the *convolution kernel*, the self-convolution of $K$ taken column by
column. For the Gaussian this is available analytically,

$$
\bar k(u) = (k * k)(u) = \int k(t)\, k(u - t)\, dt
          = \frac{1}{2\sqrt{\pi}}\, e^{-u^2/4},
$$

the density of a normal with variance 2, and the categorical kernels have their own closed
forms obtained by summing the product over the level set. The double sum runs over the full
matrix including its diagonal, since $\int \hat f^2$ is a property of the estimate built
from all $n$ points.

### Regression criteria

For regression the analogue is the leave-one-out mean squared residual,

$$
\mathrm{CV}_{ls}(h, \lambda) = \frac{1}{n} \sum_{i=1}^{n}
\bigl(Y_i - \hat m_{-i}(X_i)\bigr)^2 .
$$

Computing this naively suggests $n$ separate fits, but for a linear smoother the leave-one-out
residual is available from the full fit alone. Since $\hat{\mathbf{m}} = H\mathbf{y}$ with
$H$ the smoother matrix of the [regression page](regression.md),

$$
Y_i - \hat m_{-i}(X_i) = \frac{Y_i - \hat m(X_i)}{1 - H_{ii}},
$$

so that

$$
\mathrm{CV}_{ls} = \frac{1}{n}\sum_{i=1}^n
\left(\frac{Y_i - \hat m(X_i)}{1 - H_{ii}}\right)^{\!2},
$$

an identity that holds exactly, not asymptotically. The leverages $H_{ii}$ come out of the
same Cholesky factorization that produced the fit.

An alternative avoids leave-one-out altogether by penalizing model complexity directly. The
corrected Akaike criterion of Hurvich, Simonoff and Tsai (1998) is

$$
\mathrm{AIC}_c(h, \lambda) = \log \hat\sigma^2
 + \frac{1 + \operatorname{tr}(H)/n}{1 - \bigl(\operatorname{tr}(H) + 2\bigr)/n},
$$

where $\hat\sigma^2 = \frac{1}{n}\sum_i (Y_i - \hat m(X_i))^2$ is the residual variance of
the full fit and $\operatorname{tr}(H)$ is the effective degrees of freedom. The penalty
diverges as $\operatorname{tr}(H) \to n - 2$, which prevents the criterion from chasing an
interpolating fit.

## What cross validation buys us

It would be reasonable to regard all of this as mere tuning, a constant to be pinned down
before the real work begins. It turns out to be considerably more than that.

Recall from the [previous page](mixed-data.md#what-the-smoothing-buys) that the discrete part
of the asymptotic bias is

$$
\lambda \sum_{\tilde x^d \,:\, d_{\tilde x, x} = 1}
\bigl[m(x^c, \tilde x^d) - m(x)\bigr] \frac{f(x^c, \tilde x^d)}{f(x)} .
$$

If the variable is irrelevant, so that $m$ does not change when its level changes, every
bracket vanishes and the bias is identically zero for *any* $\lambda$. The criterion is then
free to choose $\lambda$ purely to reduce variance, and variance falls monotonically as more
observations are pooled. It therefore drives $\lambda$ to its upper bound, at which point,
as we saw, the variable is removed from the estimate altogether.

Hall, Li and Racine (2007) made this precise and proved the corresponding statement for
continuous covariates, where the selected bandwidth diverges rather than shrinking at the
$n^{-1/(p+4)}$ rate a relevant variable would command. In both cases the column is, in
effect, deleted from the model, and we did not have to decide this ourselves. The authors
describe the effect as *automatic dimensionality reduction* and report that it is effective
at realistic sample sizes rather than merely in the limit.

Recalling our
[table of rates](smoothing.md#several-covariates-and-how-fast-we-can-hope-to-learn), this is
worth a great deal. A variable carrying no signal is not merely tolerated; it is removed, so
it costs nothing in effective dimension. Including a covariate whose relevance we are unsure
of is far less dangerous than the table alone would suggest.

It also tells us how to read a fitted model. A bandwidth sitting at or near its upper bound
is not a failure of the optimizer and not a bug. It is the criterion telling us, in the only
language available to it, that the variable does not belong.

## Selection as an optimization problem

These criteria are functions of $(h, \lambda)$ that we wish to minimize over
$\prod_d [0, \infty) \times \prod_d [0, \bar\lambda_d]$. Traditionally they are evaluated on
a grid or handed to a derivative-free search, both of which scale poorly as columns
accumulate, since every column contributes a parameter and a grid over $D$ parameters costs
exponentially in $D$.

KernelJax treats selection as what it plainly is, a smooth constrained optimization problem.
Each parameter is mapped to an unconstrained coordinate, bandwidths through a softplus and
categorical parameters through a scaled logistic carrying $\mathbb{R}$ onto
$(0, \bar\lambda_d)$, the criterion is differentiated through that map by automatic
differentiation, and L-BFGS minimizes it from several starting points. Continuous bandwidths
and categorical smoothing parameters are selected jointly, as they must be, since the
criterion depends on all of them at once.

Because the criterion is an ordinary differentiable function rather than a closed procedure,
we are also free to embed it in a larger model and learn its parameters alongside everything
else.

## Conclusions

We began by asking what it means to estimate a function without assuming its shape, and
arrived at a single construction that answers the question for densities and regressions
alike. A kernel converts distance into weight. A bandwidth sets the scale of that weight,
contributing a bias of order $h^2$ and a variance of order $1/nh$, and the balance between
them fixes both the optimal bandwidth and the $n^{-2/(4+d)}$ rate that no estimator can beat.
Fitting local polynomials rather than local constants removes the design term $m'f'/f$ from
that bias at no cost in variance, and repairs the boundary along with it. Kernels for
unordered and ordered categories extend the whole apparatus to variables with no notion of
distance, and because they enter as a product, the discrete covariates never touch the
convergence rate. Cross validation then supplies every smoothing parameter from the data,
and along the way removes the variables that were not helping.

To see these estimators at work, the [Home](../index.md) page shows them applied to data,
and [Installation](../installation.md) covers getting set up. For fuller treatments, Li and
Racine (2007) is the standard reference for mixed-type kernel methods, and Fan and Gijbels
(1996) for local polynomial modelling.

## References

- Fan, J., & Gijbels, I. (1996). [*Local Polynomial Modelling and Its
  Applications*](https://doi.org/10.1201/9780203748725). Chapman and Hall.
- Hall, P., Li, Q., & Racine, J. S. (2007). [Nonparametric estimation of regression
  functions in the presence of irrelevant
  regressors](https://doi.org/10.1162/rest.89.4.784). *The Review of Economics and
  Statistics*, 89(4), 784-789.
- Hall, P., Racine, J. S., & Li, Q. (2004). [Cross-validation and the estimation of
  conditional probability densities](https://doi.org/10.1198/016214504000000548). *Journal
  of the American Statistical Association*, 99(468), 1015-1026.
- Hurvich, C. M., Simonoff, J. S., & Tsai, C. L. (1998). [Smoothing parameter selection in
  nonparametric regression using an improved Akaike information
  criterion](https://doi.org/10.1111/1467-9868.00125). *Journal of the Royal Statistical
  Society, Series B*, 60(2), 271-293.
- Li, Q., & Racine, J. S. (2007). [*Nonparametric Econometrics: Theory and
  Practice*](https://press.princeton.edu/books/hardcover/9780691121611/nonparametric-econometrics).
  Princeton University Press.
