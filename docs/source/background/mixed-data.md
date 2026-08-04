# Mixed-type data

So far we have quietly assumed that every covariate is continuous, and real datasets rarely
oblige. A wage regression carries years of experience alongside region, union status and
education level. Nothing we have said applies to these, because the distance $x - X_i$ that
our kernel consumes is meaningless for a variable whose values are labels.

This page builds kernels for categorical variables, first for unordered levels and then for
ordered ones, combines them with the continuous kernel of the [first page](smoothing.md)
into a single product weight, and states the asymptotic theory for the resulting estimator.
That theory contains the payoff: the discrete covariates do not enter the convergence rate
at all.

Write $X_i = (X_i^c, X_i^d)$ for the split of a covariate vector into its $p$ continuous and
$q$ discrete components.

## Why splitting is not enough

The traditional response is to split the sample. We divide the data by category and smooth
the continuous variables separately within each cell. This is consistent and it imposes no
functional form, so it is a legitimate thing to do, but the cost is easy to quantify.

Suppose the $q$ discrete covariates have $c_1, \dots, c_q$ levels. The cells are formed from
their cross product, so there are $\prod_j c_j$ of them and each retains roughly
$n / \prod_j c_j$ observations. From the [previous page](regression.md), the variance of a
local polynomial fit is inversely proportional to the effective local sample size, so
splitting inflates every variance by the factor $\prod_j c_j$ while leaving the bias
untouched. Three variables with four levels each give 64 cells, so a sample of 10,000 leaves
about 150 observations in each, and the standard errors are eight times what the full sample
would support. Looking back at our
[table of rates](smoothing.md#several-covariates-and-how-fast-we-can-hope-to-learn), this is
not a comfortable position.

The insight of Aitchison and Aitken (1976) is that the trade we already accepted for
continuous variables is available here too. Instead of discarding observations from other
categories, we can *downweight* them. An observation from a neighboring region still tells
us something about how wages vary with experience, and using it at reduced weight buys us
variance at the cost of bias, exactly as widening a bandwidth does. All we need is a kernel
suited to labels.

## Unordered categories

For a variable whose $c$ levels have no natural ordering, Aitchison and Aitken proposed

$$
L_\lambda(x, X_i) = \begin{cases}
1 - \lambda & x = X_i \\[4pt]
\dfrac{\lambda}{c - 1} & x \neq X_i
\end{cases}
$$

with $\lambda \in \left[0, \frac{c-1}{c}\right]$. This is a proper weighting scheme in the
sense that it sums to one over the level set, since one level matches and $c-1$ do not:

$$
\sum_{s=0}^{c-1} L_\lambda(s, X_i) = (1 - \lambda) + (c-1)\cdot\frac{\lambda}{c-1} = 1 .
$$

Here $\lambda$ plays the part of a bandwidth, and we learn a great deal by examining the two
ends of its range.

At $\lambda = 0$ the kernel is simply an indicator. Only observations sharing the category
contribute anything, and we recover exactly the splitting estimator described above.

At $\lambda = \frac{c-1}{c}$ something more interesting happens. The matching branch equals
$1 - \frac{c-1}{c} = \frac{1}{c}$, while the non-matching branch equals
$\frac{c-1}{c} \big/ (c-1) = \frac{1}{c}$. The two coincide, every level receives identical
weight $1/c$, and since a constant factor cancels from the numerator and denominator of any
ratio estimator, the variable exerts no influence on our estimate whatsoever. It has been
smoothed away entirely.

So the single parameter $\lambda$ carries us continuously from using a variable in full to
discarding it, and where we land is determined by the data. This is a more powerful
statement than it may appear, and [the next page](selection.md) returns to it.

For estimators built as ratios of kernel sums, such as regressions, the normalization is
unnecessary because it cancels, and a simpler variant is often used,

$$
L_\lambda(x, X_i) = \begin{cases} 1 & x = X_i \\ \lambda & x \neq X_i \end{cases}
\qquad \text{with } \lambda \in [0, 1],
$$

which reaches complete smoothing at $\lambda = 1$ rather than at $(c-1)/c$ and does not
require the level count at all.

## Ordered categories

When the levels *are* ordered, as with an education level or a rating scale, the unordered
kernel discards information by giving every non-matching level the same weight. We should
prefer adjacent levels to count for more than distant ones. Wang and van Ryzin (1981)
proposed

$$
L_\lambda(x, X_i) = \begin{cases}
1 - \lambda & x = X_i \\[4pt]
\tfrac{1}{2}(1 - \lambda)\, \lambda^{|x - X_i|} & x \neq X_i
\end{cases}
$$

with $\lambda \in [0, 1]$, so that weight decays geometrically in the number of levels
separating the two values. The factor of $\tfrac{1}{2}$ divides the mass between the two
directions, and the weights sum to one over the integers:

$$
(1 - \lambda) + 2 \sum_{d=1}^{\infty} \tfrac{1}{2}(1-\lambda)\lambda^{d}
 = (1 - \lambda) + (1-\lambda)\frac{\lambda}{1 - \lambda}
 = (1 - \lambda) + \lambda = 1 .
$$

As before, $\lambda = 0$ returns us to splitting, and as $\lambda \to 1$ every weight tends
to zero at the same rate, so in a ratio estimator the variable is smoothed away.

KernelJax also offers the normalized Li and Racine variant,

$$
L_\lambda(x, X_i) = \frac{1 - \lambda}{1 + \lambda}\, \lambda^{|x - X_i|},
$$

which handles matching and non-matching levels with a single expression. It is normalized on
the same support, by the same geometric series,

$$
\frac{1-\lambda}{1+\lambda}\left(1 + 2\sum_{d=1}^\infty \lambda^d\right)
 = \frac{1-\lambda}{1+\lambda}\cdot\frac{1+\lambda}{1-\lambda} = 1 .
$$

## The product kernel

We now have three kernels and a dataset that may need all of them at once. The weight
attaching observation $i$ to the evaluation point $x$ is simply their product,

$$
\begin{aligned}
K_{h,\lambda}(x, X_i) = {}&
  \prod_{d \in \mathcal{C}} \frac{1}{h_d}\,
  k\!\left(\frac{x_d - X_{id}}{h_d}\right) \\[4pt]
 &\times \prod_{d \in \mathcal{U}} L_{\lambda_d}(x_d, X_{id})
  \prod_{d \in \mathcal{O}} L_{\lambda_d}(x_d, X_{id}),
\end{aligned}
$$

taken over the continuous, unordered and ordered index sets, with its own smoothing
parameter for every column.

This product form is what makes mixed-type data tractable without a proliferation of special
cases. Each column contributes exactly one factor, determined solely by its kind, so any
estimator we write in terms of $K_{h,\lambda}$ never needs to know how the columns are
divided up. Substituting this weight into the local polynomial problem of the
[previous page](regression.md) gives a regression accepting arbitrary mixtures of variable
types, and every estimator in KernelJax is assembled in precisely this way.

## What the smoothing buys

The asymptotic theory for this estimator is due to Racine and Li (2004), and it is worth
stating carefully because it contains the entire justification for the approach.

Writing $\hat m$ for the mixed-data estimator with smoothing parameters $(h, \lambda)$, and
assuming $h \to 0$, $\lambda \to 0$ and $n h^p \to \infty$, their Theorem 2.1 gives

$$
\sqrt{n h^p}\,\bigl(\hat m(x) - m(x) - B(h, \lambda)\bigr)
\;\xrightarrow{d}\; \mathcal{N}\bigl(0, \Omega(x)\bigr),
$$

with asymptotic variance

$$
\Omega(x) = \frac{\sigma^2(x)\, R(k)^p}{f(x)}
$$

and a bias that separates into a continuous and a discrete part,

$$
\begin{aligned}
B(h, \lambda) = {}& h^2 \mu_2(k)
  \left\{ \frac{\nabla f(x)^\top \nabla m(x)}{f(x)}
         + \frac{\operatorname{tr}\bigl[\nabla^2 m(x)\bigr]}{2} \right\} \\[6pt]
 &+ \lambda \sum_{\tilde x^d \,:\, d_{\tilde x, x} = 1}
   \bigl[m(x^c, \tilde x^d) - m(x)\bigr]\,
   \frac{f(x^c, \tilde x^d)}{f(x)} .
\end{aligned}
$$

The inner sum runs over those level combinations differing from $x^d$ in exactly one
component, so the discrete bias is a weighted average of how much $m$ moves when a single
category is changed. It is precisely zero when $m$ does not depend on that variable, which is
the fact the next page exploits.

Three consequences are worth drawing out.

The first, and the reason all of this is worth doing, is that the normalization is
$\sqrt{n h^p}$, involving only $p$, the number of *continuous* covariates. The discrete
covariates do not appear in the rate. Splitting, by contrast, replaces $n$ with the cell
count $n / \prod_j c_j$ and so converges more slowly by a constant factor that grows
geometrically in the number of discrete variables. Smoothing the categorical variables keeps
the whole sample in play.

The second is that the discrete bias is $O(\lambda)$, linear rather than quadratic, while
the continuous bias is $O(h^2)$. Balancing the two therefore calls for $\lambda \propto h^2$.
Since the variance is unaffected by $\lambda$ to leading order, and the optimal continuous
bandwidth is $h \propto n^{-1/(p+4)}$ as on the first page, the optimal smoothing parameter
for a relevant categorical variable behaves like $\lambda \propto n^{-2/(p+4)}$. It goes to
zero faster than $h$ does.

The third is that the continuous part of the bias still carries the design term
$\nabla f^\top \nabla m / f$ that we met on the previous page, because the estimator analysed
above is local constant. Fitting local linear in the continuous directions removes it, at no
cost in variance, exactly as before.

## Where next

Our estimator is now general enough for real data, and we know how fast it converges. But it
has acquired a smoothing parameter for every column and we have not yet said where any of
them come from. [Bandwidth selection](selection.md) answers that, and shows that the answer
does more work than one might expect.

## References

- Aitchison, J., & Aitken, C. G. G. (1976). [Multivariate binary discrimination by the
  kernel method](https://doi.org/10.1093/biomet/63.3.413). *Biometrika*, 63(3), 413-420.
- Li, Q., & Racine, J. S. (2007). [*Nonparametric Econometrics: Theory and
  Practice*](https://press.princeton.edu/books/hardcover/9780691121611/nonparametric-econometrics).
  Princeton University Press.
- Racine, J. S., & Li, Q. (2004). [Nonparametric estimation of regression functions with
  both categorical and continuous
  data](https://doi.org/10.1016/S0304-4076(03)00157-X). *Journal of Econometrics*, 119(1),
  99-130.
- Wang, M. C., & van Ryzin, J. (1981). [A class of smooth estimators for discrete
  distributions](https://doi.org/10.1093/biomet/68.1.301). *Biometrika*, 68(1), 301-309.
