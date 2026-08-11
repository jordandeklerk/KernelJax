# Mixed-type data

So far we have quietly assumed that every covariate is continuous, and real datasets rarely
oblige. A wage regression carries years of experience alongside region, union status and
education level. Nothing we have said applies to these, because the distance $x - X_i$ that
our kernel consumes is meaningless for a variable whose values are labels.

This page builds kernels for categorical variables, first for unordered levels and then for
ordered ones, combines them with the continuous kernel of the [first page](smoothing.md)
into a single product weight, and states the asymptotic theory for the resulting estimator.
That theory contains the payoff, since the discrete covariates do not enter the convergence rate
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
splitting inflates the variance at a cell of probability $\pi$ by $1/\pi$, which is
$\prod_j c_j$ when the cells are equally populated and worse than that in the sparse ones,
and it leaves the bias untouched. Three variables with four levels each give 64 cells, so a
balanced sample of 10,000 leaves about 150 observations in each, and the standard errors are
eight times what the full sample would support. Looking back at our
[table of rates](smoothing.md#how-fast-can-we-learn), this is
not a comfortable position.

The insight of [Aitchison and Aitken (1976)](https://doi.org/10.1093/biomet/63.3.413) is that the trade we already accepted for
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

with $\lambda \in \left[0, \frac{c-1}{c}\right]$. That upper limit is not arbitrary. It is
exactly the point past which a non-matching level would outweigh a matching one, since
$1 - \lambda \ge \lambda/(c-1)$ rearranges to $\lambda \le (c-1)/c$.

The kernel is a proper weighting scheme in the sense that it sums to one over the level set,
since one level matches and $c-1$ do not,

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

For estimators built as ratios of kernel sums, such as regressions, the normalization cancels
from the fitted value and is unnecessary, and a simpler variant is often used,

$$
L_\lambda(x, X_i) = \begin{cases} 1 & x = X_i \\ \lambda & x \neq X_i \end{cases}
\quad \text{with } \lambda \in [0, 1],
$$

which reaches complete smoothing at $\lambda = 1$ rather than at $(c-1)/c$ and needs no level
count in its own definition. This is the form Racine and Li analyze, and it will matter below.
KernelJax ships the normalized version instead, since a density estimate needs the
normalization that a regression discards.

## Ordered categories

When the levels *are* ordered, as with an education level or a rating scale, the unordered
kernel discards information by giving every non-matching level the same weight. We should
prefer adjacent levels to count for more than distant ones.
[Wang and van Ryzin (1981)](https://doi.org/10.1093/biomet/68.1.301) proposed

$$
L_\lambda(x, X_i) = \begin{cases}
1 - \lambda & x = X_i \\[4pt]
\tfrac{1}{2}(1 - \lambda)\, \lambda^{|x - X_i|} & x \neq X_i
\end{cases}
$$

with $\lambda \in [0, 1)$, so that weight decays geometrically in the number of levels
separating the two values. The factor of $\tfrac{1}{2}$ divides the mass between the two
directions, and the weights sum to one over the integers,

$$
(1 - \lambda) + 2 \sum_{d=1}^{\infty} \tfrac{1}{2}(1-\lambda)\lambda^{d}
 = (1 - \lambda) + (1-\lambda)\frac{\lambda}{1 - \lambda}
 = (1 - \lambda) + \lambda = 1 .
$$

As before, $\lambda = 0$ returns us to splitting. The other end of the range is less obliging
than it looks. The common factor $(1 - \lambda)$ does drive every weight to zero, but a
constant factor cancels from a ratio estimator, and what survives is a relative weight of $1$
on the matching level against $\tfrac{1}{2}$ on every other one, at any distance. This kernel
therefore never lets go of an ordered variable, however large $\lambda$ grows, and at
$\lambda = 1$ exactly it does not smooth the variable away so much as annihilate the estimate,
every weight being zero at once.

The variant KernelJax uses by default repairs this. [Li and Racine (2007)](https://press.princeton.edu/books/hardcover/9780691121611/nonparametric-econometrics) handle
matching and
non-matching levels with a single expression,

$$
L_\lambda(x, X_i) = \frac{1 - \lambda}{1 + \lambda}\, \lambda^{|x - X_i|},
\quad \lambda \in [0, 1),
$$

whose relative weights are $\lambda^{|x - X_i|}$ with no exception carved out for a match. As
$\lambda \to 1$ every level is weighted equally and the variable really is smoothed away,
which is the behavior we relied on for the unordered case and could not get from Wang and van
Ryzin. It is normalized on the same support, by the same geometric series,

$$
\frac{1-\lambda}{1+\lambda}\left(1 + 2\sum_{d=1}^\infty \lambda^d\right)
 = \frac{1-\lambda}{1+\lambda}\cdot\frac{1+\lambda}{1-\lambda} = 1 .
$$

Both of those sums run over the whole integer lattice, and a real ordered factor has only $c$
levels. The tail beyond them is cut off, so the weights sum to strictly less than one at every
level, worst at the endpoints. A regression is unaffected, since the shortfall cancels in the
ratio, but a density is not. An ordered column in KernelJax at $\lambda = 0.2, 0.5, 0.8$
returns estimated probabilities summing to $0.913$, $0.737$ and $0.401$, where the unordered
kernel of the previous section sums to one exactly.

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

The categorical columns enter through the weight and through nothing else. The polynomial
basis is still built from the continuous coordinates alone, because a difference
$X_{is} - x_s$ has no meaning for a label, and coding ordered levels as numbers and regressing
on them would impose exactly the functional form we are trying to avoid.

## What the smoothing buys

The asymptotic theory for the local constant version of this estimator is due to
[Racine and Li (2004)](https://doi.org/10.1016/S0304-4076(03)00157-X), and it is worth stating carefully because it contains the entire justification for
the approach.

Two conventions come with the theorem. It is stated for a single $h$ shared by the $p$
continuous columns and a single $\lambda$ shared by the discrete ones, the authors noting that
per-column parameters are a straightforward generalization, while KernelJax carries one of
each per column. And it is stated for the unnormalized kernels, $1$ on a match against
$\lambda$ otherwise and its ordered analogue $\lambda^{|x - X_i|}$, rather than for the
normalized forms displayed above. Neither convention touches the rates, but the second does
change the constant multiplying the discrete bias, as we note below.

Writing $\hat m$ for the mixed-data estimator with smoothing parameters $(h, \lambda)$, and
assuming $h \to 0$, $\lambda \to 0$ and $n h^p \to \infty$, which is how the authors describe
the effect of their formal conditions, their Theorem 2.1 gives

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

The sum runs over those level combinations one step from $x^d$, meaning one unordered
component changed or one ordered component moved to an adjacent level, since a two-level move
carries $\lambda^2$ and is of higher order. So the discrete bias is a weighted sum, with
density ratios for weights, of how much $m$ moves when a single category is changed. The terms
belonging to a given column are precisely zero when $m$ does not depend on that column,
whatever the other columns do, and that is the fact the next page exploits.

The coefficient $\lambda$ in front belongs to the unnormalized kernels the theorem is stated
for. Expanding the normalized forms to first order gives $\lambda/(c-1)$ for Aitchison and
Aitken and $\lambda/2$ for Wang and van Ryzin, while Li and Racine reproduce $\lambda$
exactly. The order in $\lambda$ is the same either way, so nothing below depends on which was
used, but the constant is not interchangeable.

Three consequences are worth drawing out.

The first, and the reason all of this is worth doing, is that the normalization is
$\sqrt{n h^p}$, involving only $p$, the number of *continuous* covariates. The discrete
covariates do not appear in the rate. Racine and Li put it plainly, that the convergence
rate is the same as in the case where the regressors are continuous only.

It is worth being careful about what this does and does not beat. Splitting reaches the same
rate in $n$, with $n$ replaced by the cell count, and its leading variance constant is not
merely comparable but identical. The $f(x)$ sitting in $\Omega(x)$ is the joint density and
mass function, so $f(x) = \pi(x^d) f(x^c \mid x^d)$, and substituting turns the smoothed
variance into the split-sample variance with $n \pi(x^d)$ observations. Since $\lambda \to 0$,
this is only to be expected, because the estimator is converging on the splitting estimator.
What smoothing buys at first order is therefore not a variance saving on a relevant discrete
variable. It is a finite-sample saving, which is the argument we opened with and is
substantial at the sample sizes anyone actually has, and it is a first-order saving on an
*irrelevant* discrete variable, where $\lambda$ does not shrink at all and the column drops
out of $f(x)$ entirely. The [next page](selection.md) is where that second case is settled.

The second is that the discrete bias is $O(\lambda)$, linear rather than quadratic, while
the continuous bias is $O(h^2)$. Balancing the two calls for $\lambda \propto h^2$, and in
this case the proportionality is exact rather than heuristic. Minimizing the leading term of
the cross-validation objective gives $\lambda_0 = B_2 h_0^2 / (2 B_3)$ for constants
$B_2, B_3$ depending on the design, and solving for the pair yields

$$
h_0 = c_1\, n^{-1/(4+p)}, \quad \lambda_0 = c_2\, n^{-2/(4+p)} .
$$

So the optimal smoothing parameter of a relevant categorical variable goes to zero at twice
the exponent of the continuous bandwidth, and both are governed by $p$ alone. This is the
interior minimizer of the leading term, so it presumes that minimizer is interior and
nonnegative, and it presumes at least one continuous covariate to balance against. With
purely discrete regressors there is no $h$ in the problem, the trade is between an
$O(\lambda^2)$ squared bias and an $O(\lambda/n)$ variance term, and Racine and Li obtain
$O_p(n^{-1})$ for the cross-validated $\lambda$ rather than the $n^{-1/2}$ that setting
$p = 0$ above would suggest.

The third is that the continuous part of the bias still carries the design term
$\nabla f^\top \nabla m / f$ that we met on the previous page, because the estimator analysed
above is local constant. Fitting local linear in the continuous directions removes it at no
cost in leading interior variance, exactly as before.

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
