# Kernel smoothing

Welcome! These four pages are a gentle introduction to kernel smoothing, the set of ideas
that KernelJax implements. No prior exposure to nonparametric statistics is assumed, and
they are meant to be read in order.

We begin here by asking what it means to estimate a function without assuming its shape,
build a density estimator out of the humble histogram, and derive in full the bias and
variance that determine how its *bandwidth* must be chosen.
[Kernel regression](regression.md) then turns these ideas into an estimator of a regression
function. [Mixed-type data](mixed-data.md) extends the construction to categorical
variables, and [Bandwidth selection](selection.md) shows how every smoothing parameter can
be chosen from the data.

## What do we mean by nonparametric?

Suppose we observe pairs $(X_1, Y_1), \dots, (X_n, Y_n)$ and wish to describe how $Y$
depends on $X$. The quantity we are after is the *regression function*

$$
m(x) = \mathbb{E}[Y \mid X = x],
$$

the average value of $Y$ among observations whose covariate equals $x$.

A *parametric* approach commits to a shape for $m$ before looking at the data. Linear
regression asserts that $m(x) = \beta_0 + \beta_1 x$, reducing the problem to estimating two
numbers. When the assertion is correct this is an excellent thing to do, and our estimates
converge at the familiar $\sqrt{n}$ rate. When it is incorrect, however, no amount of data
will save us. Writing $\beta^*$ for the population least squares coefficients, our estimator
converges to $\beta^*$, and the approximation error
$\sup_x |m(x) - \beta_0^* - \beta_1^* x|$ is a fixed positive number that does not shrink
with $n$.

A *nonparametric* approach makes a much weaker assumption. We suppose only that $m$ lies in
a smoothness class, typically that it is twice continuously differentiable, and let the
sample tell us its shape. The object we are estimating is now an entire function rather than
a short vector of coefficients, and this is the source of both the flexibility we gain and
the difficulties we shall have to confront. The same distinction applies when estimating a
probability density $f$, and it will turn out that both problems are solved by the same
device.

The underlying idea is disarmingly simple. To estimate $m$ at a point $x$, we average the
responses of those observations whose covariates lie *near* $x$. Everything that follows
comes from taking that word seriously.

## Building a density estimator

Let us start with densities, where the reasoning is easiest to see. We return to regression
on the next page, once the machinery is in place.

Given a sample $X_1, \dots, X_n$ drawn from a density $f$, the oldest estimator is one we
have all drawn by hand. We partition the line into bins of width $h$ and report, for each
bin, the fraction of observations landing in it divided by $h$. This histogram is a
perfectly respectable estimator and it does converge to $f$, but it has two defects. They
are worth naming carefully, because repairing them leads us directly to the kernel
estimator.

The first defect is that a histogram is not smooth. It is constant within each bin and jumps
at every boundary, and no quantity of data removes those jumps. It does converge as the bins
shrink, but more slowly than what we are about to build, because pinning bins to a fixed grid
rather than to the evaluation point leaves a bias of order $h$ where the kernel estimator
achieves $h^2$. The second is subtler and more troubling. The estimate at a point depends on where
we happened to place the bins. Two analysts using the same bin width but different origins
will report different values at the same $x$, which is an artifact of our grid rather than
anything to do with the data.

Both defects stem from the bins being fixed in advance. So suppose that, instead, we center
the window on the evaluation point itself. For any $x$ we count the observations within
distance $h$ and divide by the width of the window,

$$
\hat f(x) = \frac{1}{2hn} \#\{i : |X_i - x| \le h\}
        = \frac{1}{nh} \sum_{i=1}^{n} w\!\left(\frac{x - X_i}{h}\right),
$$

where $w(u) = \tfrac{1}{2}$ for $|u| \le 1$ and zero otherwise. This is already an
improvement. Our estimate no longer depends on an arbitrary origin, and it is defined at
every $x$ rather than only on a grid.

It is still not smooth, though, and we can see exactly why. An observation contributes
nothing until it comes within distance $h$ of $x$, at which moment it abruptly contributes
its full weight. If we want a smooth estimate, we should let that weight decay, so that
observations close to $x$ count for more than distant ones and nothing enters or leaves the
sum discontinuously. We therefore ask for a function $k$ that is nonnegative and symmetric about zero, so that
weight depends only on distance, and that satisfies

$$
\int k(u)\, du = 1, \quad
\mu_2(k) = \int u^2 k(u)\, du < \infty, \quad
R(k) = \int k(u)^2\, du < \infty .
$$

Symmetry does two jobs below. It makes $\int u\, k(u)\, du = 0$, which kills the first-order
term of the expansion, and it lets us replace $k(-v)$ by $k(v)$ after a change of variable.
A function meeting these requirements with $\mu_2(k) \neq 0$ is a *second-order kernel*.
Nonnegativity is what keeps $\hat f$ a density; higher-order kernels give it up to buy a
faster rate, at the price of estimates that can dip below zero. The estimator $k$ produces,

$$
\hat f(x) = \frac{1}{nh} \sum_{i=1}^{n} k\!\left(\frac{x - X_i}{h}\right)
          = \frac{1}{n} \sum_{i=1}^{n} k_h(x - X_i),
\quad
k_h(u) = \tfrac{1}{h}\, k(u/h),
$$

is the *kernel density estimator*. KernelJax uses the Gaussian kernel
$k(u) = (2\pi)^{-1/2} e^{-u^2/2}$ by default, for which $\mu_2(k) = 1$ and
$R(k) = \int k^2 = 1/(2\sqrt{\pi}) \approx 0.2821$.

There is a pleasant way to read this expression. We are placing a small bump of area $1/n$
at each observation and adding them up. Where observations cluster the bumps overlap and
our estimate is large; where they are sparse it is small. The parameter $h$, which we call
the *bandwidth*, controls how wide each bump is, and it is to that parameter that we now
turn.

## The bias of the estimator

We now have an estimator and no way of choosing the one number it depends on. To choose it
we need to know what it costs us, so we should ask in what ways an estimator can be wrong.

There are two, and they are quite different. An estimator can be systematically off, sitting
away from the truth on average however much data we collect, which we call its *bias*. Or it
can be unstable, landing far from its own average on any particular sample, which we call
its *variance*. Both matter, and the bandwidth turns out to move them in opposite
directions, which is the whole reason choosing it is delicate. This section computes the
bias and the next computes the variance.

Three assumptions carry us through both. We take $f$ to be twice continuously
differentiable, which is what makes the Taylor expansion below legitimate. We work at a
point $x$ in the interior of the support, since at an edge the window is one-sided and the
leading bias is larger, a problem we meet properly on the [next page](regression.md). And we
let $h \to 0$ while $nh \to \infty$ as the sample grows. The first of that pair says the window closes in on
the evaluation point, so the estimate becomes local. The second says it must not close so
fast that the window empties, and the variance calculation will give that condition an exact reading.

Two pieces of notation recur from here on. Writing $a_n = O(b_n)$ means $a_n / b_n$ stays
bounded, so $a_n$ is at most of the size of $b_n$; writing $a_n = o(b_n)$ means
$a_n / b_n \to 0$, so $a_n$ is negligible beside $b_n$. A term recorded as $o(h^2)$ is
therefore one we may ignore once $h$ is small enough, and the expressions below are exact
only to the order shown.

Since the $X_i$ are identically distributed, the expectation of the sum is $n$ times the
expectation of one term,

$$
\mathbb{E}[\hat f(x)] = \mathbb{E}\!\left[k_h(x - X)\right]
                      = \int \frac{1}{h} k\!\left(\frac{x-u}{h}\right) f(u)\, du .
$$

Substituting $u = x + hv$, so that $du = h\, dv$ and the factor $1/h$ cancels,

$$
\mathbb{E}[\hat f(x)] = \int k(-v) f(x + hv)\, dv = \int k(v) f(x + hv)\, dv ,
$$

using the symmetry of $k$. Now expand $f(x + hv)$ in a Taylor series about $x$,

$$
f(x + hv) = f(x) + hv f'(x) + \tfrac{1}{2} h^2 v^2 f''(x) + o(h^2),
$$

and integrate term by term against $k$. The first term contributes $f(x)$ because $k$
integrates to one, the second vanishes because $\int v\, k(v)\, dv = 0$, and the third
leaves $\tfrac{1}{2} h^2 f''(x) \mu_2(k)$. Hence

$$
\operatorname{Bias}[\hat f(x)] = \frac{h^2}{2}\, \mu_2(k)\, f''(x) + o(h^2).
$$

Two features deserve comment. The bias is $O(h^2)$ and does not involve $n$ at all, so
collecting more data does nothing to reduce it if $h$ is held fixed. And it is proportional
to $f''(x)$, which tells us something intuitive. The estimate is pulled down at peaks, where
$f'' < 0$, and pushed up in troughs. Smoothing flattens curvature, exactly as we should
expect from averaging over a neighborhood.

## The variance of the estimator

The bias tells us where the estimator sits on average. It says nothing about how far any
one sample can land from that average, and since we only ever have one sample, that is the
half of the cost we feel. It is also the half the bandwidth pushes the other way.

Because the observations are independent, the variance of the average is $1/n$ times the
variance of a single term,

$$
\operatorname{Var}[\hat f(x)] = \frac{1}{n}\operatorname{Var}\!\left[k_h(x - X)\right]
 = \frac{1}{n}\left(\mathbb{E}\!\left[k_h(x - X)^2\right]
   - \bigl(\mathbb{E}[k_h(x - X)]\bigr)^2\right).
$$

The second moment succumbs to the same substitution as before,

$$
\begin{aligned}
\mathbb{E}\!\left[k_h(x - X)^2\right]
 = \int \frac{1}{h^2} k\!\left(\frac{x-u}{h}\right)^{\!2} f(u)\, du
  &= \frac{1}{h} \int k(v)^2 f(x + hv)\, dv \\[4pt]
 &= \frac{R(k)\, f(x)}{h} + O(1),
\end{aligned}
$$

writing $R(k) = \int k(v)^2 dv$. The squared first moment is $\bigl(f(x) + O(h^2)\bigr)^2 =
O(1)$, which is negligible beside a term of order $1/h$. Therefore

$$
\operatorname{Var}[\hat f(x)] = \frac{R(k)\, f(x)}{nh} + o\!\left(\frac{1}{nh}\right).
$$

The quantity $nh$ is worth dwelling on. It is, up to a constant, the number of observations
falling inside the effective window, so the variance behaves like that of an average over
$nh$ points. This is the *effective local sample size*, and the requirement $nh \to \infty$
is precisely the statement that it must grow.

## Trading bias against variance

Our two expressions point in opposite directions. The bias grows with $h$ and the variance
shrinks with it. Taking $h$ small leaves us nearly unbiased but wildly variable; taking $h$
large gives a stable estimate biased toward a flat line. To choose between them we need a
single criterion, and the usual one is the *mean integrated squared error*,

$$
\mathrm{MISE}(h) = \mathbb{E} \int \bigl(\hat f(x) - f(x)\bigr)^2 dx
 = \int \operatorname{Bias}[\hat f(x)]^2 dx + \int \operatorname{Var}[\hat f(x)]\, dx .
$$

Substituting the two results above, and using $\int f = 1$ in the variance term, gives the
asymptotic form

$$
\mathrm{AMISE}(h) = \frac{h^4}{4}\, \mu_2(k)^2\, R(f'') + \frac{R(k)}{nh},
\quad R(f'') = \int f''(x)^2 dx ,
$$

taking $R(f'')$ to be finite, which is what lets the pointwise expansions be integrated.

Differentiating with respect to $h$ and setting the result to zero,

$$
h^3 \mu_2(k)^2 R(f'') - \frac{R(k)}{n h^2} = 0
\quad\Longrightarrow\quad
h_{\text{opt}} = \left(\frac{R(k)}{\mu_2(k)^2 R(f'')\, n}\right)^{1/5}
 \propto n^{-1/5},
$$

and substituting back yields the error attained at that bandwidth,

$$
\mathrm{AMISE}(h_{\text{opt}})
 = \frac{5}{4}\,\bigl(\mu_2(k)^2 R(f'')\bigr)^{1/5} R(k)^{4/5}\, n^{-4/5}.
$$

It is worth being careful about which rate is which, since both appear in the literature.
The *squared* error decays as $n^{-4/5}$, so the error itself, on the scale of $f$, decays
as $n^{-2/5}$. It is the latter we hold against the $n^{-1/2}$ of a correctly
specified parametric model, though that comparison is a heuristic one, since a root
integrated error and a pointwise parametric error are not the same loss. The gap between
$n^{-2/5}$ and $n^{-1/2}$ is nonetheless the price we pay for having declined to assume a
shape.

The formula for $h_{\text{opt}}$ also carries a sting. It depends on $R(f'')$, a functional
of the very density we are trying to estimate. We cannot simply look it up, and this is why
bandwidth selection is a genuine statistical problem rather than a matter of convention. The
[final page](selection.md) of this section is devoted to it.

## Why the kernel hardly matters

One consequence of the AMISE expression shapes how the library is built. The kernel enters
the attainable error only through the factor

$$
C(k) = \mu_2(k)^{2/5} R(k)^{4/5},
$$

with everything else depending on $f$ and $n$ alone. Comparing two kernels therefore amounts
to comparing two numbers. For the Gaussian, $\mu_2 = 1$ and $R = 1/(2\sqrt{\pi})$, giving
$C \approx 0.3633$. For the Epanechnikov kernel $k(u) = \tfrac{3}{4}(1 - u^2)$ on $[-1,1]$,
which minimizes $C$ over all second-order kernels, $\mu_2 = 1/5$ and $R = 3/5$, giving
$C \approx 0.3491$.

Since $\mathrm{AMISE} \propto C(k)\, n^{-4/5}$, two kernels reach the same error when
$C_1 n_1^{-4/5} = C_2 n_2^{-4/5}$, so the sample sizes stand in the ratio
$n_1/n_2 = (C_1/C_2)^{5/4}$. Taking the Gaussian first and the Epanechnikov second,
$C_{\mathrm{gau}}/C_{\mathrm{epa}} \approx 1.041$ and

$$
\frac{n_{\mathrm{gau}}}{n_{\mathrm{epa}}} \approx 1.041^{5/4} \approx 1.051 ,
$$

so the Gaussian needs about 5 percent more data to match the optimal kernel, a difference
few analyses would ever notice. Changing the bandwidth by 5 percent, by contrast, is
nothing at all, while changing it by a factor of two transforms the answer. Our effort is
far better spent selecting $h$ well than agonizing over the shape of $k$.

## Several covariates

With $d$ continuous covariates we take a product of kernels, one per coordinate, each with
its own bandwidth,

$$
\hat f(x) = \frac{1}{n \prod_{j=1}^d h_j} \sum_{i=1}^n
\prod_{j=1}^d k\!\left(\frac{x_j - X_{ij}}{h_j}\right).
$$

Repeating the expansions above coordinate by coordinate gives

$$
\begin{aligned}
\operatorname{Bias}[\hat f(x)] &= \frac{\mu_2(k)}{2} \sum_{j=1}^d h_j^2
  \frac{\partial^2 f(x)}{\partial x_j^2}
  + o\!\left(\textstyle\sum_j h_j^2\right), \\[6pt]
\operatorname{Var}[\hat f(x)] &= \frac{R(k)^d f(x)}{n \prod_j h_j}
  + o\!\left(\frac{1}{n\prod_j h_j}\right).
\end{aligned}
$$

The structure is unchanged, but the variance denominator is now $n \prod_j h_j$ rather than
$nh$. The effective local sample size is the count of observations in a $d$-dimensional box,
and it shrinks geometrically in $d$. Balancing squared bias against variance as before gives
$h_j \propto n^{-1/(d+4)}$ and $\mathrm{AMISE} \propto n^{-4/(d+4)}$, so the error on the
scale of $f$ decays at $n^{-2/(d+4)}$.

## How fast can we learn?

The deterioration above is not an artifact of the kernel estimator. [Stone (1980)](https://projecteuclid.org/journals/annals-of-statistics/volume-8/issue-6/Optimal-Rates-of-Convergence-for-Nonparametric-Estimators/10.1214/aos/1176345206.full)
established the best pointwise rate any estimator can achieve uniformly over a broad
smoothness class.

Fix a nonnegative integer $k$ and a constant $p > k$. After translating the point of
interest to the origin, write $g_k$ for the degree-$k$ Taylor polynomial of $g$ there,
and let $\Theta$ collect functions that are $k$ times continuously differentiable and
satisfy

$$
\bigl| g(x) - g_k(x) \bigr| \le M \lVert x \rVert^{p}
$$

on a neighborhood of the origin. The exponent $p$ measures local smoothness on a
continuous scale rather than simply counting derivatives. When $p$ is a positive
integer and $k = p - 1$, this condition follows, on a convex neighborhood, from an
appropriate bound on the derivatives of order $p$.

Let $m \le k$ and suppose the target is a derivative of order $m$, with $m = 0$
corresponding to the function value itself. Under regularity conditions on the design
density and response distribution, Stone shows that the optimal uniform pointwise rate is

$$
n^{-r}, \qquad r = \frac{p - m}{2p + d}.
$$

Optimal here has a precise minimax meaning. There are estimators that attain this rate
uniformly over the function class, while no sequence of estimators can converge uniformly
over the class at a strictly faster rate. Stone proves the result for an unknown regression
function and obtains an analogous result for density estimation.

Our twice continuously differentiable setting corresponds locally to $p = 2$, $k = 1$
and $m = 0$, giving

$$
r = \frac{2}{4+d}.
$$

This is the same rate obtained from the bias-variance calculation above. For regression,
Stone establishes achievability using a polynomial least-squares fit over a shrinking
neighborhood, a local polynomial estimator of the kind introduced on the
[next page](regression.md). For density estimation he uses a suitably chosen kernel
estimator. These rates are therefore not peculiar to the calculations above. They attain
the optimal rate permitted by the smoothness class.

Writing out a few values makes the effect of dimension vivid.

| Covariates $d$ | Optimal rate | $n$ giving the same asymptotic rate factor as $n=100$ for $d=1$ |
| --- | --- | --- |
| 1 | $n^{-2/5}$ | 100 |
| 2 | $n^{-1/3}$ | 251 |
| 5 | $n^{-2/9}$ | 3,981 |
| 10 | $n^{-1/7}$ | 398,107 |

The right-hand column equates only the dimension-dependent rate terms,

$$
n_d^{-2/(4+d)} = 100^{-2/5},
$$

which gives

$$
n_d = 100^{(4+d)/5}.
$$

The comparison ignores constants and should therefore be read as an illustration of
asymptotic scaling rather than an exact finite-sample equivalence. Even so, the growth is
punishing. Within Stone's smoothness class there is no estimator that avoids this
dimension dependence.

What can be done is to impose additional structure. If the function is additive across
coordinates, depends on only a few linear combinations of them, or depends on only a
subset of the covariates, the relevant rate can instead be governed by a smaller effective
dimension. The last of these is particularly relevant to KernelJax, since bandwidth
selection can effectively smooth out uninformative variables, as discussed in
[Bandwidth selection](selection.md).

The intuition behind this *curse of dimensionality* is geometric. In a uniform
$d$-dimensional unit cube, a hypercube containing a fraction $q$ of the volume has side
length $q^{1/d}$. For $q = 0.01$ and $d = 10$,

$$
0.01^{1/10} \approx 0.63,
$$

so a neighborhood containing just 1 percent of the sample spans about 63 percent of the
range of every coordinate. Local averaging rapidly stops being very local.

This is the backdrop against which the ability to discard uninformative variables
automatically becomes so valuable.

## Where next

We now have a kernel, a bandwidth, and exact leading-order expressions for what the
bandwidth costs us in either direction. [Kernel regression](regression.md) puts them to work
estimating $m$.

## References

- Silverman, B. W. (1986). [*Density Estimation for Statistics and Data
  Analysis*](https://doi.org/10.1201/9781315140919). Chapman and Hall.
- Stone, C. J. (1980). [Optimal rates of convergence for nonparametric
  estimators](https://doi.org/10.1214/aos/1176345206). *The Annals of Statistics*, 8(6),
  1348-1360.
- Wand, M. P., & Jones, M. C. (1995). [*Kernel
  Smoothing*](https://doi.org/10.1201/b14876). Chapman and Hall.
