# Acknowledgments

No statistical library begins from nothing. KernelJax builds on statistical methods, reference implementations, and open-source software developed by others. This page documents those influences and the standards the library uses to validate its own implementation.

## Software

KernelJax does not incorporate source code from the projects below. Their influence is methodological, numerical, or editorial.

### np (R)

The [np](https://github.com/JeffreyRacine/R-Package-np) R package by Tristen Hayfield and [Jeffrey S. Racine](https://experts.mcmaster.ca/people/racinej) has been the reference implementation for mixed-type kernel smoothing for two decades, and it defines the numerical standard KernelJax holds itself to.

The estimator families, kernel parameterizations, and cross-validation criteria in KernelJax closely follow those exposed by `np` and the literature it implements.

More concretely, KernelJax mirrors the following from `np`.

* **Estimators.** {func}`~kerneljax.density`, {func}`~kerneljax.cdf`, and {func}`~kerneljax.local_poly` cover the ground of `npudens`, `npudist`, and `npreg`. The conditional family {func}`~kerneljax.cdensity`, {func}`~kerneljax.cdist`, {func}`~kerneljax.cquantile`, and {func}`~kerneljax.cmode` covers `npcdens`, `npcdist`, `npqreg`, and `npconmode`.
* **Kernels.** The {class}`~kerneljax.AitchisonAitken`, {class}`~kerneljax.WangVanRyzin`, and {class}`~kerneljax.LiRacine` categorical kernels follow the parameterizations used by `np` for unordered and ordered variables.
* **Selection criteria.** The likelihood, least-squares, and distribution cross-validation criteria, along with the normal-reference rules, implement the same objectives used by `np`. KernelJax differs in how those objectives are optimized, using gradient-based rather than derivative-free search.

Every estimator and criterion is checked numerically against `np` during development. At matched inputs, machine-precision agreement is the standard rather than statistical closeness.

The [code conventions](conventions.md) page describes how those comparisons fit into development.

For more on `np`, see the [associated paper](https://doi.org/10.18637/jss.v027.i05) by Hayfield and Racine or the textbook [*Nonparametric Econometrics: Theory and Practice*](https://press.princeton.edu/books/hardcover/9780691121611/nonparametric-econometrics) by Li and Racine, which develops much of the theory implemented by both packages.

### CVXPY (Python)

[CVXPY](https://www.cvxpy.org/) by Steven Diamond, Stephen Boyd, and the CVXPY community has influenced the structure and presentation of KernelJax's documentation. Its documentation provides a useful example of how mathematically sophisticated software can introduce the high-level interface without hiding the underlying methods.

You can learn more about CVXPY [on GitHub](https://github.com/cvxpy/cvxpy) or through the [associated paper](https://www.jmlr.org/papers/v17/15-408.html) by Diamond and Boyd.

## Built with

KernelJax is built on several open-source projects.

* [JAX](https://docs.jax.dev/) provides automatic differentiation, transformations, and compilation.
* [NumPy](https://numpy.org/) provides the array conventions accepted throughout the API.
* [jaxtyping](https://docs.kidger.site/jaxtyping/) provides shape annotations for the public API.
* [Matplotlib](https://matplotlib.org/) is used to produce the figures in this documentation.
* [Sphinx](https://www.sphinx-doc.org/) and [sphinx-immaterial](https://sphinx-immaterial.readthedocs.io/) build and style these pages.

## Statistical references

The following papers describe the methods and algorithms implemented in KernelJax.

* Aitchison, J., & Aitken, C. G. G. (1976). "Multivariate binary discrimination by the kernel method." *Biometrika*, 63(3), 413-420.
  [DOI:10.1093/biomet/63.3.413](https://doi.org/10.1093/biomet/63.3.413)

* Hall, P., Racine, J. S., & Li, Q. (2004). "Cross-validation and the estimation of conditional probability densities." *Journal of the American Statistical Association*, 99, 1015-1026.
  [DOI:10.1198/016214504000000548](https://doi.org/10.1198/016214504000000548)

* Hall, P., Li, Q., & Racine, J. S. (2007). "Nonparametric estimation of regression functions in the presence of irrelevant regressors." *The Review of Economics and Statistics*, 89(4), 784-789.
  [DOI:10.1162/rest.89.4.784](https://doi.org/10.1162/rest.89.4.784)

* Hayfield, T., & Racine, J. S. (2008). "Nonparametric Econometrics: The np Package." *Journal of Statistical Software*, 27(5).
  [DOI:10.18637/jss.v027.i05](https://doi.org/10.18637/jss.v027.i05)

* Hurvich, C. M., Simonoff, J. S., & Tsai, C.-L. (1998). "Smoothing parameter selection in nonparametric regression using an improved Akaike information criterion." *Journal of the Royal Statistical Society, Series B*, 60(2), 271-293.
  [DOI:10.1111/1467-9868.00125](https://doi.org/10.1111/1467-9868.00125)

* Li, Q., Lin, J., & Racine, J. S. (2013). "Optimal bandwidth selection for nonparametric conditional distribution and quantile functions." *Journal of Business & Economic Statistics*, 31(1), 57-65.
  [DOI:10.1080/07350015.2012.738955](https://doi.org/10.1080/07350015.2012.738955)

* Liu, D. C., & Nocedal, J. (1989). "On the limited memory BFGS method for large scale optimization." *Mathematical Programming*, 45, 503-528.
  [DOI:10.1007/BF01589116](https://doi.org/10.1007/BF01589116)

* Nocedal, J. (1980). "Updating quasi-Newton matrices with limited storage." *Mathematics of Computation*, 35, 773-782.
  [DOI:10.1090/S0025-5718-1980-0572855-7](https://doi.org/10.1090/S0025-5718-1980-0572855-7)

* Racine, J. S., & Li, Q. (2004). "Nonparametric estimation of regression functions with both categorical and continuous data." *Journal of Econometrics*, 119(1), 99-130.
  [DOI:10.1016/S0304-4076(03)00157-X](https://doi.org/10.1016/S0304-4076%2803%2900157-X)

* Stone, C. J. (1980). "Optimal rates of convergence for nonparametric estimators." *The Annals of Statistics*, 8(6), 1348-1360.
  [DOI:10.1214/aos/1176345206](https://doi.org/10.1214/aos/1176345206)

* Wang, M.-C., & van Ryzin, J. (1981). "A class of smooth estimators for discrete distributions." *Biometrika*, 68(1), 301-309.
  [DOI:10.1093/biomet/68.1.301](https://doi.org/10.1093/biomet/68.1.301)

## Use of AI tools

Agentic coding tools have been used during development of both KernelJax and its documentation. AI-generated changes are treated like any other contribution. They are reviewed, revised, tested, and validated before being merged.

Numerical implementations are still required to reproduce the relevant `np` reference results at matched inputs, and documentation examples are executed to verify the values they report.

These tools are used to accelerate development, not as a substitute for review or numerical validation.
