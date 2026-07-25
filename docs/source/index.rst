KernelJAX
=========

**KernelJAX** is a low-level JAX library for nonparametric kernel smoothing of
mixed-type data, aimed at researchers who want to develop new methodology from
composable, differentiable building blocks. Each kernel, bandwidth selector, and
smoother is designed to be understood, modified, and recombined, not just
called. Through JAX and XLA, estimators built from these pieces run natively on
CPUs, GPUs, and TPUs with automatic differentiation and vectorized operations,
and fit naturally alongside the broader JAX ecosystem.

.. warning::

   KernelJAX is in early development. APIs are unstable and subject to change.

Installation
------------

.. code-block:: bash

   pip install kerneljax

Indices
-------

* :ref:`genindex`
* :ref:`search`
