metric-learn: Metric Learning in Python
=======================================
|License| |PyPI version|

Distance metrics are widely used in the machine learning literature.
Traditionally, practicioners would choose a standard distance metric
(Euclidean, City-Block, Cosine, etc.) using a priori knowledge of
the domain.
Distance metric learning (or simply, metric learning) is the sub-field of
machine learning dedicated to automatically constructing optimal distance
metrics.

This package contains efficient Python implementations of several popular
metric learning algorithms.

.. toctree::
   :caption: Algorithms
   :maxdepth: 1

   metric_learn.covariance
   metric_learn.lmnn
   metric_learn.itml
   metric_learn.sdml
   metric_learn.lsml
   metric_learn.nca
   metric_learn.lfda
   metric_learn.rca

Each metric supports the following methods:

-  ``fit(...)``, which learns the model.
-  ``transformer()``, which returns a transformation matrix
   :math:`L \in \mathbb{R}^{D \times d}`, which can be used to convert a
   data matrix :math:`X \in \mathbb{R}^{n \times d}` to the
   :math:`D`-dimensional learned metric space :math:`X L^{\top}`,
   in which standard Euclidean distances may be used.
-  ``transform(X)``, which applies the aforementioned transformation.
-  ``metric()``, which returns a Mahalanobis matrix
   :math:`M = L^{\top}L` such that distance between vectors ``x`` and
   ``y`` can be computed as :math:`\left(x-y\right)M\left(x-y\right)`.


Installation and Setup
======================

Run ``pip install metric-learn`` to download and install from PyPI.

Alternately, download the source repository and run:

-  ``python setup.py install`` for default installation.
-  ``python setup.py test`` to run all tests.

**Dependencies**

-  Python 2.7+, 3.4+
-  numpy, scipy, scikit-learn
-  (for running the examples only: matplotlib)

**Notes**

If a recent version of the Shogun Python modular (``modshogun``) library
is available, the LMNN implementation will use the fast C++ version from
there. The two implementations differ slightly, and the C++ version is
more complete.

Navigation
----------

:ref:`genindex` | :ref:`modindex` | :ref:`search`

.. toctree::
   :maxdepth: 4
   :hidden:

   Package Overview <metric_learn>

.. |PyPI version| image:: https://badge.fury.io/py/metric-learn.svg
   :target: http://badge.fury.io/py/metric-learn
.. |License| image:: http://img.shields.io/:license-mit-blue.svg?style=flat
   :target: http://badges.mit-license.org
