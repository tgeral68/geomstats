"""
The n-dimensional Poincare Hyperbolic space
"""

import logging
import math

import geomstats.backend as gs

from geomstats.geometry.embedded_manifold import EmbeddedManifold
from geomstats.geometry.minkowski_space import MinkowskiMetric
from geomstats.geometry.minkowski_space import MinkowskiSpace
from geomstats.geometry.riemannian_metric import RiemannianMetric

class HyperbolicSpace(EmbeddedManifold):
    """
    Class for the n-dimensional Hyperbolic space
    as embedded in (n+1)-dimensional Minkowski space.

    By default, points are parameterized by their extrinsic (n+1)-coordinates.
    """

    def __init__(self, dimension):
        assert isinstance(dimension, int) and dimension > 0
        super(HyperbolicSpace, self).__init__(
                dimension=dimension,
                embedding_manifold=MinkowskiSpace(dimension+1))

class HyperbolicMetric(RiemannianMetric):

    def __init__(self, dimension):
        super(HyperbolicMetric, self).__init__(
                dimension=dimension,
                signature=(dimension, 0, 0))
