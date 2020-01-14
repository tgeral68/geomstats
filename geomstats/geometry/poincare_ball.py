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
from geomstats.geometry.hyperbolic_space import HyperbolicSpace

class PoincareBall(HyperbolicSpace):
    """
    Class for the n-dimensional PoincareBall
    """

    def __init__(self, dimension):
        assert isinstance(dimension, int) and dimension > 0
        super(HyperbolicSpace, self).__init__(
                dimension=dimension,
                embedding_manifold=MinkowskiSpace(dimension+1))

class PoincareMetric(RiemannianMetric):

    """
    Class for the Poincareball metric
    """

    def __init__(self, dimension):
        dimension = dimension
        # super(RiemannianMetric, self).__init__(
        #         dimension=dimension,
        #         signature=(dimension, 0, 0))

    def add(x, y):
        nx = gs.sum(x ** 2, dim=-1, keepdim=True).expand_as(x)
        ny = gs.sum(y ** 2, dim=-1, keepdim=True).expand_as(x)
        xy = (x * y).sum(-1, keepdim=True).expand_as(x)
        return ((1 + 2 * xy + ny) * x + (1 - nx) * y) / (1 + 2 * xy + nx * ny)

    def log(self, k, x=None):
        kpx = self.add(-k, x)
        norm_kpx = kpx.norm(2, -1, keepdim=True).expand_as(kpx)
        norm_k = k.norm(2, -1, keepdim=True).expand_as(kpx)
        res = (1 - norm_k ** 2) * ((gs.arc_tanh(norm_kpx))) * (kpx / norm_kpx)
        if (0 != len((norm_kpx == 0).nonzero())):
            res[norm_kpx == 0] = 0
        return res

    def exp(self, k, x=None):
        norm_k = k.norm(2, -1, keepdim=True).expand_as(k)
        lambda_k = 1 / (1 - norm_k ** 2)
        norm_x = x.norm(2, -1, keepdim=True).expand_as(x)
        direction = x / norm_x
        factor = gs.tanh(lambda_k * norm_x)
        res = self.add(k, direction * factor)
        if (0 != len((norm_x == 0).nonzero())):
            res[norm_x == 0] = k[norm_x == 0]
        return res

    def dist(self, x,y):

        x_norm = gs.clamp(gs.sum(x ** 2, dim=-1), 0, 1 - 1e-3)
        y_norm = gs.clamp(gs.sum(y ** 2, dim=-1), 0, 1 - 1e-3)
        d_norm = gs.sum((x - y) ** 2, dim=-1)
        cc = 1 + 2 * d_norm / ((1 - x_norm) * (1 - y_norm))
        dist = gs.log(cc + gs.sqrt(cc ** 2 - 1))
        #ctx.save_for_backward(x, y, dist)
        return dist
