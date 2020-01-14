"""
Unit tests for numpy backend.
"""

import importlib
import os
import unittest
import warnings

import geomstats.backend as gs
from geomstats.geometry.poincare_ball import *


class TestPoincareBall(unittest.TestCase):
    _multiprocess_can_split_ = True

    @classmethod
    def setUpClass(cls):
        cls.initial_backend = os.environ['GEOMSTATS_BACKEND']
        os.environ['GEOMSTATS_BACKEND'] = 'torch'
        importlib.reload(gs)

    @classmethod
    def tearDownClass(cls):
        os.environ['GEOMSTATS_BACKEND'] = cls.initial_backend
        importlib.reload(gs)

    def setUp(self):
        warnings.simplefilter('ignore', category=ImportWarning)

        self.dimension = 2
        self.poincare_ball = PoincareBall(self.dimension)
        self.metric = PoincareMetric(self.dimension)

    def test_dist(self):
        x = gs.array([0.5, 0.5])
        y = gs.array([0.5, -0.5])
        result = self.metric.dist(x,y).item()

        print(result)

        expected = gs.array([2.887270927429199]).item()

        self.assertTrue(gs.allclose(result, expected))

if __name__ == '__main__':
        unittest.main()
