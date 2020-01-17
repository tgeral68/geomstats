"""
Unit tests for pytorch backend.
"""

import os
import unittest
import warnings

import geomstats.backend as gs
import geomstats.tests

from geomstats.geometry.special_orthogonal_group import SpecialOrthogonalGroup


@geomstats.tests.pytorch_only
class TestBackendPytorch(geomstats.tests.TestCase):
    def test_sampling_choice(self):
        res = gs.random.choice(10, (5, 1, 3))
        self.assertAllClose(res.shape, [5, 1, 3])

    def test_inverse_hyperbolic_function(self):
        expected  = gs.array([2.2924316695611777, 0.29567305, 0.3095196])

        x_cos = gs.array([5.])
        x_tan_sin = gs.array([0.3])
        acosh = gs.arccosh(x_cos)
        asinh = gs.arcsinh(x_tan_sin)
        atanh = gs.arctanh(x_tan_sin)
        result = [acosh.item(), asinh.item(), atanh.item()]
        print(expected, result)
        self.assertAllClose(result, expected)