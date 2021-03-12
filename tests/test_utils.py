import unittest
import numpy as np

from openTSNE.utils import clip_point_to_disc


class TestClipping(unittest.TestCase):
    def test_circular_clip_do_nothing(self):
        x = np.array([[0, 1], [1, 0], [0.05, 0.05]], dtype=np.float64)
        x_clipped, mask = clip_point_to_disc(x, 1)
        np.testing.assert_almost_equal(x, x_clipped)
        np.testing.assert_almost_equal(mask, [0, 0, 0])

    def test_circular_clip_do_clipping1(self):
        x = np.array([[0, 1], [5, 0], [0.05, 0.05]], dtype=np.float64)
        x_clipped, mask = clip_point_to_disc(x, 1)
        np.testing.assert_almost_equal([[0, 1], [1, 0], [0.05, 0.05]], x_clipped)
        np.testing.assert_almost_equal(mask, [0, 1, 0])

    def test_circular_clip_do_clipping2(self):
        x = np.array([[0, 1], [1, 0], [1, 1]], dtype=np.float64)
        x_clipped, mask = clip_point_to_disc(x, 1)
        v = np.cos(np.radians(45))
        np.testing.assert_almost_equal([[0, 1], [1, 0], [v, v]], x_clipped)
        np.testing.assert_almost_equal(mask, [0, 0, 1])
