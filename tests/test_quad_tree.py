import unittest

import numpy as np

from openTSNE.quad_tree import QuadTree


class TestQuadTreeBounds(unittest.TestCase):
    """Points outside the root bounding box must be rejected.

    The descent in ``add_point_to`` selects the child on the side of the node
    center the point falls on. That child contains the point only when the point
    is inside the parent's box; otherwise the box never shrinks around it and
    the descent runs until the process dies.

    """

    def test_add_point_below_lower_bound(self):
        rs = np.random.RandomState(0)
        x = np.ascontiguousarray(rs.randn(2000, 1))
        tree = QuadTree(x)

        below = np.ascontiguousarray(np.array([x.min() - 1e-3]))
        with self.assertRaises(ValueError):
            tree.add_point(below)

    def test_add_point_above_upper_bound(self):
        rs = np.random.RandomState(0)
        x = np.ascontiguousarray(rs.randn(2000, 1))
        tree = QuadTree(x)

        above = np.ascontiguousarray(np.array([x.max() + 1e-3]))
        with self.assertRaises(ValueError):
            tree.add_point(above)

    def test_add_point_outside_in_a_single_dimension(self):
        rs = np.random.RandomState(0)
        x = np.ascontiguousarray(rs.randn(500, 2))
        tree = QuadTree(x)

        # The root box is a cube sized by the widest dimension, so it is wider
        # than the data in every other dimension
        center = (x.max(axis=0) + x.min(axis=0)) / 2
        half_length = (x.max(axis=0) - x.min(axis=0)).max() / 2

        outside = np.ascontiguousarray(
            np.array([center[0] - half_length - 1e-3, center[1]])
        )
        with self.assertRaises(ValueError):
            tree.add_point(outside)

    def test_add_point_outside_data_range_but_inside_bounding_box(self):
        rs = np.random.RandomState(0)
        x = np.ascontiguousarray(np.column_stack([rs.randn(500), rs.randn(500) * 10]))
        tree = QuadTree(x)

        # The box is sized by the second dimension, leaving room around the
        # first, so a point beyond the data in that dimension is still in bounds
        narrow_min = x[:, 0].min()
        center = (x.max(axis=0) + x.min(axis=0)) / 2
        half_length = (x.max(axis=0) - x.min(axis=0)).max() / 2
        assert narrow_min - 1e-3 > center[0] - half_length

        tree.add_point(np.ascontiguousarray(np.array([narrow_min - 1e-3, center[1]])))

    def test_add_points_rejects_batch_containing_an_outside_point(self):
        rs = np.random.RandomState(0)
        x = np.ascontiguousarray(rs.randn(2000, 1))
        tree = QuadTree(x)

        # One of these 500 points falls below the minimum of `x`
        new_points = np.ascontiguousarray(rs.randn(500, 1))
        assert np.any(new_points < x.min()), "reproducer no longer has an outside point"

        with self.assertRaises(ValueError):
            tree.add_points(new_points)

    def test_add_points_inside_bounding_box(self):
        rs = np.random.RandomState(0)
        x = np.ascontiguousarray(rs.randn(500, 2))
        tree = QuadTree(x)

        inside = np.ascontiguousarray(rs.uniform(-0.5, 0.5, size=(100, 2)))
        tree.add_points(inside)

    def test_add_points_with_too_few_dimensions(self):
        rs = np.random.RandomState(0)
        tree = QuadTree(np.ascontiguousarray(rs.randn(50, 3)))

        # Indexing a 2 column array up to the tree's 3 dimensions reads past the
        # end of each row
        with self.assertRaises(ValueError):
            tree.add_points(np.ascontiguousarray(np.zeros((4, 2))))

    def test_add_points_with_too_many_dimensions(self):
        rs = np.random.RandomState(0)
        tree = QuadTree(np.ascontiguousarray(rs.randn(50, 2)))

        with self.assertRaises(ValueError):
            tree.add_points(np.ascontiguousarray(np.zeros((4, 3))))

    def test_add_point_with_mismatched_dimensions(self):
        rs = np.random.RandomState(0)
        tree = QuadTree(np.ascontiguousarray(rs.randn(50, 3)))

        with self.assertRaises(ValueError):
            tree.add_point(np.ascontiguousarray(np.zeros(2)))

    def test_construction_admits_points_on_the_bounding_box_edge(self):
        """The extremes of the data define the box and must always be accepted.

        The center and side length are computed in floating point, so an extreme
        point can land a fraction of an ulp outside the box it defines.

        """
        rs = np.random.RandomState(0)
        for n_dim in (1, 2, 3):
            for scale in (1e-8, 1, 1e8):
                x = np.ascontiguousarray(rs.randn(500, n_dim) * scale)
                QuadTree(x)


if __name__ == "__main__":
    unittest.main()
