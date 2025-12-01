from unittest import TestCase

import matplotlib.path as mpl_path
import numpy as np

from models.polygons import point_in_polygon


class PolygonsTests(TestCase):
    polygon = mpl_path.Path(np.array([[0, 0], [0, 10], [10, 10], [10, 0]], dtype=np.float32))
    inside_polygon = np.array([[1, 1], [5, 5]], dtype=np.float32)
    outside_polygon = np.array([[-1, -1], [-1, 11], [11, 11], [11, -1]], dtype=np.float32)

    def test_point_in_polygon(self):
        for point in self.inside_polygon:
            x, y = point[0], point[1]
            assert point_in_polygon(x, y, self.polygon)

    def test_point_not_in_polygon(self):
        for point in self.outside_polygon:
            x, y = point[0], point[1]
            assert not point_in_polygon(x, y, self.polygon)
