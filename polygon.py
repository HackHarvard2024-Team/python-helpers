from re import S
import polyline
import numpy as np
from collections import defaultdict
import math
import sys

class Polygon():

    def __init__(self, coords: list[tuple[float, float]]):
        self._coords = coords

    @staticmethod
    def from_polyline(polyline):
        return Polygon(polyline.decode(polyline, 10))

    def __len__(self):
        return len(self._coords)

    def simplify_fromdist(self, n : int):
        """
        Simplify this polygon to a smaller number of vertexes
        """
        avg_coord0 = np.average(np.array([coord[0] for coord in self._coords]))
        avg_coord1 = np.average(np.array([coord[1] for coord in self._coords]))
        distances = {}
        for coord in self._coords:
            distances[coord] = ((coord[0] - avg_coord0) ** 2 + (coord[1] - avg_coord1) ** 2) ** 1/2
        out = [v[0] for v in sorted(distances.items(), key = lambda item : item[1], reverse = True)]
        return Polygon(out[0:n])
    
    def simplify_minmax(self, n : int):
        """
        calculate center point
        find 4 furthest points away, in terms of magnitude of vector
        """
        coords = self._coords
        min0 = min([coord[0] for coord in coords])
        min1 = min([coord[1] for coord in coords])
        max0 = max([coord[0] for coord in coords])
        max1 = max([coord[1] for coord in coords])
        return Polygon([
            (min0, min1),
            (min0, max1),
            (max0, min1),
            (max0, max0)
        ])
    
    def simplify(self, n : int):
        pass

    def __contains__(self, to_check : tuple[float, float]):
        coords = self._coords
        min0 = min([c[0] for c in coords])
        max0 = max([c[0] for c in coords])
        min1 = min([c[1] for c in coords])
        max1 = max([c[1] for c in coords])
        if to_check[0] < min0 or to_check[0] > max0 or to_check[1] < min1 or to_check[1] > max1:
            return False
        # Then use ray casting algorithm
        count = 0
        for i in range(len(coords)):
            point1 = coords[i]
            point2 = coords[(i + 1) % len(coords)]
            if (to_check[0] > min(point1[0], point2[0]) and
                to_check[0] < max(point1[0], point2[0]) and 
                to_check[1] < max(point1[1], point2[1])
                ):
                intercept_1 = (to_check[0] - point1[0]) * (point2[1] - point1[1]) / (point2[0] - point1[0]) + point1[1]
                if (point1[1] == point2[1] or to_check[1] <= intercept_1):
                    count += 1
        print(count)
        return count % 2 != 0

    


import unittest

class TestPointInNgon(unittest.TestCase):

    def __init__(self, methodName = "test"):
        super().__init__(methodName)
        self.trapezoid = Polygon([(0, 1.0), (1, 0), (2, 0), (0,3)])
        self.point1 = (10, 10) # not inside
        self.point2 = (1.5, 0.5) # inside

    def test(self):
        self.assertFalse(self.point1 in self.trapezoid)
        self.assertTrue(self.point2 in self.trapezoid)


if __name__ == "__main__":
    unittest.main()



    
