import unittest
import numpy as np
import birdepy as bd


class TestUtility(unittest.TestCase):
    def test_should_sort_single_path(self):
        result = bd.probability(20, 25, 1.0, [0.8, 0.4, 0.01, 0.001], model='Verhulst', method='expm')

        assert np.isclose(result[0,0], 0.08189476)
        
