import unittest
import numpy as np
import birdepy as bd


class TestUtility(unittest.TestCase):
    def test_probability_expm(self):
        jump_times, pop_sizes = bd.simulate.continuous([0.75, 0.25, 1/1000, 0], 'Verhulst', 10, seed= 2021, t_max=1)
        assert np.isclose(jump_times[1], 0.1733207523280825)
        assert np.isclose(jump_times[-1], 0.9173238186709972)
        assert pop_sizes[5] == 13
        assert pop_sizes[7] == 13
        
