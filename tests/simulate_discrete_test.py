import unittest
import numpy as np
import birdepy as bd


class TestUtility(unittest.TestCase):
    def test_simulate_exact(self):
        result = bd.simulate.discrete(1, 'Poisson', 0, times=[0, 1, 3, 4, 5], seed=2021)

        assert result[0] == 0
        assert result[1] == 0
        assert result[2] == 1
        assert result[3] == 2
        assert result[4] == 2

    def test_simulate_ea(self):
        result = bd.simulate.discrete(1, 'Poisson', 0, times=[0, 1, 3, 4, 5], method='ea', seed=2021)
        
        assert result[0] == 0
        assert result[1] == 2
        assert result[2] == 3
        assert result[3] == 3
        assert result[4] == 4
        
    def test_simulate_ma(self):
        result = bd.simulate.discrete([0.75, 0.25, 0.02, 1], 'Ricker', 10, [0, 10, 100],
                                      survival=True, seed=2021)        
        assert result[0] == 10
        assert result[1] == 47
        assert result[2] == 39
  
    def test_simulate_gwa(self):
        result = bd.simulate.discrete([0.75, 0.25, 1/1000, 0], 'Verhulst', 10,
                             [0, 10, 100], method='gwa', seed=2021)
        
        assert result[0] == 10
        assert result[1] == 288
        assert result[2] == 688
