import unittest
import numpy as np
import birdepy as bd


class TestUtility(unittest.TestCase):
    def test_estimate_dnm(self):
        t_data = list(range(100))
        p_data = bd.simulate.discrete([0.75, 0.25, 0.02, 1], 'Ricker', 10, t_data,
                                      survival=True, seed=2021)
        result = bd.estimate(t_data, p_data, [0.5, 0.5, 0.05], [[0,1], [0,1], [0, 0.1]],
                              framework='dnm', model='Ricker', idx_known_p=[3], known_p=[1])
        
        assert np.isclose(result.p[0], 0.7477238086602771)
        assert np.isclose(result.p[1], 0.2150482925947323)
        assert np.isclose(result.p[2], 0.0227452130348903)
        
 
    def test_estimate_dnm(self):
        t_data = list(range(100))
        p_data = bd.simulate.discrete([0.75, 0.25, 0.02, 1], 'Ricker', 10, t_data,
                                      survival=True, seed=2021)
        result = bd.estimate(t_data, p_data, [0.5, 0.5, 0.05], [[0,1], [0,1], [0, 0.1]],
                              framework='dnm', model='Ricker', idx_known_p=[3], known_p=[1])
        
        assert np.isclose(result.p[0], 0.7477238086602771)
        assert np.isclose(result.p[1], 0.2150482925947323)
        assert np.isclose(result.p[2], 0.0227452130348903)
        
    def test_estimate_abc(self):
        t_data = list(range(100))
        p_data = bd.simulate.discrete([0.75, 0.25, 0.02, 1], 'Ricker', 10, t_data,
                                      survival=True, seed=2021)
        
        result = bd.estimate(t_data, p_data, [0.5], [[0,1]], framework='abc',
                  model='Ricker', idx_known_p=[1, 2, 3],
                  known_p=[0.25, 0.02, 1], display=False, seed=2021,
                  max_its=2, k = 10)
        
        assert np.isclose(result.p[0], 0.742410416185772)
 
