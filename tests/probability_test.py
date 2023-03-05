import unittest
import numpy as np
import birdepy as bd


class TestUtility(unittest.TestCase):
    def test_probability_expm(self):
        result = bd.probability(20, 25, 1.0, [0.8, 0.4, 0.01, 0.001], model='Verhulst', method='expm')[0,0]

        assert np.isclose(result, 0.08189476)
        
    def test_probability_da(self):
        result = bd.probability(19, 27, 1.0, [0.5, 0.3, 0.01, 0.01], model='Verhulst', method='da')[0, 0]
        
        assert np.isclose(result, 0.016040426614336103)
        
    def test_probability_erlang(self):
        result = bd.probability(19, 27, 1.0, [0.5, 0.3, 0.01, 0.01], model='Verhulst', method='da')[0, 0]
        
        assert np.isclose(result, 0.0161337966847677)
    
    def test_probability_gwa(self):
        result = bd.probability(19, 27, 1.0, [0.5, 0.3, 0.01, 0.01], model='Verhulst', method='gwa')[0, 0]
        
        assert np.isclose(result, 0.014646030484734228)
  
    def test_probability_gwasa(self):
        result = bd.probability(19, 27, 1.0, [0.5, 0.3, 0.01, 0.01], model='Verhulst', method='gwasa')[0, 0]
        
        assert np.isclose(result, 0.014622270048744283)
 
    def test_probability_ilt(self):
        result = bd.probability(19, 27, 1.0, [0.5, 0.3, 0.01, 0.01], model='Verhulst', method='ilt')[0, 0]
        
        assert np.isclose(result, 0.01618465415009876)
  
    def test_probability_oua(self):
        result = bd.probability(19, 27, 1.0, [0.5, 0.3, 0.01, 0.01], model='Verhulst', method='oua')[0, 0]
        
        assert np.isclose(result, 0.021627234315268227)
        
    def test_probability_sim(self):
        
        result = bd.probability(19, 27, 1.0, [0.5, 0.3, 0.01, 0.01], model='Verhulst', method='sim', k=1000, seed=42)[0, 0]

        assert np.isclose(result, 0.017)
 
    def test_probability_uniform(self):
        result = bd.probability(19, 27, 1.0, [0.5, 0.3, 0.01, 0.01], model='Verhulst', method='uniform')[0, 0]
        
        assert np.isclose(result, 0.016191045442910168)
