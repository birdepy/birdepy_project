import birdepy as bd
print(bd.probability(19, 27, 1.0, [0.5, 0.3, 0.02, 0.01], model='Verhulst', method='ilt')[0][0])