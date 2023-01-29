import unittest
from birdepy.utility import data_sort


class TestUtility(unittest.TestCase):
    def test_should_sort_single_path(self):
        result = data_sort([0, 1.1, 1.6, 2.1], [10, 12, 100, 188])

        assert result.get((10, 12, 1.1)) == 1
        assert result.get((12, 100, 0.5)) == 1
        assert result.get((100, 188, 0.5)) == 1

    def test_should_sort_multi_path(self):
        result = data_sort(
            [[0, 1.1, 1.6, 2.1], [0, 1.1, 1.6, 2.1]],
            [[10, 12, 100, 188], [10, 12, 100, 188]],
        )

        assert result.get((10, 12, 1.1)) == 2
        assert result.get((12, 100, 0.5)) == 2
        assert result.get((100, 188, 0.5)) == 2
