"""Compatibility checks for current NumPy, mpmath, and installed packages."""

from unittest.mock import patch

import mpmath as mp
import numpy as np
import pytest

import birdepy as bd
from birdepy import interface_aug
from birdepy import utility as ut


@pytest.mark.parametrize('precision', [None, 60])
@pytest.mark.parametrize('fails', [False, True])
def test_ilt_precision_is_applied_and_restored(precision, fails):
    expected = 1000 if precision is None else precision

    def invert(*args):
        assert mp.mp.dps == expected
        assert mp.mpf(1) + mp.mpf('1e-40') != 1
        if fails:
            raise ValueError('inversion failed')
        return np.array([[0.5]])

    options = {} if precision is None else {'precision': precision}
    with mp.workdps(15), patch('birdepy.probability_ilt.probability_ilt', invert):
        if fails:
            with pytest.raises(ValueError, match='inversion failed'):
                bd.probability(1, 1, 1, [1, 1], model='linear', method='ilt', **options)
        else:
            assert bd.probability(1, 1, 1, [1, 1], model='linear', method='ilt', **options)[0, 0] == 0.5
        assert mp.mp.dps == 15


@pytest.mark.parametrize('precision', [None, 60])
def test_ilt_augmentation_applies_precision_without_leaking_it(precision):
    def augment(*args):
        assert mp.mp.dps == (100 if precision is None else precision)
        assert mp.mpf(1) + mp.mpf('1e-40') != 1
        return {1: [1, 1, 1]}

    options = {} if precision is None else {'precision': precision}
    with mp.workdps(15), patch('birdepy.aug_ilt.aug_bld_ilt', augment):
        _, data = interface_aug.f_fun_bld(
            {}, [1, 1], lambda z, p: z, lambda z, p: z, 'ilt', 'ilt',
            [], [], 'linear', [0, 10], 1e-6, 1e-6, options,
        )
        assert data == {1: [1, 1, 1]}
        assert mp.mp.dps == 15


def test_ilt_matches_analytic_linear_probability_at_explicit_precision():
    with mp.workdps(20):
        actual = bd.probability(1, 2, .5, [.8, .3], model='linear',
                                method='ilt', precision=60, laplace_method='talbot',
                                lentz_eps=1e-12)
        expected = bd.probability(1, 2, .5, [.8, .3], model='linear', method='gwa')
        np.testing.assert_allclose(actual, expected, rtol=1e-6)
        assert mp.mp.dps == 20


def test_saddlepoint_high_precision_fallback_restores_context():
    # The machine-precision probability underflows; the mpmath fallback is required.
    with mp.workdps(25):
        actual = bd.probability(2000, 2, 1, [1, 1], model='linear', method='gwasa')
        assert actual[0, 0] == 0
        assert mp.mp.dps == 25


def test_cme_rebuilds_coefficients_when_precision_changes():
    with mp.workdps(15):
        ut.cme(lambda s: 1 / (s + 1), 1, 25)
    with mp.workdps(60):
        after_low_precision = ut.cme(lambda s: 1 / (s + 1), 1, 25)
        cached_eta = ut.cme_eta.copy()
        # Changing k forces a fresh construction, giving an independent cache check.
        ut.cme(lambda s: 1 / (s + 1), 1, 15)
        fresh = ut.cme(lambda s: 1 / (s + 1), 1, 25)
        assert ut.cme_eta == cached_eta
        assert after_low_precision == fresh
        assert fresh == pytest.approx(np.exp(-1), rel=1e-3)
