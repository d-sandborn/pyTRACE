import pytest
import numpy as np
from pyTRACE import trace, integrate_column


def test_dummy():
    """Are internal tests working?"""
    assert 1 == 1


def test_trace_matlab():
    """Is pyTRACE giving identical results to the TRACEv1 check values?"""
    output = trace(
        output_coordinates=np.array([[0, 0, 0], [0, 0, 0]]),
        dates=np.array([2000, 2200]),
        predictor_measurements=np.array([[35, 20], [35, 20]]),
        predictor_types=np.array([1, 2]),
        atm_co2_trajectory=9,
    )
    assert output.canth.data[0] - 47.7869 < 0.00001


def test_integrate_column():
    """Is the integration routine interpolating correctly, and not
    extrapolating beyond define bottom bound?"""
    integral = integrate_column(
        integrand=np.array([1, 2, 3]),
        salinity=np.array([35, 35, 35]),
        temperature=np.array([4, 4, 4]),
        depth=np.array([100, 200, 300]),
        lat=0,
        bottom=250,
    )
    assert integral - 321347.0731205583 < 0.00001
