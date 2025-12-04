import pytest
import numpy as np
from TRACE import trace, integrate_column


def test_dummy():
    """Are internal tests working?"""
    assert 1 == 1


def test_trace_matlab():
    """Is TRACE giving identical results to the TRACEv1 check values?"""
    output = trace(
        output_coordinates=np.array([[0, 0, 0], [0, 0, 0]]),
        dates=np.array([2000, 2200]),
        predictor_measurements=np.array([[35, 20], [35, 20]]),
        predictor_types=np.array([1, 2]),
        atm_co2_trajectory=9,
    )
    assert np.abs(output.canth.data[0] - 47.7868563)  < 0.00001
    assert np.abs(output.canth.data[1] - 79.8749319) < 0.00001


def test_trace_matlab_no_temperature():
    """Is TRACE giving identical results to no-T TRACEv1 check values?"""
    output = trace(
        output_coordinates=np.array([[0, 0, 0], [0, 0, 0]]),
        dates=np.array([2000, 2010]),
        predictor_measurements=np.array([[35], [35]]),
        predictor_types=np.array([1]),
        atm_co2_trajectory=1,
    )
    assert np.abs(output.canth.data[0] - 56.0591388) < 0.00001
    assert np.abs(output.canth.data[1] - 66.4566880) < 0.00001


def test_integrate_column():
    """Is the integration routine interpolating correctly?"""
    integral = integrate_column(
        integrand=np.array([1, 2, 3]),
        salinity=np.array([35, 35, 35]),
        temperature=np.array([4, 4, 4]),
        depth=np.array([100, 200, 300]),
        lat=0,
        bottom=250,
    )
    assert integral - 321347.0731205583 < 0.00001
