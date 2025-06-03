import pytest
import numpy as np
from pyTRACE import trace


def test_dummy():
    assert 1 == 1


def test_trace_matlab():
    output = trace(
        output_coordinates=np.array([[0, 0, 0], [0, 0, 0]]),
        dates=np.array([2000, 2200]),
        predictor_measurements=np.array([[35, 20], [35, 20]]),
        predictor_types=np.array([1, 2]),
        atm_co2_trajectory=9,
    )
    assert output.canth.data[0] - 47.7869 < 0.0001
