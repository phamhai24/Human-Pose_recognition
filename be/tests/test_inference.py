import numpy as np
import pytest

from be.app.inference import validate_sequence, validate_probabilities


@pytest.mark.parametrize('value', [np.zeros((9, 132)), np.zeros((10, 131)), np.full((10, 132), np.nan)])
def test_reject_invalid_sequence(value):
    with pytest.raises(ValueError):
        validate_sequence(value)


def test_valid_sequence_is_float32_batch():
    result = validate_sequence(np.zeros((10, 132)))
    assert result.shape == (1, 10, 132)
    assert result.dtype == np.float32


@pytest.mark.parametrize('value', [[1, 0], [float('nan')] * 6, [-1, 1, 0, 0, 0, 1], [0] * 6])
def test_reject_invalid_probabilities(value):
    with pytest.raises(ValueError):
        validate_probabilities(value)


def test_probabilities_preserve_class_order():
    assert validate_probabilities([.1, .2, .3, .1, .2, .1]) == pytest.approx([.1, .2, .3, .1, .2, .1])
