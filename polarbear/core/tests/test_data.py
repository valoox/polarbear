import numpy as np
from ..data import view, View, DataSet


def test_isinstance():
    a = np.array([1, 2, 3])
    assert isinstance(a, View)
    assert not isinstance(12, View)
    assert isinstance(np.array(12.), View)


def test_view_fromarray():
    a = np.array([1, 2, 3])
    v = view(a)
    assert isinstance(v, np.ndarray)


def test_dataset():
    a = DataSet([[[1, 2, 3], [4, 5, 6]]], (range(10), ['a', 'b'], None))
    assert a.shape == (10, 2, 3)