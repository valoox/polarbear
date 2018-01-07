import numpy as np
from ..view import view, View


def test_isinstance():
    a = np.array([1, 2, 3])
    assert isinstance(a, View)
    assert not isinstance(12, View)
    assert isinstance(np.array(12.), View)


def test_view_fromarray():
    a = np.array([1, 2, 3])
    v = view(a)
    assert isinstance(v, np.ndarray)


