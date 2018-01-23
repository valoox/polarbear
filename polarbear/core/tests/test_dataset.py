from numpy.testing import assert_allclose
from polarbear.core.index import *
from polarbear.core.data import DataSet


def test_indexing():
    ds = DataSet(
        (10 * np.arange(26))[:, None] + np.arange(10)[None, :],
        ([chr(ord('a') + i) for i in range(26)],
         [i*10 for i in range(10)])
    )
    sub = ds[['a', 'c', 'x'], 30:60]
    assert_allclose(sub.data, [[3, 4, 5], [23, 24, 25], [233, 234, 235]])
