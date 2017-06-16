"""Represents index, i.e. a collection of values being used as labels on an
axis of a View"""

import numpy as np

from polarbear.compat import abstract


class Index(abstract(object)):
    """The base index abstraction"""
    def __new__(cls, *args, **kwargs):
        """Constructor"""
        #TODO


class _Array1d(Index):
    """Wraps a 1D array as an index"""
    def __init__(self, data):
        """Constructor"""
        self.data = data


class IntIndex(_Array1d):
    """Integral index

    Parameters
    ----------
    data: np.ndarray[Int](n,)
        The indices
    copy: bool, optional
        Whether to copy the index (default=False)
    """
    def __init__(self, data, copy=False):
        """Constructor"""
        data = np.array(data, copy=copy)
        if not np.issubdtype(data.dtype, int):
            raise TypeError('Expected integer, got {}'.format(data.dtype))
        super(IntIndex, self).__init__(data=data)


class Range(Index):
    """Lazily represents a range"""
    def __init__(self, n, dtype=np.intp):
        """Constructor"""
        self.n = n
        self.dtype = dtype
