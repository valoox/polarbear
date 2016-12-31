"""Implementation of the various N-dimensional frames

A Frame is conceptually a single-typed array, along with an index.

The 'array' constituting the data only has to match the 'is_arraylike' function,
but can itself be a view, or any other type of object matching the provided
interface.

The Frame then simply offers ways of converting queries to the index to the
corresponding queries on the underlying data, and providing a uniform access
layer to the data itself, when transforming/viewing it.
"""
import numpy as np

from polarbear import index

import polarbear.data.arraylike as arraylike


class NDFrame(object):
    """Represents the base of the N-Dimensional Frame, which is simply a wrapper
    around a pair (data, axes), where data is an n-dimensional dataset (such as
    a numpy.ndarray or similar), and axes is an n-dimensions Index.

    Parameters
    ----------
    data: Array-like, n-dimensional
        The data being wrapped itself
    axes: NDIndex, n-dimensional
        The dimensions of the data being wrapped.
    """
    __slots__ = ('data', 'axes')

    def __init__(self, data, axes: index.NDIndex):
        """Constructor"""
        self.axes = axes
        self.data = arraylike.validate(data, axes)

    @property
    def ndim(self) -> int:
        """Number of dimensions in the dataset"""
        return self.axes.ndim

    @property
    def dtype(self) -> np.dtype:
        """The data-type of the data in the dataset"""
        return self.data.dtype
