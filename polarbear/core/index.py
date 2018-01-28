"""Implements indexing with labels"""

import numpy as np

from .arr import asorted


class BaseIndex(object):
    """The base index class

    Parameters
    ----------
    labels: sorted np.ndarray[Any](n)
        The labels of the index
    """
    __slots__ = ('labels',)

    def __new__(cls, labels: np.ndarray, copy: bool=True):
        return object.__new__(cls)

    def __init__(self, labels: np.ndarray, copy: bool=True):
        """Constructor"""
        self.labels = np.array(labels, copy=copy).ravel(order='C')

    @property
    def size(self) -> int:
        """The length of the index"""
        return self.labels.shape[0]


class Sorted(BaseIndex):
    """A sorted index, where data is prealably sorted

    Parameters
    ----------
    labels: np.ndarray[Any](n)
        The (sorted) labels of the index
    skip_check: bool, optional
        If set to True, the sorted check is skipped
    copy: bool, optional
        Whether to copy the labels
    """
    def __init__(self, labels: np.ndarray, skip_check: bool=False,
                 copy: bool=True):
        """Constructor"""
        if not skip_check:
            assert asorted(labels), 'Index MUST be sorted !'
        super(Sorted, self).__init__(labels=labels, copy=copy)


class Index(BaseIndex):
    """A 1-dimensional, not necessarily sorted

    Parameters
    ----------

    """
    def __new__(cls, labels: np.ndarray, copy: bool=True):
        """Constructor"""
        if asorted(labels):
            return Sorted(labels, skip_check=True, copy=copy)
        self = BaseIndex.__new__(cls, labels, copy=copy)
        return self

    def __init__(self, labels: np.ndarray, copy: bool=True):
        """Constructor"""
        order = np.argsort(labels, axis=0)
        labels = np.array(labels, copy=copy)
        super(Index, self).__init__(labels[order], copy=False)
        self.order = order
        
