"""Represents index, i.e. a collection of values being used as labels on an
axis of a View"""
import numpy as np

from polarbear.compat import abstract, abstractproperty, abstractmethod


class Index(abstract(object)):
    """The base index abstraction"""
    def __new__(cls, *args, **kwargs):
        """Constructor"""
        #TODO

    @abstractproperty
    def dtype(self):
        """The type of the data in the index

        Returns
        -------
        dtype: np.dtype
            The data type of the index
        """
        pass

    @abstractmethod
    def index(self, value):
        """Gets the index of the provided value

        Parameters
        ----------
        value: Selector[self.dtype]
            The value for which the index is required

        Returns
        -------
        index: Selector[int]
            The corresponding selection of index/indices
        """
        return ()

    @abstractmethod
    def __getitem__(self, item):
        """Gets the value of the index corresponding to the item

        Parameters
        ----------
        item: Selector[int]
            The index to fetch

        Returns
        -------
        index: Selector[self.dtype]
            The corresponding selector
        """
        return ()


class _Array1d(Index):
    # TODO: Add sorter array ?
    """Wraps a 1D array as an index"""
    def __init__(self, data):
        """Constructor"""
        self.data = data

    @property
    def dtype(self):
        """The type of the data"""
        return self.data.dtype

    def __getitem__(self, item):
        """self.__getitem__(x) <=> self[x]"""
        return self.data[item]


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
        self._dtype = dtype

    @property
    def dtype(self):
        """The data type of the data"""
        return self._dtype

    def index(self, value):
        """Gets the index of the provided item"""
        return value

    def __getitem__(self, item):
        """Get the labels of the provided indices"""
        return item
