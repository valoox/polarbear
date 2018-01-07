"""The `DataSet` is the main container for the different data.

Conceptually, it is simply a sequence of pair `View` and `Index`.

Note that these `DataSet`s are therefore from a single uniform datatype; for
multiple types of data, there is also the `DataFrame`, which combines
different dataset with the same indices along a different index.
"""
import numpy as np
from .view import view, View
from .labels import make_index, Labels


class DataSet(object):
    """Simple implementation of the dataset

    Parameters
    ----------
    data: View instance (or convertible to it)
        The data being held in the dataset
    index: Index instance (or convertible to it), if anything
        The index, if any; if none, or if it doesn't cover the entire shape
        of the data, simple passthroughs are added for missing axes
    dtype: np.dtype, optional [default=None]
        The type of the data
    order: str, optional [default='K']
        The order of the data
    copy: bool, optional [default=False]
        Whether data should be copied when building the object. Note that
        depending on the compatibility of the dtype, this might be ignored
        and data would _still_ be copied to make it consistent.
    """
    # Simply works as a pair (data, index)
    __slots__ = ('data', 'index')

    def __init__(self, data, index=None, dtype: np.dtype=None,
                 order: str='K', copy: bool=False):
        """Constructor"""
        self.data = view(data, dtype=dtype, order=order, copy=copy)
        self.index = make_index(self.data.shape, index)

    @property
    def shape(self) -> tuple:
        """The shape of the data

        Returns
        -------
        shape: tuple[int]
            The shape of the data
        """
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        """The type of the data

        Returns
        -------
        data_type: np.ndtype
            The type of the data in the dataset
        """
        return self.data.dtype

    def __getitem__(self, item):
        """Gets one or more item from the dataset

        Parameters
        ----------
        item: Index selectors
            The selectors, using the index of this dataset

        Returns
        -------
        out: dtype | DataSet
            Either a single element, or a subset of the dataset
        """
        idx, item = self.index.select(item)
        data = self.data[item]
        if idx is None:
            return data
        return DataSet(data, index=idx)

    def __setitem__(self, item, value):
        """Sets the value provided

        Parameters
        ----------
        item: Selector[labels]
            The selector on the index of this dataset
        value: array-like[dtype]
            The values to set in this position
        """
        _, item = self.index.select(item)
        self.data[item] = value

    class _AtWrapper(object):
        """Simple wrapper to access values in a DataSet by indices,
        while still retaining labels

        Parameters
        ----------
        ds: DataSet instance
            The dataset being worked on
        """
        __slots__ = ('ds',)

        def __init__(self, ds):
            """Constructor"""
            self.ds = ds

        def __getitem__(self, item):
            """Access to the data by INDEX"""
            data = self.ds.data[item]
            idx = self.ds.index.at(item)
            return DataSet(data, idx)

    @property
    def at(self):
        """A simple accessor to get the data given its INDICES, while still
        retrieving the index data

        Returns
        -------
        at: _AtWrapper instance
            A simple wrapper to get access to the data by indices
        """
        return self._AtWrapper(self)
