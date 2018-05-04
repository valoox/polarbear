"""The contains the implementation for the generic datasets, as well as
ways of building and interacting with common use-cases (such as series)
"""
from abc import ABCMeta, abstractmethod
from typing import Tuple

import numpy as np

from .index import Index, as_index
from .buffer import Buffer, buffer


class Dataset(object, metaclass=ABCMeta):
    """The base dataset interface"""
    # TODO: Define the interface
    @abstractmethod
    @property
    def labels(self) -> Tuple[Index]:
        """The set of indices which form the labels of this dataset

        Returns
        -------
        labels: Tuple[Index]
            The indices forming the labels along each dimensions
        """
        return ()

    @property
    def shape(self) -> Tuple[int]:
        """Represents the logical shape of the Dataset. Note that this is
        NOT NECESSARILY the same as the data shape. For instance, a Series
        with an `n`-sized index and a `(n, k)` data buffer will have a
        dataset shape of (n,), whereas its data.shape will be (n, k).

        Returns
        -------
        shape: Tuple[int]
            The shape of the dataset
        """
        return tuple(idx.size for idx in self.labels)

    @property
    def ndim(self) -> int:
        """The number of dimensions in the dataset.
        Just like the shape, this is the number of dimensions of the dataset,
        which is not necessarily the same as the data dimensions

        Returns
        -------
        ndim: int
            The number of dimensions in the dataset
        """
        return len(self.labels)


class Uniform(Dataset):
    """A specialisation of the DataSet which contains a single Buffer of data
    (by opposition to DataFrame-like datasets which contains several of
    these, possibly of several different types)
    """
    @abstractmethod
    @property
    def data(self) -> Buffer:
        """Returns the n-dimensional data buffer

        Returns
        -------
        bfr: Buffer instance
            The buffer data
        """
        return None

    @property
    def dtype(self) -> np.dtype:
        """The data-type of the inner data

        Returns
        -------
        dtype: np.dtype instance
            The datatype of the inner data
        """
        return self.data.dtype


class Series(Uniform):
    """The series is a very simple 1D dataset, i.e. a pair (index, values)
    where index is a 1D index[N] and values is a Buffer[N, ...]

    Parameters
    ----------
    index: Index instance
        The index for the data
    values: Buffer instance, first dimension being N
        The values being indexed
    """
    __slots__ = ('_index', '_values')

    def __init__(self, index: Index, values: Buffer):
        """Constructor"""
        self._index = as_index(index)
        self._values = buffer(values, shape_hint=(self._index.size,))
        assert self._values.shape[0] == self._index.size

    @property
    def size(self) -> int:
        """The size of the series is the length of the series index

        Returns
        -------
        len: int
            The length of the series
        """
        return self.index.size

    def __len__(self) -> int:
        """The length of the series is its indexed first dimension

        Returns
        -------
        length: int
            The length of the series
        """
        return self.size

    @property
    def index(self) -> Index:
        """Read access to the index

        Returns
        -------
        index: Index instance
            The index of that series
        """
        return self._index

    @property
    def data(self) -> Buffer:
        """Read access to the buffer

        Returns
        -------
        bfr: Buffer instance
            The buffer holding the data
        """
        return self._values

    @property
    def labels(self) -> Tuple[Index]:
        """The labels of the series"""
        return self.index,

    def __getitem__(self, item):
        """Access """
