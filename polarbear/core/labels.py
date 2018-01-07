"""Represents index, i.e. a collection of values being used as labels"""

from abc import ABCMeta, abstractmethod
import numpy as np


class Labels(metaclass=ABCMeta):
    """The base `Index` class defining the general interface for all indices

    Its core is the `select` method which converts a set of selectors on the
    index labels (e.g. a list of strings) into the selector on the underlying
    view.
    """
    @abstractmethod
    def select(self, item):
        """This is the main method of the index, which converts a set of
        selector on the labels to the corresponding set of selectors on the
        underlying box of data

        Parameters
        ----------
        item: Selector | tuple[Selector]
            The selectors on the labels

        Returns
        -------
        idx: Index instance | None
            The selected Index
        sel: Selector | tuple[Selector]
            The corresponding set of selectors on the underlying view
        """
        return self, slice(None)

    @abstractmethod
    def at(self, item):
        """Gets the 'sub'-index at the provided indices

        Parameters
        ----------
        item: Selector | tuple[Selector]
            Selectors on the INDICES of the labels, rather than the labels
            themselves

        Returns
        -------
        idx: Index instance
            The corresponding 'sub'-index at the specified indices
        """
        return self


class Axis(Labels):
    """Represents a single axis of a Labels (although this can still be
    acting as a Label colleciton of its own, e.g. for 1-dimensional data)
    """
    @classmethod
    def new(cls, data, dtype: np.dtype=None, copy: bool=False):
        """Converts something into an axis

        """
        if isinstance(data, Axis):
            return data
        elif data is None:
            return Passthrough()
        # TODO !!!!

    @abstractmethod
    def select1d(self, s):
        """Selects from a single selector

        Parameters
        ----------
        s: Selector instance
            The selector itself, on the LABELS

        Returns
        -------
        idx: Axis instance
            The selected Axis, if any
        item: Selector instance
            The corresponding selector, on the INDICES of the underlying view
        """
        pass

    @property
    @abstractmethod
    def dtype(self):
        """The type of labels on this axis

        Returns
        -------
        dtype: np.dtype
            The type of labels on this axis
        """
        return ()

    def select(self, item):
        """Implements the full selection, if the entire Index is a single
        Axis

        Parameters
        ----------
        item: Selector | tuple[Selector]
            The selection on the labels

        Returns
        -------
        idx: Index instance
            The selected Index, if any
        item: Selector | tuple[Selector]
            The selection on the underlying data
        """
        if isinstance(item, tuple):
            if len(item) != 1:
                raise IndexError(
                    'Cannot index 1-dimensional axis with '
                    f'{len(item)}-dimensional index: {item}'
                )
            item = item[0]
        axis, item = self.select1d(item)
        return axis, item


class Passthrough(Axis):
    """This is a somewhat degenerate implementation of the Index, which acts
    as if there was no index at all;
    """
    def select1d(self, s):
        """Selection is relatively straightforward, as this is simply passing
        through the data; only subtlety is to specify whether or not this
        should yield itself or if the selector is 'unique'

        Parameters
        ----------
        s: Selector[int]
            The selector itself

        Returns
        -------
        axis: Axis instance | None
            Whether there is something to index back
        s: Selector[int]
            The selector on the underlying data
        """
        if isinstance(s, int):
            return None, s
        return Passthrough(), s

    def at(self, item):
        """Selects a subset of the index
        Being a simple passthrough, this is always unchanged

        Parameters
        ----------
        item: Selector[int]
            The selector on the data

        Returns
        -------
        itself: Index instance
            The Index corresponding to the selection
        """
        return Passthrough()

    @property
    def dtype(self):
        """The type of labels along this axis

        Returns
        -------
        dtype: np.dtype
            The type of labels along this axis
        """
        return np.int


class Sorted(Axis):
    """A simple sorted collection of labels, which can be searched efficiently
    for elements

    Parameters
    ----------
    data: array-like, sorted
        The labels of the axis
    dtype: np.dtype, optional [default=None]
        The data type of the labels; if none, infers it from the data
    copy: bool, optional [default=False]
        Whether the data should be copied when building it
    """
    def __init__(self, data, dtype: np.dtype=None, copy: bool=False):
        """Constructor"""
        self.data = np.array(data, dtype=dtype, copy=copy)

    @property
    def dtype(self):
        """The type of labels along that axis"""
        return self.data.dtype

    def select1d(self, s):
        """Selects along the data"""


class Map(Axis):
    """A simple map between a set of labels and their corresponding indices

    """


class Array(Axis):
    """This is a generic numpy array, in no particular order. If data is
    sorted, then Sorted should be preferred.

    Parameters
    ----------
    labels: sequence[T]
        The labels of the index
    dtype: np.dtype
        The type of the values in the index

    """
    def __init__(self, labels, dtype: np.dtype=None, copy: bool=False):
        """"""
        # TODO !!!!
        pass


# TODO: !!!!!!!
# TODO: ! Big difficulty will be ADVANCED INDEXING
# TODO: ! as it will change the STRUCTURE of the index itsel
# TODO: !!!!!!
class Cartesian(Labels, tuple):
    """Simple implementation of a Labels as a cartesian product of a set of
    axes

    Parameters
    ----------
    axes: iterable[Axis]
        The different axes of the indexing; each is completely independent
    """

    def __new__(cls, *axes):
        """Constructor"""
        return tuple.__new__(cls, (Axis.new(ax) for ax in axes))

    @property
    def ndim(self):
        """The number of dimensions of the index

        Returns
        -------
        n: int
            The number of dimensions of the index
        """
        return len(self)

    def __getitem__(self, idx):
        """Gets access to one or more of the axes, as an Index

        Parameters
        ----------
        idx: int | iterable[int] | slice
            The item(s) to retrieve
        """
        if isinstance(idx, int):
            return self[idx]
        return Cartesian(*self[idx])

    def select(self, item):
        """The selection corresponding to each of the indices

        Parameters
        ----------
        item: tuple[Selector]
            The selectors for each of the axes of the index

        Returns
        -------
        idx: Labels | None
            The index corresponding to this selection, if any
        item: tuple[Selector[int]]
            The selector for the underlying array
        """
        # TODO!!!!

    def at(self, item):
        """Selects the index at the provided indices

        Parameters
        ----------
        item: Selector[int] | tuple[Selector[int]]
           The selection being made on the index itself

        Returns
        -------
        idx: Index instance
            The selected sub-index
        """
        if not isinstance(item, tuple):
            item = (item,)
        if len(item) > len(self):
            raise IndexError(
                f'Too many indices ({len(item)}) for Index ({len(self)} dims)'
            )
        axes = tuple(ax.at(s) for ax, s in zip(self, item))
        axes += tuple()
