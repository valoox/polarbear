from abc import ABCMeta, abstractmethod

import numpy as np

from .index import indexing


class Labels(tuple):
    """Represents a set of labelled axes

    Parameters
    ----------
    shape: tuple[int]
        The shape of the inner data being labelled
    axes: collection[Axis]
        The labels for each of the axes
    """
    def __new__(cls, shape: tuple, *axes):
        """Constructor from a set of axes"""
        if len(axes) > len(shape):
            raise IndexError(
                f"Invalid set of axes: got {len(axes)} label axes to index "
                f"{len(shape)}-dimensional data"
            )
        ax = (
                tuple(Axis.new(ax, size=n) for n, ax in zip(shape, axes)) +
                tuple(NoLabel(size=n) for n in shape[len(axes):])
        )
        return tuple.__new__(cls, ax)

    @property
    def shape(self):
        """This is the shape of the data"""
        return tuple(ax.size for ax in self)

    def select(self, item):
        """Maps a selector on the output labels on to the inner data

        Parameters
        ----------
        item: tuple[Selector]
            The selection being made

        Returns
        -------
        labels: Labels instance | None
            The labels for the output being generated
        idx: Selector[int]
            The selector for the underlying data
        """
        adv, idx = indexing(item, self.shape)

        print(adv, idx)
        # TODO !!!

    def at(self, idx):
        """Selects the labels by index

        Parameters
        ----------
        idx: Selector[int]
            The selection being made on the labels

        Returns
        -------
        labels: Labels instance
            The labels corresponding to the selection
        """
        index = Indexing(idx, self.shape)
        adv, out = index.output()
        # TODO !!!


class Axis(object, metaclass=ABCMeta):
    """Represents a single axis of labels"""
    @classmethod
    def new(cls, data, size: int, dtype: np.dtype=None, copy: bool =False):
        """Creates a new axis matching the provided data"""
        if not hasattr(data, '__len__') or len(data) != size:
            return Index(np.array([data for _ in range(size)]), dtype=dtype)
        return Index(np.array(data, dtype=dtype, copy=copy))

    @property
    @abstractmethod
    def size(self) -> int:
        """The size of the axis"""
        return 0


class Index(Axis):
    """A simple 1D array as an Axis

    Parameters
    ----------
    arr: np.ndarray
        The labels themselves
    """
    def __init__(self, values, dtype=None):
        self._arr = np.array(values, dtype=dtype)
        assert values.ndim == 1

    @property
    def size(self):
        """The size of the index"""
        return self._arr.shape

    def __getitem__(self, item):
        """Gets the provided item"""
        out = self._arr[item]
        if isinstance(out, np.ndarray):
            return Index(out)
        return out
