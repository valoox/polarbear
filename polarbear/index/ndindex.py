import abc
from typing import Iterable

from .index1d import Index


class NDIndex(object, metaclass=abc.ABCMeta):
    """A N-dimensional index"""

    @abc.abstractproperty
    def ndim(self) -> int:
        """The number of dimensions in the index"""
        return 0

    @abc.abstractmethod
    def axes(self) -> Iterable[Index]:
        """Returns the sequence of 1D axes composing that NDIndex

        Returns
        -------
        axes: Iterable[Index], length == self.ndim
            The one-dimensional axes composing the dimensions of that NDIndex.
        """
        return ()


class Axes(tuple, NDIndex):
    """Simple implementation of an NDIndex as a tuple of 1D indices"""
    def __new__(cls, *axes):
        """Constructor"""
        assert all(isinstance(ax, Index) for ax in axes)
        self = tuple.__new__(cls, *axes)
        return self

    @property
    def ndim(self) -> int:
        """The number of dimensions in the axes"""
        return len(self)

    def axes(self) -> Iterable[Index]:
        """Returns the sequence of axes composing this index"""
        return self

