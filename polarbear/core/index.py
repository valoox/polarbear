"""This defines and implement the Index interface, which is essentially a 1D
searchable collection of keys
"""
from abc import ABCMeta, abstractmethod

from typing import Any


class Index(object, metaclass=ABCMeta):
    """Represents a 1D index over an axis

    This is the basis for all indices, which can themselves be arbitrarily
    complex, but always represent this simple 1d mapping
    """
    @property
    def size(self) -> int:
        """The size of the index, in number of elements

        Returns
        -------
        size: int
            The number of elements in the index
        """
        return len(self)

    @abstractmethod
    def __len__(self) -> int:
        """The size of the index, in number of elements"""
        return 0


def as_index(data: Any, *, size: int=None) -> Index:
    """Converts the data to an index

    Parameters
    ----------
    data: Any
        The data being converted to an index
    size: int
        The size of the required index

    Returns
    -------
    idx: Index instance
        The Index instance corresponding to the provided data
    """
    # TODO: FILL THAT
