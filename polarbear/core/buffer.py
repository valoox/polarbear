"""This defines the Buffer interface, which is essentially isomorphic to a
numpy array, while also allowing alternative implementations (such as lazy
computations, or on-disk storage)
"""
from abc import ABCMeta, abstractmethod
from typing import Any, Tuple

import numpy as np


class Buffer(object, metaclass=ABCMeta):
    """The generic Buffer interface"""
    @abstractmethod
    @property
    def dtype(self) -> np.dtype:
        """The type of the data held in the buffer

        Returns
        -------
        dtype: np.dtype instance
            The type of the data held in the buffer
        """
        return np.dtype(None)

    @abstractmethod
    @property
    def shape(self) -> Tuple[int]:
        """Returns the (logical) shape of the buffer, in number of elements

        Returns
        -------
        dimensions: tuple[int]
            The (logical) dimensions of the buffer
        """
        return ()

    @property
    def size(self) -> int:
        """The total number of elements in the buffer. This will typically be
        the product of all dimensions, but could be lower for non-dense
        storage for instance.

        Returns
        -------
        size: int
            The total number of elements in this buffer
        """
        return np.product(self.shape, dtype=int)

    @property
    def ndim(self) -> int:
        """The number of dimensions of the array

        Returns
        -------
        ndim: int
            The total number of dimensions of the buffer
        """
        return len(self.shape)

    @abstractmethod
    def take(self, indices,
             axis: int=None,
             out: np.ndarray=None,
             mode: str='raise') -> np.ndarray:
        """Takes a (dense) set of values from this buffer

        Parameters
        ----------
        indices: np.ndarray | int
            The indices of the values to extract
        axis: int, optional (default=None)
            The axis along which the selection is made. Default works on the
            'flattened' version, i.e. the 'absolute' index of each value in
            [0; self.size)
        out: np.ndarray[self.dtype]
            The output where data is written
        mode: str in {'raise', 'wrap', 'clip'}, optional [default='raise']
            Specifies how out-of-bounds indices will behave.
            * 'raise' -- raise an error (default)
            * 'wrap' -- wrap around
            * 'clip' -- clip to the range

        Returns
        -------
        out: np.ndarray
            The output values, as a dense numpy array
            If out is passed, it is used; otherwise, a new one is generated
            with the same dtype as self.
        """
        indices = np.array(indices, dtype=int)
        if out is None:
            if axis is None:
                shape = indices.shape
            else:
                shape = tuple(
                    dim for i, dim in enumerate(self.shape) if i != axis
                )
            out = np.empty(shape, dtype=self.dtype)
        if axis is None:
            # TODO: flat?
            self.flat.readat(indices, out=out)
        else:
            # TODO: view?
            idx = tuple(slice(None) for _ in range(axis)) + (indices, Ellipsis)
            self.view[idx].copyto(out)
        return out

    @abstractmethod
    def __getitem__(self, item):
        """Gets """


def buffer(data: Any, shape_hint: Tuple[int]=()) -> Buffer:
    """Generates a new buffer from the provided data

    Parameters
    ----------
    data: Any
        The data being used to represent the buffer
    shape_hint: tuple[int], optional
        The partial shape(s) of the data to match. This is used to broadcast
        data and assert consistency, raising an exception if the data is
        inconsistent with the hint and cannot be broadcasted.
        As in the broadcast convention, passing in '1' as a dimension will
        keep the data dimension, whatever that is.

    Returns
    -------
    bfr: Buffer instance
        The buffer corresponding to this data
    """
    if isinstance(data, Buffer):
        return data
    else:
        raise ValueError("Invalid data")
