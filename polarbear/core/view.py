"""The View is the core interface of the entire library, which defines an
n-dimensional array along with a set of index axes
"""
from abc import abstractmethod, abstractproperty

import numpy as np

from polarbear.compat import abstract
import polarbear.core.index as index


class View(abstract(object)):
    """The base `view` interface being shared by all data containers and
    pipes.

    This conceptually acts like an inner (n-dimensional) array with `n`
    index axes.
    """
    @abstractproperty
    def shape(self):
        """The shape of the data

        Returns
        -------
        shape: tuple[int]
            The shape of the data
        """
        ()

    @abstractproperty
    def dtype(self):
        """The type of the data manipulated"""
        pass

    @abstractmethod
    def read(self, into):
        """Reads the entire view into the provided buffer

        Parameters
        ----------
        into: np.ndarray[T: self.dtype](self.shape)
            The buffer in which the value are being copied. Note that dtype
            MUST be compatible with the inner dtype

        Returns
        -------
        into: np.ndarray[T](self.shape)
            The buffer, once data has been written into it
        """
        pass

    def to_array(self, order='C'):
        """Converts this to a numpy array

        Parameters
        ----------
        order: str, optional
            The order of the returned array

        Returns
        -------
        arr: np.ndarray[self.dtype](self.shape)
            Representation of this, as a numpy array
        """
        arr = np.empty(self.shape, dtype=self.dtype, order=order)
        return self.read(into=arr)

    @property
    def ndim(self):
        """Number of dimensions in the view

        Returns
        -------
        ndim: int
            The number of dimensions
        """
        return len(self.shape)

    @abstractproperty
    def axes(self):
        """Returns a sequence of `ndim` axes corresponding to each of the
        axes of the View

        Returns
        -------
        axes: tuple[Index]
            The axes of the view
        """
        return ()

    @abstractproperty
    def aview(self):
        """Returns a simple array-like view, which can be indexed as a
        numpy array and behaves like it in every way.

        Most indexed/labelled operations are simply mapped to this view

        Returns
        -------
        view: np.ndarray-like[self.dtype](self.shape)
            The view on which operations are carried out
        """
        pass

    def __getitem__(self, item):
        """Access to a selection in the view"""
        if not isinstance(item, tuple):
            item = (item,)
        iitem = tuple(ax(at) for ax, at in zip(self.axes, item))
        return self.aview[iitem]


class HCube(View):
    """The HyperCube is the simplest implementation of the `View` interface,
    and simply wraps a numpy.ndarray along with a set of indices for each of
    the axes

    Parameters
    ----------
    data: np.ndarray[dtype]
        The inner data of the HCube
    axes: tuple[Index]
        The indices for each of the axes
    copy: bool, optional (default=False)
        If set, this will copy the data
    dtype: np.dtype, optional
        The data type to use. Default is to keep the one in `data`
    broadcast: bool, optional
        If true (the default), this will broadcast the data to be consistent
        with the indices
    """
    def __init__(self, data, axes, copy=False, dtype=None, broadcast=True):
        """Constructor"""
        self.data = np.array(data, dtype=dtype, copy=copy)
        self._axes = ()
        for i, ax in enumerate(axes):
            ax = index.Index(ax)
            # Consistent: same length or broadcast
            l = self.data.shape[i]
            if l == len(ax) or (broadcast and l == 1):
                self._axes += (ax,)
            else:
                raise ValueError(
                    "Inconsistent dimension {i}: data is of shape "
                    "{self.data.shape[i]}, but axis is of length {ax.size}",
                    i=i, self=self, ax=ax
                )
        rem = self.data.shape[len(self._axes):]
        self._axes += tuple(index.Range(l) for l in rem)
        if broadcast:
            self.data = np.broadcast_to(
                self.data, tuple(len(ax) for ax in self._axes)
            )

    @property
    def axes(self):
        """The indices/axes for the hypercube"""
        return self._axes

    @property
    def shape(self):
        """The shape is the shape of the underlying array"""
        return self.data.shape

    @property
    def ndim(self):
        """There are as many dimensions as the underlying array"""
        return self.data.ndim

    @property
    def aview(self):
        """Hypercube simple refer to the inner data as the numpy view"""
        return self.data
