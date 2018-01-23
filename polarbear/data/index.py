"""The core module for Indexing and manipulations
"""
from abc import ABCMeta, abstractmethod
from collections import namedtuple

import numpy as np


# tuple[int, int, Selector]
# Parser-friendly version of a single selector
# itarget: index of the targetd axis in the input
# iselect: index of the selector in the original selector
# selector: None, slice, array, etc
# ndim: int, the number of dimensions created by the selector
Selector = namedtuple('Selector', ('itarget', 'iselect', 'selector', 'ndim'))


def normalise(items, shape: tuple):
    """Pushes a single selector

    Parameters
    ----------
    items: Selector
        The tuple of selector to normalise
    shape: tuple[int]
        The shape of the input array

    Returns
    -------
    out: tuple
        The normalised tuple
    """
    if not isinstance(items, tuple):
        items = (items,)
    front = []
    back = []
    _at = front
    i = 0
    for s in items:
        if s is None:
            _at.append(None)
        elif s is Ellipsis:
            if _at is back:
                raise IndexError(
                    "an index can only have a single ellipsis ('...')"
                )
            _at = back
        elif isinstance(s, slice):
            i += 1
            _at.append(s)
        elif isinstance(s, (list, np.ndarray)):
            i += 1
            _at.append(np.array(s))
        else:  # Atom
            i += 1
            _at.append(s)
    out = front + [slice(None) for _ in range(len(shape) - i)] + back
    return tuple(out)


def indexing(sel, shape: tuple) -> tuple:
    """Converts a tuple of selections to more explicit Selectors

    Parameters
    ----------
    sel: selector
        The raw selector
    shape: tuple[int]
        The shape of the input

    Returns
    -------
    out: tuple[Selector]
        The selector and their input/output index
        The selectors are returned in the order they will be returned
        from the selection.
        Each tuple is:
          * The index of its input axis
          * The index of this selector in the original selection
          * The selector proper
          * The number of dimensions of the output axe/is
    """
    selection = normalise(sel, shape=shape)
    advanced = 1 < sum(isinstance(s, np.ndarray) for s in selection)
    out = ()
    i = 0
    adv = ()
    for j, s in enumerate(selection):
        if s is None:
            # New axis
            out += (Selector(-1, j, None, 1),)
        elif isinstance(s, np.ndarray) and advanced:
            # Advanced indexing
            adv += ((i, j, s),)
            i += 1
        elif isinstance(s, (slice, np.ndarray)):
            # Slice/Array (single)
            ndim = s.ndim if isinstance(s, np.ndarray) else 1
            out += (Selector(i, j, s, ndim),)
            i += 1
        else:
            # Atom
            out += (Selector(i, j, s, 0),)
            i += 1
    # Broadcasting all arrays in the advanced indexing
    bcst = np.broadcast_arrays(*(arr for _, _, arr in adv))
    nmax = max(b.ndim for b in bcst)
    adv = tuple(Selector(i, j, arr, -nmax) for (i, j, _), arr in zip(adv, bcst))
    return adv, out


class Axis(object, metaclass=ABCMeta):
    """The basis class for all the different axes."""
    @property
    @abstractmethod
    def size(self) -> int:
        """The size of the axis"""
        return 0

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """Data-type of the axis"""
        return np.dtype('L')

    @abstractmethod
    def apply(self, sel: Selector, axes: list) -> tuple:
        """Applies the provided selector to this axis

        Parameters
        ----------
        sel: Selector instance
            The selector being applied
        axes: list[Axis]
            The output axes, modified by side-effect

        Returns
        -------
        isel: Selector[int]
            The corresponding int selector
        """
        pass


class Range(Axis):
    """
    This is used as a simple axis itself, assuming 'no labels', i.e. a similar
    indexing

    Parameters
    ----------
    n: int
        The size of the axis
    """
    def __init__(self, n: int):
        """Constructor"""
        self.n = n

    @property
    def dtype(self) -> np.dtype:
        """dtype of the index"""
        return np.dtype('L')

    @property
    def size(self) -> int:
        """The size of the axis"""
        return self.n

    def isel(self, s: Selector) -> Axis:
        """Selects a subset of the axis

        Parameters
        ----------
        s: Selector[int] instance
            Normalised selector

        Returns
        -------
        ax: Axis instance
            The selected instance
        """


class Index(Axis):
    """A simple sorted array of labels

    Parameters
    ----------
    arr: np.ndarray[T](n,)
        The array of labels
    """
    def __init__(self, arr: np.ndarray):
        """Constructor"""
        arr = np.array(arr)
        assert arr.ndim == 1
        self.data = arr

    @property
    def dtype(self) -> np.dtype:
        """The data-type of the axis"""
        return self.data.dtype

    @property
    def size(self) -> int:
        """The size of the Index"""
        return len(self.data)


class Axes(tuple):
    """Axes <=> tuple[Axis]"""
    def __new__(cls, *args):
        """Constructor"""
        assert all(isinstance(ax, Axis) for ax in args)
        return tuple.__new__(cls, args)

    @property
    def shape(self):
        """The shape of axes"""
        return tuple(ax.size for ax in self)

    def __add__(self, other):
        """Concatenates two tuples"""
        return Axes(*tuple.__add__(self, other))

    def __getitem__(self, item):
        """Slices axes"""
        return Axes(*tuple.__getitem__(self, item))

    def __repr__(self):
        """Representation of this"""
        return 'Axes({})'.format(', '.join(repr(ax) for ax in self))

    def iselect(self, tup: tuple):
        """Applies a tuple of selectors to a set of axes

        Parameters
        ----------
        tup: tuple[selectors]
            The selection being made

        Returns
        -------
        oaxes: Axes instance
            The resulting axes of the selection
        itup: tuple[selector[int]]
            The mapped selection
        """
        out = [None for _ in tup]
        adv, idx = indexing(tup, self.shape)
        oaxes = []
        # TODO adv
        # Idx:
        for sel in idx:
            ax = self[sel.itarget]
            s = ax.apply(sel, oaxes)
            out[sel.iselect] = s
        return Axes(oaxes), tuple(out)
