"""The View is the core interface of the entire library, which defines an
n-dimensional array-like structure, which can be accessed via:
 - integers, or lists or slices of integers
 - boolean masks
in order to mimick (and support) numpy arrays.
"""
from abc import ABCMeta, abstractmethod

import numpy as np


class _MetaView(ABCMeta):
    """A type of abstract base, which also recognises np.arrays as being
    of its type (although they are not).
    """
    # Global dict used to convert arbitrary data into views in `new`
    # Use with caution ! This is REALLY global, and can have VERY unforeseen
    # consequences. Be the reasonable adult we're all expecting you are.
    # If you definitely know what you're doing, this should contain entries
    # in the form:
    #  T: (fn(T, dtype=None, order='K', copy=False) -> U)
    # such that:
    # View.new(t: T, dtype=DTYPE, order=ORDER, copy=COPY) -> U
    convert_from = {}

    def __instancecheck__(cls, instance):
        """Checks that 'instance' is of this type

        Parameters
        ----------
        instance: Any
            The instance being checked

        Returns
        -------
        isinstance: bool
            Whether `instance` is an instance of `cls`
        """
        if cls is View and isinstance(instance, np.ndarray):
            return True
        # Fall back to default
        return type.__instancecheck__(cls, instance)

    def new(cls, data, dtype: np.dtype=None, order: str='K', copy: bool=False):
        """Method reading 'arbitrary' data into a View

        Parameters
        ----------
        data: np.ndarray | View | list | ...
            The data to be converted
        dtype: np.dtype | str | None [default=None]
            The type that data should be interpreted as
        order: str {'C', 'F', 'K'} [default='K']
            The order to use for the data:
             * 'C' is C-contiguous (row-first)
             * 'F' is Fortran-contiguous (columns-first)
             * 'K' is to Keep it as-is (default; 'C' when converting)
        copy: bool, optional [default=False]
            Whether to force a copy when reading the data

        Returns
        -------
        dat: np.ndarray | View instance
            The view being created
        """
        if isinstance(data, View):
            return data.astype(dtype=dtype, order=order, copy=copy)
        fn = cls.convert_from.get(type(data))
        if fn is not None:
            return fn(data, dtype=dtype, order=order, copy=copy)
        return np.array(data, dtype=dtype, order=order, copy=copy)


def view(data, dtype: np.dtype=None, order: str='K', copy: bool=False):
    """Reads arbitrary `data` into a view

    Parameters
    ----------
    data: np.ndarray | View | list | ...
        The data to be converted
    dtype: np.dtype | str | None [default=None]
        The type that data should be interpreted as
    order: str {'C', 'F', 'K'} [default='K']
        The order to use for the data:
         * 'C' is C-contiguous (row-first)
         * 'F' is Fortran-contiguous (columns-first)
         * 'K' is to Keep it as-is (default; 'C' when converting)
    copy: bool, optional [default=False]
        Whether to force a copy when reading the data

    Returns
    -------
    dat: np.ndarray | View instance
        The view being created
    """
    return View.new(data, dtype=dtype, order=order, copy=copy)


class View(object, metaclass=_MetaView):
    """A `view` is the basic abstraction corresponding to a numpy array in
    its interface, but which can also be used for other (e.g. sparse,
    or indirect) data structures.

    All of the other abstractions here are built assuming inner 'views',
    and should be able to handle any extension compatible with it, although
    they would be numpy arrays 99% of the time.

    Although the implementation is left to vary, the core concept covered
    here is that of an n-dimensional 'square' of data of a single type.
    """
    @property
    def ndim(self) -> int:
        """The number of dimensions in this view

        Returns
        -------
        ndim: int
            The number of dimensions
        """
        return len(self.shape)

    @property
    @abstractmethod
    def shape(self) -> tuple:
        """The shape of the data

        Returns
        -------
        shape: tuple[int]
            The shape of the data
        """
        return ()

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """The type of the data manipulated"""
        pass

    @abstractmethod
    def astype(self, dtype, order: str='K', casting: str = 'unsafe',
               subok: bool=True, copy: bool=True):
        """Returns a new instance of the View with the provided dtype.

        Please refer to `np.ndarray.astype` documentation for details

        Parameters
        ----------
        dtype: np.ndtype | str
            The dtype to convert this to
        order: str {'C', 'F', 'K', 'A'} [default='K']
            The memory order to return:
             * 'K' keeps it as unchanged as possible (default)
             * 'C' is C-contiguous (row first)
             * 'F' is Fortran-contiguous (columns first)
             * 'A' is automatic: 'F' is all subarrays are 'F'-contiguous and
               'C' otherwise;
        casting: str {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
            The type of casting: refer to numpy doc for details;
            default is `"unsafe"` for legacy reasons
        subok: bool, optional [default=True]
            If True, then sub-classes will be passed-through; this is only
            relevant for np.arrays, and ignored for most abstract Views
        copy: bool, optional [default=True]
            Whether to copy the data (default); if set to False with
            unchanged `dtype`, `order` and `subok` subtype, this will simply
            return the array unchanged

        Returns
        -------
        arr: np.ndarray | View
            The changed array/view
        """
        pass

    def copy(self, order: str='K'):
        """Performs a copy of the given view, with the provided order

        Parameters
        ----------
        order: str {'A', 'C', 'F', 'K'} [default='K']
            The order of the copy:
             * 'A' is auto: 'F' is all subarrays are 'F', 'C' otherwise
             * 'C' is C-contiguous (rows first)
             * 'F' is Fortran-contiguous (columns first)
             * 'K' is to Keep it as close as possible (default)

        Returns
        -------
        copy: Self instance
            A copy of this data
        """
        return self.astype(
            self.dtype, order=order, casting='same', subok=True, copy=True
        )

    @abstractmethod
    def __getitem__(self, item):
        """The accessor: this _only_ accepts integers.

        Parameters
        ----------
        item: int, tuple[int], list[int | tuple[int] | slice], ...
            The indexing into the view

        Returns
        -------
        data: self.dtype | View
            Either the item proper, or the subset of this.
        """
        pass

    # @abstractmethod
    # def read(self, into):
    #     """Reads the entire view into the provided buffer
    #
    #     Parameters
    #     ----------
    #     into: np.ndarray[T: self.dtype](self.shape)
    #         The buffer in which the value are being copied. Note that dtype
    #         MUST be compatible with the inner dtype
    #
    #     Returns
    #     -------
    #     into: np.ndarray[T](self.shape)
    #         The buffer, once data has been written into it
    #     """
    #     pass
    #
    # def to_array(self, order='C'):
    #     """Converts this to a numpy array
    #
    #     Parameters
    #     ----------
    #     order: str, optional
    #         The order of the returned array
    #
    #     Returns
    #     -------
    #     arr: np.ndarray[self.dtype](self.shape)
    #         Representation of this, as a numpy array
    #     """
    #     arr = np.empty(self.shape, dtype=self.dtype, order=order)
    #     return self.read(into=arr)

    def __getitem__(self, item):
        """Access to a selection in the view"""
        if not isinstance(item, tuple):
            item = (item,)
        iitem = tuple(ax(at) for ax, at in zip(self.axes, item))
        return self.aview[iitem]
