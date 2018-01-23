"""The `DataSet` is the main container for the different data.

Conceptually, it is simply a sequence of pair `View` and `Index`.

Note that these `DataSet`s are therefore from a single uniform datatype; for
multiple types of data, there is also the `DataFrame`, which combines
different dataset with the same indices along a different index.
"""
from abc import ABCMeta, abstractmethod
import numpy as np

from .label import Labels


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


class DataSet(object):
    """Simple implementation of the dataset

    Parameters
    ----------
    data: View instance (or convertible to it)
        The data being held in the dataset
    labels: Labels instance (or convertible to it), if anything
        The labels, if any; if none, or if it doesn't cover the entire shape
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
    __slots__ = ('data', 'labels')

    def __init__(self, data, labels=None, dtype: np.dtype=None,
                 order: str='K', copy: bool=False):
        """Constructor"""
        self.data = view(data, dtype=dtype, order=order, copy=copy)
        self.labels = Labels(self.data.shape, *labels)

    @property
    def shape(self) -> tuple:
        """The shape of the data

        Returns
        -------
        shape: tuple[int]
            The shape of the data
        """
        # This is the shape of the LABELS, which might be different
        # from the shape of the DATA, e.g. if some is broadcasted.
        return self.labels.shape

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
        idx, item = self.labels.select(item)
        data = self.data[item]
        if idx is None:
            return data
        return DataSet(data, labels=idx)

    def __setitem__(self, item, value):
        """Sets the value provided

        Parameters
        ----------
        item: Selector[labels]
            The selector on the index of this dataset
        value: array-like[dtype]
            The values to set in this position
        """
        _, item = self.labels.select(item)
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
            idx = self.ds.labels.at(item)
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
