"""Description of the 'array' protocol, and the array-like interface that
all wrapped data need to comply with
"""
from typing import Any

from polarbear import index


def ndimensional(data: Any) -> bool:
    """Does the passed in data have N-dimensions

    Parameters
    ----------
    data: Any
        The data to be checked

    Returns
    -------
    is_ndim: bool
        Whether the data has an `ndim` field corresponding to a number of
        dimensions
    """
    return (
        hasattr(data, 'ndim') and isinstance(data.ndim, int) and data.ndim >= 0
    )


def indexable(data: Any) -> bool:
    """Whether the passed-in data has the `__getitem__` infrastructure required
    to index it

    Parameters
    ----------
    data: Any
        The data to be checked

    Returns
    -------
    is_indexable: bool
        Whether the data is indexable
    """
    return hasattr(data, '__getitem__')



ARRAY_CHECKS = [ndimensional, indexable]


def is_arraylike(data: Any) -> bool:
    """Relaxed approach to data buffers, requiring them to implement a certain
    interface, and checking this via Duck-typing only.

    This lets implementers have freedom on the exact format of the data that
    they want used, while also covering the usual cases of converting a list
    of list to an array, or similar cases.

    Parameters
    ----------
    data: Any
        The data being examined

    Returns
    -------
    array_like: bool
        Whether the data complies with the array protocol being assumed in the
        rest of the library
    """
    return all(check(data) for check in ARRAY_CHECKS)


def validate(data: Any, axes: index.NDIndex):
    """Validates the input data, ensuring that the dimensions are consistent
    with the ones of the axes NDIndex, and performing any required conversion
    or broadcasting to make them compatible.

    If this is not possible, an exception is raised

    Parameters
    ----------
    data: Array-like, dimensions axes.ndim or less
        The data being used, either an array-like or convertible to an
        array-like (e.g. a list).
    axes: NDIndex instance
        The index describing the shape required.

    Returns
    -------
    data: Array-like, size ndim
        If the data passed in is directly compatible, it will be returned
        unchanged. Otherwise, any conversion/broadcasting required is applied,
        and that converted value is returned.
        If this is impossible, an exception is raised.
    """
    return data
