"""This contains the formalisation of the INDEXING, as documented here:

https://docs.scipy.org/doc/numpy/user/basics.indexing.html

Although this can also be performed for arbitrary Labels mapping, and even be
combined symbolically outside any underlying array.
"""

from abc import ABCMeta
import numpy as np


class Selector(object, metaclass=ABCMeta):
    """This is the base class for all types of `Selector`s
    """
    # TODO
    

class Selector1D(object, metaclass=ABCMeta):
    """This represents a selector along a SINGLE axis.

    Although the basic indexing will be entirely covered by simple
    cartesian products of Selector1D (see `Basic` for details),
    these interact with advanced indexing (see `Advanced` for details) in
    non-obvious ways.

    Selector1D only cover the basic indexing techniques, leaving the advanced
    ones for `Advanced`.

    The only classes of Selector1D are therefore:
     - Atoms (i.e. a single element)
     - Slices
     - Broadcasts (None/`np.newaxis`)
    """
    # TODO !!!
