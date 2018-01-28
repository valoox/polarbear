import numpy as np
cimport numpy as np


cpdef int asorted(np.ndarray x):
    """Asserts whether 'x' is sorted
    
    Parameters
    ----------
    x: np.ndarray[Any](n)
        The array to check
    
    Returns
    -------
    sorted: bool
        Whether the array is sorted
    """
    cdef:
        int i = 0
        int n = x.shape[0]
    for i in range(n - 1):
        if x[i] > x[i+1]:
            return False
    return True

