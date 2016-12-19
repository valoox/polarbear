import numpy as np


class Int(int):
    """Simply an integer"""
    def __new__(cls, i):
        return int.__new__(cls, i)


# class Slice(slice):
#     """Simply a slice"""
#     def __new__(cls, *args, **kwargs):
#         return slice.__new__(cls, *args, **kwargs)


class Indices(np.ndarray):
    """A one-dimensional array of integer indices"""
    def __new__(cls, indices):
        arr = np.array(indices, dtype=int)
        return np.ndarray.__new__(
            data=arr.data,
            shape=(arr.size,),
            dtype=int
        )
