from polarbear.select.oned import Int, Indices #Slice, Indices


def test_int():
    i = Int(2)
    assert i == 2


# def test_slice():
#     l = Slice(1)
#     assert l == slice(1, None, None)
#     u = Slice(stop=10)
#     assert u == slice(None, 10, None)
#     b = Slice(1, 10)
#     assert b == slice(1, 10)


def test_indices():
    a = Indices([1,2,3])
    assert a.size == 3
    assert a.shape == (3,)
