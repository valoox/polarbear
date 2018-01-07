import numpy as np
from ..labels import *


class TestPassthrough(object):
    def test_select(self):
        p = Passthrough()
        idx, sel = p.select([1, 2, 3])
        assert isinstance(idx, Passthrough)
        assert idx is not p
        assert sel == [1, 2, 3]

    def test_select_unit(self):
        p = Passthrough()
        idx, sel = p.select(12)
        assert idx is None
        assert sel == 12

    def test_at(self):
        p = Passthrough()
        q = p.at([1, 2, 3])
        assert isinstance(q, Passthrough)
        # 'At' ignores unit of selection
        r = p.at(12)
        assert isinstance(r, Passthrough)
