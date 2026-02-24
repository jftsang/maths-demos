import numpy as np
from differentials import discretise

def test_discretise():
    xs, *_ = discretise(0, 1, nx=101, periodic=True)
    assert xs.shape == (101, )
