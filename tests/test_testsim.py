from __future__ import division

from numpy.testing import assert_array_equal

from funcsim.testsim import simulator


def test_signature():

    def stat(data):
        return 8

    assert_array_equal(simulator(stat, 3, 10, 10), [8] * 9)
    assert_array_equal(simulator(stat, 3, 10, 1e1), [8] * 9)

def test_sorting():
    i = [9]

    def stat(data):
        # Python 3: nonlocal i.
        i[0] -= 1
        return i[0]

    assert_array_equal(simulator(stat, 3, 10, 10), range(9))
