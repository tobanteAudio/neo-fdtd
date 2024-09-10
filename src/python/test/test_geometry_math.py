import numpy as np

from pffdtd.geometry.math import find_third_vertex, transform_point


def test_find_third_vertex():
    v = find_third_vertex([0, 0, 0], [2, 0, 0])
    assert np.allclose(v, ([1, +1.73205081, 0], [1, -1.73205081, 0]))

    v = find_third_vertex([0, 0, 1.2], [3, 0, 1.2])
    assert np.allclose(v, ([1.5, +2.59807621, 1.2], [1.5, -2.59807621, 1.2]))


def test_transform_point():
    p = transform_point([0, 0, 0], [1, 1, 1], [0, 0, 0], [0, 0, 0])
    assert np.allclose(p, [0, 0, 0])

    p = transform_point([1, 2, 3], [1, 1, 1], [0, 0, 0], [0, 0, 0])
    assert np.allclose(p, [1, 2, 3])

    p = transform_point([1, 2, 3], [1, 1, 1], [0, 0, 0], [-1, -2, -4])
    assert np.allclose(p, [0, 0, -1])
