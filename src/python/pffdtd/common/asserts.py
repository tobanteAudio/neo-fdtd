# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2021 Brian Hamilton
"""Miscellaneous python/numpy assertions (mostly vestigial)
"""
import numpy as np


def assert_np_array_float(x):
    assert isinstance(x, np.ndarray)
    assert x.dtype in [np.dtype('float32'), np.dtype('float64')]


# python 'int' is arbitrary size
def assert_is_int(x):
    assert isinstance(x, (int, np.integer))
