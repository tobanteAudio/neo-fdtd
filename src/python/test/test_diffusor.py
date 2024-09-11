# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

import numpy as np

from pffdtd.diffusor.qrd import quadratic_residue_diffuser


def test_quadratic_residue_diffuser():
    w = quadratic_residue_diffuser(5, depth=None)
    assert np.allclose(w, [0, 1, 4, 4, 1])

    w = quadratic_residue_diffuser(7, depth=None)
    assert np.allclose(w, [0, 1, 4, 2, 2, 4, 1])

    w = quadratic_residue_diffuser(11, depth=None)
    assert np.allclose(w, [0, 1, 4, 9, 5, 3, 3, 5, 9, 4, 1])

    w = quadratic_residue_diffuser(5, depth=10)
    assert np.allclose(w, [0, 2.5, 10, 10, 2.5])
