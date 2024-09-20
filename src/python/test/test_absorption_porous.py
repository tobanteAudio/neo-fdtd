# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch
import numpy as np
import pytest

from pffdtd.absorption.air import Air, air_density, sound_velocity
from pffdtd.absorption.porous import porous_absorber


@pytest.mark.parametrize('offset_zeros', [True, False])
@pytest.mark.parametrize(
    'thickness,flow_resistivity,expected',
    [
        (0.1, 3000, [0.0, 0.00914955, 0.0798299, 0.25952981, 0.53513877, 0.77518233, 0.90934681, 0.95555376, 0.97817786, 0.99485394, 0.99822579]),
        (0.1, 5000, [0.0, 0.0, 0.05994547, 0.25596251, 0.59086384, 0.86550292, 0.97620624, 0.97769105, 0.99157503, 0.99930661, 0.999125]),
        (0.1, 8000, [0.0, 0.0, 0.05577502, 0.27697129, 0.6626998, 0.93604009, 0.99884517, 0.98875809, 0.99722687, 0.99939574, 0.99926252]),
    ]
)
def test_absorption_porous(offset_zeros, thickness, flow_resistivity, expected):
    frequency = 1000*(2.0**np.arange(-6, 5))
    absorber = porous_absorber(thickness, flow_resistivity, frequency, offset_zeros=offset_zeros)
    assert np.allclose(absorber, expected)
    if offset_zeros:
        assert np.count_nonzero(absorber) == absorber.shape[0]
