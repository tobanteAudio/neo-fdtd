# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch
from dataclasses import dataclass

import click
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

from pffdtd.absorption.air import Air, air_density, sound_velocity, wave_number_in_air
from pffdtd.geometry.math import difference_over_sum


@dataclass
class PorousLayer:
    thickness: np.float64
    flow_resistivity: np.float64


def delaney_bazley_term_x(density, frequency, sigma):
    """Calculate Delaney & Bazley's term X
    """
    return (density * frequency) / sigma


def absorber_props(air: Air, flow_resistivity: float, frequency):
    """Characteristic absorber impedance and wave number
    """
    x = delaney_bazley_term_x(air.density, frequency, flow_resistivity)

    # Complex impedance
    z = air.impedance
    z_abs = z * ((1.0 + 0.0571 * x**-0.754) + 1j * (-0.087 * x**-0.732))

    # Complex wave number
    k = wave_number_in_air(air, frequency)
    k_abs = k * ((1.0 + 0.0978 * x**-0.7) + 1j*(-0.189 * x**-0.595))

    return (z_abs, k_abs)


def reflectivity_as_alpha(refl):
    """Convert reflectivity to absorption
    """
    alpha = 1.0 - np.abs(refl)**2.0
    alpha[alpha < 0.0] = 0.0
    return alpha


def porous_absorber(
    thickness: float,
    flow_resistivity: float,
    frequency: ArrayLike,
    angle=45.0,
    temperature=20.0,
    pressure=1.0,
    offset_zeros=False,
) -> np.ndarray:
    density = air_density(pressure, temperature)
    velocity = sound_velocity(temperature)
    air = Air(
        temperature=temperature,
        pressure=pressure,
        density=density,
        velocity=velocity,
        impedance=velocity*density,
        tau_over_c=(2.0*np.pi)/velocity
    )

    minus_i = complex(0.0, -1.0)
    angle_rad = np.deg2rad(angle)
    sin_phi = np.sin(angle_rad)
    cos_phi = np.cos(angle_rad)

    k_air = wave_number_in_air(air, frequency)

    # Characteristic absorber impedance and wave number
    (z_abs, wave_no_abs) = absorber_props(air, flow_resistivity, frequency)
    wave_no_abs_y = k_air * sin_phi
    wave_no_abs_x = np.sqrt(wave_no_abs**2 - wave_no_abs_y**2)

    # Angle of propagation within porous layer
    beta_porous = np.rad2deg(np.sin(np.abs(wave_no_abs_y / wave_no_abs)))

    # Intermediate term for porous impedance calculation
    porous_wave_no = wave_no_abs * thickness
    cot_porous_wave_no = np.cos(porous_wave_no) / np.sin(porous_wave_no)

    # Impedance at absorber surface
    z_abs_surface = minus_i * z_abs * \
        (wave_no_abs / wave_no_abs_x) * cot_porous_wave_no

    # Calculate absorption coefficient for porous absorber with no air gap
    abs_refl = difference_over_sum(
        (z_abs_surface / air.impedance) * cos_phi, 1.0)
    alpha = reflectivity_as_alpha(abs_refl)

    if offset_zeros:
        alpha[alpha == 0.0] = np.finfo(alpha.dtype).eps

    return alpha


@click.command(name='porous', help='Plot porous absorption properties.')
@click.option('--temperature', default=20, type=float)
@click.option('--thickness', default=0.1, type=float)
def main(temperature, thickness):
    angle = 45
    frequency = np.linspace(20, 20_000, 1024*16)
    absorber_3000 = porous_absorber(thickness, 3000, frequency, angle, temperature)
    absorber_5000 = porous_absorber(thickness, 5000, frequency, angle, temperature)
    absorber_8000 = porous_absorber(thickness, 8000, frequency, angle, temperature)

    thickness_cm = int(thickness*100)
    plt.semilogx(frequency, absorber_3000, label=f"{thickness_cm}cm 3000")
    plt.semilogx(frequency, absorber_5000, label=f"{thickness_cm}cm 5000")
    plt.semilogx(frequency, absorber_8000, label=f"{thickness_cm}cm 8000")
    plt.grid(which='both')
    plt.legend()
    plt.show()
