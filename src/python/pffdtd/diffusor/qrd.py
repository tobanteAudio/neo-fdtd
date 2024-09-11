# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

import numpy as np


def quadratic_residue_diffuser(prime, depth=None):
    n = np.mod(np.arange(0, prime, 1)**2, prime)
    if depth:
        n = n / np.max(n)
        return n*depth
    return n


def main():
    n = 13
    c = 343
    well_width = 0.0254*2
    design_frequency = 400

    design_wavelength = c/design_frequency
    design_depth = design_wavelength/2
    plate_frequency = design_frequency*n
    fmin = design_frequency/2
    fmax = c/(well_width*2)
    seat_distance = design_wavelength*3

    w = quadratic_residue_diffuser(n, design_depth)

    print(f"prime     = {n}")
    print(f"width     = {well_width*100:.2f} cm")
    print(f"depth     = {design_depth*100:.2f} cm")
    print(f"seat      = {seat_distance*100:.2f} cm")
    print("")

    print(f"scatter   = {fmin:.2f} Hz")
    print(f"diffuse   = {design_frequency:.2f} Hz")
    print(f"HF cutoff = {fmax:.2f} Hz")
    print(f"plate     = {plate_frequency:.2f} Hz")
    print("")

    print(f"wells     = {np.round(w*100,2)} cm")
    print(f"max depth = {np.max(w)*100:.2f} cm")


if __name__ == "__main__":
    main()
