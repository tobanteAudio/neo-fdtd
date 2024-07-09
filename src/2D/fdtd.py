import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def add_diffusor(width, max_depth, in_mask, X, Y):
    depths = np.array([0.0, 0.25, 1.0, 0.5, 0.5, 1.0, 0.25])
    depths = np.array([0.0,0.06,0.25,0.56,1.0,0.5,0.12,0.94,0.81,0.81,0.94,0.12,0.5,1,0.56,0.25,0.06])
    assert depths.shape[0]==17
    prime = depths.shape[0]
    n = int(6/width)
    for w in range(n):
        xs = 2+w*width
        xe = xs+width
        ys = 1
        ye = ys+depths[w % prime] * max_depth
        in_mask[(X >= xs) & (Y >= ys) & (X < xe) & (Y < ye)] = False

    return in_mask


def main():
    c = 343  # speed of sound m/s (20degC)
    fmax = 4_000  # Hz
    PPW = 7.7  # points per wavelength at fmax
    duration = 0.05  # seconds
    refl_coeff = 0.9  # reflection coefficient

    Bx, By = 10.0, 10.0  # box dims (with lower corner at origin)
    x_in, y_in = Bx*0.5, By*0.5  # source input position
    R_dome = By*0.5  # heigh of dome (to be centered on roof of box)

    draw = True
    add_dome = False
    apply_rigid = True
    apply_loss = True

    if apply_loss:
        assert apply_rigid

    if add_dome:
        Lx = Bx
        Ly = By+R_dome
    else:
        Lx = Bx
        Ly = By

    # calculate grid spacing, time step, sample rate
    dx = c/fmax/PPW  # grid spacing
    dt = np.sqrt(0.5)*dx/c
    SR = 1/dt

    Nx = int(np.ceil(Lx/dx)+2)  # number of points in x-dir
    Ny = int(np.ceil(Ly/dx)+2)  # number of points in y-dir
    Nt = int(np.ceil(duration/dt))  # number of time-steps to compute

    print(f'SR = {SR:.3f} Hz')
    print(f'Δx = {dx*100:.5f} cm / {dx*1000:.2f} mm')
    print(f'Nx = {int(Nx)} Ny = {int(Ny)} Nt = {int(Nt)}')

    # x and y sampling points
    xv = np.arange(0, Nx) * dx - 0.5 * dx
    yv = np.arange(0, Ny) * dx - 0.5 * dx
    X, Y = np.meshgrid(xv, yv, indexing='ij')

    # Mask for 'interior' points
    in_mask = np.zeros((Nx, Ny), dtype=bool)
    in_mask[(X >= 0) & (Y >= 0) & (X < Bx) & (Y < By)] = True

    if add_dome:
        in_mask[(X - 0.5 * Bx)**2 + (Y - By)**2 < R_dome**2] = True

    in_mask = add_diffusor(dx*4, 0.6, in_mask, X, Y)

    if apply_rigid:
        # Calculate number of interior neighbours (for interior points only)
        K_map = np.zeros((Nx, Ny), dtype=int)
        K_map[1:-1, 1:-1] += in_mask[2:, 1:-1]
        K_map[1:-1, 1:-1] += in_mask[:-2, 1:-1]
        K_map[1:-1, 1:-1] += in_mask[1:-1, 2:]
        K_map[1:-1, 1:-1] += in_mask[1:-1, :-2]
        K_map[~in_mask] = 0
        ib = np.where((K_map.flat > 0) & (K_map.flat < 4))[0]
        Kib = K_map.flat[ib]

    # Grid forcing points
    inx = int(np.round(x_in / dx + 0.5) + 1)
    iny = int(np.round(y_in / dx + 0.5) + 1)
    assert in_mask[inx, iny]

    if draw:
        draw_mask = np.nan*in_mask
        draw_mask[in_mask] = 1

    if apply_loss:
        # calculate specific admittance γ (g)
        assert abs(refl_coeff) <= 1.0
        g = (1-refl_coeff)/(1+refl_coeff)
        lf = 0.5*np.sqrt(0.5)*g  # a loss factor

    # Set up an excitation signal
    u_in = np.zeros(Nt, dtype=np.float64)
    u_in[0] = 1.0

    # Nh = int(np.ceil(5 * SR / fmax))
    # n = np.arange(Nh)
    # u_in[:Nh] = 0.5 - 0.5 * np.cos(2 * np.pi * n / Nh)
    # u_in[:Nh] *= np.sin(2 * np.pi * n / Nh)

    u0 = np.zeros((Nx, Ny), dtype=np.float64)
    u1 = np.zeros((Nx, Ny), dtype=np.float64)
    u2 = np.zeros((Nx, Ny), dtype=np.float64)

    print(f"{u2.shape=}")
    print(f"{in_mask.shape=}")
    print(f"{ib.shape=}")
    print(f"{Kib.shape=}")
    print(f"{Kib[0]}")

    frames = []

    for nt in range(Nt):
        # AIR
        left = u1[0:-2, 1:-1]
        right = u1[2:, 1:-1]
        top = u1[1:-1, 2:]
        bottom = u1[1:-1, 0:-2]
        last = u2[1:-1, 1:-1]
        mask = in_mask[1:-1, 1:-1]
        u0[1:-1, 1:-1] = mask * (0.5 * (left + right + top + bottom) - last)

        if apply_rigid:
            left = u1.flat[ib - 1]
            right = u1.flat[ib + 1]
            top = u1.flat[ib + Ny]
            bottom = u1.flat[ib - Ny]

            last1 = u1.flat[ib]
            last2 = u2.flat[ib]

            u0.flat[ib] = (2 - 0.5 * Kib) * last1 + 0.5 * (left + right + top + bottom) - last2

            if apply_loss:
                u0.flat[ib] = (u0.flat[ib] + lf * (4 - Kib) * u2.flat[ib]) / (1 + lf * (4 - Kib))

        u0[inx, iny] = u0[inx, iny] + u_in[nt]

        u2 = u1.copy()
        u1 = u0.copy()

        if nt < 2000:
            frames.append(u0.copy())

    print(f"last: u0={u0[inx, iny]} u1={u1[inx, iny]} u2={u2[inx, iny]}")

    fig = plt.figure()

    def draw_func(i):
        plt.cla()
        plt.clf()
        plt.xlim([np.min(xv), np.max(xv)])
        plt.ylim([np.min(yv), np.max(yv)])
        img = plt.imshow(
            (frames[i] * draw_mask).T,
            extent=(np.min(xv), np.max(xv), np.min(yv), np.max(yv)),
            cmap="bone",
            aspect="equal",
            origin="lower"
            # vmin=0.0,
            # vmax=0.05
        )
        # Add minorticks on the colorbar to make
        # it easy to read the values off the colorbar.
        color_bar = fig.colorbar(img, extend = 'both')

        color_bar.minorticks_on()

    ani = FuncAnimation(fig, draw_func, frames=len(frames), interval=16)
    plt.show()


if __name__ == "__main__":
    main()
