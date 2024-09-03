# SPDX-License-Identifier: MIT
"""Miscellaneous python/numpy functions.  Not all used or useful.
"""
from typing import Any


import numpy as np
import numpy.linalg as npl
from numpy import cos, sin, pi


EPS = np.finfo(np.float64).eps


def rotmatrix_ax_ang(Rax: Any, Rang: float):
    assert isinstance(Rax, np.ndarray)
    assert Rax.shape == (3,)
    assert type(Rang) is float

    Rax = Rax/npl.norm(Rax)  # to be sure

    theta = Rang/180.0*pi  # in rad
    # see https://en.wikipedia.org/wiki/Rotation_matrix
    R = np.array([[cos(theta) + Rax[0]*Rax[0]*(1-cos(theta)), Rax[0]*Rax[1]*(1-cos(theta)) - Rax[2]*sin(theta), Rax[0]*Rax[2]*(1-cos(theta)) + Rax[1]*sin(theta)],
                  [Rax[1]*Rax[0]*(1-cos(theta)) + Rax[2]*sin(theta), cos(theta) + Rax[1]*Rax[1]*(
                      1-cos(theta)), Rax[1]*Rax[2]*(1-cos(theta)) - Rax[0]*sin(theta)],
                  [Rax[2]*Rax[0]*(1-cos(theta)) - Rax[1]*sin(theta), Rax[2]*Rax[1]*(1-cos(theta)) + Rax[0]*sin(theta), cos(theta) + Rax[2]*Rax[2]*(1-cos(theta))]])
    assert npl.norm(npl.inv(R)-R.T) < 1e-8
    return R


def rotate_xyz_deg(thx_d, thy_d, thz_d):
    # R applies Rz then Ry then Rx (opposite to wikipedia)
    # rotations about x,y,z axes, right hand rule

    thx = np.deg2rad(thx_d)
    thy = np.deg2rad(thy_d)
    thz = np.deg2rad(thz_d)

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(thx), -np.sin(thx)],
                   [0, np.sin(thx), np.cos(thx)]])

    Ry = np.array([[np.cos(thy), 0, np.sin(thy)],
                   [0, 1, 0],
                   [-np.sin(thy), 0, np.cos(thy)]])

    Rz = np.array([[np.cos(thz), -np.sin(thz), 0],
                   [np.sin(thz), np.cos(thz), 0],
                   [0, 0, 1]])

    R = Rx @ Ry @ Rz

    return R, Rx, Ry, Rz


def rotate_az_el_deg(az_d, el_d):
    # R applies Rel then Raz
    # Rel is rotation about negative y-axis, right hand rule
    # Raz is rotation z-axis, right hand rule
    # this uses matlab convention
    # note, this isn't the most general rotation
    _, _, Ry, Rz = rotate_xyz_deg(0, -el_d, az_d)
    Rel = Ry
    Raz = Rz
    R = Raz @ Rel

    return R, Raz, Rel


def dot2(v):
    return dotv(v, v)


def dotv(v1, v2):
    return np.sum(v1*v2, axis=-1)


def vecnorm(v1):
    return np.sqrt(dot2(v1))


def normalise(v1, eps=EPS):
    return (v1.T/(vecnorm(v1)+eps)).T


def ind2sub3d(ii, Nx, Ny, Nz):
    iz = ii % Nz
    iy = (ii - iz)//Nz % Ny
    ix = ((ii - iz)//Nz-iy)//Ny
    return ix, iy, iz


def rel_diff(x0, x1):
    # relative difference at machine epsilon level
    return (x0-x1)/(2.0**np.floor(np.log2(x0)))


def iceil(x):
    return np.int_(np.ceil(x))


def iround(x):
    return np.int_(np.round(x))


def to_ixy(x, y, Nx, Ny, order="row"):
    if order == "row":
        return x*Ny+y
    return y*Nx+x


def point_on_circle(center, radius: float, angle: float):
    """
    Calculate the coordinates of a point on a circle arc.

    Parameters:
    center (tuple): (x, y) coordinates of the center of the circle.
    radius: Radius of the circle.
    angle: Angle in radians.

    Returns:
    tuple: (p_x, p_y) coordinates of the point on the circle arc.
    """
    x, y = center
    p_x = x + radius * np.cos(angle)
    p_y = y + radius * np.sin(angle)
    return (p_x, p_y)
