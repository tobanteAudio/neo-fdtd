# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch
import click
import numpy as np


def is_primitive_root(prime, g):
    if g in (0, 1):
        return False
    powers = set()
    for i in range(1, prime):
        powers.add(pow(g, i, prime))
    return len(powers) == prime - 1


def find_primitive_root(prime):
    for g in range(2, prime):
        if is_primitive_root(prime, g):
            return g
    return None


def primitive_root_diffuser(prime, g=None, depth=None):
    if g:
        assert is_primitive_root(prime, g)
    else:
        g = find_primitive_root(prime)
    n = np.mod(g**np.arange(0, prime-1), prime)
    if depth:
        n = n / np.max(n)
        n *= depth
    return n, g


@click.command(name='prd', help='Design PRD diffusors.')
def main():
    print(primitive_root_diffuser(17))
