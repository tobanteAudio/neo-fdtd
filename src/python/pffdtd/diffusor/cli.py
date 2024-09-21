# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

import click

from pffdtd.diffusor import measurement
from pffdtd.diffusor import prd
from pffdtd.diffusor import qrd


@click.group(help='Diffusor.')
def diffusor():
    pass


diffusor.add_command(measurement.main)
diffusor.add_command(prd.main)
diffusor.add_command(qrd.main)
