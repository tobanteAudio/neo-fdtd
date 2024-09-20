# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

import click

from pffdtd.diffusor.measurement import main as measurement
from pffdtd.diffusor.prd import main as prd
from pffdtd.diffusor.qrd import main as qrd


@click.group(help='Diffusor.')
def diffusor():
    pass


diffusor.add_command(measurement)
diffusor.add_command(prd)
diffusor.add_command(qrd)
