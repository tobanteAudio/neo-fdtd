# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

import click

from pffdtd.sim2d import run
from pffdtd.sim2d import process_outputs


@click.group(help="2D wave-equation.")
def sim2d():
    pass


sim2d.add_command(process_outputs.main)
sim2d.add_command(run.main)
