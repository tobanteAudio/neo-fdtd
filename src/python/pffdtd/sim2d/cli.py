# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

import click

from pffdtd.sim2d import process_outputs
from pffdtd.sim2d import report
from pffdtd.sim2d import run


@click.group(help='2D wave-equation.')
def sim2d():
    pass


sim2d.add_command(process_outputs.main)
sim2d.add_command(report.main)
sim2d.add_command(run.main)
