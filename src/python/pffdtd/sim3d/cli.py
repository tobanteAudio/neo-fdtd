# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

import click

from pffdtd.sim3d import process_outputs


@click.group(help="3D wave-equation.")
def sim3d():
    pass


sim3d.add_command(process_outputs.main)
