# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

import click

from pffdtd.analysis.cli import analysis
from pffdtd.materials.cli import materials
from pffdtd.sim2d.cli import sim2d
from pffdtd.sim3d.cli import sim3d


@click.group()
@click.option('--verbose', is_flag=True, help='Print debug output.')
@click.pass_context
def main(ctx, verbose):
    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose


main.add_command(analysis)
main.add_command(materials)
main.add_command(sim2d)
main.add_command(sim3d)
