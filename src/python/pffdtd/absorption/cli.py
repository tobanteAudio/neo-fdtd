# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch
import click

from pffdtd.absorption.admittance import main as admittance
from pffdtd.absorption.porous import main as porous


@click.group(help='Absorption.')
def absorption():
    pass


absorption.add_command(admittance)
absorption.add_command(porous)
