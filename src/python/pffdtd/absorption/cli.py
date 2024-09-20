# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch
import click

from pffdtd.absorption.admittance import main as admittance


@click.group(help='Absoroption.')
def absorption():
    pass


absorption.add_command(admittance)
