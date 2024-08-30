import click

from pffdtd.analysis.response import main as response
from pffdtd.analysis.room_modes import main as room_modes
from pffdtd.analysis.t60 import main as t60
from pffdtd.analysis.waterfall import main as waterfall


@click.group(help="Analysis.")
def analysis():
    pass


analysis.add_command(response)
analysis.add_command(room_modes)
analysis.add_command(t60)
analysis.add_command(waterfall)
