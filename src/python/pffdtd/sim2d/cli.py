import click

from pffdtd.sim2d.engine import Engine2D


@click.group(help="2D wave-equation.")
def sim2d():
    pass


@sim2d.command(help="Run simulation.")
@click.option('--sim_dir', type=click.Path(exists=True))
@click.option('--video', is_flag=True)
def run(sim_dir, video):
    engine = Engine2D(sim_dir=sim_dir, video=video)
    engine.run()
    engine.save_output()
