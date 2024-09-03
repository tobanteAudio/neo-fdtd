import click

from pffdtd.sim2d.engine import Engine2D


@click.command(name="run", help="Run simulation.")
@click.option('--sim_dir', type=click.Path(exists=True))
@click.option('--out', default="out.h5")
@click.option('--video', is_flag=True)
def main(sim_dir, out, video):
    engine = Engine2D(sim_dir=sim_dir, out=out, video=video)
    engine.run()
    engine.save_output()
