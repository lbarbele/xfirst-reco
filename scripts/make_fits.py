#!/usr/bin/python

import click
from xfirst.data import make_fits

@click.command()
@click.option('--datadir', type = click.Path(exists = True), required = True)
@click.option('--out', type = click.Path(), required = True)
@click.option('--max-train', type = click.IntRange(1, 4000000), required = False, default = None)
@click.option('--max-val', type = click.IntRange(1, 4000000), required = False, default = None)
@click.option('--max-test', type = click.IntRange(1, 4000000), required = False, default = None)
@click.option('--min-depth', type = click.FloatRange(0., 2000.), required = False, default = 0.)
@click.option('--max-depth', type = click.FloatRange(0., 2000.), required = False, default = 2000.)
@click.option('--workers', type = click.IntRange(1, 2048), default = None, required = False)
@click.option('--verbose/--no-verbose', default = True)
def main(**kwargs):
  make_fits(**kwargs)
  return 0

if __name__ == '__main__':
  main()
