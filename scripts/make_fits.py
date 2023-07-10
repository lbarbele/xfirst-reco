#!/usr/bin/python

import click
from xfirst.data import make_fits

@click.command()
@click.option('--datadir', type = click.Path(exists = True), required = True)
@click.option('--workers', type = click.IntRange(1, 2048), default = None, required = False)
@click.option('--verbose/--no-verbose', default = True)
def main(**kwargs):
  make_fits(**kwargs)
  return 0

if __name__ == '__main__':
  main()
