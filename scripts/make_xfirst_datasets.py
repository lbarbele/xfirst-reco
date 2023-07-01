#!/usr/bin/python

import click
from xfirst.data import make_xfirst_datasets

@click.command()
@click.option('--data', type = click.Path(exists = True), required = True)
@click.option('--out', type = click.Path(), required = True)
@click.option('--max-train', type = click.IntRange(1, 4000000), required = False, default = None)
@click.option('--max-val', type = click.IntRange(1, 4000000), required = False, default = None)
@click.option('--max-test', type = click.IntRange(1, 4000000), required = False, default = None)
@click.option('--verbose/--no-verbose', default = True)
def main(**kwargs):
  make_xfirst_datasets(**kwargs)
  return 0

if __name__ == '__main__':
  main()
