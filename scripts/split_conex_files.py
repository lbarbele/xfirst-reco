#!/usr/bin/python

import click
from xfirst.data import split_conex_files
    
@click.command()
@click.option('--ds', type = (str, click.IntRange(1, 4000)), required = True, multiple = True)
@click.option('--datadir', type = click.Path(exists = True, dir_okay = True), required = True)
@click.option('--out', type = click.Path(exists = False), required = True)
@click.option('--verbose/--no-verbose', default = True)
def main(**kwargs):
  split_conex_files(**kwargs)
  return 0

if __name__ == '__main__':
  main()
