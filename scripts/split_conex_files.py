#!/usr/bin/python

import pathlib
import click

from xfirst.data import split_conex_files
from xfirst.util import json_dump
    
@click.command()
@click.option('--datadir', type = click.Path(exists = True, dir_okay = True), required = True)
@click.option('--nfiles', type = (str, click.IntRange(1, 4000)), required = True, multiple = True)
def main(**kwargs):
  d = split_conex_files(**kwargs)
  o = pathlib.Path(kwargs['datadir']).resolve()/'conex.json'
  json_dump(d, o)
  return 0

if __name__ == '__main__':
  main()
