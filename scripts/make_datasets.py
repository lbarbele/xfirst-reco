#!/usr/bin/python

import click

import xfirst.data as data

@click.command()
@click.option('--datadir', type = click.Path(exists = True, dir_okay = True), required = True)
@click.option('--nshowers', type = (str, click.IntRange(1, 4000000)), required = True, multiple = True)
@click.option('--verbose/--no-verbose', default = True)
def main(**kwargs):
  data.make_datasets(**kwargs)
  return 0

if __name__ == '__main__':
  main()
