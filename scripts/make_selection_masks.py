#!/usr/bin/python

import click

import xfirst

@click.command()
@click.option('--datadir', type = click.Path(exists = True), required = True)
@click.option('--verbose/--no-verbose', default = True)
def main(**kwargs):
  xfirst.data.make_selection_masks(**kwargs)
  return 0

if __name__ == '__main__':
  main()
