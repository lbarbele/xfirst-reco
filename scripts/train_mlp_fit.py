import os
import re
from typing import Mapping, Sequence

import click

import xfirst

@click.command()
@click.option('--datadir', type = click.Path(exists = True, dir_okay = True), required = True)
@click.option('--layers', type = str, required = True)
@click.option('--save', type = click.Path(), required = False, default = None)
@click.option('--cuts', type = str, required = False, default = ','.join(xfirst.config.cut.names()))
@click.option('--nshowers', type = (str, click.IntRange(1, None)), required = True, multiple = True)
@click.option('--verbose/--no-verbose', default = True)
def main(
  datadir: str | os.PathLike,
  layers: str,
  save: str | os.PathLike | None = None,
  cuts: str | Sequence[str] = xfirst.config.cut.names(),
  nshowers: Mapping[xfirst.config.dataset_t, int] | int | None = None,
  verbose: bool = True,
):
  
  layers = [int(i) for i in re.sub(',|-|\.|\/', ':', layers).split(':')]
  cuts = [xfirst.config.cut.get(c) for c in xfirst.util.strlist(cuts)]

  x = xfirst.profile_functions.usp().parameter_names
  y = 'Xfirst'
  
  xfirst.util.echo(verbose, f'training on cut configurations {[c.name for c in cuts]}')

  for cut in cuts:
    xfirst.util.echo(verbose, f'\nprocessing cut configuration {cut.name}')

    savepath = None if save is None else cut.path(f'{save}/mlp-fit-' + '-'.join(map(str, layers)))
    backuppath = None if save is None else savepath/'backup'

    # load normalized data. drop bad fits on train and validation sets
    data = xfirst.data.load_fits(
      datadir  = datadir,
      cut      = cut,
      columns  = x,
      xfirst   = True,
      norm     = x,
      drop_bad = {'train': True, 'validation': True, 'test': False},
      shuffle  = {'train': True, 'validation': True, 'test': False},
      nshowers = nshowers,
      verbose  = verbose,
    )

    # fit the multilayer perceptron
    model = xfirst.models.multilayer_perceptron_regressor(
      input = len(x),
      layers = layers,
      backup_dir = backuppath,
      verbose = verbose,
    ).fit(data, x, y)

    # save model
    if save is not None:
      model.eval(
        data = data,
        save = savepath,
        plot = True
      )

if __name__ == '__main__':
  main()
