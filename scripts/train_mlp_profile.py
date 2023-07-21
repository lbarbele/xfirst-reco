import tensorflow

import os
import re
from typing import Mapping

import click

import xfirst

@click.command()
@click.option('--datadir', type = click.Path(exists = True, dir_okay = True), required = True)
@click.option('--layers', type = str, required = True)
@click.option('--cut', type = str, required = True)
@click.option('--batch-size', type = click.IntRange(8, 2048), required = False, default = 128)
@click.option('--save', type = click.Path(), required = False, default = None)
@click.option('--nshowers', type = (str, click.IntRange(1, None)), required = True, multiple = True)
@click.option('--fits/--no-fits', default = False)
@click.option('--verbose/--no-verbose', default = True)
def main(
  datadir: str | os.PathLike,
  layers: str,
  cut: str,
  batch_size: int = 128,
  save: str | os.PathLike | None = None,
  nshowers: Mapping[xfirst.config.dataset_t, int] | int | None = None,
  fits: bool = False,
  verbose: bool = True,
):
  
  layers = [int(i) for i in re.sub(',|-|\.|\/', ':', layers).split(':')]
  cut = xfirst.config.cut.get(cut)

  x_prof = xfirst.data.load_depths(datadir, cut).index.to_list()
  x_fits = xfirst.profile_functions.usp().parameter_names if (fits is True) else []
  x = x_prof + x_fits
  y = 'Xfirst'

  xfirst.util.echo(verbose, f'processing profiles from cut {cut.name}')

  data = xfirst.data.load_profiles(
    datadir  = datadir,
    cut      = cut,
    nshowers = nshowers,
    norm     = True,
    xfirst   = True,
    fits     = x_fits if (fits is True) else None,
    drop_bad = {'train': True, 'validation': True, 'test': False},
    verbose  = verbose,
  )

  model = xfirst.models.multilayer_perceptron_regressor(
    input = len(x),
    layers = layers,
    verbose = verbose,
    batch_size = batch_size,
  ).fit(data, x, y)

  if save is not None:
    model.eval(
      data = data,
      save = cut.path(f'{save}/mlp-' + ('both-' if fits else 'profile-') + '-'.join(map(str, layers))),
      plot = True
    )

if __name__ == '__main__':
  main()
