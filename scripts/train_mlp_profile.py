import os
import re
from typing import Mapping

import click

import xfirst

@click.command()
@click.option('--datadir', type = click.Path(exists = True, dir_okay = True), required = True)
@click.option('--layers', type = str, required = True)
@click.option('--cut', type = str, required = True)
@click.option('--save', type = click.Path(), required = False, default = None)
@click.option('--nshowers', type = (str, click.IntRange(1, None)), required = True, multiple = True)
@click.option('--verbose/--no-verbose', default = True)
def main(
  datadir: str | os.PathLike,
  layers: str,
  cut: str,
  save: str | os.PathLike | None = None,
  nshowers: Mapping[xfirst.config.dataset_t, int] | int | None = None,
  verbose: bool = True,
):
  
  layers = re.sub(',|-|\.|\/', ':', layers).split(':')
  cut = xfirst.config.cut.get(cut)

  x = xfirst.data.load_depths('data', cut).index.to_list()
  y = 'Xfirst'

  xfirst.util.echo(f'processing profiles from cut {cut.name}')

  data = xfirst.data.load_profiles(
    datadir  = datadir,
    cut      = cut,
    nshowers = nshowers,
    norm     = True,
    xfirst   = True,
    verbose  = verbose,
  )

  model = xfirst.models.multilayer_perceptron_regressor(
    input = len(x),
    layers = layers,
    verbose = verbose,
  ).fit(data, x, y)

  if save is not None:
    model.eval(
      data = data,
      save = cut.path(f'{save}/mlp-fit-' + '-'.join(map(str, layers))),
      plot = True
    )

if __name__ == '__main__':
  main()
