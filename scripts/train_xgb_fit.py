import os
from typing import Mapping, Sequence

import click

import xfirst

@click.command()
@click.option('--datadir', type = click.Path(exists = True, dir_okay = True), required = True)
@click.option('--save', type = click.Path(), required = False, default = None)
@click.option('--cuts', type = str, required = False, multiple = True, default = xfirst.config.cut_names)
@click.option('--nshowers', type = (str, click.IntRange(1, None)), required = False, multiple = True)
@click.option('--verbose/--no-verbose', default = True)
def main(
  datadir: str | os.PathLike,
  save: str | os.PathLike | None = None,
  cuts: Sequence[str] = xfirst.config.cut_names,
  nshowers: Mapping[xfirst.config.dataset_t, int] | int | None = None,
  verbose: bool = True,
):
  
  cutlist = [xfirst.config.get_cut(c) for c in cuts]
  features = xfirst.profile_functions.usp().parameter_names
  target = 'Xfirst'

  ret = {}

  xfirst.util.echo(verbose, f'training on cut configurations {[c.name for c in cutlist]}')

  for cut in cutlist:
    xfirst.util.echo(verbose, f'\nprocessing cut configuration {cut.name}')
    xfirst.util.echo(verbose, '+ loading data')

    # load data
    train, validation, test, normalization = xfirst.data.load_fits(
      datadir = datadir,
      cut = cut,
      columns = features,
      norm = features,
      nshowers = nshowers,
      drop = {'train': True, 'validation': True, 'test': False},
      xfirst = True,
    )

    xfirst.util.echo(verbose, '+ training the gradient boosting regressor')

    # fit a gradient boosting regressor and add to return dict
    ret[cut.name] = xfirst.models.gradient_boosting_regressor().fit(
      train = (train[features], train[target]),
      validation = (validation[features], validation[target]),
      verbose = verbose,
    )

    if save is not None:
      
      # save model
      outdir = ret[cut.name].save(f'{save}/xgb/range-{cut.min_depth}-{cut.max_depth}')

      # compute and save predictions
      xfirst.util.echo(verbose, f'+ computing predictions')
      
      results = ret[cut.name].test(test[features], test[target])
      results['lgE'] = test['lgE']
      results['good'] = test['good']

      for p in results.index.levels[0]:
        xfirst.util.parquet_save(outdir/'predictions'/p, results.loc[p])

      # create plots and save
      xfirst.viz.draw_predictions(results.loc[results['good']]).savefig(outdir/'predictions.pdf')

      # save normalization
      xfirst.util.parquet_save(outdir/'normalization', normalization)

      xfirst.util.echo(verbose, f'+ model saved to {outdir}')

  return ret if len(ret) > 1 else next(iter(ret.values()))

if __name__ == '__main__':
  main()
