import concurrent.futures
import itertools
import os
import pathlib
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from .. import config
from .. import util
from .. import profile_functions

from . import conex

#
# helper functions
#

def get_good_fits(
  fits: pd.DataFrame,
  cut: config.cut_t,
) -> pd.Series:

  c = config.get_cut(cut)

  mask = True

  # fit has converged
  mask &= (0.99 < fits.status) & (fits.status < 1.01)
  # no nans or inf
  mask &= np.isfinite(fits).all(axis = 1)
  # xmax is within cut range
  mask &= (c.min_depth + 10 < fits.Xmax) & (fits.Xmax < c.max_depth - 10)
  # params, errors, status, chi2, and ndf are all positive
  mask &= (fits > 0).all(axis = 1)

  return mask

#
# data transformations
#

def normalize(
  data: Mapping[config.dataset_t, pd.DataFrame],
  columns: str | Sequence[str]
) -> Mapping[Literal[config.dataset_t, 'norm'], pd.DataFrame]:
  
  norm = pd.DataFrame({
    'mean': data['train'].loc[:, columns].mean(axis = 0),
    'std': data['train'].loc[:, columns].std(axis = 0),
  })

  for df in data.values():
    df.loc[:, columns] = (df.loc[:, columns] - norm['mean']) / norm['std']

  return {**data, 'normalization': norm}

#
# dataset loaders
#

def load_profiles(
  datadir: str | os.PathLike,
  cut: config.cut_t | str | None = None,
  datasets: Sequence[config.dataset_t] = config.datasets,
  particles: config.particle_t | Sequence[config.particle_t] = config.particles,
) -> dict[config.dataset_t, np.ndarray] | pd.DataFrame:
  
  profdir = pathlib.Path(datadir).resolve()/'profiles'

  # read depths and determine depth cuts
  depths = np.load(profdir/'depths.npy')
  columns = [f'Edep_{i}' for i in range(len(depths))]

  if cut is not None:
    xrange = config.get_cut(cut)
    xslice = util.get_range(depths, xrange.min_depth, xrange.max_depth)
    columns = columns[xslice]
    depths = depths[xslice]

  # load data
  ret = {d: util.hdf_load(path = profdir/d, key = particles, columns = columns) for d in util.strlist(datasets)}
    
  return {**ret, 'depths': depths}

def load_fits(
  datadir: str | os.PathLike,
  cut: config.cut_t | str,
  datasets: config.dataset_t | Sequence[config.dataset_t] = config.datasets,
  particles: config.particle_t | Sequence[config.dataset_t] = config.particles,
  columns: str | Sequence[str] | None = None,
  xfirst: bool = False,
  norm: str | Sequence[str] | None = None,
  drop_bad: bool = False,
) -> pd.DataFrame | dict[pd.DataFrame]:
  
  cutrange = config.get_cut(cut)
  path = f'{datadir}/fits/range-{cutrange.min_depth}-{cutrange.max_depth}'

  fits = {d: util.hdf_load(f'{path}/{d}', key = util.strlist(particles), columns = columns) for d in util.strlist(datasets)}

  if xfirst is True:
    xfdata = {d: util.hdf_load(f'{datadir}/xfirst/{d}', key = particles, columns = columns) for d in util.strlist(datasets)}
    fits = {d: fits[d].join(xfdata[d]) for d in util.strlist(datasets)}

  if drop_bad is True:
    for df in fits.values():
      df.drop(df.index[df.status < 0.99], inplace = True)

  if norm is not None:
    fits = normalize(fits, norm)

  return util.collapse(fits)

def load_xfirst(
  datadir: str | os.PathLike,
  datasets: config.dataset_t | Sequence[config.dataset_t] = config.datasets,
  particles: config.particle_t | Sequence[config.dataset_t] = config.particles,
  columns: str | Sequence[str] | None = None,
) -> pd.DataFrame | dict[pd.DataFrame]:

  xfdata = {d: util.hdf_load(f'{datadir}/xfirst/{d}', key = particles, columns = columns) for d in util.strlist(datasets)}
  return util.collapse(xfdata)

#
# data generation
#

def make_fits(
  datadir: str | os.PathLike,
  workers: int | None = None,
  verbose: bool = True,
):
  
  basedir = pathlib.Path(datadir).resolve()
  batches = workers or os.cpu_count() or 1
  function = profile_functions.usp

  util.echo(verbose, f'generating fits for cut configurations {[c.name for c in config.cuts]}')

  for c in config.cuts:
    util.echo(verbose, f'\nprocessing cut configuration {c.name}')

    for d, p in itertools.product(config.datasets, config.particles):
      
      profiles, depths = load_profiles(datadir = basedir, cut = c, datasets = d, particles = p).values()

      ys = [y.to_numpy() for y in util.split(profiles, batches = batches)]
      xs = [itertools.repeat(np.copy(depths), len(y)) for y in ys]
      fs = [function() for _ in ys]

      with concurrent.futures.ProcessPoolExecutor(batches) as exec:
        fits = exec.map(function.get_fits, fs, xs, ys)
        fits = pd.concat(fits, ignore_index = True, copy = False)
        fits.index.name = 'id'
        # update status by applying cuts
        fits.status = get_good_fits(fits, c).astype(fits.status.dtype)
        # save
        util.hdf_save(basedir/f'fits/range-{c.min_depth}-{c.max_depth}/{d}', fits, p, verbose)

def make_conex_split(
  datadir: str | os.PathLike,
  nshowers: dict[config.dataset_t, int],
  verbose: bool = True,
) -> dict[config.dataset_t, dict[config.particle_t, list[str]]]:
  
  basedir = pathlib.Path(datadir).resolve(strict = True)
  ret = {d: {} for d in config.datasets}

  util.echo(verbose, f'splitting conex files under directory {basedir}')
  
  for p in config.particles:
    files = set(map(str, basedir.glob(f'conex/{p}*/**/*.root')))
    for d in config.datasets:
      ret[d][p] = conex.parser(files, [], nshowers[d]).files
      files -= set(ret[d][p])

  util.json_save(ret, basedir/'conex.json', verbose)

  return ret

def make_datasets(
  datadir: str | os.PathLike,
  nshowers: Mapping[config.dataset_t, int],
  verbose: bool = True,
) -> None:
  
  basedir = pathlib.Path(datadir).resolve()
  branches = ('Xfirst', 'lgE')
  nsh = dict(nshowers)

  # split conex files
  cxpaths = make_conex_split(basedir, nsh)

  # profiles
  util.echo(verbose, 'generating profile datasets')
  for d, p in itertools.product(config.datasets, config.particles):
    data = conex.parser(files = cxpaths[d][p], branches = ['Edep'], nshowers = nsh[d], concat = True).get_table()
    util.hdf_save(basedir/'profiles'/d, data, p, verbose)

  # depths
  util.echo(verbose, '\nsaving slant depth profile')
  depths = conex.parser(files = cxpaths[d][p], branches = ['Xdep'], nshowers = 1, concat = True)[0]
  util.np_save(basedir/'profiles'/'depths', depths, verbose)

  # xfirst
  util.echo(verbose, '\ngenerating xfirst datasets')
  for d, p in itertools.product(config.datasets, config.particles):
    data = conex.parser(files = cxpaths[d][p], branches = branches, nshowers = nsh[d], concat = True).get_table()
    util.hdf_save(basedir/'xfirst'/d, data, p, verbose)
