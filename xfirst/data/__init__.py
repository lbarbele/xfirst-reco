import os
import pathlib
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from .. import config
from .. import util

from . import conex

# *
# * helper functions
# *

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

# *
# * data transformations
# *

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

# *
# * dataset loaders
# *

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
    for d in util.strlist(datasets):
      bad = util.hdf_load(f'{path}/{d}', key = util.strlist(particles), columns = 'status')
      bad = bad.status < 0.99
      bad = bad.index[bad]
      fits[d].drop(bad, inplace = True)

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
