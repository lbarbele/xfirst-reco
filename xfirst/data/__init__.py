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
  profiles = {}
  for d in util.strlist(datasets):
    profiles[d] = util.hdf_load(path = profdir/d, key = particles, columns = columns)
    
  return {**profiles, 'depths': depths}

def load_fits(
  datadir: str | os.PathLike,
  cut: config.cut_t | str,
  datasets: config.dataset_t | Sequence[config.dataset_t] = config.datasets,
  particles: config.particle_t | Sequence[config.dataset_t] = config.particles,
  columns: str | Sequence[str] | None = None,
  xfirst: bool = False,
  norm: str | Sequence[str] | None = None,
  drop_bad: bool | Mapping[config.dataset_t, bool] = False,
  nshowers: int | Mapping[config.dataset_t, int] | None = None, 
) -> pd.DataFrame | dict[config.dataset_t, pd.DataFrame]:
  
  cut = config.get_cut(cut)
  datasets = util.strlist(datasets)
  particles = util.strlist(particles)
  columns = util.strlist(columns)
  drop_bad = dict.fromkeys(datasets, drop_bad) if isinstance(drop_bad, bool) else {d: drop_bad[d] for d in datasets}
  nshowers = dict.fromkeys(datasets, nshowers) if isinstance(nshowers, int | None) else {d: nshowers[d] for d in datasets}
  path = f'{datadir}/fits/range-{cut.min_depth}-{cut.max_depth}'

  fits = {}

  for d in datasets:

    if drop_bad[d] is True and nshowers[d] is not None:
      fitsdata = []

      for p in particles:
        status = util.hdf_load(f'{path}/{d}', key = p, columns = 'status').status
        nrows = status[status > 0.99].index[nshowers[d]]
        data = util.hdf_load(f'{path}/{d}', key = p, columns = columns, nrows = nrows)
        badindices = data.index[status[:nrows] < 0.99]
        data.drop(badindices, inplace = True)
        fitsdata.append(data)

      fits[d] = pd.concat(fitsdata, keys = particles, copy = False)
    else:
      fits[d] = util.hdf_load(f'{path}/{d}', key = particles, columns = columns, nrows = nshowers[d])
      if xfirst is True:
        xfdata = util.hdf_load(f'{datadir}/xfirst/{d}', key = particles, nrows = nshowers[d])
        fits[d] = fits[d].join(xfdata)

      if drop_bad[d] is True:
        status = util.hdf_load(f'{path}/{d}', key = particles, columns = 'status').status
        fits[d].drop(fits.index[status < 0.99], inplace = True)
      elif any(drop_bad.values()) and columns is not None and not 'status' in columns:
        status = util.hdf_load(f'{path}/{d}', key = particles, columns = 'status')
        fits[d] = fits[d].join(status)

  if norm is not None:
    fits = normalize(fits, norm)

  return util.collapse(fits)

def load_xfirst(
  datadir: str | os.PathLike,
  datasets: config.dataset_t | Sequence[config.dataset_t] = config.datasets,
  particles: config.particle_t | Sequence[config.dataset_t] = config.particles,
  columns: str | Sequence[str] | None = None,
) -> pd.DataFrame | dict[pd.DataFrame]:

  xfdata = {}
  for d in util.strlist(datasets):
    xfdata[d] = util.hdf_load(f'{datadir}/xfirst/{d}', key = particles, columns = columns)

  return util.collapse(xfdata)
