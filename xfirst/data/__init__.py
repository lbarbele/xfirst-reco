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
  cut: config.cut | str,
) -> pd.Series:

  cut = config.cut.get(cut)

  mask = True

  # fit has converged
  mask &= (0.99 < fits.status) & (fits.status < 1.01)
  # no nans or inf
  mask &= np.isfinite(fits).all(axis = 1)
  # xmax is within cut range
  mask &= (cut.min_depth + 10 < fits.Xmax) & (fits.Xmax < cut.max_depth - 10)
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

def load_depths(
  datadir: str | os.PathLike,
  cut: config.cut | str | None = None,
):
  
  path = pathlib.Path(f'{datadir}/profiles/depths.npy').resolve(strict = True)
  x = np.load(path)
  x = pd.Series(x, index = [f'Edep_{i}' for i in range(x.shape[0])])

  if cut is not None:
    cut = config.cut.get(cut)
    out = x[(x < cut.min_depth) | (x > cut.max_depth)].index
    x.drop(out, inplace = True)

  return x

def load_profiles(
  datadir: str | os.PathLike,
  cut: config.cut | str | None = None,
  datasets: str | Sequence[config.dataset_t] = config.datasets,
  particles: config.particle_t | Sequence[config.particle_t] = config.particles,
  xfirst: bool = False,
  fits: str | Sequence[str] | None = None,
  drop_bad: bool | Mapping[config.dataset_t, bool] = False,
  nmax_rescale: bool = False,
  norm: bool = False,
  nshowers: int | Mapping[config.dataset_t, int] | None = None,
  verbose: bool = False,
) -> dict[config.dataset_t, np.ndarray] | pd.DataFrame:
  
  profdir = pathlib.Path(datadir).resolve()/'profiles'
  fitsdir = config.cut.get(cut).path(f'{datadir}/fits')
  xfirstdir = pathlib.Path(f"{datadir}/xfirst").resolve()
  datasets = util.strlist(datasets)
  particles = util.strlist(particles)
  drop_bad = dict.fromkeys(datasets, drop_bad) if isinstance(drop_bad, bool) else dict(drop_bad)
  nshowers = dict.fromkeys(datasets, nshowers) if isinstance(nshowers, int | None) else dict(nshowers)

  depths = load_depths(datadir, cut)

  profiles = {}

  util.echo(verbose, f'+ loading profiles of {datasets} datasets from {profdir}')
  util.echo(verbose and xfirst, f'+ loading xfirst data from {xfirstdir}')
  util.echo(verbose and fits, f'+ loading profile fits from {fitsdir}')
  util.echo(verbose and any(drop_bad.values()), f'+ dropping profiles with bad fits from {[d for d, v in drop_bad.items() if v is True]} datasets')

  for d in datasets:
    if (any(drop_bad.values()) is True) and (nshowers[d] is not None):
      profdata = []

      for p in particles:
        status = util.hdf_load(fitsdir/d, key = p, columns = 'status').status
        nrows = status[status > 0.99].index[nshowers[d]]

        data = util.hdf_load(profdir/d, p, nrows, depths.index)

        if fits is not None:
          fitsdata = util.hdf_load(fitsdir/d, p, nrows, fits).astype('float32')
          data = data.join(fitsdata)

        if xfirst is True:
          xfdata = util.hdf_load(xfirstdir/d, p, nrows)
          data = data.join(xfdata)

        if drop_bad[d] is True:
          badindices = data.index[status[:nrows] < 0.99]
          data.drop(badindices, inplace = True)
        else:
          data = data.join(status[:nrows].astype('float32'))

        profdata.append(data)

      profiles[d] = pd.concat(profdata, keys = particles, copy = False)

    else:
      profiles[d] = util.hdf_load(profdir/d, particles, nshowers[d], depths.index)

      if fits is not None:
        fitsdata = util.hdf_load(fitsdir/d, particles, nshowers[d], fits).astype('float32')
        profiles[d] = profiles[d].join(fitsdata)

      if xfirst is True:
        xfdata = util.hdf_load(xfirstdir/d, particles, nshowers[d])
        profiles[d] = profiles[d].join(xfdata)

      if drop_bad[d] is True:
        status = util.hdf_load(fitsdir/d, key = particles, columns = 'status').status
        profiles[d].drop(profiles[d].index[status < 0.99], inplace = True)
      elif any(drop_bad.values()) and (fits is None or 'status' not in fits):
        status = util.hdf_load(fitsdir/d, key = particles, nrows = nshowers[d], columns = 'status').astype('float32')
        profiles[d] = profiles[d].join(status)

  if nmax_rescale is True:
    for d in datasets:
      nmx = profiles[d][depths.index].max(axis = 1)
      profiles[d][depths.index] = profiles[d][depths.index].div(nmx, axis = 0)
      profiles[d].insert(profiles[d].shape[1], 'lgNmx', np.log(nmx))

  if norm is True:
    norm = list(depths.index)
    if nmax_rescale is True: norm += ['lgNmx']
    if fits is not None: norm += util.strlist(fits)
    util.echo(True, f'+ normalizing columns {norm}')
    profiles = normalize(profiles, norm)
    
  return {**profiles, 'depths': depths}

def load_fits(
  datadir: str | os.PathLike,
  cut: config.cut | str,
  datasets: config.dataset_t | Sequence[config.dataset_t] = config.datasets,
  particles: config.particle_t | Sequence[config.dataset_t] = config.particles,
  columns: str | Sequence[str] | None = None,
  xfirst: bool = False,
  norm: str | Sequence[str] | None = None,
  drop_bad: bool | Mapping[config.dataset_t, bool] = False,
  nshowers: int | Mapping[config.dataset_t, int] | None = None,
  verbose: bool = False,
) -> pd.DataFrame | dict[config.dataset_t, pd.DataFrame]:
  
  cut = config.cut.get(cut)
  datasets = util.strlist(datasets)
  particles = util.strlist(particles)
  columns = util.strlist(columns)
  drop_bad = dict.fromkeys(datasets, drop_bad) if isinstance(drop_bad, bool) else dict(drop_bad)
  nshowers = dict.fromkeys(datasets, nshowers) if isinstance(nshowers, int | None) else dict(nshowers)
  path = cut.path(f'{datadir}/fits')

  fits = {}

  util.echo(verbose, f'+ loading fits of {datasets} datasets from {path}')
  util.echo(verbose and xfirst, f'+ loading xfirst data from {pathlib.Path(f"{datadir}/xfirst").resolve()}')
  util.echo(verbose and any(drop_bad.values()), f'+ dropping bad fits from {[d for d, v in drop_bad.items() if v is True]} datasets')

  for d in datasets:

    if (any(drop_bad.values()) is True) and (nshowers[d] is not None):
      fitsdata = []

      for p in particles:
        status = util.hdf_load(f'{path}/{d}', key = p, columns = 'status').status
        nrows = status[status > 0.99].index[nshowers[d]]

        data = util.hdf_load(f'{path}/{d}', key = p, columns = columns, nrows = nrows)
        if xfirst is True:
          xfdata = util.hdf_load(f'{datadir}/xfirst/{d}', key = p, nrows = nrows)
          data = data.join(xfdata)

        if drop_bad[d] is True:
          badindices = data.index[status[:nrows] < 0.99]
          data.drop(badindices, inplace = True)
        else:
          data = data.join(status[:nrows])

        fitsdata.append(data)

      fits[d] = pd.concat(fitsdata, keys = particles, copy = False)
    else:
      fits[d] = util.hdf_load(f'{path}/{d}', key = particles, columns = columns, nrows = nshowers[d])

      if xfirst is True:
        xfdata = util.hdf_load(f'{datadir}/xfirst/{d}', key = particles, nrows = nshowers[d])
        fits[d] = fits[d].join(xfdata)

      if drop_bad[d] is True:
        status = util.hdf_load(f'{path}/{d}', key = particles, columns = 'status').status
        fits[d].drop(fits[d].index[status < 0.99], inplace = True)
      elif any(drop_bad.values()) and columns is not None and not 'status' in columns:
        status = util.hdf_load(f'{path}/{d}', key = particles, nrows = nshowers[d], columns = 'status')
        fits[d] = fits[d].join(status)

  if norm is not None:
    util.echo(True, f'+ normalizing columns {norm}')
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
