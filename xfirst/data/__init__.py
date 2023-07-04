import collections
import concurrent.futures
import itertools
import os
import pathlib

import numpy as np
import pandas as pd

from .. import config
from .. import util
from ..profile_functions import usp

from . import conex

def load_fits(
  datadir: str,
  datasets: str | list[str] = config.datasets,
  particles: str | list[str] = config.particles,
  nshowers: dict[str, int] | None = None,
) -> pd.DataFrame:
  
  particles = util.as_list(particles)
  nshw = collections.defaultdict(lambda: None, {} if nshowers is None else nshowers)

  ret = []

  for dsname in util.as_list(datasets):
    data = [util.parquet_load(f'{datadir}/{dsname}/{prm}.parquet', nshw[dsname]) for prm in particles]
    data = pd.concat(data, keys = particles)
    ret.append(data)

  return ret if len(ret) > 1 else ret[0]

def load_profiles(
  datadir: str,
  datasets: str | list[str] = config.datasets,
  particles: str | list[str] = config.particles,
  nshowers: dict[str, int] | None = None,
  min_depth: float | None = None,
  max_depth: float | None = None,
  return_depths: bool = False,
  format: str = 'np',
) -> dict[str, np.ndarray] | pd.DataFrame:
  
  particles = util.as_list(particles)
  nshw = collections.defaultdict(lambda: None, {} if nshowers is None else nshowers)

  # read depths and determine depth cuts
  depths = util.npz_load(f'{datadir}/depths.npz')['depths']
  il, ir = util.get_range(depths, min_depth, max_depth)
  columns = [f'Edep_{i}' for i in range(len(depths))][il:ir]

  ret = []
  for dsname in util.as_list(datasets):
    # read data
    data = {prm: util.npz_load(f'{datadir}/{dsname}/{prm}.npz')[prm] for prm in particles}
    # apply cuts
    if nshw[dsname] is not None or il is not None or ir is not None:
      data = {prm: np.copy(v[:nshw[dsname], il:ir]) for prm, v in data.items()}
    # format data
    if format == 'np':
      ret.append(data if len(particles) > 1 else data[particles[0]])
    elif format == 'pd':
      ret.append(util.df_from_dict(data, particles, columns))
    else:
      raise RuntimeError(f'load_profiles: unsupported format {format}')

  # append depth to the return value
  if return_depths:
    ret.append(np.copy(depths[il:ir]))

  return ret if len(ret) > 1 else ret[0]

def load_xfirst(
  datadir: str,
  datasets: str | list[str] = config.datasets,
  particles: str | list[str] = config.particles,
  nshowers: dict[str, int] | None = None,
) -> pd.DataFrame:
  
  return load_fits(datadir, datasets, particles, nshowers)

def make_fits(
  datadir: str,
  out: str,
  max_train: int | None = None,
  max_val: int | None = None,
  max_test: int | None = None,
  min_depth: float | None = None,
  max_depth: float | None = None,
  workers: int | None = None,
  verbose: bool = True,
) -> None:
  
  workers = os.cpu_count() if workers is None else workers
  fcn = usp # class, not instance

  load_args = {
    'datadir': datadir,
    'nshowers': {'train': max_train, 'validation': max_val, 'test': max_test},
    'min_depth': min_depth,
    'max_depth': max_depth,
    'return_depths': True,
    'format': 'np',
  }

  for dsname in config.datasets:
    if verbose: print(f'parsing {dsname} dataset')

    for prm in config.particles:
      profiles, depths = load_profiles(datasets = dsname, particles = prm, **load_args)

      split_at = np.repeat(profiles.shape[0]//workers, workers - 1)
      split_at += (np.arange(workers - 1) < profiles.shape[0]%workers)
      split_at = split_at.cumsum()

      ys = np.split(profiles, split_at)
      fs = [fcn() for _ in ys]
      xs = [itertools.repeat(depths, len(y)) for y in ys]

      with concurrent.futures.ProcessPoolExecutor(workers) as exec:
        fits = exec.map(fcn.get_fits, fs, xs, ys)
        fits = np.concatenate(list(fits))
        fits = pd.DataFrame(fits, columns = fcn().columns, index = pd.Index(range(len(fits)), name = 'id'))

        file = f'{out}/{dsname}/{prm}.parquet'
        util.parquet_save(fits, file)

        if verbose: print(f'+ {prm} fits saved to {file}')

# *
# * functions to extract data from conex files 
# *

def make_conex_split(
  datadir: str | os.PathLike,
  nshowers: dict[config.dataset_t, int],
) -> dict[config.dataset_t, dict[config.particle_t, list[str]]]:
  
  dir = pathlib.Path(datadir).resolve(strict = True)
  ret = {d: {} for d in config.datasets}
  
  for p in config.particles:
    files = set(map(str, dir.glob(f'conex/{p}*/**/*.root')))
    for d in config.datasets:
      ret[d][p] = conex.parser(files, [], nshowers[d]).files
      files -= set(ret[d][p])

  return dict(ret)

def make_datasets(
  datadir: str | os.PathLike,
  nshowers: dict[config.dataset_t, int] | None = None,
  refresh: bool = False,
  verbose: bool = True,
) -> None:
  dir = pathlib.Path(datadir).resolve()
  nsh = collections.defaultdict(lambda: None, {} if nshowers is None else dict(nshowers))
  bcs = ('Xfirst', 'lgE')

  # split conex files, if hasn't been done already
  conexjson = dir/'conex.json'
  if not conexjson.exists() or refresh:
    util.echo(verbose, 'splitting conex files')
    conexpaths = make_conex_split(datadir, nshowers)
    util.json_dump(conexpaths, conexjson)
    util.echo(verbose, f'+ conex json saved to {conexjson}\n')
  else:
    conexpaths = util.json_load(conexjson)

  # profiles
  util.echo(verbose, 'generating profile datasets')

  for d, p in itertools.product(config.datasets, config.particles):
    data = conex.parser(files = conexpaths[d][p], branches = ['Edep'], nshowers = nsh[d], concat = True).get_table('np')
    util.np_save(dir/'profiles'/d/p, data, verbose)

  # depths
  util.echo(verbose, '\nsaving slant depth profile')
  depths = conex.parser(files = conexpaths[d][p], branches = ['Xdep'], nshowers = 1, concat = True)[0]
  util.np_save(dir/'profiles'/'depths', depths, verbose)

  # xfirst
  util.echo(verbose, '\ngenerating xfirst datasets')
  for d, p in itertools.product(config.datasets, config.particles):
    data = conex.parser(files = conexpaths[d][p], branches = bcs, nshowers = nsh[d], concat = True).get_table('pd')
    util.parquet_save(dir/'xfirst'/d/p, data, verbose)
