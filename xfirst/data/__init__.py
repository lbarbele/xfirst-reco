import collections
from concurrent.futures import ProcessPoolExecutor
import itertools
import os
import pathlib
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from .. import config
from .. import util
from .. import profile_functions

from . import conex

#
# dataset loaders
#

def load_profiles(
  datadir: str | os.PathLike,
  datasets: config.dataset_t | Sequence[config.dataset_t] = config.datasets,
  particles: config.particle_t | Sequence[config.particle_t] = config.particles,
  nshowers: dict[config.dataset_t, int] | None = None,
  cut: config.cut_t | str | dict | Sequence[float | int] | None = None,
  return_depths: bool = False,
  format: Literal['np', 'pd'] = 'np',
) -> dict[config.dataset_t, np.ndarray] | pd.DataFrame:
  
  dir = pathlib.Path(datadir).resolve()
  pts = [particles] if isinstance(particles, str) else particles
  dts = [datasets] if isinstance(datasets, str) else datasets
  nsh = collections.defaultdict(lambda: None, {} if nshowers is None else dict(nshowers))
  cut = config.cuts.get(cut)

  # read depths and determine depth cuts
  depths = np.load(dir/'profiles/depths.npy')
  il, ir = util.get_range(depths, cut.min_depth, cut.max_depth)
  columns = [f'Edep_{i}' for i in range(len(depths))][il:ir]

  ret = []

  for d in dts:
    # read data
    data = {p: np.load(dir/f'profiles/{d}/{p}.npy', mmap_mode = 'r') for p in pts}
    # apply cuts
    data = {prm: np.copy(v[:nsh[d], il:ir]) for prm, v in data.items()}
    # format data
    if format == 'np':
      ret.append(data if len(pts) > 1 else data[pts[0]])
    elif format == 'pd':
      ret.append(util.df_from_dict(data, pts, columns))
    else:
      raise RuntimeError(f'load_profiles: unsupported format {format}')
    
  # append depths array to the return value
  if return_depths:
    ret.append(np.copy(depths[il:ir]))

  return ret if len(ret) > 1 else ret[0]

def load_tables(
  dir: str | os.PathLike,
  datasets: config.dataset_t | Sequence[config.dataset_t] = config.datasets,
  particles: config.particle_t | Sequence[config.particle_t] = config.particles,
  nshowers: dict[config.dataset_t, int] | None = None,
) -> pd.DataFrame | list[pd.DataFrame] :
  
  dir = pathlib.Path(dir).resolve()
  pts = [particles] if isinstance(particles, str) else particles
  dts = [datasets] if isinstance(datasets, str) else datasets
  nsh = collections.defaultdict(lambda: None, {} if nshowers is None else dict(nshowers))

  ret = []

  for d in dts:
    with ProcessPoolExecutor(1) as exec:
      data = exec.map(util.parquet_load, [dir/d/p for p in pts], itertools.repeat(nsh[d], len(pts)))
      data = pd.concat(data, keys = pts, copy = False)
      ret.append(data)

  return ret if len(ret) > 1 else ret[0]

def load_fits(
  datadir: str | os.PathLike,
  cut: config.cut_t | str | dict | Sequence[float | int],
  datasets: config.dataset_t | Sequence[config.dataset_t] = config.datasets,
  particles: config.particle_t | Sequence[config.particle_t] = config.particles,
  nshowers: dict[config.dataset_t, int] | None = None,
) -> pd.DataFrame | list[pd.DataFrame] :
  
  c = config.cuts.get(cut)
  p = pathlib.Path(f'{datadir}/fits/range-{c.min_depth}-{c.max_depth}').resolve()

  if not p.exists():
    raise RuntimeError(f'load_fits: fits for cut range [{c.min_depth}, {c.max_depth}] do not exist')

  return load_tables(p, datasets, particles, nshowers)

def load_xfirst(
  datadir: str | os.PathLike,
  datasets: config.dataset_t | Sequence[config.dataset_t] = config.datasets,
  particles: config.particle_t | Sequence[config.particle_t] = config.particles,
  nshowers: dict[config.dataset_t, int] | None = None,
) -> pd.DataFrame | list[pd.DataFrame] :
  
  return load_tables(f'{datadir}/xfirst', datasets, particles, nshowers)

#
# generate fits 
# 

def make_fits(
  datadir: str | os.PathLike,
  nshowers: dict[config.dataset_t, int] | None = None,
  workers: int | None = None,
  verbose: bool = True,
):
  
  dir = pathlib.Path(datadir).resolve()
  wrk = os.cpu_count() if workers is None else workers
  nsh = collections.defaultdict(lambda: None, {} if nshowers is None else dict(nshowers))
  fcn = profile_functions.usp

  util.echo(verbose, f'generating fits for cut configurations {[c.name for c in config.cuts]}')

  for c in config.cuts:
    util.echo(verbose, f'\nprocessing cut configuration {c.name}')

    for d, p in itertools.product(config.datasets, config.particles):
      
      profiles, depths = load_profiles(
        datasets = d,
        particles = p,
        cut = c,
        datadir = dir,
        nshowers = nsh,
        return_depths = True,
        format = 'np',
      )

      ys = util.split(profiles, batches = wrk)
      fs = [fcn() for _ in ys]
      xs = [itertools.repeat(np.copy(depths), len(y)) for y in ys]

      with ProcessPoolExecutor(workers) as exec:
        fits = exec.map(fcn.get_fits, fs, xs, ys)
        fits = np.concatenate(list(fits))
        fits = pd.DataFrame(fits, columns = fcn().columns, index = pd.Index(range(len(fits)), name = 'id'))
        util.parquet_save(dir/f'fits/range-{c.min_depth}-{c.max_depth}/{d}/{p}', fits, verbose)

#
# extract data from conex files 
#

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
