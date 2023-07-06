import concurrent.futures
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
# data transformations
#

def good_fits_mask(
  fits: pd.DataFrame,
  cut: config.cut_t | str,
) -> pd.Series:
  
  c = config.get_cut(cut)

  fits.replace([np.inf, -np.inf], np.nan, inplace = True)
  
  mask = True
  mask &= (fits.status > 0.99)
  mask &= ~(fits.isna().any(axis = 1))
  mask &= (c.min_depth+10 < fits.Xmax) & (fits.Xmax < c.max_depth-10)
  mask &= (fits > 0).all(axis = 1)
  
  return mask

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
  format: Literal['np', 'pd', 'mm'] = 'np',
) -> dict[config.dataset_t, np.ndarray] | pd.DataFrame:
  
  profdir = pathlib.Path(datadir).resolve()/'profiles'
  pts = [particles] if isinstance(particles, str) else particles
  dts = [datasets] if isinstance(datasets, str) else datasets
  nsh = {} if nshowers is None else dict(nshowers)
  cut = config.get_cut(cut)

  # read depths and determine depth cuts
  depths = np.load(profdir/'depths.npy')
  il, ir = util.get_range(depths, cut.min_depth, cut.max_depth)
  columns = [f'Edep_{i}' for i in range(len(depths))][il:ir]

  ret = []

  for d in dts:
    # read data
    data = {p: np.load(profdir/f'{d}/{p}.npy', mmap_mode = 'r') for p in pts}
    # apply cuts
    data = {p: v[:nsh.get(d), il:ir] for p, v in data.items()}
    # format data
    if format == 'mm':
      ret.append(data if len(pts) > 1 else data[pts[0]])
    elif format == 'pd':
      ret.append(util.df_from_dict(data, pts, columns))
    elif format == 'np':
      ret.append({p: np.copy(data[p]) for p in pts} if len(pts) > 1 else np.copy(data[pts[0]]))
    else:
      raise RuntimeError(f'load_profiles: unsupported format {format}')
    
  # append depths array to the return value
  if return_depths:
    ret.append(np.copy(depths[il:ir]))

  return ret if len(ret) > 1 else ret[0]

def load_tables(
  path: str | os.PathLike,
  datasets: config.dataset_t | Sequence[config.dataset_t] = config.datasets,
  particles: config.particle_t | Sequence[config.particle_t] = config.particles,
  nshowers: dict[config.dataset_t, int] | None = None,
  columns: str | Sequence[str] | None = None,
) -> pd.DataFrame | list[pd.DataFrame] :
  
  tabdir = pathlib.Path(path).resolve()
  pts = [particles] if isinstance(particles, str) else particles
  dts = [datasets] if isinstance(datasets, str) else datasets
  nsh = {} if nshowers is None else dict(nshowers)

  ret = []

  for d in dts:
    with concurrent.futures.ProcessPoolExecutor(1) as exec:
      if len(pts) == 1:
        data = exec.submit(util.parquet_load, tabdir/d/pts[0], nsh.get(d), columns)
        data = data.result()
      else:
        ps = [tabdir/d/p for p in pts]
        ns = itertools.repeat(nsh.get(d), len(ps))
        cs = itertools.repeat(columns, len(ps))
        data = exec.map(util.parquet_load, ps, ns, cs)
        data = pd.concat(data, keys = pts, copy = False, names = ['particle'])
        
      ret.append(data)

  return ret if len(ret) > 1 else ret[0]

def load_fits(
  datadir: str | os.PathLike,
  cut: config.cut_t | str | dict | Sequence[float | int],
  datasets: config.dataset_t | Sequence[config.dataset_t] = config.datasets,
  particles: config.particle_t | Sequence[config.particle_t] = config.particles,
  nshowers: dict[config.dataset_t, int] | None = None,
  columns: str | Sequence[str] | None = None,
  drop_bad: bool = False,
  xfirst: bool = False,
) -> pd.DataFrame | list[pd.DataFrame] :
  
  c = config.get_cut(cut)
  p = pathlib.Path(f'{datadir}/fits/range-{c.min_depth}-{c.max_depth}').resolve()

  if not p.exists():
    raise RuntimeError(f'load_fits: fits for cut range [{c.min_depth}, {c.max_depth}] do not exist')
  
  ret = load_tables(p, datasets, particles, nshowers, columns)

  if xfirst:
    xf = load_xfirst(datadir, datasets, particles, nshowers)
    ret = ret.join(xf) if len(datasets) == 1 else [a.join(b) for a, b in zip(ret, xf)]

  if drop_bad:
    for df in (ret if isinstance(ret, list) else [ret]):
      df.drop(df.index[~good_fits_mask(df, c)], inplace = True)

  return ret

def load_xfirst(
  datadir: str | os.PathLike,
  datasets: config.dataset_t | Sequence[config.dataset_t] = config.datasets,
  particles: config.particle_t | Sequence[config.particle_t] = config.particles,
  nshowers: dict[config.dataset_t, int] | None = None,
  columns: str | Sequence[str] | None = None,
) -> pd.DataFrame | list[pd.DataFrame] :
  
  return load_tables(f'{datadir}/xfirst', datasets, particles, nshowers, columns)

#
# generate fits 
# 

def make_fits(
  datadir: str | os.PathLike,
  nshowers: dict[config.dataset_t, int] | None = None,
  workers: int | None = None,
  verbose: bool = True,
):
  
  basedir = pathlib.Path(datadir).resolve()
  batches = os.cpu_count() if workers is None else workers
  fcn = profile_functions.usp

  util.echo(verbose, f'generating fits for cut configurations {[c.name for c in config.cuts]}')

  for c in config.cuts:
    util.echo(verbose, f'\nprocessing cut configuration {c.name}')

    for d, p in itertools.product(config.datasets, config.particles):
      
      profiles, depths = load_profiles(
        datasets = d,
        particles = p,
        cut = c,
        datadir = basedir,
        nshowers = nshowers,
        return_depths = True,
        format = 'np',
      )

      ys = util.split(profiles, batches = batches)
      fs = [fcn() for _ in ys]
      xs = [itertools.repeat(np.copy(depths), len(y)) for y in ys]

      with concurrent.futures.ProcessPoolExecutor(batches) as exec:
        fits = exec.map(fcn.get_fits, fs, xs, ys)
        fits = np.concatenate(list(fits))
        fits = pd.DataFrame(fits, columns = fcn().columns, index = pd.Index(range(len(fits)), name = 'id'))
        util.parquet_save(basedir/f'fits/range-{c.min_depth}-{c.max_depth}/{d}/{p}', fits, verbose)

#
# extract data from conex files 
#

def make_conex_split(
  datadir: str | os.PathLike,
  nshowers: dict[config.dataset_t, int],
) -> dict[config.dataset_t, dict[config.particle_t, list[str]]]:
  
  basedir = pathlib.Path(datadir).resolve(strict = True)
  ret = {d: {} for d in config.datasets}
  
  for p in config.particles:
    files = set(map(str, basedir.glob(f'conex/{p}*/**/*.root')))
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
  
  basedir = pathlib.Path(datadir).resolve()
  profdir = basedir/'profiles'
  xfstdir = basedir/'xfirst'
  branches = ('Xfirst', 'lgE')
  nsh = {} if nshowers is None else dict(nshowers)

  # split conex files, if hasn't been done already
  conexjson = basedir/'conex.json'
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
    data = conex.parser(files = conexpaths[d][p], branches = ['Edep'], nshowers = nsh.get(d), concat = True).get_table('np')
    util.np_save(profdir/d/p, data, verbose)

  # depths
  util.echo(verbose, '\nsaving slant depth profile')
  depths = conex.parser(files = conexpaths[d][p], branches = ['Xdep'], nshowers = 1, concat = True)[0]
  util.np_save(profdir/'depths', depths, verbose)

  # xfirst
  util.echo(verbose, '\ngenerating xfirst datasets')
  for d, p in itertools.product(config.datasets, config.particles):
    data = conex.parser(files = conexpaths[d][p], branches = branches, nshowers = nsh.get(d), concat = True).get_table('pd')
    util.parquet_save(xfstdir/d/p, data, verbose)
