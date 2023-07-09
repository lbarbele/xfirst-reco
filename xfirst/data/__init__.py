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

def get_nshowers(input) -> dict[config.dataset_t, int | None]:

  ret = dict.fromkeys(config.datasets, None)

  if isinstance(input, int):
    if input < 0:
      raise ValueError(f'get_nshowers: invalid nshowers value {input}')
    elif input > 0:
      ret = dict.fromkeys(config.datasets, input)
  elif isinstance(input, Sequence):
    ret.update(dict(input))
  elif isinstance(input, Mapping):
    ret.update(input)
  elif input is not None:
    raise ValueError(f'get_nshowers: invalid input type {type(input)}')
  
  for k in ret.keys():
    if not k in config.datasets:
      raise ValueError(f'get_nshowers: invalid key {k}')

  return ret

#
# data transformations
#

def normalize(datasets, columns: Sequence[str | int]):
  
  if isinstance(datasets, list):
    a = datasets[0]
    bs = datasets
  elif isinstance(datasets, dict):
    a = datasets['train']
    bs = [v for v in datasets.values()]
  else:
    a = datasets
    bs = [datasets]

  if isinstance(a, pd.DataFrame):
    means = a.loc[:, columns].mean(axis = 0)
    stds = a.loc[:, columns].std(axis = 0)

    for b in bs:
      b.loc[:, columns] = (b.loc[:, columns] - means) / stds

  elif isinstance(a, np.ndarray):
    means = a[:, columns].mean(axis = 0)
    stds = a[:, columns].std(axis = 0)

    for b in bs:
      b[:, columns] = (b[:, columns] - means) / stds
  
  else:
    raise ValueError('normalize: unsupported input')
  
  stats = pd.DataFrame({'mean': means, 'std': stds})

  return [*bs, stats]

#
# dataset loaders
#

def load_profiles(
  datadir: str | os.PathLike,
  datasets: config.dataset_t | Sequence[config.dataset_t] = config.datasets,
  particles: config.particle_t | Sequence[config.particle_t] = config.particles,
  nshowers: Mapping[config.dataset_t, int] | None = None,
  cut: config.cut_t | str | dict | Sequence[float | int] | None = None,
  return_depths: bool = False,
  format: Literal['np', 'pd', 'mm'] = 'np',
) -> dict[config.dataset_t, np.ndarray] | pd.DataFrame:
  
  profdir = pathlib.Path(datadir).resolve()/'profiles'
  pts = [particles] if isinstance(particles, str) else particles
  dts = [datasets] if isinstance(datasets, str) else datasets
  nsh = get_nshowers(nshowers)
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
  nshowers: Mapping[config.dataset_t, int] | int | None = None,
  columns: str | Sequence[str] | None = None,
) -> list[pd.DataFrame] :
  
  tabdir = pathlib.Path(path).resolve()
  pts = [particles] if isinstance(particles, str) else particles
  dts = [datasets] if isinstance(datasets, str) else datasets
  nsh = get_nshowers(nshowers)

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

  return ret

def load_fits(
  datadir: str | os.PathLike,
  cut: config.cut_t | str,
  datasets: config.dataset_t | Sequence[config.dataset_t] = config.datasets,
  particles: config.particle_t | Sequence[config.particle_t] = config.particles,
  nshowers: Mapping[config.dataset_t, int] | int | None = None,
  columns: str | Sequence[str] | None = None,
  drop: Mapping[config.dataset_t, bool] | bool = False,
  xfirst: bool = False,
  norm: Sequence[str] | None = None,
) -> pd.DataFrame | list[pd.DataFrame] :
  
  c = config.get_cut(cut)
  path = pathlib.Path(f'{datadir}/fits/range-{c.min_depth}-{c.max_depth}').resolve()

  if not path.exists():
    raise RuntimeError(f'load_fits: fits for cut {c.name} [{c.min_depth}, {c.max_depth}] do not exist')
  
  if drop is not False:
    masks = load_masks(datadir, cut, datasets, particles, nshowers, drop)
    nsh = masks[-1]
    masks = masks[:-1]
    dts = [datasets] if isinstance(datasets, str) else datasets
    drops = len(masks)*[True] if drop is True else [drop.get(d) for d in dts]
  else:
    nsh = nshowers

  fits = load_tables(path, datasets, particles, nsh, columns)

  if xfirst is True:
    xf = load_xfirst(datadir, datasets, particles, nsh)
    fits = [a.join(b) for a, b in zip(fits, xf)] if (len(fits) > 1) else [fits[0].join(xf)]

  if drop is not False:
    for f, m, b in zip(fits, masks, drops):
      if b is True:
        f.drop(f.index[~m], inplace = True)
      else:
        f.insert(len(f.columns), 'good', m)

  if norm:
    fits = normalize(fits, columns = norm)

  return fits if len(fits) > 1 else fits[0]

def load_masks(
  datadir: str | os.PathLike,
  cut: config.cut_t | str | dict | Sequence[float | int],
  datasets: config.dataset_t | Sequence[config.dataset_t] = config.datasets,
  particles: config.particle_t | Sequence[config.particle_t] = config.particles,
  nshowers: dict[config.dataset_t, int] | int | None = None,
  drop: Mapping[config.dataset_t, bool] | bool = False,
) -> pd.DataFrame | list[pd.DataFrame]:
  
  c = config.get_cut(cut)
  path = pathlib.Path(f'{datadir}/masks/range-{c.min_depth}-{c.max_depth}').resolve()
  dts = [datasets] if isinstance(datasets, str) else datasets
  pts = [particles] if isinstance(particles, str) else particles
  nsh = get_nshowers(nshowers)
  dropdict = {d: drop for d in config.datasets} if isinstance(drop, bool) else dict(drop)

  if not path.exists():
    raise RuntimeError(f'load_masks: masks for cut {c.name} [{c.min_depth}, {c.max_depth}] do not exist')

  if any(dropdict.values()):

    masks = []

    for d in dts:
      if drop.get(d) is True and nsh.get(d) is not None:
        n = nsh[d]

        ms = load_tables(path, d, particles)[0]['good']
        ms = [ms[p] for p in pts]
        ct = [m[m].index[n] for m in ms]
        mx = max(ct)
        ms = [m[:mx] for m in ms]

        for c, m in zip(ct, ms):
          m[c:] = False

        masks.append(pd.concat(ms, keys = pts))
        nsh[d] = mx
      else:
        masks.append(load_tables(path, d, particles, nsh)[0]['good'])
  else:

    masks = load_tables(path, datasets, particles, nshowers)
    masks = [df['good'] for df in masks]

  return [*masks, nsh]

def load_xfirst(
  datadir: str | os.PathLike,
  datasets: config.dataset_t | Sequence[config.dataset_t] = config.datasets,
  particles: config.particle_t | Sequence[config.particle_t] = config.particles,
  nshowers: dict[config.dataset_t, int] | int | None = None,
  columns: str | Sequence[str] | None = None,
) -> pd.DataFrame | list[pd.DataFrame] :
  
  ret = load_tables(f'{datadir}/xfirst', datasets, particles, nshowers, columns)
  return ret if len(ret) > 1 else ret[0]

#
# data generation
#

def make_selection_masks(
  datadir: str | os.PathLike,
  cuts: str | config.cut_t | Sequence[str | config.cut_t] = config.cuts,
  verbose: bool = True,
) -> None:
  
  cs = [config.get_cut(c) for c in cuts] if isinstance(cuts, Sequence) else config.get_cut(cuts)
  util.echo(verbose, f'generating cut masks for cut configurations {[c.name for c in cs]}')

  for c in cs:
    util.echo(verbose, f'\nprocessing cut configuration {c.name}')
    for d in config.datasets:
      for p in config.particles:
        fits = load_fits(datadir, cut = c, particles = p, datasets = d)
        fits.replace([np.inf, -np.inf], np.nan, inplace = True)

        mask = True
        mask &= (fits.status > 0.99)
        mask &= ~(fits.isna().any(axis = 1))
        mask &= (c.min_depth+10 < fits.Xmax) & (fits.Xmax < c.max_depth-10)
        mask &= (fits > 0).all(axis = 1)

        mask = mask.to_frame('good')
        file = f'{datadir}/masks/range-{c.min_depth}-{c.max_depth}/{d}/{p}'
        util.parquet_save(file, mask, verbose)

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
