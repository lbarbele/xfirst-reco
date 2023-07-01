import collections
import concurrent.futures
import itertools
import os
import pathlib
import typing

import numpy as np
import pandas as pd

from .. import util
from ..profile_functions import usp

from . import conex

def load_profiles(
  datadir: str,
  datasets: typing.Union[str, typing.List[str]] = ['train', 'validation', 'test'],
  particles: typing.Union[str, typing.List[str]] = ['p', 'He', 'C', 'Si', 'Fe'],
  nshowers: dict = None,
  min_depth: float = None,
  max_depth: float = None,
  return_depths: bool = False,
  format: str = 'np',
) -> typing.Union[dict, pd.DataFrame]:
  
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

def make_fits(
  datadir: str,
  out: str,
  max_train: int = None,
  max_val: int = None,
  max_test: int = None,
  min_depth: float = None,
  max_depth: float = None,
  workers: int = None,
  verbose: bool = True,
):
  
  datasets = ['train', 'validation', 'test']
  particles = ['p', 'He', 'C', 'Si', 'Fe']
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

  for dsname in datasets:
    if verbose: print(f'parsing {dsname} dataset')

    for prm in particles:
      profiles, depths = load_profiles(datasets = dsname, particles = prm, **load_args)

      batch_size = profiles.shape[0]//workers + int(profiles.shape[0]%workers > 0)
      split_at = np.arange(1, workers)*batch_size

      data_slices = np.split(profiles, split_at)
      n_list = [len(y) for y in data_slices]
      z_list = [zip(itertools.repeat(depths, len(y)), y) for y in data_slices]

      with concurrent.futures.ProcessPoolExecutor(workers) as exec:
        fits = exec.map(fcn.get_fits, itertools.repeat(fcn(), workers), z_list, n_list, itertools.repeat('np', workers))
        fits = np.concatenate(list(fits))
        fits = pd.DataFrame(fits, columns = fcn().columns, index = pd.Index(range(len(fits)), name = 'id'))

        file = pathlib.Path(f'{out}/{dsname}/{prm}.parquet').resolve()
        os.makedirs(file.parent, exist_ok = True)
        fits.to_parquet(file)

        if verbose: print(f'+ {prm} fits saved to {file}')

def make_profile_datasets(
  data: str,
  out: str,
  max_train: int = None,
  max_val: int = None,
  max_test: int = None,
  verbose: bool = True,
):
  dataset_paths = util.json_load(data)
  nshowers = {'train': max_train, 'validation': max_val, 'test': max_test}

  for dsname, ds in dataset_paths.items():
    if verbose: print(f'parsing {dsname} dataset')

    for prm, files in ds.items():
      parser = conex.parser(files = files, branches = ['Edep'], nshowers = nshowers[dsname], concat = True)
      data = parser.get_table('np')
      file = pathlib.Path(f'{out}/{dsname}/{prm}.npz').resolve()
      util.npz_save(file, **{prm: data})

      if verbose: print(f'+ {prm} data saved to {file}')
  
  depths = conex.parser(files = dataset_paths[dsname][prm], branches = ['Xdep'], nshowers = 1, concat = True)[0]
  util.npz_save(f'{out}/depths.npz', depths = depths)

def make_xfirst_datasets(
  data: str,
  out: str,
  max_train: int = None,
  max_val: int = None,
  max_test: int = None,
  verbose: bool = True,
):
  branches = ['Xfirst', 'lgE', 'Nmx', 'Xmx']
  dataset_paths = util.json_load(data)
  nshowers = {'train': max_train, 'validation': max_val, 'test': max_test}

  for dsname, ds in dataset_paths.items():
    if verbose: print(f'parsing {dsname} dataset')

    for prm, files in ds.items():
      parser = conex.parser(files = files, branches = branches, nshowers = nshowers[dsname], concat = True)
      data = parser.get_table('pd')
      file = pathlib.Path(f'{out}/{dsname}/{prm}.parquet').resolve()
      os.makedirs(file.parent, exist_ok = True)
      data.to_parquet(file)

      if verbose: print(f'+ {prm} data saved to {file}')

def split_conex_files(
  ds,
  datadir: str,
  verbose: bool = False,
  particles: typing.List[str] = ['p', 'He', 'C', 'Si', 'Fe'],
  out: str = None,
) -> dict:
  
  datadir = pathlib.Path(datadir).resolve()

  sizes = ds if isinstance(ds, dict) else dict(ds)
  globs = {p: f'{datadir}/{p}*/*/*.root' for p in particles}
  paths = {p: util.get_file_list(g) for p, g in globs.items()}

  # check if there are enought files to generate the datasets

  min_files = 0
  for n in sizes.values():
    min_files += n

  for prm, files in paths.items():
    if len(files) < min_files:
      raise RuntimeError(f'not enough files for particle {prm}')

  # print an overview of the input files

  if verbose:
    print('overview of the input files:')
    for prm, files in paths.items():
      print(f'+ {prm}: {len(files)} files under {globs[prm]}')

  # consume the paths dictionary to create the datasets

  datasets = {dsname: {} for dsname in sizes.keys()}
  for dsname, size in sizes.items():
    for prm in paths.keys():
      datasets[dsname][prm] = paths[prm][:size]
      paths[prm] = paths[prm][size:]

  # perform checks

  if verbose:
    print('checking files: ')

  for prm in particles:
    l = [f for dsname in sizes.keys() for f in datasets[dsname][prm]]
    if len(l) != len(set(l)):
      raise RuntimeError(f'file overlap for particle {prm}, this is a bug')

  if verbose:
    print('+ no overlap')

  for dsname, size in sizes.items():
    for prm, l in datasets[dsname].items():
      if len(l) != size:
        raise RuntimeError(f'wrong value in dataset {dsname}.{prm}')
      
  if verbose:
    print('+ sizes are correct')

  # save output file

  if out is not None:
    util.json_dump(datasets, out)
    
  return datasets