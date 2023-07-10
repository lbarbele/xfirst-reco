from collections.abc import Sequence
import itertools
import json
import os
import pathlib
from typing import overload, Any, Iterable, Mapping

import numpy as np
import pandas as pd

# *
# * misc
# *

def collapse(input: Mapping[Any, Any]) -> Mapping[Any, Any] | Any:

  return input if len(input) > 1 else next(iter(input.values()))

def echo(verbose: bool, msg) -> None:

  if verbose: print(msg)
  
def get_range(values: np.ndarray, min_value: float | None = None, max_value: float | None = None) -> tuple[int | None, int | None]:

  if min_value is not None and max_value is not None and min_value >= max_value:
    raise RuntimeError('get_range: min must be smaller than max')

  ileft, iright = None, None

  if min_value is not None and min_value > values[0]:
    if min_value > values[-1]:
      raise RuntimeError('get_range: min is out of range') 
    
    ileft = (values > min_value).argmax()

  if max_value is not None and max_value < values[-1]:
    if max_value < values[0]:
      raise RuntimeError('get_range: max is out of range')
    
    iright = (values < max_value).argmin()
  
  return slice(ileft, iright)

def strlist(input: str | Sequence[str] | None) -> list[str] | None:

  if input is None:
    return None

  l = list(input.split(',') if isinstance(input, str) else input)

  if len(set(l)) != len(l):
    raise ValueError('strset: repeated values are not allowed')

  return l

@overload
def split(data: Sequence[Any], *, indices: Iterable[int]) -> list[Sequence[Any]]:
  ...
@overload
def split(data: Sequence[Any], *, sizes: Iterable[int]) -> list[Sequence[Any]]:
  ...
@overload
def split(data: Sequence[Any], *, batches: int) -> list[Sequence[Any]]:
  ...
@overload
def split(data: Sequence[Any], *, map_sizes: Mapping[Any, int]) -> Mapping[Any, Sequence[Any]]:
  ...
def split(data, *, indices = None, sizes = None, batches = None, map_sizes = None):
  args = {k: v for k, v in locals().items() if k != 'data'}

  if len([v for v in args.values() if v is not None]) != 1:
    raise ValueError(f'split: exactly one of {list(args.keys())} must be given')
  
  if sizes is not None:

    if any([s <= 0 for s in sizes]):
      raise IndexError(f'split: size of a slice cannot be <= 0')

    idx = [0, *list(itertools.accumulate(sizes))]

    if len(data) < idx[-1]:
      raise IndexError(f'split: sum of sizes ({idx[-1]}) is larger than the dataset size ({len(data)})')
    
    return [data[a:b] for a, b in itertools.pairwise(idx)]
  
  elif batches is not None:

    if batches < 1:
      raise IndexError(f'split: invalid batches count {batches}')
    if len(data) < batches:
      raise IndexError(f'split: requested more batches ({batches}) than available data ({len(data)})')

    q, r = divmod(len(data), batches)
    return split(data, sizes = r*[q+1] + (batches - r)*[q])
  
  elif indices is not None:

    idx = sorted({i if i > 0 else (len(data) + i) for i in indices})

    if len(idx) != len(indices):
      raise IndexError('split: repeated indices are not allowed')
    if any([i == 0 for i in idx]):
      raise IndexError('split: index 0 is invalid for index-based split')
    if any([i >= len(data) for i in idx]):
      raise IndexError('split: index is out of range')
    
    idx = [0, *idx, len(data)]
    return [data[a:b] for a, b in itertools.pairwise(idx)]
  
  elif map_sizes is not None:

    return dict(zip(map_sizes.keys(), split(data, sizes = map_sizes.values())))

  raise RuntimeError('split: unexpected error')

# *
# * io
# *

def hdf_save(path: str | os.PathLike, data: pd.DataFrame, key: str, verbose: bool = False) -> None:

  p = pathlib.Path(path).resolve().with_suffix('.h5')
  os.makedirs(p.parent, exist_ok = True)
  data.to_hdf(p, key = key, complevel = 1, append = True, format = 'table', data_columns = True)

  echo(verbose, f'+ hdf key "{key}" saved to {p}')

def hdf_load(path: str | os.PathLike, key: str | Sequence[str], nrows: int | None = None, columns: str | Sequence[str] | None = None) -> pd.DataFrame:

  file = pathlib.Path(path).resolve().with_suffix('.h5')
  cols = strlist(columns)
  keys = strlist(key)

  if len(keys) == 1:
    return pd.read_hdf(file, key = keys[0], stop = nrows, columns = cols)
  else:
    data = [pd.read_hdf(file, key = k, stop = nrows, columns = cols) for k in keys]
    data = pd.concat(data, keys = keys, copy = False)
    return data

def json_save(data: dict, path: str, verbose: bool = False) -> None:

  p = pathlib.Path(path).resolve()
  os.makedirs(p.parent, exist_ok = True)
  with open(p, 'w') as f:
    f.write(json.dumps(data, indent = 2))

  echo(verbose, f'+ json saved to {p}\n')

def json_load(path: str) -> dict:

  with open(path, 'r') as f:
    data = json.load(f)

  return data

def np_save(path: str | os.PathLike, data: np.ndarray, verbose: bool = False) -> None:

  f = pathlib.Path(path).resolve().with_suffix('.npy')
  os.makedirs(f.parent, exist_ok = True)
  np.save(f, data)

  echo(verbose, f'+ npy saved to {f}')

def parquet_load(path: str, nrows: int | None = None, columns: list[str] | None = None) -> pd.DataFrame:

  p = pathlib.Path(path).resolve().with_suffix('.parquet')
  d = pd.read_parquet(p, columns = columns)

  if nrows is not None:
    d.drop(d.index[nrows:], inplace = True)

  return d

def parquet_save(path: str | os.PathLike, data: pd.DataFrame, verbose: bool = False) -> None:

  p = pathlib.Path(path).resolve().with_suffix('.parquet')
  os.makedirs(p.parent, exist_ok = True)
  data.to_parquet(p)

  echo(verbose, f'+ parquet file saved to {p}')
