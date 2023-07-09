from collections.abc import Sequence
import itertools
import glob
import json
import os
import pathlib
from typing import overload, Any, Iterable, Mapping

import numpy as np
import pandas as pd

def as_list(input, check_empty: bool = True) -> list:

  ret = input if isinstance(input, list) else [input]

  if check_empty and len(ret) == 0:
    raise RuntimeError('as_list: list is empty')
  
  return ret

def df_from_dict(
  datadict: dict[str, np.ndarray],
  keys: list[str] | None = None,
  columns: list[str] | None = None,
  index_name: str = 'id',
) -> pd.DataFrame:
  
  if keys is None:
    keys = list(datadict.keys())

  dfs = []

  for k in keys:
    data = datadict[k]
    index = pd.Index(range(len(data)), name = index_name)
    dfs.append(pd.DataFrame(data, columns = columns, copy = False, index = index))

  return pd.concat(dfs, keys = keys)

def echo(verbose: bool, msg) -> None:
  if verbose:
    print(msg)
  
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
  
  return ileft, iright

def json_dump(data: dict, path: str) -> None:

  p = pathlib.Path(path).resolve()
  os.makedirs(p.parent, exist_ok = True)
  with open(p, 'w') as f:
    f.write(json.dumps(data, indent = 2))

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