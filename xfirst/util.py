import glob
import json
import os
import pathlib

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


def json_dump(data: dict, path: str) -> None:

  p = pathlib.Path(path).resolve()
  os.makedirs(p.parent, exist_ok = True)
  with open(p, 'w') as f:
    f.write(json.dumps(data, indent = 2))

def get_file_list(
  globs: str | list[str],
  sort: bool = True,
  absolute: bool = True,
) -> list[str]:
  """
  Takes as input a single glob or a list of globs, resolve them and
  returns a list of strings containing all file paths that match the
  given globs.
  """

  if isinstance(globs, str):
    files = glob.glob(globs)
    if len(files) == 0: raise RuntimeError(f'bad path "{globs}"')
    if sort: files.sort()
    if absolute: files = [str(pathlib.Path(f).resolve()) for f in files]
    return files
  elif isinstance(globs, list) or isinstance(globs, tuple):
    files = [f for g in globs for f in get_file_list(g, sort = False)]
    if sort: files.sort()
    return files
  else:
    raise RuntimeError(f'get_file_list: bad input "{globs}"')
  
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

  
def json_load(path: str) -> dict:

  with open(path, 'r') as f:
    data = json.load(f)

  return data

def npz_load(path: str) -> dict[np.ndarray]:

  with np.load(path) as data:
    ret = dict(data)

  return ret

def npz_save(path: str, **kwargs) -> None:
  
  p = pathlib.Path(path).resolve()
  os.makedirs(p.parent, exist_ok = True)
  np.savez_compressed(p, **kwargs)

def parquet_load(path: str, nrows: int | None = None) -> pd.DataFrame:

  data = pd.read_parquet(path)

  if nrows is not None:
    data.drop(data.index[nrows:], inplace = True)

  return data

def parquet_save(data: pd.DataFrame, path: str) -> None:

  p = pathlib.Path(path).resolve()
  os.makedirs(p.parent, exist_ok = True)
  data.to_parquet(p)