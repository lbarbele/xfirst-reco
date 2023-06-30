import glob
import json
import os
import pathlib
from typing import List, Union

import numpy as np

def json_dump(data, path: str):
  p = pathlib.Path(path).resolve()
  os.makedirs(p.parent, exist_ok = True)
  with open(p, 'w') as f:
    f.write(json.dumps(data, indent = 2))

def get_file_list(
  globs: Union[str, List[str], tuple],
  sort: bool = True,
  absolute: bool = True,
) -> List[str]:
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
  
def json_load(path: str):
  with open(path, 'r') as f:
    data = json.load(f)

  return data

def npz_load(path):
  with np.load(path) as data:
    ret = dict(data)
  return ret

def npz_save(path, **kwargs):
  p = pathlib.Path(path).resolve()
  os.makedirs(p.parent, exist_ok = True)
  np.savez_compressed(p, **kwargs)