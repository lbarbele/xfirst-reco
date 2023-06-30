import glob
import pathlib
from typing import List, Union

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