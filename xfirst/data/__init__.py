import pathlib
from typing import List

from . import conex

from ..util import get_file_list, json_dump

def split_conex_files(
  ds,
  datadir: str,
  verbose: bool = False,
  particles: List[str] = ['p', 'He', 'C', 'Si', 'Fe'],
  out: str = None,
) -> dict:
  
  datadir = pathlib.Path(datadir).resolve()

  sizes = ds if isinstance(ds, dict) else dict(ds)
  globs = {p: f'{datadir}/{p}*/*/*.root' for p in particles}
  paths = {p: get_file_list(g) for p, g in globs.items()}

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
    json_dump(datasets, out)
    
  return datasets