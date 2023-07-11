import itertools
import pathlib
import os
from typing import Mapping

import click

import xfirst.config as config
import xfirst.data.conex as conex
import xfirst.util as util

def make_conex_split(
  datadir: str | os.PathLike,
  nshowers: Mapping[config.dataset_t, int],
  filter: str | None = None,
  verbose: bool = True,
) -> dict[config.dataset_t, dict[config.particle_t, list[str]]]:
  """
  Splits CONEX files under directory into train, validation, and test datasets.

  The number of files in each dataset is computed according to the number of
  requested through the nshowers parameter.

  If successful, a file conex.json will be created under datadir mapping each
  dataset and primary particle to its corresponding CONEX files.

  Parameters
  ----------
  datadir : str | os.PathLike
    directory from where CONEX data will be read. The structure of this
    directory must be such data data for a given primary can be found by the glob
    pattern `datadir/conex/{primary}*/**/*.root`.

  nshowers : dict
    dicionary where keys are the dataset names (train, validation, and
    test) and the values are the corresponding number of showers requested for the
    datasets. All three datasets must be present.

  filter: str
    a string describing a filter to apply when parsing the CONEX tree. Any selection
    string valid as a selector for root's TTree::Draw is a valid filter. Effectively,
    a filter will affect the number of showers read from each file.

  verbose : bool, default True
    if true, information will be printed as data is processed.

  Returns
  -------

  dict
    A dictionary mapping each dataset name to another level of dictionaries,
    whose keys are the primary particles and levels contain lists of CONEX files
    as strings.
  """
  basedir = pathlib.Path(datadir).resolve(strict = True)
  ret = {d: {} for d in config.datasets}

  util.echo(verbose, f'splitting conex files under directory {basedir}')
  
  for p in config.particles:
    files = set(map(str, basedir.glob(f'conex/{p}*/**/*.root')))
    util.echo(verbose, f'+ {len(files)} files found under {basedir}/conex/{p}*/**/*.root, of which')
    for d in config.datasets:
      ret[d][p] = conex.parser(files = files, nshowers = nshowers[d], filter = filter).files
      files -= set(ret[d][p])
      util.echo(verbose, f'  {len(ret[d][p])} {d} files')

  util.json_save(ret, basedir/'conex.json', verbose)

  return ret

def make_datasets(
  datadir: str | os.PathLike,
  nshowers: Mapping[config.dataset_t, int],
  filter: str | None = None,
  verbose: bool = True,
) -> None:
  """
  Parses CONEX files and creates datasets of shower profiles and Xfirst data.

  CONEX data will be read according to the conex.json file, which is created
  by make_conex_split. The function will create two folders under datadir, a
  path given as argument:
  
  + profiles directory: will be filled with energy-deposit profiles extracted
  from the CONEX files. The directory will contain one .h5 file per dataset
  (train, validation, and test) whose keys are the primary particle names.
  Apart from those, a fourth file "depths.npy" will contain a numpy array
  describing the slant atmospheric depths corresponding to the energy-deposit
  profiles.

  + xfirst directory: analogous to the profiles directory, but the files will
  store a table with the columns "Xfirst" and "lgE".

  Parameters
  ----------
  datadir : str | os.PathLike
    directory from where CONEX data will be read. The structure of this
    directory must be such data data for a given primary can be found by the glob
    pattern `datadir/conex/{primary}*/**/*.root`.

  nshowers : dict
    dicionary where keys are the dataset names (train, validation, and
    test) and the values are the corresponding number of showers requested for the
    datasets. All three datasets must be present.

  filter: str
    a string describing a filter to apply when parsing the CONEX tree. Any selection
    string valid as a selector for root's TTree::Draw is a valid filter.

  verbose : bool, default True
    if true, information will be printed as data is processed.

  See also
  --------
  make_conex_split: split a list of CONEX files into datasets
  """

  basedir = pathlib.Path(datadir).resolve()
  branches = ('Xfirst', 'lgE')
  nsh = dict(nshowers)

  # split conex files
  cxpaths = make_conex_split(basedir, nsh, filter, verbose)

  # profiles
  util.echo(verbose, 'generating profile datasets')
  for d, p in itertools.product(config.datasets, config.particles):
    data = conex.parser(files = cxpaths[d][p], branches = ['Edep'], nshowers = nsh[d], concat = True, filter = filter).get_table()
    util.hdf_save(basedir/'profiles'/d, data, p, verbose)

  # depths
  util.echo(verbose, '\nsaving slant depth profile')
  depths = conex.parser(files = cxpaths[d][p], branches = ['Xdep'], nshowers = 1, concat = True)[0]
  util.np_save(basedir/'profiles'/'depths', depths, verbose)

  # xfirst
  util.echo(verbose, '\ngenerating xfirst datasets')
  for d, p in itertools.product(config.datasets, config.particles):
    data = conex.parser(files = cxpaths[d][p], branches = branches, nshowers = nsh[d], concat = True, filter = filter).get_table()
    util.hdf_save(basedir/'xfirst'/d, data, p, verbose)

@click.command()
@click.option('--datadir', type = click.Path(exists = True, dir_okay = True), required = True)
@click.option('--nshowers', type = (str, click.IntRange(1, 4000000)), required = True, multiple = True)
@click.option('--filter', type = str, required = False, default = None)
@click.option('--verbose/--no-verbose', default = True)
def main(**kwargs):
  make_datasets(**kwargs)
  return 0

if __name__ == '__main__':
  main()
