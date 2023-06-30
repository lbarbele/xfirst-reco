from typing import List, Literal, Union

import numpy as np
import pandas as pd
import ROOT

from ..util import get_file_list as _get_file_list

def get_conex_tree(
  files: Union[str, List[str]],
  tree_name: str,
  max_entries: Union[int, None] = None
) -> ROOT.TChain:
  
  if max_entries is not None and (not isinstance(max_entries, int) or max_entries < 1):
    raise RuntimeError(f'get_conex_tree: invalid max_entries parameter {max_entries}')

  chain = ROOT.TChain(tree_name, tree_name)

  for file in _get_file_list(files):
    chain.Add(file)

    if max_entries is not None and chain.GetEntries() >= max_entries:
      break

  return chain

class parser:

  def __init__(
    self,
    files: Union[str, List[str]],
    branches: List[str],
    nshowers: Union[int, None] = None,
    concat: bool = False
  ) -> None:

    self._branches = branches
    self._tree = get_conex_tree(files, 'Shower', nshowers)
    self._files = [f.GetTitle() for f in self.tree.GetListOfFiles()]
    self._nshowers = self.tree.GetEntries() if nshowers is None else nshowers
    self._data = {}
    self._updaters = []
    self._concat = concat
    self._current = 0

    self._nX = np.zeros(1, np.int32)
    self.tree.SetBranchAddress('nX', self._nX)
    self.tree.GetEntry(0)

    for branch_name in branches:
      if branch_name in ['Xdep', 'Edep']:
        self.add_special_branch(branch_name)
      else:
        self.add_branch(branch_name)

    # form a row of data
    row = [self.data[br] for br in branches]
    self._row_data = row
    self._row = np.concatenate(row, dtype = np.float32) if concat else row

    # generate column names
    self._columns = []
    for i, v in enumerate(row):
      l = len(v)
      b = branches[i]
      self._columns += [f'{b}_{j}' for j in range(l)]

  # methods

  def add_branch(self, branch_name: str) -> np.ndarray:
    
    typedict = {'I': np.int32, 'F': np.float32, 'D': np.float64}

    if branch_name in self.data:
      return self.data[branch_name]

    if not self.tree.GetListOfBranches().Contains(branch_name):
      raise RuntimeError(f'parser.add_branch: invalid branch {branch_name}')
    
    branch = self.tree.GetBranch(branch_name)

    if branch.GetNleaves() != 1:
      raise RuntimeError(f'parser.add_branch: bad branch in tree {branch.GetName()}')
    
    title, tp = branch.GetTitle().split('/')
    
    if not tp in typedict:
      raise RuntimeError(f'parser.add_branch: bad branch type {branch} {tp}')
    
    if '[' in title and ']' in title:
      title, nptstr = title.replace('[', ' ').replace(']', '').split()
      npt = self.nX if nptstr == 'nX' else int(nptstr)
    else:
      npt = 1

    value = np.zeros(npt, dtype = typedict[tp])
    self.data[branch_name] = value
    self.tree.SetBranchAddress(branch_name, value)

    return value
  
  def add_special_branch(self, branch_name: str) -> None:

    match branch_name:
      case 'Xdep':
        x = self.add_branch('X')
        self.read(0)
        self.data[branch_name] = 0.5*(x[1:]+x[:-1])
      case 'Edep':
        y =  self.add_branch('dEdX')
        self.data[branch_name] = y[:-1]
      case _:
        raise RuntimeError(f'parser.add_special_branch: invalid branch {branch_name}')
      
  def get_table(self, format: Literal['np', 'pd'] = 'np') -> Union[np.ndarray, pd.DataFrame]:
    
    data = np.zeros(shape = (self.nshowers, len(self.columns)), dtype = np.float32)
    
    for i in range(self.nshowers):
      self.read(i)
      np.concatenate(self._row_data, out = data[i])

    match format:
      case 'np':
        return data
      case 'pd':
        idx = pd.Index(range(self.nshowers), name = 'id')
        return pd.DataFrame(data, columns = self.columns, copy = False, index = idx)
      case _:
        raise RuntimeError(f'parser.get_table: invalid format {format}')
      
  def read(self, entry: int) -> None:

    self.tree.GetEntry(entry)

  # element access and iteration

  def __getitem__(self, pos: int) -> Union[List[np.ndarray], np.ndarray]:

    self.read(pos)
    if self.concat: np.concatenate(self._row_data, out = self._row)
    return self.row

  def __iter__(self):

    self._current = 0
    return self
  
  def __next__(self) -> Union[List[np.ndarray], np.ndarray]:
    if self.current >= self.nshowers: raise StopIteration
    data = self.__getitem__(self.current)
    self._current += 1
    return data
  
  # properties

  @property
  def branches(self):
    return self._branches
  
  @property
  def columns(self):
    return self._columns

  @property
  def concat(self):
    return self._concat
  
  @property
  def current(self):
    return self._current

  @property
  def data(self):
    return self._data
  
  @property
  def files(self):
    return self._files
  
  @property
  def nshowers(self):
    return self._nshowers
  
  @property
  def nX(self):
    return self._nX[0]
  
  @property
  def row(self):
    return self._row

  @property
  def tree(self):
    return self._tree