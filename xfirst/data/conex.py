import os
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import ROOT

from .. import util

def get_conex_tree(
  files: str | os.PathLike | Iterable[str | os.PathLike],
  tree_name: str,
  entries: int | None = None,
  filter: str = None,
) -> ROOT.TChain:
  
  filesiter = map(str, [files] if isinstance(files, str | os.PathLike) else files)
  chain = ROOT.TChain(tree_name, tree_name)

  count = 0

  for f in filesiter:
    chain.Add(f)

    if entries is not None:
      with ROOT.TFile(f) as rootfile:
        tree = rootfile.Get(tree_name)
        count += tree.GetEntries() if filter is None else tree.GetEntries(filter)

      if count >= entries:
        break

  if entries is not None and count < entries:
    raise RuntimeError(f'get_conex_tree: requested more entries ({entries}) than available ({count})')
  
  chain.Draw('>>selector', filter or '')
  selector = ROOT.TEventList(ROOT.gDirectory.selector)

  if entries is not None and count > entries:
    excess = ROOT.TEventList()
    for i in range(entries, selector.GetN()):
      excess.Enter(selector.GetEntry(i))
    selector.Subtract(excess)
  
  chain.SetEventList(selector)

  return chain

def get_event_list(tree: ROOT.TTree | ROOT.TChain):

  eventlist = tree.GetEventList()
  buffer = eventlist.GetList()
  count = eventlist.GetN()

  return np.frombuffer(buffer, dtype = np.int64, count = count)

class parser:

  def __init__(
    self,
    files: str | os.PathLike | Iterable[str | os.PathLike],
    branches: str | Sequence[str] | None = None,
    nshowers: int | None = None,
    concat: bool = False,
    filter: str | None = None,
  ) -> None:

    self._tree = get_conex_tree(files, 'Shower', nshowers, filter)
    self._files = [f.GetTitle() for f in self.tree.GetListOfFiles()]
    self._eventlist = get_event_list(self._tree)
    self._nshowers = len(self._eventlist)
    self._branches = util.strlist(branches) or []
    self._data = {}
    self._concat = concat
    self._current = 0

    self._nX = np.zeros(1, np.int32)
    self.tree.SetBranchAddress('nX', self._nX)
    self.read(0)

    for branch_name in self.branches:
      if branch_name in ['Xdep', 'Edep']:
        self.add_special_branch(branch_name)
      else:
        self.add_branch(branch_name)

    # form a row of data
    row = [self.data[br] for br in self.branches]
    self._row_data = row
    self._row = np.concatenate(row).astype(np.float32) if concat else row

    # generate column names
    self._columns = []
    for i, v in enumerate(row):
      l = len(v)
      b = self.branches[i]
      self._columns += [b] if (l == 1) else [f'{b}_{j}' for j in range(l)]

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

    if branch_name == 'Xdep':
      x = self.add_branch('X')
      self.read(0)
      self.data[branch_name] = 0.5*(x[1:]+x[:-1])
    elif branch_name == 'Edep':
      y =  self.add_branch('dEdX')
      self.data[branch_name] = y[:-1]
    else:
      raise RuntimeError(f'parser.add_special_branch: invalid branch {branch_name}')
      
  def get_table(self) -> pd.DataFrame:
    
    data = np.zeros(shape = (self.nshowers, len(self.columns)), dtype = np.float32)
    
    for i in range(self.nshowers):
      self.read(i)
      np.concatenate(self._row_data, out = data[i])

    idx = pd.Index(range(self.nshowers), name = 'id')
    return pd.DataFrame(data, columns = self.columns, copy = False, index = idx)
      
  def read(self, entry: int) -> None:

    self.tree.GetEntry(self.eventlist[entry])

  # element access and iteration

  def __getitem__(self, pos: int) -> list[np.ndarray] | np.ndarray:

    self.read(pos)
    if self.concat: np.concatenate(self._row_data, out = self._row)
    return self.row

  def __iter__(self):

    self._current = 0
    return self
  
  def __next__(self) -> list[np.ndarray] | np.ndarray:
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
  def eventlist(self):
    return self._eventlist
  
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