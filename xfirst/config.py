import os
import pathlib
import typing

dataset_t = typing.Literal['train', 'validation', 'test']
datasets = typing.get_args(dataset_t)

particle_t = typing.Literal['p', 'He', 'C', 'Si', 'Fe']
particles = typing.get_args(particle_t)

class cut():
  _cuts = {
    'A1': dict(min_depth = 600, max_depth = 1000),
    'A2': dict(min_depth = 350, max_depth = 1000),
    'A3': dict(min_depth = 100, max_depth = 1000),
    'B1': dict(min_depth = 650, max_depth = 1250),
    'B2': dict(min_depth = 300, max_depth = 1250),
    'B3': dict(min_depth =  50, max_depth = 1250),
    'C1': dict(min_depth = 450, max_depth = 1750),
    'C2': dict(min_depth = 100, max_depth = 1750),
    'C3': dict(min_depth =   0, max_depth = 1750),
  }

  def __init__(self, name: str, min_depth: int, max_depth: int):
    self._name = name
    self._min_depth = min_depth
    self._max_depth = max_depth

  @staticmethod
  def count():
    return len(cut._cuts)

  @staticmethod
  def get(input = None):
    if isinstance(input, cut):
      return input
    elif input is None:
      return tuple(cut(k, **v) for k, v in cut._cuts.items())
    else:
      return cut(input, **cut._cuts[input])
  
  @staticmethod
  def names():
    return tuple(c.name for c in cut.get())
  
  def path(self, base: str | os.PathLike):
    return pathlib.Path(base).resolve()/f'range-{self.min_depth}-{self.max_depth}'

  @property
  def max_depth(self):
    return self._max_depth

  @property
  def min_depth(self):
    return self._min_depth
  
  @property
  def name(self):
    return self._name
  
for c in cut.get():
  setattr(cut, c.name, c)
