from collections import namedtuple
import typing

dataset_t = typing.Literal['train', 'validation', 'test']
datasets = typing.get_args(dataset_t)

particle_t = typing.Literal['p', 'He', 'C', 'Si', 'Fe']
particles = typing.get_args(particle_t)

cut_t = namedtuple('cut_config', ('name', 'min_depth', 'max_depth'))

_cuts = (
  cut_t(
    name = 'A1',
    min_depth = 600,
    max_depth = 1000,
  ),
  cut_t(
    name = 'A2',
    min_depth = 350,
    max_depth = 1000,
  ),
  cut_t(
    name = 'A3',
    min_depth = 100,
    max_depth = 1000,
  ),
  cut_t(
    name = 'B1',
    min_depth = 650,
    max_depth = 1250,
  ),
  cut_t(
    name = 'B2',
    min_depth = 300,
    max_depth = 1250,
  ),
  cut_t(
    name = 'B3',
    min_depth = 50,
    max_depth = 1250,
  ),
  cut_t(
    name = 'C1',
    min_depth = 450,
    max_depth = 1750,
  ),
  cut_t(
    name = 'C2',
    min_depth = 100,
    max_depth = 1750,
  ),
  cut_t(
    name = 'C3',
    min_depth = 0,
    max_depth = 1750,
  ),
)

def _cuts_get(i):
  if i is None:
    return cut_t('none', None, None)
  elif isinstance(i, cut_t):
    return i
  elif isinstance(i, int):
    return _cuts(i)
  elif isinstance(i, str):
    for c in _cuts:
      if c.name == i:
        return c
  elif isinstance(i, dict):
    if 'min_depth' in i and 'max_depth' in i:
      return cut_t(i.get('name', 'unnamed'), i['min_depth'], i['max_depth'])
  elif hasattr(i, 'min_depth') and hasattr(i, 'max_depth'):
    return cut_t(getattr(i, 'name', 'unnamed'), i.min_depth, i.max_depth)
  elif isinstance(i, list | tuple) and len(i) == 2:
    if isinstance(i[0], float | int) and isinstance(i[1], float | int):
      return cut_t('unnamed', int(i[0]), int(i[1]))

  raise ValueError(f'cuts.get: non existent cut {i}')

cuts = namedtuple('cuts_tuple', [c.name for c in _cuts] + ['get'])(*_cuts, get = _cuts_get)
