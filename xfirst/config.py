from collections import namedtuple
import typing

dataset_t = typing.Literal['train', 'validation', 'test']
datasets = typing.get_args(dataset_t)

particle_t = typing.Literal['p', 'He', 'C', 'Si', 'Fe']
particles = typing.get_args(particle_t)

cut_t = namedtuple('cut_config', ('name', 'min_depth', 'max_depth'))
cuts = (
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
