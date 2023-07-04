from immutabledict import immutabledict
import typing

dataset_t = typing.Literal['train', 'validation', 'test']
datasets = typing.get_args(dataset_t)

particle_t = typing.Literal['p', 'He', 'C', 'Si', 'Fe']
particles = typing.get_args(particle_t)
