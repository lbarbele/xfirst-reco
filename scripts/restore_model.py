import pathlib
import sys

import tensorflow

import xfirst

# parse input

if len(sys.argv) != 3:
  raise RuntimeError('bad input')

if sys.argv[1] != 'A2' and sys.argv[1] != 'C3':
  raise RuntimeError('invalid cut')

cut = xfirst.config.cut.get(sys.argv[1])
base_path = pathlib.Path(sys.argv[2]) # like: .../ml-conex-xfirst/models/mlp-profile-...

# path to model

path = cut.path('models/mlp-profile-1024-1024-1024-1024')

# load model

model = xfirst.models.load(path)
model._cfg['verbosity'] = True

# load profiles for the given cut

data = xfirst.data.load_profiles(
  datadir  = 'data',
  cut      = cut,
  datasets = ['train', 'test'],
  nshowers = {'train': 1000000, 'test': 500000},
  norm     = True,
  xfirst   = True,
  fits     = None,
  drop_bad = {'train': True, 'test': False},
  shuffle  = {'train': True, 'test': False},
  verbose  = True,
)

# evaluate and save the model

model.eval(
  data = data,
  save = path/'new_save',
  plot = True,
)