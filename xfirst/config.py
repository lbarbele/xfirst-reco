from immutabledict import immutabledict

datasets = ('train', 'validation', 'test')
particles = ('p', 'He', 'C', 'Si', 'Fe')

cuts = immutabledict({
  'A1': immutabledict({
    'min_depth': 600,
    'max_depth': 1000,
  }),
  'A2': immutabledict({
    'min_depth': 350,
    'max_depth': 1000,
  }),
  'A3': immutabledict({
    'min_depth': 100,
    'max_depth': 1000,
  }),
  'B1': immutabledict({
    'min_depth': 650,
    'max_depth': 1250,
  }),
  'B2': immutabledict({
    'min_depth': 300,
    'max_depth': 1250,
  }),
  'B3': immutabledict({
    'min_depth': 50,
    'max_depth': 1250,
  }),
  'C1': immutabledict({
    'min_depth': 450,
    'max_depth': 1750,
  }),
  'C2': immutabledict({
    'min_depth': 100,
    'max_depth': 1750,
  }),
  'C3': immutabledict({
    'min_depth': 0,
    'max_depth': 1750,
  }),
})