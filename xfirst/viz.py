from typing import Sequence

import matplotlib.axes
import matplotlib.figure
import numpy as np
import pandas as pd
import seaborn as sns

from . import config
from . import profile_functions

def draw_fit_parameters(
  data: pd.DataFrame,
  fcn: profile_functions.profile_function = profile_functions.usp(),
  axes: Sequence[matplotlib.axes.Axes] | None = None,
  stat: bool = True,
  pallete: str = 'rocket',
  nbins: int = 50,
  annot: bool = True,
  limits: dict[str, tuple[float, float]] | None = None
) -> matplotlib.figure.Figure:
  
  if hasattr(data.index, 'levels'):
    particles = [p for p in config.particles if p in data.index.levels[0]]
    npart = len(particles)
  else:
    particles = None
    npart = 1
  
  ncols = fcn.npar + int(stat)
  colors = sns.color_palette(pallete, npart + 1)

  if axes is None:
    fig, axes = matplotlib.pyplot.subplots(1, ncols, figsize = (4*ncols, 4))
  else:
    fig = axes[0].figure

  labels = {
    'lgNmax': '$\log_{10}\\left(\\frac{N_\mathrm{max}}{\mathrm{GeV}\, \mathrm{g}^{-1}\, \mathrm{cm}^2}\\right)$',
      'Xmax': '$X_\mathrm{max}$ [g cm$^{-2}$]',
         'L': 'L [g cm$^{-2}$]',
         'R': 'R',
      'stat': '$\chi^2/\mathrm{ndf}$'
  }

  scales = {
    'lgNmax': 'linear',
      'Xmax': 'log',
         'L': 'log',
         'R': 'log',
      'stat': 'log',
  }

  args = {
    'stacked': True,
    'log': True,
    'label': particles,
    'color': colors[1:],
    'edgecolor': colors[0],
    'linewidth': 0.5,
    'alpha': 0.9,
  }

  def annotate(v, ax):
    if not annot: return
    ax.annotate(f'min: {v.min():.4g}', xy = (0.03, 0.95), xycoords = 'axes fraction', fontsize = 8)
    ax.annotate(f'max: {v.max():.4g}', xy = (0.03, 0.90), xycoords = 'axes fraction', fontsize = 8)

  def compute(param):
    if param == 'stat' and not 'stat' in data:
      values = data.loc[:, 'chi2']/data.loc[:, 'ndf']
    else:
      values = data.loc[:, param]

    if limits is not None and param in limits:
      l = limits[param]
    else:
      l = (values.min(), values.max())

    if scales[param] == 'log':
      values = values.loc[values > 0]
      bins = 10**np.linspace(np.log10(l[0]), np.log10(l[1]), nbins + 1)
    else:
      bins = np.linspace(l[0], l[1], nbins + 1)

    return values, bins
  
  def split(v):
    if npart > 1:
      return [v.loc[p] for p in particles]
    else:
      return v

  for ax, param in zip(axes, fcn.parameter_names):
    values, bins = compute(param)
    ax.hist(split(values), bins = bins, **args)
    ax.set_xscale(scales[param])
    ax.set_xlabel(labels[param])
    ax.set_ylabel('Count')
    annotate(values, ax)

  if stat is not None:
    ax = axes[-1]
    values, bins = compute('stat')
    ax.hist(split(values), bins = bins, **args)
    ax.set_xscale(scales['stat'])
    ax.set_xlabel(labels['stat'])
    ax.set_ylabel('Count')
    annotate(values, ax)
  
  if npart > 1:
    axes[-1].legend()

  fig.tight_layout()

  return fig
