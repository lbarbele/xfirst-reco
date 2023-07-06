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


def get_bins(limits: Sequence[float], nbins: int, log: bool = False):
  if log:
    return 10**np.linspace(np.log10(limits[0]), np.log10(limits[1]), nbins + 1)
  else:
    return np.linspace(limits[0], limits[1], nbins + 1)

def draw_predicted_versus_xfirst(
  values: Sequence[float],
  preds: Sequence[float],
  ax: matplotlib.axes.Axes | None = None,
  nbins: int = 50,
  cmap: str | int = 0,
  limits: Sequence[float] = (0.1, 400),
):
  if ax is None:
    ax = matplotlib.pyplot.figure().gca()

  bins = 2*[get_bins(limits, nbins, True)]
  norm = 'log'

  if isinstance(cmap, int):
    cmap = sns.color_palette().as_hex()[cmap]
    cmap = sns.color_palette(f'light:{cmap}', as_cmap = True)

  corr = np.corrcoef(values, preds)[0, 1]

  ax.plot(limits, limits, '--k', lw = 0.5, alpha = 0.6)
  ax.hist2d(values, preds, bins = bins, norm = norm, cmap = cmap)
  ax.set_xscale('log')
  ax.set_yscale('log')
  ax.set_xlabel('True $X_\mathrm{first}$ [g cm$^{-2}$]')
  ax.annotate(f'$\\rho = {corr:.3f}$', xy = (0.03, 0.95), xycoords = 'axes fraction')

  return ax

def draw_residuals_distribution(
  residuals: Sequence[float],
  ax: matplotlib.axes.Axes | None = None,
  nbins: int = 50,
  color: int = 0,
  limits: Sequence[float] = (-100, 100),
):
  if ax is None:
    ax = matplotlib.pyplot.figure().gca()

  bins = get_bins(limits, nbins, False)

  edgecolor = sns.color_palette('dark')[color]
  color = sns.color_palette()[color]

  ax.hist(residuals, bins, color = color, log = True, alpha = 0.5, linewidth = 0.5, edgecolor = edgecolor)
  ax.set_xlabel('Predicted $X_\mathrm{first}$ - True $X_\mathrm{first}$ [g cm$^{-2}$]')
  ax.annotate(f'$\mu = {np.mean(residuals):.3f}$', xy = (0.03, 0.95), xycoords = 'axes fraction')
  ax.annotate(f'$\sigma = {np.std(residuals):.3f}$', xy = (0.03, 0.90), xycoords = 'axes fraction')
  return ax

def draw_xfirst_distributions(
  values: Sequence[float],
  preds: Sequence[float],
  ax: matplotlib.axes.Axes | None = None,
  nbins: int = 50,
  color: int = 0,
  limits: Sequence[float] = (0.1, 400),
): 
  if ax is None:
    ax = matplotlib.pyplot.figure().gca()

  bins = get_bins(limits, nbins, True)

  edgecolor = sns.color_palette('dark')[color]
  color = sns.color_palette()[color]

  ax.hist(preds, bins, color = color, log = True, alpha = 0.5, linewidth = 0.5, edgecolor = edgecolor, label = 'Predictions')
  ax.hist(values, bins, histtype = 'step', color = edgecolor, linewidth = 1.5, alpha = 0.85, label = 'True values')
  ax.set_xscale('log')
  ax.set_xlabel('$X_\mathrm{first}$ [g cm$^{-2}$]')
  ax.legend()

  return ax

def draw_residuals_versus_xfirst(
  values: Sequence[float],
  residuals: Sequence[float],
  ax: matplotlib.axes.Axes | None = None,
  nbins: int = 50,
  cmap: str | int = 0,
  limits: Sequence[float] = (0.1, 400),
  reslimits: Sequence[float] = (-100, 100),
):
  if ax is None:
    ax = matplotlib.pyplot.figure().gca()

  bins = [get_bins(limits, nbins, True), get_bins(reslimits, nbins, False)]
  norm = 'log'

  if isinstance(cmap, int):
    cmap = sns.color_palette().as_hex()[cmap]
    cmap = sns.color_palette(f'light:{cmap}', as_cmap = True)

  ax.plot(limits, [0, 0], '--k', lw = 0.5, alpha = 0.6)
  ax.hist2d(values, residuals, bins = bins, norm = norm, cmap = cmap)
  ax.set_xscale('log')
  ax.set_xlabel('True $X_\mathrm{first}$ [g cm$^{-2}$]')

  return ax

def draw_bias_versus_energy(
  residuals: Sequence[float],
  log_energy: Sequence[float],
  ax: matplotlib.axes.Axes | None = None,
  nbins: int = 12,
  color: int = 0,
  limits: Sequence[float] = (-10, 10),
):
  if ax is None:
    ax = matplotlib.pyplot.figure().gca()

  lgemin = round(log_energy.min())
  lgemax = round(log_energy.max())
  lgebins = get_bins((lgemin, lgemax), nbins, False)
  indices = np.digitize(log_energy, lgebins)

  x = 10**(0.5*(lgebins[1:] + lgebins[:-1]))
  means = []
  stds = []
  
  for idx in range(1, nbins+1):
    bin_data = residuals.loc[indices == idx]
    means.append(np.mean(bin_data))
    stds.append(np.std(bin_data))

  means = np.array(means)
  stds = np.array(stds)

  color = sns.color_palette()[color]

  ax.plot(x, means, 'o-', color = color)
  ax.fill_between(x, means-stds, means + stds, color = color, alpha = 0.2)
  ax.plot((10**lgemin, 10**lgemax), [0, 0], '--k', lw = 0.5, alpha = 0.6)
  ax.set_xlabel('$E_0$ [eV]')
  ax.set_xscale('log')
  return ax

def draw_predictions(
  data: pd.DataFrame,
):
  particles = [p for p in config.particles if p in data.index.levels[0]]
  npart = len(particles)
  nrows = 5

  fig, axes = matplotlib.pyplot.subplots(nrows, npart, figsize = (4*npart, 4*nrows))

  for icol, p in enumerate(particles):
    data_slice = data.loc[p]
    draw_predicted_versus_xfirst(data_slice.loc[:, 'target'], data_slice.loc[:, 'predictions'], ax = axes[0, icol], cmap = icol)
    draw_xfirst_distributions(data_slice.loc[:, 'target'], data_slice.loc[:, 'predictions'], ax = axes[1, icol], color = icol)
    draw_residuals_versus_xfirst(data_slice.loc[:, 'target'], data_slice.loc[:, 'residuals'], ax = axes[2, icol], cmap = icol)
    draw_residuals_distribution(data_slice.loc[:, 'residuals'], ax = axes[3, icol], color = icol)
    draw_bias_versus_energy(data_slice.loc[:, 'residuals'], data_slice.loc[:, 'lgE'], ax = axes[4, icol], color = icol)

    axes[0, icol].set_title(p)

  axes[0, 0].set_ylabel('Predicted $X_\mathrm{first}$ [g cm$^{-2}$]')
  axes[1, 0].set_ylabel('Count')
  axes[2, 0].set_ylabel('True $X_\mathrm{first}$ - predicted $X_\mathrm{first}$ [g cm$^{-2}$]')
  axes[3, 0].set_ylabel('Count')
  axes[4, 0].set_ylabel('$\\langle X_\mathrm{first}^\mathrm{(pred)} X_\mathrm{first}^\mathrm{(true)} \\rangle$ [g cm$^{-2}$]')
  fig.tight_layout()

  return fig