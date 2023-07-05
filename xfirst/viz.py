import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import profile_functions

def draw_fit_parameters(data):

  fcn = profile_functions.usp()

  fig, axes = plt.subplots(1, fcn.npar + 1, figsize = (20, 4))

  nice_names = {
    'lgNmax': '$\log_{10}\\left(\\frac{N_\mathrm{max}}{\mathrm{GeV}\, \mathrm{g}^{-1}\, \mathrm{cm}^2}\\right)$',
    'Xmax': '$X_\mathrm{max}$ [g cm$^{-2}$]',
    'L': 'L [g cm$^{-2}$]',
    'R': 'R',
    'stat': '$\chi^2/\mathrm{ndf}$'
  }

  draw_args = {
    'multiple': 'stack',
    'hue': 'particle',
    'palette': 'rocket',
    'edgecolor': '.3',
    'linewidth': .5,
    'bins': 50,
  }

  for icol, param in enumerate(fcn.parameter_names):
    ax = axes[icol]
    sns.histplot(data = data, x = param, log_scale = (icol != 0, True), ax = ax, legend = False, **draw_args)
    ax.set_xlabel(nice_names[param])

  stat = pd.DataFrame({'stat': data.loc[:, 'chi2']/data.loc[:, 'ndf']})
  sns.histplot(data = stat, x = 'stat', log_scale = (True, True), ax = axes[-1], **draw_args)

  fig.tight_layout()