import abc
import os
import pathlib
from typing import overload, Any, Literal, Mapping, Self, Sequence

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot
import numpy as np
import pandas as pd
import xgboost

from . import config
from . import util
from . import viz

# *
# * model interface
# *

class model(abc.ABC):
  def __init__(
    self,
    verbose: bool = False,
  ) -> None:
    
    self._features = None
    self._target = None
    self._verbose = verbose

  @property
  def features(self):
    return self._features
  
  @property
  def target(self):
    return self._target

  @property
  def verbose(self):
    return self._verbose
  
  # draw validation curve

  @abc.abstractmethod
  def _draw(
    self,
    ax: matplotlib.axes.Axes
  ) -> None:
    ...

  def draw(
    self,
    ax: matplotlib.axes.Axes | None = None
  ) -> matplotlib.figure.Figure:

    if ax is None:
      fig = matplotlib.pyplot.figure()
      ax = fig.gca()
    else:
      fig = ax.figure
    
    self._draw(ax)

    return fig
  
  # evaluate model, plot, optionally save

  def eval(
    self,
    data: Mapping[config.dataset_t | Literal['normalization'], pd.DataFrame], 
    save: str | os.PathLike | None = None,
    plot: bool = False,
  ) -> pd.DataFrame:

    # compute predictions
    results = {}
    results['predictions'] = self.predict(data['test'][self.features])
    results['target'] = data['test'][self.target]
    results['residuals'] = results['predictions'] - results['target']
    results['lgE'] = data['test']['lgE']
    results['good'] = data['test']['status'] > 0.99
    results = pd.DataFrame(results)

    # draw predictions
    if plot is True:
      util.echo(self.verbose, '+ drawing predictions')
      goodresults = results.loc[results['good']]
      fig = viz.draw_predictions(goodresults)
    
    if save is not None:
      # save model
      outdir = self.save(save)
      # save predictions
      for p in results.index.levels[0]:
        res = results.loc[p]
        util.hdf_save(outdir/'predictions', data = res, key = p, verbose = self.verbose)
      # save plot
      if plot is True:
        fig.savefig(outdir/'predictions.pdf')
      # save normalization
      norm = data['normalization']
      util.hdf_save(outdir/'normalization', data = norm, key = 'normalization', verbose = self.verbose)

    return results

  # fit model

  @abc.abstractmethod
  def _fit(
    self,
    train: tuple[pd.DataFrame, pd.Series],
    validation: tuple[pd.DataFrame, pd.Series],
  ) -> None:
    ...

  def fit(
    self,
    data: Mapping[config.dataset_t, pd.DataFrame],
    x: str | Sequence[str],
    y: str,
  ) -> Self:
    
    self._features = x
    self._target = y
    
    util.echo(self.verbose, f'+ training the model on {self.features} for target {self.target}')

    train = (data['train'][x], data['train'][y])
    valid = (data['validation'][x], data['validation'][y])
    self._fit(train, valid)

    return self
  
  # predict on test data

  @abc.abstractmethod
  def _predict(
    self,
    x: pd.DataFrame
  ) -> np.ndarray:
    ...

  def predict(
    self,
    data: pd.DataFrame | Mapping[config.dataset_t, pd.DataFrame],
  ) -> np.ndarray:
    
    util.echo(self.verbose, f'+ computing predictions')

    x = data if isinstance(data, pd.DataFrame) else data['train']
    x = x[self.features]

    y = self._predict(x)
    y = pd.Series(y, index = x.index)
    return y

  # save model

  @abc.abstractmethod
  def _save(
    self,
    path: pathlib.Path,
  ) -> pathlib.Path:
    ...

  def save(
    self,
    path: str | os.PathLike,
  ) -> pathlib.Path:
    
    outdir = pathlib.Path(path).resolve()
    os.makedirs(outdir, exist_ok = True)

    self._save(outdir)
    util.echo(self.verbose, f'+ model saved under {outdir}')

    fig = self.draw()
    fig.savefig(outdir/'validation_curve.pdf')
    matplotlib.pyplot.close(fig)
    util.echo(self.verbose, f'+ validation curve saved to {outdir}/validation_curve.pdf')

    return outdir

# *
# * XGboost wrapper
# *

class gradient_boosting_regressor(model):
  def __init__(
    self,
    verbose: bool = True,
    **kwargs
  ) -> None:
    
    super().__init__(verbose = verbose)
    
    kwargs.setdefault('n_estimators', 1000)
    kwargs.setdefault('n_jobs', os.cpu_count())
    kwargs.setdefault('early_stopping_rounds', 7)

    self._xgb = xgboost.XGBRegressor(**kwargs)

  def _draw(
    self,
    ax: matplotlib.axes.Axes,
  ) -> None:

    yt = self.xgb.evals_result()['validation_0']['rmse']
    yv = self.xgb.evals_result()['validation_1']['rmse']
    x = np.arange(len(yt)) + 1

    ax.plot(x, yv, '-', label = 'Validation', color = 'navy', alpha = 0.7)
    ax.plot(x, yt, '--', label = 'Train', color = 'orange')
    ax.set_ylabel('Root mean squared error')
    ax.set_xlabel('$n_\mathrm{trees}$')
    ax.legend()
  
  def _fit(
    self,
    train: tuple[pd.DataFrame, pd.Series],
    validation: tuple[pd.DataFrame, pd.Series],
  ) -> Self:
    
    self.xgb.fit(*train, eval_set = [train, validation], verbose = self.verbose)
  
  def _predict(
    self,
    x: pd.DataFrame,
  ) -> np.ndarray:
    
    return self.xgb.predict(x)
  
  def _save(
    self,
    path: pathlib.Path,
  ) -> None:
    
    self.xgb.save_model(path/'model.ubj')

  @property
  def xgb(self):
    return self._xgb