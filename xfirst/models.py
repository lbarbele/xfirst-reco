import abc
import os
import pathlib
from typing import Literal

import matplotlib.axes
import matplotlib.pyplot
import numpy as np
import pandas as pd
import xgboost

# *
# * model classes
# *

class model(abc.ABC):
  @abc.abstractmethod
  def draw(self):
    ...

  @abc.abstractmethod
  def fit(self, train, validation):
    ...

  @abc.abstractmethod
  def predict(self, x):
    ...

  @abc.abstractmethod
  def save(self, path: str | os.PathLike) -> None:
    ...

  def test(self, x, y, format: Literal['np', 'pd'] = 'pd'):
    
    if len(x) != len(y):
      raise ValueError('model.test: x and y have different lengths')
    
    results = np.zeros(shape = (len(x), 3), dtype = np.float32)
    results[:, 0] = self.predict(x)
    results[:, 1] = y
    results[:, 2] = results[:, 0] - results[:, 1]

    if format == 'np':
      return results
    elif format == 'pd':
      cols = ['predictions', 'target', 'residuals']
      idx = x.index if isinstance(x, pd.DataFrame) else y.index if isinstance(y, pd.DataFrame) else None
      return pd.DataFrame(results, columns = cols, index = idx)
    else:
      raise ValueError(f'model.test: invalid format {format}')

class gradient_boosting_regressor(model):
  def __init__(self, **kwargs):

    kwargs.setdefault('n_estimators', 1000)
    kwargs.setdefault('n_jobs', os.cpu_count())
    kwargs.setdefault('early_stopping_rounds', 7)

    self._xgb = xgboost.XGBRegressor(**kwargs)

  def draw(self, ax: matplotlib.axes.Axes | None = None) -> matplotlib.axes.Axes:
    ytrain = self.xgb.evals_result()['validation_0']['rmse']
    yval = self.xgb.evals_result()['validation_1']['rmse']
    x = np.arange(len(ytrain)) + 1

    if ax is None:
      ax = matplotlib.pyplot.gca()

    ax.plot(x, yval, '-', label = 'Validation', color = 'navy', alpha = 0.7)
    ax.plot(x, ytrain, '--', label = 'Train', color = 'orange')
    ax.legend()
    ax.set_ylabel('Root mean squared error')
    ax.set_xlabel('$n_\mathrm{trees}$')

    return ax

  def fit(self, train: tuple, validation: tuple):
    self.xgb.fit(X = train[0], y = train[1], eval_set = [train, validation], verbose = True)
    return self
  
  def predict(self, x):
    return self.xgb.predict(X = x)
  
  def save(self, path: str | os.PathLike):
    p = pathlib.Path(path).resolve()
    os.makedirs(p.parent, exist_ok = True)
    self.xgb.save_model(p.with_suffix('.ubj'))

  @property
  def xgb(self):
    return self._xgb