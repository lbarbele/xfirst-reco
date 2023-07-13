import abc
import os
import pathlib
from typing import Iterable, Self, Sequence

import keras
import keras.callbacks
import keras.layers
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
# * helper functions
# *

def get_keras_callbacks(
  early_stopping: int | None = 35,
  reduce_lr: int | None = 10,
  verbose: bool = True
) -> list[keras.callbacks.Callback]:
  callbacks = []

  if early_stopping is not None:
    callbacks.append(
      keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        patience = early_stopping,
        verbose = verbose,
        min_delta = 1e-7,
        mode = 'min',
        restore_best_weights = True
      )
    )

  if reduce_lr is not None:
    callbacks.append(
      keras.callbacks.ReduceLROnPlateau(
        monitor = 'val_loss',
        factor = 0.1,
        patience = reduce_lr,
        verbose = verbose,
        mode = 'min',
        min_delta = 1e-5,
      )
    )

  return callbacks

# *
# * model interfaces
# *

class model(abc.ABC):
  def __init__(
    self,
    verbose: bool = False,
  ) -> None:
    
    self._cfg = {}
    self._features = None
    self._history = None
    self._target = None
    self._verbose = verbose

  # properties: common to all models

  @property
  def cfg(self):
    return self._cfg

  @property
  def features(self):
    return self._features
  
  @property
  def history(self):
    return self._history
  
  @property
  def target(self):
    return self._target

  @property
  def verbose(self):
    return self._verbose
  
  # abstract methods: require implementation

  @abc.abstractmethod
  def _fit(self, train: tuple[pd.DataFrame, pd.Series], validation: tuple[pd.DataFrame, pd.Series]) -> dict:
    ...

  @abc.abstractmethod
  def _predict(self, x: pd.DataFrame) -> np.ndarray:
    ...

  @abc.abstractmethod
  def _save(self, path: pathlib.Path) -> pathlib.Path:
    ...

  # model interface methods: common to all models

  # draw validation curve
  def draw(self, ax: matplotlib.axes.Axes | None = None) -> matplotlib.figure.Figure:

    if ax is None:
      fig = matplotlib.pyplot.figure()
      ax = fig.gca()
    else:
      fig = ax.figure
    
    ax.plot(self.history['x'], self.history['val_loss'], '-', label = 'Validation', c = 'navy', alpha = 0.7)
    ax.plot(self.history['x'], self.history['loss'], '--', label = 'Train', c = 'orange')
    ax.legend()
    ax.set_ylabel('RMS error')
    ax.set_xlabel(self.history['xlabel'])

    return fig
  
  # evaluate model, optionally plot, optionally save
  def eval(self, data: config.datadict_t, save: config.path_t | None = None, plot: bool = False) -> pd.DataFrame:

    test = data['test']

    # compute predictions
    results = {}
    results['predictions'] = self.predict(test[self.features])
    results['target'] = test[self.target]
    results['residuals'] = results['predictions'] - results['target']
    results['lgE'] = test['lgE']

    if 'status' in test.columns: # for fit data
      results['good'] = test['status'] > 0.99

    results = pd.DataFrame(results)

    # draw predictions
    if plot is True:
      util.echo(self.verbose, '+ drawing predictions')
      if 'good' in results.columns:
        fig = viz.draw_predictions(results.loc[results['good']])
      else:
        fig = viz.draw_predictions(results)
    
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
      if 'normalization' in data:
        norm = data['normalization']
        util.hdf_save(outdir/'normalization', data = norm, key = 'normalization', verbose = self.verbose)

    return results

  # fit model
  def fit(self, data: config.datadict_t, x: str | Sequence[str], y: str) -> Self:
    
    self._features = x
    self._target = y

    self.cfg['features'] = x
    self.cfg['target'] = y
    
    util.echo(self.verbose, f'+ training the model on {self.features} for target {self.target}')

    train = (data['train'][x], data['train'][y])
    valid = (data['validation'][x], data['validation'][y])
    self._history = self._fit(train, valid)

    return self
  
  # predict on test data
  def predict(self, data: pd.DataFrame | config.datadict_t) -> np.ndarray:
    
    util.echo(self.verbose, f'+ computing predictions')

    x = data if isinstance(data, pd.DataFrame) else data['train']
    x = x[self.features]

    y = self._predict(x)
    y = pd.Series(y, index = x.index)

    return y

  # save model
  def save(self, path: config.path_t) -> pathlib.Path:
    
    outdir = pathlib.Path(path).resolve()
    os.makedirs(outdir, exist_ok = True)

    self._save(outdir)
    util.echo(self.verbose, f'+ model saved under {outdir}')

    fig = self.draw()
    fig.savefig(outdir/'validation_curve.pdf')
    matplotlib.pyplot.close(fig)
    util.echo(self.verbose, f'+ validation curve saved to {outdir}/validation_curve.pdf')

    util.json_save(self.cfg, outdir/'config')
    util.echo(self.verbose, f'+ model configuration saved to {outdir}/config.json')

    util.json_save(self.history, path/'history')
    util.echo(self.verbose, f'+ training history saved to {outdir}/history.json')

    return outdir
  
class neural_network(model):
  def __init__(
    self,
    batch_size: int = 32,
    epochs: int = 1000,
    verbose: bool = True,
  ) -> None:
    
    super().__init__(verbose = verbose)

    self.cfg['batch_size'] = batch_size

    self._batch_size = batch_size
    self._callbacks = get_keras_callbacks(verbose = verbose)
    self._epochs = epochs
    self._verbosity_level = int(self.verbose) * (2 - int(util.interactive()))
    self._model = None

  def _fit(self, train, validation) -> None:

    history = self.nn.fit(
      x = train[0],
      y = train[1],
      batch_size = self._batch_size,
      epochs = self._epochs,
      verbose = self._verbosity_level,
      callbacks = self._callbacks,
      validation_data = validation,
    )

    history = {
      'loss': [float(i) for i in np.sqrt(history.history['loss'])],
      'val_loss': [float(i) for i in np.sqrt(history.history['val_loss'])],
      'xlabel': 'Epoch',
      'x': history.epoch,
      **{k: [float(i) for i in v] for k, v in history.history.items() if k not in ['loss', 'val_loss']},
    }

    return history
  
  def _predict(self, x) -> np.ndarray:
    return self.nn.predict(x, verbose = self._verbosity_level).flatten()
  
  def _save(self, path) -> None:
    self.nn.save(path/'model')

  @property
  def nn(self):
    return self._model
  
  @nn.setter
  def nn(self, value: keras.Model):
    self._model = value

# *
# * XGboost wrapper
# *

class gradient_boosting_regressor(model):
  def __init__(self, verbose: bool = True, **kwargs) -> None:
    
    super().__init__(verbose = verbose)
    
    kwargs.setdefault('n_estimators', 1000)
    kwargs.setdefault('n_jobs', os.cpu_count())
    kwargs.setdefault('early_stopping_rounds', 7)

    self._xgb = xgboost.XGBRegressor(**kwargs)

  def _fit(self, train, validation) -> dict:
    
    m = self.xgb.fit(*train, eval_set = [train, validation], verbose = self.verbose)

    return {
      'loss': m.evals_result()['validation_0']['rmse'],
      'val_loss': m.evals_result()['validation_1']['rmse'],
      'xlabel': '$n_\mathrm{trees}$',
      'x': [i+1 for i in range(m.get_booster().num_boosted_rounds())]
    }

  def _predict(self, x) -> np.ndarray:
    return self.xgb.predict(x)
  
  def _save(self, path) -> None:
    self.xgb.save_model(path/'model.ubj')

  @property
  def xgb(self):
    return self._xgb
  
# *
# * mlp wrapper
# *

class multilayer_perceptron_regressor(neural_network):
  def __init__(
    self,
    input: int,
    layers: Iterable[int],
    optimizer: str = 'adam',
    batch_size: int = 32,
    epochs: int = 1000,
    verbose: bool = True,
  ) -> None:
    
    super().__init__(batch_size, epochs, verbose)

    layers = [keras.layers.Dense(u, 'relu') for u in layers]
    layers.insert(0, keras.Input(shape = input))
    layers.append(keras.layers.Dense(1))

    self.nn = keras.Sequential(layers)
    self.nn.compile(optimizer = optimizer, loss = 'mse')

class recurrent_network(neural_network):
  def __init__(
    self,
    input: tuple[int, int],
    recurrent_layers: Sequence[int],
    dense_layers: Sequence[int],
    bidirectional: bool = False,
    optimizer: str = 'adam',
    batch_size: int = 32,
    epochs: int = 1000,
    verbose: bool = True,
  ):
    
    super().__init__(batch_size, epochs, verbose)

    self.nn = keras.models.Sequential()
    self.nn.add(keras.Input(shape = input))
    for i, u in enumerate(recurrent_layers):
      l = keras.layers.LSTM(u, return_sequences = i < len(recurrent_layers) - 1)
      self.nn.add(keras.layers.Bidirectional(l) if bidirectional is True else l)
    for u in dense_layers:
      self.nn.add(keras.layers.Dense(u, 'relu'))
    self.nn.add(keras.layers.Dense(1))

    self.nn.compile(optimizer = optimizer, loss = 'mse')
