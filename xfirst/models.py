import abc
import collections
import os
import pathlib
from typing import Any, Iterable, Self, Sequence

import keras
import keras.callbacks
import keras.layers
import keras.utils
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
# * model interfaces
# *

class model(abc.ABC):
  implementations = {}

  @classmethod
  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    model.implementations[cls.__name__] = cls

  def __init__(
    self,
    backend: Any,
    cfg: dict = {}, 
    verbose: bool = False,
  ) -> None:
    
    self._backend = backend
    self._cfg = cfg
    self._verbose = verbose

    self._features = None
    self._history = None
    self._target = None

  # properties: common to all models

  @property
  def backend(self):
    return self._backend

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

  @staticmethod
  @abc.abstractmethod
  def _load(path: pathlib.Path) -> Self:
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

    if self.history is None:
      raise RuntimeError('model.draw: model history empty, did you call fit before?')

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

  # load saved model
  @classmethod
  def load(cls, path: str | os.PathLike, verbose: bool = True) -> Self:

    path = pathlib.Path(path).resolve(strict = True)

    # if called from base model interface, read class name from config.json
    if cls is model:
      name = util.json_load(path/'config.json')['name']
      cls = model.implementations[name]
    
    m = cls._load(path)
    m._cfg = util.json_load(path/'config.json')
    m._history = util.json_load(path/'history.json')
    m._features = m.cfg['features']
    m._target = m.cfg['target']
    m._verbose = verbose

    return m

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

    self._cfg['name'] = type(self).__name__
    util.json_save(self.cfg, outdir/'config')
    util.echo(self.verbose, f'+ model configuration saved to {outdir}/config.json')

    util.json_save(self.history, outdir/'history')
    util.echo(self.verbose, f'+ training history saved to {outdir}/history.json')

    return outdir
  
class neural_network(model):
  def __init__(
    self,
    backend: Any,
    cfg: dict = {},
    batch_size: int = 32,
    epochs: int = 1000,
    backup_dir: str | os.PathLike | None = None,
    verbose: bool = True,
  ) -> None:

    cfg = {
      'batch_size': batch_size,
      'epochs': epochs,
      'verbosity': verbose * (2 - util.interactive()),
      'backup_dir': backup_dir,
    }

    super().__init__(backend = backend, cfg = cfg, verbose = verbose)

  def _fit(self, train, validation) -> None:

    callbacks = [
      keras.callbacks.EarlyStopping(patience = 35, verbose = self.verbose, restore_best_weights = True),
      keras.callbacks.ReduceLROnPlateau(patience = 10, verbose = self.verbose)
    ]

    if self.cfg.get('backup_dir') is not None:
      callbacks.append(keras.callbacks.BackupAndRestore(self.cfg['backup_dir']))
    
    keras_history = self.backend.fit(
      x = train[0],
      y = train[1],
      batch_size = self.cfg['batch_size'],
      epochs = self.cfg['epochs'],
      verbose = self.cfg['verbosity'],
      callbacks = callbacks,
      validation_data = validation,
    )

    history = dict()
    history['loss'] = [float(i) for i in np.sqrt(keras_history.history['loss'])]
    history['val_loss'] = [float(i) for i in np.sqrt(keras_history.history['val_loss'])]
    history['xlabel'] = 'Epoch'
    history['x'] = keras_history.epoch

    for k in keras_history.history:
      if not k in history:
        history[k] = [float(v) for v in keras_history.history[k]]

    return history

  @staticmethod
  def _load(path) -> Self:

    nn = keras.models.load_model(path/'model')
    nn = neural_network(backend = nn)
    return nn
  
  def _predict(self, x) -> np.ndarray:
    return self.backend.predict(x, verbose = self.cfg['verbosity']).flatten()
  
  def _save(self, path) -> None:
    self.backend.save(path/'model')

# *
# * XGboost wrapper
# *

class gradient_boosting_regressor(model):
  def __init__(
      self,
      *,
      n_estimators: int = 1000,
      n_jobs: int = os.cpu_count(),
      early_stopping_rounds: int = 7,
      verbose: bool = True, 
      **kwargs
    ) -> None:
    
    xgb = xgboost.XGBRegressor(
      n_estimators = n_estimators,
      n_jobs = n_jobs,
      early_stopping_rounds = early_stopping_rounds,
      **kwargs
    )

    super().__init__(backend = xgb, verbose = verbose)

  def _fit(self, train, validation) -> dict:
    
    m = self.backend.fit(*train, eval_set = [train, validation], verbose = self.verbose)

    history = dict()
    history['loss'] = m.evals_result()['validation_0']['rmse']
    history['val_loss'] = m.evals_result()['validation_1']['rmse']
    history['xlabel'] = '$n_\mathrm{trees}$'
    history['x'] = [i+1 for i in range(m.get_booster().num_boosted_rounds())]
    return history
  
  @staticmethod
  def _load(path) -> Self:
    
    xgbmodel = gradient_boosting_regressor()
    xgbmodel.backend.load_model(path/'model.ubj')
    return xgbmodel

  def _predict(self, x) -> np.ndarray:
    return self.backend.predict(x)
  
  def _save(self, path) -> None:
    self.backend.save_model(path/'model.ubj')

# *
# * mlp wrapper
# *

class multilayer_perceptron_regressor(neural_network):
  def __init__(
    self,
    input: int,
    layers: Iterable[int],
    *,
    optimizer: str = 'adam',
    batch_size: int = 32,
    epochs: int = 1000,
    backup_dir: str | os.PathLike | None = None,
    verbose: bool = True,
  ) -> None:
    
    mlp = keras.models.Sequential()
    mlp.add(keras.Input(shape = input))
    for u in layers:
      mlp.add(keras.layers.Dense(u, 'relu'))
    mlp.add(keras.layers.Dense(1))

    mlp.compile(optimizer = optimizer, loss = 'mse', jit_compile = True)

    super().__init__(backend = mlp, batch_size = batch_size, epochs = epochs, backup_dir = backup_dir, verbose = verbose)

class recurrent_network(neural_network):
  def __init__(
    self,
    input: tuple[int | None, int],
    recurrent_layers: Iterable[int],
    dense_layers: Iterable[int],
    *,
    bidirectional: bool = False,
    optimizer: str = 'adam',
    batch_size: int = 32,
    epochs: int = 1000,
    backup_dir: str | os.PathLike | None = None,
    verbose: bool = True,
  ) -> None:
    
    rnn = keras.models.Sequential()
    rnn.add(keras.Input(shape = input))
    for i, u in enumerate(recurrent_layers):
      l = keras.layers.LSTM(u, return_sequences = i < len(recurrent_layers) - 1)
      rnn.add(keras.layers.Bidirectional(l) if bidirectional is True else l)
    for u in dense_layers:
      rnn.add(keras.layers.Dense(u, 'relu'))
    rnn.add(keras.layers.Dense(1))

    rnn.compile(optimizer = optimizer, loss = 'mse', jit_compile = True)

    super().__init__(backend = rnn, batch_size = batch_size, epochs = epochs, backup_dir = backup_dir, verbose = verbose)

class convolutional_network(neural_network):
  layer = collections.namedtuple('convolutional_network_layer', ('filters', 'kernel', 'pooling'))

  def __init__(
    self,
    input: tuple[int | None, int],
    conv_layers: Iterable[tuple[int, int, int | None] | layer],
    dense_layers: Iterable[int] = [],
    *,
    optimizer: str = 'adam',
    batch_size: int = 32,
    epochs: int = 1000,
    backup_dir: str | os.PathLike | None = None,
    verbose: bool = True,
  ) -> None:
    
    cnn = keras.models.Sequential()
    # input
    cnn.add(keras.layers.Input(shape = input))
    # convolutions + optional max pooling after each layer
    for l in conv_layers:
      l = convolutional_network.layer(*l)
      cnn.add(keras.layers.Conv1D(l.filters, l.kernel, activation = 'relu'))
      if l.pooling is not None:
        cnn.add(keras.layers.MaxPooling1D(l.pooling))
    # global pooling
    cnn.add(keras.layers.GlobalAveragePooling1D())
    # dense layers
    for u in dense_layers:
      cnn.add(keras.layers.Dense(u, 'relu'))
    # output
    cnn.add(keras.layers.Dense(1))

    cnn.compile(optimizer = optimizer, loss = 'mse', jit_compile = True)

    super().__init__(backend = cnn, batch_size = batch_size, epochs = epochs, backup_dir = backup_dir, verbose = verbose)

# *
# * model loader
# *

def load(path: str | os.PathLike, verbose: bool = True) -> model:
  return model.load(path, verbose)
