import abc
import inspect

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# *
# * virtual interface
# *

class profile_function(abc.ABC):
  """
  Base class for the profile functions defined below.
  """
  _props = {}
  _impl = {}

  def __init_subclass__(cls, function_name, **kwargs):
    super().__init_subclass__(**kwargs)

    if hasattr(inspect, 'getargspec'):
      pnames = inspect.getargspec(cls.__call__).args[2:]
    elif hasattr(inspect, 'getfullargspec'):
      pnames = inspect.getfullargspec(cls.__call__).args[2:]
    else:
      raise RuntimeError('in ProfileFunction __init_subclass__: unable to get arg spec')
    
    profile_function._props[cls.__name__] = {
      'name': function_name,
      'parameter_names': pnames,
    }

    profile_function._impl[function_name] = cls

  # abstract methods
  
  @abc.abstractmethod
  def __call__(x, *parameters):
    pass

  @abc.abstractmethod
  def guess(x, y):
    pass

  # static methods

  @staticmethod
  def from_name(name: str, **kwargs):
    if name in profile_function._impl:
      return profile_function._impl[name](**kwargs)
    else:
      raise RuntimeError(f'profile_function: "{name}" is not implemented')
    
  # properties
  
  @property
  def chi2(self):
    return getattr(self, '_chi2', None)
  
  @property
  def columns(self):
    return self.parameter_names + [f'e_{p}' for p in self.parameter_names] + ['status', 'chi2', 'ndf']

  @property
  def errors(self):
    return getattr(self, '_errors', None)
  
  @property
  def fit_status(self):
    return getattr(self, '_fit_status', None)

  @property
  def fitted_data(self):
    return getattr(self, '_fitted_data', None)

  @property
  def name(self):
    return profile_function._props[self.__class__.__name__]['name']
  
  @property
  def ndf(self):
    return getattr(self, '_ndf', None)

  @property
  def npar(self): 
    return len(self.parameter_names)

  @property
  def parameter_names(self):
    return list(profile_function._props[self.__class__.__name__]['parameter_names'])
  
  @property
  def parameters(self):
    return getattr(self, '_params', None)
  
  # methods

  def draw(self, color = 'navy', axis = None):
    
    if self.fitted_data is not None:
      x, y = self.fitted_data
      f = self(x, *self.parameters)

      if axis is None:
        axis = plt.gca()
      
      axis.fill_between(x, y*1e-6, color = color, alpha = 0.5, lw = 0)
      axis.plot(x, f*1e-6, '-', color = color, alpha = 0.7, lw = 2)
      axis.set_xlabel('Slant atmospheric depth [g cm$^{-2}$]')
      axis.set_ylabel('d$E/$d$X$ [PeV g$^{-1}$ cm$^2$]')
      axis.set_ylim([0, None])
    
      return axis

  def fit(self, x: np.ndarray, y: np.ndarray, concat: bool = False):

    # number of degrees of freedom
    self._ndf = len(x) - self.npar

    # scipy requires float64 for x and y and p0
    xf = x if x.dtype >= np.float64 else np.array(x, dtype = np.float64)
    yf = y if y.dtype >= np.float64 else np.array(y, dtype = np.float64)

    try:
      # poissonian ansatz as in https://arxiv.org/abs/1111.0504, but without cutting the profiles
      s = 1e-2*np.sqrt(yf*np.sum(yf))
      p0 = self.guess(xf, yf)
      popt, pcov, info, mesg, ierr = curve_fit(self, xf, yf, p0, s, full_output = True)
      self._params = popt
      self._fit_status = (1 <= ierr and ierr <= 4)
      self._errors = np.sqrt(pcov.diagonal()) if self._fit_status else np.zeros(self.npar, dtype = np.float32)
      self._chi2 = np.sum(((yf - self(xf, *popt)) / s)**2)
      self._fitted_data = (x, y)

      self.fit_callback()
      
      if concat:
        return np.concatenate([popt, self._errors, [self._fit_status, self._chi2, self._ndf]], dtype = np.float32)
      else:
        return popt, self._errors, self._fit_status, self._chi2, self._ndf
    except:
      if concat:
        return np.zeros(2*self.npar + 3, dtype = np.float32)
      else:
        return self.guess(x, y), self.guess(x, y), 0., 0., 0.
      
  def fit_callback(self):
    pass
      
  def get_fits(self, x, y = None, format: str = 'np'):

    iter = x if y is None else zip(x, y)
    data = [self.fit(xi, yi, concat = True) for xi, yi in iter]

    if format == 'np':
      return np.array(data, dtype = np.float32)
    elif format == 'pd':
      idx = pd.Index(range(len(data)), name = 'id')
      return pd.DataFrame(data, columns = self.columns, index = idx)
    else:
      raise RuntimeError(f'parser.get_table: invalid format {format}')

# *
# * implementations
# *
    
class usp(profile_function, function_name = 'usp'):
  """
  Universal Shower Profile function, functionally equivalent to a Gaisser-Hillas
  of four parameters, but written in a form that the distribution of the shape
  parameters is less correlated and without large tails.
  """
  def __call__(self, x, lgNmax, Xmax, L, R):
    absr = np.abs(R)
    z = np.maximum(1 + (absr/L) * (x - Xmax), 1e-5)
    return np.exp(lgNmax + (1 + np.log(z) - z) / absr**2)
  
  def fit_callback(self):
    self._params[3] = np.abs(self._params[3])
  
  def guess(self, x, y):
    return np.array([np.log(np.max(y)), x[np.argmax(y)], 200, 0.25], dtype = np.float64)
  
class gaisser_hillas(profile_function, function_name = 'gaisser-hillas'):
  """
  Usual Gaisser-Hillas profile with four parameters.
  """
  def __call__(self, x, logNmax, Xmax, X0, Lambda):
    xx = np.maximum((x - X0) / Lambda, 1e-5)
    yy = np.maximum((Xmax - X0) / Lambda, 1e-5)
    return np.exp(logNmax -xx + yy*(1 + np.log(xx/yy)))
  
  def guess(self, x, y):
    return np.array([np.log(np.max(y)), x[y.argmax()], -100, 80], dtype = np.float64)