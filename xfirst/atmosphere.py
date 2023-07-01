import abc

import numpy as np

class atmosphere(abc.ABC):
  def _get_depth(self, h: float) -> float:

    if h < self.h[0]:
      raise RuntimeError(f'atmosphere.get_depth: invalid height {h}')
    
    l = np.digitize(h, self.h) - 1

    if l < 4:
      return self.a[l] + self.b[l]*np.exp(-h/self.c[l])
    else:
      return max(self.a[4] - self.b[4]*h/self.c[4], 0.)
  
  def _get_height(self, x):

    if x > self.x[0] or x < self.x[-1]:
      raise RuntimeError(f'atmosphere.get_height: invalid depth {x}')
    
    l = np.digitize(x, self.x) - 1

    if l < 4:
      return self.c[l]*np.log(self.b[l]/(x - self.a[l]))
    else:
      return self.c[4]*(self.a[4] - x)/self.b[4]

  def get_depth(self, height):

    hs = height if isinstance(height, np.ndarray) else [height]
    xs = [self._get_depth(h) for h in hs]

    if len(xs) == 0:
      raise RuntimeError('atmosphere.get_depth: heights is empty')

    return np.array(xs, dtype = np.float64) if len(xs) > 1 else xs[0]
  
  def get_height(self, depth):
    
    xs = depth if isinstance(depth, np.ndarray) else [depth]
    hs = [self._get_height(x) for x in xs]

    if len(hs) == 0:
      raise RuntimeError('atmosphere.get_depth: depths is empty')
    
    return np.array(hs, dtype = np.float64) if len(hs) > 1 else hs[0]

class us_standard(atmosphere):
  def __init__(self):
    self.h = np.array([0, 4, 10, 40, 100, np.inf], dtype = np.float64)
    self.x = np.array([1036.1009, 631.10088, 271.69999, 3.0394999, 0.00128292, 0], dtype = np.float64)
    self.a = np.array([-186.555305, -94.919, 0.6128, 0.0, 0.01128292], dtype = np.float64)
    self.b = np.array([1222.6562, 1144.9069, 1305.5948, 540.1778, 1.], dtype = np.float64)
    self.c = np.array([9.9418638, 8.7815355, 6.3614304, 7.7217016, 1e4], dtype = np.float64)