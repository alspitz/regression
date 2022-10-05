import os

import numpy as np

try:
  from fastmath import fastcos, fastsin
except ImportError:
  FM_AVAIL = False
else:
  FM_AVAIL = True

try:
  from SSGPR.model.ssgpr import SSGPR as SSGPRLD
except ImportError:
  pass
else:
  class SSGPRLinesD:
    def __init__(self, *, N_feats, lengths, amp, noisevar):
      self.N_feats = N_feats
      self.lengths = lengths
      self.amp = amp
      self.noisevar = noisevar

      self.spectrals = np.random.normal(loc=0.0, scale=1.0, size=(self.N_feats * len(lengths)))

    def train(self, xs, ys):
      n_outputs = ys.shape[1]
      self.models = []
      for i in range(n_outputs):
        model = SSGPRLD(self.N_feats, optimize_freq=False)
        model.add_data(xs, ys[:, i, np.newaxis])
        model.update_parameters(np.hstack((np.log(self.lengths), 0.5 * np.log(self.amp), 0.5 * np.log(self.noisevar), self.spectrals)))
        self.models.append(model)

    def test(self, xs):
      return np.array([model.predict(xs)[0].flatten() for model in self.models]).T

class SSGPR:
  def __init__(self, *, N_feats, lengths, sig_n=1.0, sig_f=1.0, include_constant=True, include_linear=False, **kwargs):
    self.N_feats = N_feats
    self.sig_n = sig_n
    self.sig_f = sig_f
    self.include_constant = include_constant
    self.include_linear = include_linear
    # Random Fourier Features
    np.random.seed(0)
    self.rf = (1.0 / lengths) * np.random.normal(loc=0.0, scale=1.0, size=(self.N_feats, len(lengths)))

  def _Xmat(self, xs, usefastmath=False):
    rf_feats = self.rf.dot(xs.T).T

    if usefastmath and FM_AVAIL:
      cs = fastcos(rf_feats)
      ss = fastsin(rf_feats)
    else:
      cs = np.cos(rf_feats)
      ss = np.sin(rf_feats)

      if usefastmath:
        print("WARNING: usefastmath=True, but fastmath not found! Using normal trigs")

    output_feats = self.sig_f * np.hstack((cs, ss)) / np.sqrt(self.N_feats)

    X = np.hstack((output_feats,))

    # Add constant offset feature.
    if self.include_constant:
      X = np.hstack((X, np.ones((X.shape[0], 1))))

    if self.include_linear:
      X = np.hstack((X, xs))

    return X

  def train(self, xs, ys):
    if len(ys.shape) == 1:
      ys = ys[:, np.newaxis]

    X = self._Xmat(xs)
    #Xinv = np.linalg.pinv(X)


    Xinv = np.linalg.inv(X.T.dot(X) + (self.sig_n ** 2) * np.eye(X.shape[1])).dot(X.T)

    ws = []
    for i in range(ys.shape[1]):
      ws.append(Xinv.dot(ys[:, i]))

    self.W = np.array(ws).T

    return np.mean(np.abs(ys - X.dot(self.W)), axis=0)

  def save(self, outdir):
    np.savetxt(os.path.join(outdir, "rf.out"), self.rf)
    np.savetxt(os.path.join(outdir, "W.out"), self.W)

  def load(self, outdir):
    self.rf = np.loadtxt(os.path.join(outdir, "rf.out"))
    self.W = np.loadtxt(os.path.join(outdir, "W.out"))

  def gradient(self, xp):
    """
      Returns the gradient at a single point xp
      Return is matrix of shape (d_y, d_x)
    """

    d_x = len(xp)
    N = self.N_feats
    total_feats = 2 * N + 1 + d_x
    assert self.W.shape[0] == total_feats

    rf_feats = self.rf.dot(xp)

    dydx = np.vstack((
      -np.sin(rf_feats)[:, np.newaxis] * self.sig_f * self.rf / np.sqrt(N),
       np.cos(rf_feats)[:, np.newaxis] * self.sig_f * self.rf / np.sqrt(N),
       np.zeros(d_x),
       np.eye(d_x)
     ))

    return self.W.T.dot(dydx)

  #def regress_incremental(self, dataset):
  #  X = self.get_Xmat(dataset)
  #  A = np.zeros((X.shape[1], X.shape[1]))
  #  b = np.zeros((X.shape[1]))
  #  for i, row in enumerate(X):
  #    y = dataset.angaccel_error[:, i]
  #    A += np.outer(row, row)
  #    b += row.dot(y)
  #    W = np.linalg.inv(A).dot(b)
  #  W = np.array(ws).T
  #  print(W)
  #  return W

  def test(self, xs, **kwargs):
    X = self._Xmat(xs, **kwargs)
    return X.dot(self.W)

class Linear:
  def __init__(self, *, include_constant=True, sig_n=1.0):
    self.include_constant = include_constant
    self.sig_n = sig_n

  def _Xmat(self, xs):
    if self.include_constant:
      return np.hstack((xs, np.ones((xs.shape[0], 1))))
    else:
      return xs

  def train(self, xs, ys):
    X = self._Xmat(xs)
    #Xinv = np.linalg.pinv(X)
    self.W = np.linalg.pinv(X.T.dot(X) + (self.sig_n ** 2) * np.eye(X.shape[1])).dot(X.T.dot(ys))

    #ws = []
    #for i in range(ys.shape[1]):
      #ws.append(Xinv.dot(ys[:, i]))

    #self.W = np.array(ws).T

    return np.mean(np.abs(ys - X.dot(self.W)), axis=0)

  def test(self, xs):
    X = self._Xmat(xs)
    return X.dot(self.W)

  def gradient(self, xs):
    # Remove the constant ones component.
    return self.W[:-1].T

  def save(self, outdir):
    np.savetxt(os.path.join(outdir, "linearweights.out"), self.W)

class StatsMan:
  def __init__(self):
    self.fs = {'mean' : np.mean, 'std' : np.std}
    self.data = {}
    for fname in self.fs:
      self.data[fname] = []

  def startrun(self):
    self.vals = []

  def add(self, v):
    self.vals.append(v)

  def finishrun(self):
    for fname, f in self.fs.items():
      self.data[fname].append(f(self.vals))

  def finish(self):
    for fname in self.fs:
      self.data[fname] = np.array(self.data[fname])
      setattr(self, fname, self.data[fname])

def cross2d(x):
  if len(x.shape) == 1:
    x = x[np.newaxis, :]

  x12 = x[:, 0] ** 2
  x22 = x[:, 1] ** 2

  f1 = np.exp(-10 * x12)
  f2 = np.exp(-50 * x22)
  f3 = 1.25 * np.exp(-5 * (x12 + x22))

  return np.maximum(np.maximum(f1, f2), f3)

def test():
  from python_utils.plotu import Plot3D, Subplot

  import matplotlib.pyplot as plt
  from matplotlib import cm
  fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

  N_test = 41
  xs = np.linspace(-1, 1, N_test)
  xs, ys = np.meshgrid(xs, xs)
  xs = xs.flatten()
  ys = ys.flatten()
  x_test = np.array((xs, ys)).T

  y_test = cross2d(x_test)

  N_train = 400
  sig_n = 0.01
  x_train = np.random.uniform(-1, 1, size=(N_train, 2))
  y_train = cross2d(x_train) + sig_n * np.random.normal(size=(N_train,))

  N_feats = 200
  sig_n = 0.1
  lengths = np.array((0.2, 0.10))
  model = SSGPR(N_feats=N_feats, sig_n=sig_n, lengths=lengths)
  train_error = model.train(x_train, y_train)
  y_pred = model.test(x_train).flatten()
  y_pred_test = model.test(x_test).flatten()

  test_error = np.mean(np.abs(y_pred_test - y_test))

  print("Train error is", train_error)
  print("Test error is", test_error)

  #ax.plot_surface(x_test[:, 0].reshape((N_test, N_test)), x_test[:, 1].reshape((N_test, N_test)), y_test.reshape((N_test, N_test)), linewidth=0, cmap=cm.coolwarm, antialiased=True)
  #ax.plot(x_train[:, 0], x_train[:, 1], zs=y_train, linewidth=0, marker='o', markersize=1)
  ax.plot(x_test[:, 0], x_test[:, 1], zs=y_test, linewidth=0, marker='o', markersize=1, label="True")
  ax.plot(x_test[:, 0], x_test[:, 1], zs=y_pred_test, linewidth=0, marker='o', markersize=1, label="Model")
  plt.legend()
  plt.show()

if __name__ == "__main__":
  import sys
  test()
  sys.exit(0)
  from python_utils.plotu import Subplot

  n = 50
  N = 200

  N_test = N // 10

  train_errors = StatsMan()
  test_errors = StatsMan()
  model_errors = StatsMan()
  nll_stats = StatsMan()

  var_mle = StatsMan()

  allstats = [
      train_errors,
      test_errors,
      model_errors,
      nll_stats,
      var_mle
  ]

  #sig_ns = np.linspace(0.0001, 0.1, 31)
  sig_ns = np.geomspace(1e-2, 4, 101)

  sig_n_true = 1.0

  N_trials = 50

  for sig_n in sig_ns:
    [stats.startrun() for stats in allstats]

    print(sig_n)
    for i in range(N_trials):
      #w = np.array((1., 2, 3))
      # Assuming w (model) drawn from multivariate normal with identity variance.
      w = np.random.normal(size=(n,))
      X = np.random.normal(size=(N, n))

      #Y = X.dot(w)
      #Yobs = Y + sig_n_true * np.random.normal(size=N)
      Y = X.dot(w) + sig_n_true * np.random.normal(size=N)

      what = np.linalg.inv(X.T.dot(X) + sig_n * sig_n * np.eye(n)).dot(X.T).dot(Y)
      Ypred = X.dot(what)

      Xtest = np.random.normal(size=(N_test, n))
      Ytest = Xtest.dot(w) + sig_n_true * np.random.normal(size=N_test)
      Ytest_pred = Xtest.dot(what)

      tes = Y - Ypred
      nll = 0.5 * (tes.dot(tes) / sig_n ** 2 + N * np.log(2 * np.pi * sig_n ** 2))
      # Add likelihood of prior...?
      nll += 0.5 * what.dot(what)

      mle_var = (1.0 / N) * tes.dot(tes)

      train_error = np.linalg.norm(tes) / N
      test_error = np.linalg.norm(Ytest - Ytest_pred) / N_test
      model_error = np.linalg.norm(w - what)

      train_errors.add(train_error)
      test_errors.add(test_error)
      model_errors.add(model_error)
      nll_stats.add(nll)

      var_mle.add(mle_var)
      #print("Train error:", train_error)
      #print("Test  error:", test_error)

      #print("True :", w)
      #print("Model:", what)

    [stats.finishrun() for stats in allstats]

  [stats.finish() for stats in allstats]

  std_params = dict(alpha=0.2)

  err_plot = Subplot("Errors", "sig_n")
  err_plot.add(sig_ns, train_errors.mean, label="Train Errors")
  err_plot.envelope(sig_ns, train_errors.mean, train_errors.std, **std_params)
  err_plot.add(sig_ns, test_errors.mean, label="Test Errors")
  err_plot.envelope(sig_ns, test_errors.mean, test_errors.std, **std_params)
  err_plot.legend()

  mep = Subplot("Model Errors", "sig_n")
  mep.add(sig_ns, model_errors.mean)
  mep.envelope(sig_ns, model_errors.mean, model_errors.std, **std_params)

  nllp = Subplot("Negative Log Likelihood", "sig_n")
  nllp.add(sig_ns, nll_stats.mean)
  nllp.envelope(sig_ns, nll_stats.mean, nll_stats.std, **std_params)
  nllp.axs[0].set_yscale('symlog')
  nllp.axs[0].set_xscale('log')

  mlevarp = Subplot("MLE of variance", "sig_n")
  mlevarp.add(sig_ns, var_mle.mean)
  mlevarp.envelope(sig_ns, var_mle.mean, var_mle.std, **std_params)

  err_plot.show()
