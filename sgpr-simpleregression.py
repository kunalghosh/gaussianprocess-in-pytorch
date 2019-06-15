import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import torch
from collections import namedtuple
import numpy as np

plt.ion()

consts = namedtuple("consts", "Ntrain Ntest noisestd")
consts.Ntrain = 500
consts.Ntest = 500
consts.noisestd = 0.3


# data generating function
def f(x):
    return 5 * np.exp(-.5 * x**2 / 1.3**2)


# training data
x = torch.linspace(-5, 5, consts.Ntrain).unsqueeze(0)
y = f(x)

y_noisy = y + torch.randn_like(y) * consts.noisestd
x_test = torch.linspace(-5, 5, consts.Ntest).unsqueeze(0)

plt.scatter(x.numpy(), y_noisy.numpy())
