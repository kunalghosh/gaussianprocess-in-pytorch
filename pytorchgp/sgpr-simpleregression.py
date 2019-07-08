import pdb
from collections import namedtuple

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from torch.nn.functional import softplus

from kernels import SquaredExp
from models import GPR, SGPR, GPRCholesky
from visualize import visualize1d

matplotlib.use('Agg')

# plt.ion()

consts = namedtuple("consts", "Ntrain Ntest noisestd")
consts.Ntrain = 500
consts.Ntest = 500
consts.noisestd = 0.3


# data generating function
def f(x):
    return 5 * np.exp(-.5 * x**2 / 1.3**2)


# def f(x):
#     return 2 * np.sin(x)

# training data
x = torch.linspace(-5, 5, consts.Ntrain).unsqueeze(-1)
y = f(x)

# x = (x - x.mean()) / x.std()

y_noisy = y + torch.randn_like(y) * consts.noisestd
x_test = torch.linspace(-5, 5, consts.Ntest).unsqueeze(-1)

# x_test = ( x_test - x.mean() ) / x.std()
# plt.scatter(x.numpy(), y_noisy.numpy())
kernel = SquaredExp()
M = 20
kmeans = KMeans(n_clusters=M, random_state=0).fit(x)
z = torch.Tensor(kmeans.cluster_centers_)
model = SGPR(kernel=kernel, M=M, Z=z, D_out=1)

# m, S = model(x, y)
# print(f"m {m.shape}")
# print(f"S {S.shape}")
# var = torch.diag(S)
# visualize1d(x, y, y_noisy, x_test, m, var, "GPR-prior-regression.pdf")


def train(model, x, y_noisy, n_iters=501, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(n_iters):
        opt.zero_grad()
        # model.forward(x, y)
        # loss = -model.nll(x, y)
        loss = model.loss(x, y)
        loss.backward()
        opt.step()
        print(f"iter {epoch}: Marginal log likelihood: {-loss.item()}")
        print(f"Kernel lenghtscale {model.kernel.lengthscale.item()}")
        print(f"Kernel prefactor {model.kernel.prefactor.item()}")
        print(f"noise std {model.noisestd.item()}")
        if not epoch % 500:
            m, S = model.predict(x_test, full_cov=False)
            visualize1d(x, y, y_noisy, x_test, m, S,
                        f"{model}-regression-{epoch}.pdf")


train(model, x, y_noisy, n_iters=5001, lr=5e-3)
m, S = model.predict(x_test, full_cov=False)
print(f"m {m.shape}")
print(f"S {S.shape}")
visualize1d(x, y, y_noisy, x_test, m, S, f"{model}-regression.pdf")
