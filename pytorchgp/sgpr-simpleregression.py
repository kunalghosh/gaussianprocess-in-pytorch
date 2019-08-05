from collections import namedtuple

import matplotlib
import numpy as np
import torch
from sklearn.cluster import KMeans

from torch.optim.lr_scheduler import StepLR
from pytorchgp.kernels import SquaredExp
from pytorchgp.models import SGPR, SGPRDGP
from pytorchgp.visualize import visualize1d

matplotlib.use('Agg')

# plt.ion()

consts = namedtuple("consts", "Ntrain Ntest noisestd")
consts.Ntrain = 500
consts.Ntest = 500
consts.noisestd = 0.3


# data generating function
def f(x):
    return 2 * np.exp(-.5 * x**2 / 1.3**2)


# def f(x):
#     return 2 * np.sin(x)

# training data
x = torch.linspace(-5, 5, consts.Ntrain).unsqueeze(-1)
y = f(x)
x_mean, x_std = x.mean(), x.std()
x = (x - x_mean) / x_std

y_noisy = y + torch.randn_like(y) * consts.noisestd
# y_noisy_mean, y_noisy_std = y_noisy.mean(), y_noisy.std()
# y_noisy = (y_noisy - y_noisy_mean) / y_noisy_std
# y = (y-y_noisy_mean) / y_noisy_std
x_test = torch.linspace(-5, 5, consts.Ntest).unsqueeze(-1)

x_test = (x_test - x_mean) / x_std
# plt.scatter(x.numpy(), y_noisy.numpy())
kernel = SquaredExp()
M = 20
kmeans = KMeans(n_clusters=M, random_state=0).fit(x)
z = torch.Tensor(kmeans.cluster_centers_)
# model = SGPRDGP(kernel=kernel, M=M, Z=z, D_out=1)
model = SGPR(kernel=kernel, M=M, Z=z, D_out=1)


# m, S = model(x, y)
# print(f"m {m.shape}")
# print(f"S {S.shape}")
# var = torch.diag(S)
# visualize1d(x, y, y_noisy, x_test, m, var, "GPR-prior-regression.pdf")

for p in model.named_parameters():
    print(p)

def train(model, x, y_noisy, n_iters=501, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # sched = StepLR(opt, step_size=1000, gamma=0.5)
    for epoch in range(n_iters):
        opt.zero_grad()
        # model.forward(x, y)
        # loss = -model.nll(x, y)
        loss = model.loss(x, y)
        loss.backward()
        opt.step()
        # sched.step()
        print(f"iter {epoch}: Marginal log likelihood: {-loss.item()}")
        print(f"Kernel lenghtscale {model.kernel.lengthscale.item()}")
        print(f"Kernel prefactor {model.kernel.prefactor.item()}")
        print(f"noise std {model.noisestd.item()}")
        if not epoch % 500:
            m, S = model.predict(x_test, full_cov=False)
            m = m.squeeze(0)
            S = S.unsqueeze(-1)
            visualize1d(x, y, y_noisy, x_test, m, S,
                        f"{model}-regression-{epoch}.pdf")


train(model, x, y_noisy, n_iters=14001, lr=1e-3)

for p in model.named_parameters():
    print(p)
# train(model, x, y_noisy, n_iters=2001, lr=1e-3)
# train(model, x, y_noisy, n_iters=2001, lr=1e-4)
m, S = model.predict(x_test, full_cov=False)
m = m.squeeze(0)
S = S.unsqueeze(-1)
print(f"m {m.shape}")
print(f"S {S.shape}")
print(S)
visualize1d(x, y, y_noisy, x_test, m, S, f"{model}-regression.pdf")
