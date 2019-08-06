import pdb
import math

import torch
import torch.nn as nn
from torch.nn.init import uniform_

from pytorchgp.models import SGPRDGP
from pytorchgp.utils import eye
from pytorchgp.utils.params import Log1pe


class DGP(nn.Module):
    def __init__(self, kernel=None, M=20, Z=None):
        """
        :Args
        kernel : The gaussian process kernel.
        M : List of inducing points one per DGP layer
        Z : List of inducing locations for the first layer.
        """
        super(DGP, self).__init__()
        self.transform = Log1pe()
        self.layers = nn.ModuleList(
            [SGPRDGP(kernel, M, Z, D_out=1),
             SGPRDGP(kernel, M, Z, D_out=1)])

    def __str__(self):
        return "DGP"

    def forward(self, x):
        """
        In the case of DGP just returns the function values.
        """
        f = x
        for layer in self.layers:
            m, S = layer.forward(f)
            L = torch.cholesky(S + eye(*S.shape[:-1]) * 1e-3)
            eps = torch.empty(*m.shape).normal_()
            f = m + L.transpose(-2, -1) @ eps
        return f

    def nll(self, x, y, n_samples=1):
        if y.ndimension() < 3:
            y = y.reshape(self.layers[-1].D_out, -1, 1)
        N = y.shape[-2]
        nll = 0
        noisestd = self.layers[-1].noisestd
        for sample in range(n_samples):
            # posterior function value
            fL = self.forward(x)
            logpdf = -0.5 * (
                N * torch.sum(torch.log(noisestd**2)) +
                (y - fL).transpose(-2, -1)
                @ (torch.eye(N) * 1 / noisestd**2) @ (y - fL) +
                N * torch.log(2 * torch.tensor(math.pi))).squeeze()
            nll += logpdf

        print(f"nll shape {nll.shape} must be a scalar")
        return nll / n_samples

    def elbo(self, x, y):
        nll = self.nll(x, y)
        kl = self.kl()
        print(f"nll {nll} kl {kl}")
        return nll - kl

    def loss(self, x, y):
        return -self.elbo(x, y)

    def kl(self):
        kl = 0
        for layer in self.layers:
            kl += layer.kl().sum()
        return kl

    def predict(self, xtest, n_samples=30, full_cov=True):
        f = 0
        for sample in range(n_samples):
            f += self.forward(xtest)
        return f.squeeze(0) / n_samples
