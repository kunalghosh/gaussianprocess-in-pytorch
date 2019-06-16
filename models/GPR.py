import pdb
import torch
import torch.nn as nn
from torch.nn.init import uniform_
import math


class GPR(nn.Module):
    def __init__(self, kernel):
        super(GPR, self).__init__()
        self.kernel = kernel
        # self.noisestd = nn.Parameter(torch.tensor(0.2))
        self.noisestd = self.noise_std = nn.Parameter(
            torch.exp(uniform_(torch.empty(1), -3., 0.)))

    def forward(self, X, y):
        """
        Returns posterior mean and variance
        """
        self.x, self.y, self.N = X, y, X.shape[-2]

        jitter = 1e-5 * torch.eye(self.N)
        Kxx = self.kernel(X, X) + jitter
        Kxx_noise = Kxx + self.noisestd**2 * torch.eye(self.N)

        self.L = torch.cholesky(Kxx_noise)
        a, _ = torch.solve(self.y.unsqueeze(0), self.L.unsqueeze(0))
        alpha, _ = torch.solve(a, self.L.unsqueeze(0))
        self.alpha = alpha.squeeze(0)
        # self.Kxx_inv = Kxx_noise.inverse()
        m = Kxx @ alpha
        v, _ = torch.solve(Kxx.unsqueeze(0), self.L.unsqueeze(0))
        S = Kxx + jitter - v.transpose(
            -2, -1) @ v  # Kxx.transpose(-2, -1) @ self.Kxx_inv @ Kxx
        return m, S

    def nll(self):
        y = self.y
        const = -0.5 * self.N * torch.log(2 * torch.tensor(math.pi))
        data_fit = -0.5 * y.t() @ self.alpha  # self.Kxx_inv @ y
        complexity = - torch.sum(torch.log(torch.diag(self.L)))
        # complexity = -torch.trace(self.L)
        print(
            f"nll terms datafit : {data_fit.detach().numpy()},\n complexity : {complexity}"
        )
        return data_fit + complexity + const

    def predict(self, xtest, full_cov=True):
        x = self.x
        y = self.y
        Kt = self.kernel(x, xtest)
        Ktt = self.kernel(xtest, xtest)
        Kxx = self.kernel(x, x)
        Kxx_inv = (Kxx + self.noisestd**2 * torch.eye(self.N)).inverse()
        m = Kt.transpose(-2, -1) @ Kxx_inv @ y
        S = Ktt - Kt.transpose(-2, -1) @ Kxx_inv @ Kt
        if full_cov is False:
            S = torch.diag(S)
        return m, S
