import pdb
import torch
import torch.nn as nn
from torch.nn.init import uniform_
import math


class GPR(nn.Module):
    def __init__(self, kernel):
        super(GPR, self).__init__()
        self.kernel = kernel
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
        Kxx_noise_inv = Kxx_noise.inverse()
        self.Kxx = Kxx
        self.Kxx_noise = Kxx_noise
        self.Kxx_noise_inv = Kxx_noise_inv

        m = Kxx @ Kxx_noise_inv @ y
        S = Kxx - Kxx @ Kxx_noise_inv @ Kxx
        return m, S

    def nll(self, X, y):
        y = self.y
        m, S = self.predict(X, y)
        const = -0.5 * self.N * torch.log(2 * torch.tensor(math.pi))
        data_fit = -0.5 * y.t() @ self.Kxx_noise_inv @ y
        complexity = -torch.trace(torch.cholesky(self.Kxx_noise))
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

        m = Kt.transpose(-2, -1) @  Kxx_inv @ y
        S = Ktt - Kt.transpose(-2, -1) @ Kxx_inv @ Kt
        S = S.squeeze(0)
        if full_cov is False:
            S = torch.diag(S)
        return m, S
