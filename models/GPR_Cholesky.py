import pdb
import torch
import torch.nn as nn
from torch.nn.init import uniform_
import math
import torch.distributions.constraints as constraints
from torch.nn.functional import softplus


class GPRCholesky(nn.Module):
    def __init__(self, kernel):
        super(GPRCholesky, self).__init__()
        self.kernel = kernel
        self.noisestd = nn.Parameter(torch.exp(uniform_(torch.empty(1), 0., 1.)))

    def __str__(self):
        return "GPRCholesky"

    def forward(self, X, y):
        """
        Returns posterior mean and variance
        """
        self.x, self.y, self.N = X, y, X.shape[-2]

        self.Kxx = self.kernel(X, X) + torch.eye(self.N) * 1e-5
        jitter = torch.eye(self.N) * self.noisestd ** 2
        L = torch.cholesky(self.Kxx + jitter)
        a, _ = torch.solve(y.unsqueeze(0), L.unsqueeze(0))
        alpha, _ = torch.solve(a, L.t().unsqueeze(0))
        m = alpha.squeeze(0)
        S = L
        return m, S

    def nll(self, X, y):
        alpha, L = self.forward(X, y)
        const = -0.5 * self.N * torch.log(2 * torch.tensor(math.pi))
        data_fit = -0.5 * y.t() @ alpha
        complexity = -torch.trace(L)
        print(
            f"nll terms datafit : {data_fit.detach().numpy()},\n complexity : {complexity}"
        )
        return data_fit + complexity + const

    def loss(self, x, y):
        return -self.nll(x, y)

    def predict(self, xtest, full_cov=True):
        x = self.x
        y = self.y
        alpha, L = self.forward(x, y)

        Kt = self.kernel(x, xtest)
        Ktt = self.kernel(xtest, xtest)

        m = Kt.transpose(-2, -1) @ alpha
        v, _ = torch.solve(Kt.unsqueeze(0), L.unsqueeze(0))
        v = v.unsqueeze(0)
        S = Ktt - v.transpose(-2, -1) @ v
        S = S.squeeze()
        print(m.shape, S.shape)
        if full_cov is False:
            S = torch.diag(S)
        return m, S
