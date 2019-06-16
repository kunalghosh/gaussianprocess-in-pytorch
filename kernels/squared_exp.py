import torch
import torch.nn as nn
from torch.nn.init import uniform_


class SquaredExp(nn.Module):
    def __init__(self):
        super(SquaredExp, self).__init__()
        # self.lengthscale = nn.Parameter(torch.tensor(0.1))
        # self.prefactor = nn.Parameter(torch.tensor(1.))  # output std_dev
        self.lengthscale = nn.Parameter(torch.exp(uniform_(torch.empty(1, 1), -3, 0)))
        self.prefactor = nn.Parameter(torch.exp(uniform_(torch.empty(1, 1), -1, 1)))

    def forward(self, X1, X2=None):
        if X2 is None:
            X2 = X1

        if X1.ndimension() < 3:
            X1 = X1.unsqueeze(0)
        if X2.ndimension() < 3:
            X2 = X2.unsqueeze(0)

        X1 = X1.unsqueeze(-2)
        X2 = X2.unsqueeze(-2)
        l2 = ((X1 - X2.transpose(1, 2))**2).sum(dim=-1)
        exp_term = torch.exp(-0.5 * l2 / self.lengthscale**2)
        return (self.prefactor**2 * exp_term).squeeze()
