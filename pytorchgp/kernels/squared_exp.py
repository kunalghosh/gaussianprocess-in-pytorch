import torch
import torch.nn as nn
from torch.nn.init import uniform_

from pytorchgp.utils.params import Log1pe

# class SquaredExp(nn.Module):
#     def __init__(self):
#         super(SquaredExp, self).__init__()
#         # self.lengthscale = nn.Parameter(torch.tensor(0.1))
#         # self.prefactor = nn.Parameter(torch.tensor(1.))  # output std_dev
#         self.lengthscale = nn.Parameter(torch.exp(uniform_(torch.empty(1, 1), -0.5, 1.0))) # 1, 2.5
#         self.prefactor = nn.Parameter(torch.exp(uniform_(torch.empty(1, 1), 0.5, 1.5)))

#     def forward(self, X1, X2=None):
#         if X2 is None:
#             X2 = X1

#         if X1.ndimension() < 3:
#             X1 = X1.unsqueeze(0)
#         if X2.ndimension() < 3:
#             X2 = X2.unsqueeze(0)

#         X1 = X1.unsqueeze(-2)
#         X2 = X2.unsqueeze(-2)
#         l2 = ((X1 - X2.transpose(1, 2))**2).sum(dim=-1)
#         exp_term = torch.exp(-0.5 * l2 / self.lengthscale**2)
#         return (self.prefactor**2 * exp_term).squeeze()


class SquaredExp(nn.Module):
    def __init__(self):
        super(SquaredExp, self).__init__()
        # self.lengthscale = nn.Parameter(torch.tensor(0.1))
        # self.prefactor = nn.Parameter(torch.tensor(1.))  # output std_dev

        # self.lengthscale = Param(uniform_(torch.empty(1, 1), -0.5, 1.0))
        # self.prefactor = Param(uniform_(torch.empty(1, 1), 0.5, 1.5))

        self.transform = Log1pe()
        self._lengthscale = nn.Parameter(self.transform._inverse(uniform_(torch.empty(1, 1), 0.001, 0.05)))
        self._prefactor = nn.Parameter(self.transform._inverse(uniform_(torch.empty(1, 1), 0.1, 0.5)))

    @property
    def lengthscale(self):
        return self.transform(self._lengthscale)

    @property
    def prefactor(self):
        return self.transform(self._prefactor)

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


# class SquaredExp(nn.Module):
#     def __init__(self, D_out=20):
#         super(SquaredExp, self).__init__()
#         self.lengthscale = nn.Parameter(uniform_(torch.empty(D_out, 1, 1), -1, -0.5))
#         self.prefactor = nn.Parameter(uniform_(torch.empty(D_out, 1, 1), 0.5, 1.5))

#     def forward(self, X, Z):
#         gamma = torch.exp(self.lengthscale)
#         if Z is None:
#             Z = X
#         scaled_X = X / gamma ** 2
#         scaled_Z = Z / gamma ** 2
#         X2 = (scaled_X ** 2).sum(-1, keepdim = True)
#         Z2 = (scaled_Z ** 2).sum(-1, keepdim = True)
#         XZ = scaled_X.matmul(scaled_Z.transpose(1, 2))
#         r2 = (X2 - 2 * XZ + Z2.transpose(1, 2)).clamp(min = 1e-6)
#         return self.prefactor ** 2 * torch.exp(-0.5 * r2)
