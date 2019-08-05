import math

import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.nn import functional as F
from torch.nn.init import uniform_

from pytorchgp.models import SGPR
from pytorchgp.utils import eye
from pytorchgp.utils.params import Log1pe


class SGPRDGP(SGPR):
    def __init__(self, kernel=None, M=20, Z=None, D_in=1, D_out=1):
        """
        :Args
        kernel : The gaussian process kernel.
        M : The number of inducing points.
        D_out : The output dimension of the inducing points.
        """
        super(SGPRDGP, self).__init__(kernel, M, Z)

        self.transform = Log1pe()
        self._noisestd = nn.Parameter(
            self.transform._inverse(uniform_(torch.empty(1), 2, 3)))

        # Variational mean and cholesky variance
        self.qm = nn.Parameter(torch.empty(D_out, M, 1).normal_())
        self._qL = nn.Parameter(torch.empty(D_out, M, M).normal_().tril())

        if Z is None:
            raise AttributeError(
                "The inducing locations Z must be initialized.")
        else:
            self.z = nn.Parameter(Z)

        if D_out < self.z.shape[-1]:
            # then do PCA
            Z = PCA(n_components=D_out).fit_transform(Z)
            self.z = nn.Parameter(Z)
        else:
            # append empty dimensions to Z
            # F.pad pads with zeros to the (left, right)
            # so (0, 3) would mean pad right (i.e. last dimension)
            # with 3 columns of zeros.
            self.z = nn.Parameter(F.pad(self.z, (0, D_out - self.z.shape[-1])))

        self.M = M
        self.D_out = D_out

    @property
    def noisestd(self):
        return self.transform(self._noisestd)

    @property
    def qL(self):
        return self._qL

    def __str__(self):
        return "SGPR_DGP"

    def forward(self, x):

        N = x.shape[-2]
        self.N = N

        z = self.z
        Kxx = self.kernel(x, x)
        Kxx += eye(*Kxx.shape[:-1]) * 1e-6

        Kxz = self.kernel(x, z)

        Kzz = self.kernel(z, z)
        Kzz += eye(*Kzz.shape[:-1]) * 1e-6

        self.Kzz = Kzz

        A = Kxz @ Kzz.inverse()
        m = A @ self.qm
        S_q = self.qL.tril() @ self.qL.tril().transpose(-2, -1)
        S = Kxx + A @ (S_q - Kzz) @ A.transpose(-2, -1)
        return m, S

    def nll(self, x, y):
        if y.ndimension() < 3:
            y = y.reshape(self.D_out, -1, 1)
        # posterior mean and Variance
        m, S = self.forward(x)
        N = self.N
        logpdf = -0.5 * (N * torch.sum(torch.log(self.noisestd**2)) +
                         (y - m).transpose(-2, -1)
                         @ (torch.eye(N) * 1 / self.noisestd**2) @ (y - m) +
                         N * torch.log(2 * torch.tensor(math.pi))).squeeze()
        trace_term = -0.5 * torch.einsum('kii', S) / self.noisestd**2
        return logpdf + trace_term

    def elbo(self, x, y):
        nll = self.nll(x, y)
        kl = self.kl()
        print(f"nll {nll} kl {kl}")
        return nll - kl

    def loss(self, x, y):
        return -self.elbo(x, y)

    def kl(self):
        D_out = self.D_out
        return self.kl_multivariate_normal(self.qm, self.qL.tril(),
                                           torch.zeros(D_out, self.M, 1),
                                           eye(D_out, self.M))

    def kl_multivariate_normal(self, m0, L0, m1, S1):
        """
        KL between a gaussian N(m,S) and N(,I)
        KL(q||p)
        """
        S0 = L0 @ L0.transpose(-2, -1)
        M = self.M
        print(f"k is {M}")
        trace_term = torch.einsum('kii', S1.inverse() @ S0)
        mean_term = (
            (m1 - m0).transpose(-2, -1) @ S1.inverse() @ (m1 - m0)).squeeze()
        print(S1.shape)
        log_det_S1 = torch.sum(
            torch.log(torch.stack([torch.diag(_)
                                   for _ in S1])))  # S1 is Identity
        log_det_S0 = torch.sum(
            torch.log(torch.stack([torch.diag(_)**2 for _ in L0])))
        log_det_term = log_det_S1 - log_det_S0
        print(f"logdet S0 = {log_det_S0} S1 = {log_det_S1}")
        print(
            f"trace term {trace_term} mean_term {mean_term} logdet {log_det_term}"
        )
        return 0.5 * (trace_term + mean_term - M + log_det_term).squeeze()

    def predict(self, xtest, full_cov=True):
        m, S = self.forward(xtest)
        S = S + eye(*S.shape[:-1]) * self.noisestd
        if not full_cov:
            S = torch.einsum('kii->i', S)
        return m, S
