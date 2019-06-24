import pdb
import math

import torch
import torch.nn as nn
from torch.nn.init import uniform_

from models import SGPR
from utils import eye


class SGPRDGP(SGPR):
    def __init__(self, kernel=None, M=20, Z=None, D_in=1, D_out=1):
        """
        :Args
        kernel : The gaussian process kernel.
        M : The number of inducing points.
        D_out : The output dimension of the inducing points.
        """
        super(SGPRDGP, self).__init__(kernel, M, Z)

        self.noisestd = nn.Parameter(
            torch.exp(uniform_(torch.empty(1), 0., 1.)))
        # Variational mean and cholesky variance
        self.qm = nn.Parameter(torch.empty(D_out, M, 1).normal_())
        self.qL = nn.Parameter(torch.empty(D_out, M, M).normal_()).tril()
        if Z is None:
            self.z = None
        else:
            self.z = nn.Parameter(Z)
        self.M = M
        self.D_out = D_out

    def __str__(self):
        return "SGPR_DGP"

    def forward(self, x):
        # if self.z is None:
        #     # self.z = torch.clone(x[:self.M, :])
        #     kmeans = KMeans(n_clusters=self.M, random_state=0).fit(x)
        #     self.z = nn.Parameter(torch.Tensor(kmeans.cluster_centers_))

        N = x.shape[-2]
        self.N = N
        M = self.M

        z = self.z
        Kxx = self.kernel(x, x)  # + eye(*Kxx.shape) * 1e-3
        Kxz = self.kernel(x, z)
        Kzz = self.kernel(z, z)  # + eye(*Kzz.shape) * 1e-3

        self.Kzz = Kzz

        A = Kxz @ Kzz.inverse()
        m = A @ self.qm
        S_q = self.qL @ self.qL.transpose(-2, -1)
        S = Kxx + A @ (S_q - Kzz) @ A.transpose(-2, -1)
        return m, S

    def nll(self, x, y):
        if y.ndimension() < 3:
            y = y.reshape(self.D_out, -1, 1)
        # posterior mean and Variance
        m, S = self.forward(x)
        # mvn = MultivariateNormal(m, torch.eye(N) * self.noisestd**2)
        # logpdf = mvn.log_prob(y).sum()
        N = self.N
        # k, _ = m.shape
        logpdf = -0.5 * (
            torch.sum(torch.log(self.noisestd**2)) +
            (y - m).transpose(-2, -1) @ (torch.eye(N) * 1 / self.noisestd**2) @ (y - m) +
            N * torch.log(2 * torch.tensor(math.pi))).squeeze()
        trace_term = -0.5 * (1 / self.noisestd**2) * torch.einsum('kii', S)
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
        return self.kl_multivariate_normal(self.qm, self.qL @ self.qL.transpose(-2, -1),
                                           torch.zeros(D_out, self.M, 1),
                                           eye(D_out, self.M))

    def kl_multivariate_normal(self, m0, S0, m1, S1):
        """
        KL between a gaussian N(m,S) and N(,I)
        KL(q||p)
        """
        M = self.M
        print(f"k is {M}")
        trace_term = torch.einsum('kii', S1.inverse() @ S0)
        mean_term = ( (m1 - m0).transpose(-2, -1) @ S1.inverse() @ (m1 - m0) ).squeeze()
        print(S1.shape)
        log_det_S1 = torch.stack([torch.logdet(_) for _ in S1 + 1e-5 * torch.eye(M) * torch.randn(M)**2])
        log_det_S0 = torch.stack([torch.logdet(_) for _ in S0 + 1e-5 * torch.eye(M) * torch.randn(M)**2])
        log_det_term = log_det_S1 - log_det_S0
        print(f"logdet S0 = {log_det_S0} S1 = {log_det_S1}")
        print(
            f"trace term {trace_term} mean_term {mean_term} logdet {log_det_term}"
        )
        return 0.5 * (trace_term + mean_term - M + log_det_term).squeeze()

    def predict(self, xtest, full_cov=True):
        N = self.N
        m, S = self.forward(xtest)
        if not full_cov:
            S = torch.diag(S)
        return m, S
