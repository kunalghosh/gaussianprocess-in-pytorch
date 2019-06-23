import pdb

import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.init import uniform_

from models import GPR


class SGPR(GPR):
    def __init__(self, kernel, M=20, D_out=1):
        """
        :Args
        kernel : The gaussian process kernel.
        M : The number of inducing points.
        D_out : The output dimension of the inducing points.
        """
        super(SGPR, self).__init__(kernel)

        self.noisestd = nn.Parameter(
            torch.exp(uniform_(torch.empty(1), 0., 1.)))
        # Variational mean and cholesky variance
        self.qm = nn.Parameter(torch.empty(M, 1).uniform_(-3, 1))
        self.qL = nn.Parameter(torch.empty(M, M).normal_()).tril()
        self.z = None
        self.M = M

    def __str__(self):
        return "SGPR"

    def forward(self, x):
        if self.z is None:
            # self.z = torch.clone(x[:self.M, :])
            kmeans = KMeans(n_clusters=self.M, random_state=0).fit(x)
            self.z = torch.Tensor(kmeans.cluster_centers_)

        z = self.z
        Kxx = self.kernel(x, x)
        Kxz = self.kernel(x, z)
        Kzz = self.kernel(z, z)

        self.Kzz = Kzz

        A = Kxz @ Kzz.inverse()
        m = A @ self.qm
        S_q = self.qL.t() @ self.qL
        S = Kxx + A @ (S_q - Kzz) @ A.transpose(-2, -1)
        return m, S

    def nll(self, x, y):
        N = x.shape[-2]
        # posterior mean and Variance
        m, S = self.forward(x)
        mvn = MultivariateNormal(m, torch.eye(N) * self.noisestd**2)
        logpdf = mvn.log_prob(y).sum()
        trace_term = -0.5 * (1 / self.noisestd**2) * torch.trace(S)
        return logpdf + trace_term

    def elbo(self, x, y):
        return (self.nll(x, y) + self.kl())

    def loss(self, x, y):
        return -self.elbo(x, y)

    def kl(self):
        return self.kl_multivariate_normal(self.qm,
                                           self.qL.t() @ self.qL,
                                           torch.zeros(self.M, 1), self.Kzz)

    def kl_multivariate_normal(self, m0, S0, m1, S1):
        """
        KL between a gaussian N(m,S) and N(,I)
        KL(q||p)
        """
        k, _ = m0.shape
        trace_term = torch.trace(S1.inverse() @ S0)
        mean_term = (m1 - m0).t() @ S1.inverse() @ (m1 - m0)
        log_det_term = torch.log(torch.det(S1)) - torch.log(torch.det(S0))
        return 0.5 * (trace_term + mean_term - k + log_det_term)

    def predict(self, xtest, full_cov=True):
        m, S = self.forward(xtest)
        if not full_cov:
            S = torch.diag(S)
        return m, S
