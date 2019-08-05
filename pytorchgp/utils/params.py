import numpy as np
import torch
import torch.nn as nn
from torch.distributions import transforms

MyModule = nn.Module

# def newgetter(self, name):
#     if '_parameters' in self.__dict__:
#         _parameters = self.__dict__['_parameters']
#         if name in _parameters:
#             if hasattr(_parameters[name], 'transform'):
#                 print("forwarded")
#                 return _parameters[name].transform(_parameters[name])
#             else:
#                 print("no forward yet")
#                 return _parameters[name]
#     if '_buffers' in self.__dict__:
#         _buffers = self.__dict__['_buffers']
#         if name in _buffers:
#             return _buffers[name]
#     if '_modules' in self.__dict__:
#         modules = self.__dict__['_modules']
#         if name in modules:
#             if hasattr(modules[name], 'transform'):
#                 print("modules forwarded")
#                 return modules[name].transform(modules[name])
#             else:
#                 print("modules no forward yet")
#                 return modules[name]
#     raise AttributeError("'{}' object has no attribute '{}'".format(
#         type(self).__name__, name))


class settings:
    pass


settings.torch_float = torch.float32
settings.numpy_float = np.float32


class LowerTriangular:
    def __init__(self, N, num_matrices=1, squeeze=False):
        self.N = N
        self.num_matrices = num_matrices  # store this for reconstruction.
        self.squeeze = squeeze

        if self.squeeze and (num_matrices != 1):
            raise ValueError(
                "cannot squeeze matrices unless num_matrices is 1.")

    def forward(self, x):
        fwd = np.zeros((self.num_matrices, self.N, self.N),
                       dtype=settings.numpy_float)
        indices = np.tril_indices(self.N, 0)
        z = np.zeros(len(indices[0])).astype(int)
        for i in range(self.num_matrices):
            fwd[(z + i, ) + indices] = x[i, :]
        return fwd.squeeze(axis=0) if self.squeeze else fwd

    def backward(self, y):
        if self.squeeze:
            y = y[None, :, :]
        ind = np.tril_indices(self.N)
        return np.vstack([y_i[ind] for y_i in y])

    def forward_tensor(self, x):
        fwd = torch.zeros((self.num_matrices, self.N, self.N),
                          dtype=settings.torch_float)
        indices = np.tril_indices(self.N, 0)
        z = np.zeros(len(indices[0])).astype(int)
        for i in range(self.num_matrices):
            fwd[(z + i, ) + indices] = x[i, :]
        return fwd.squeeze(dim=0) if self.squeeze else fwd

    def backward_tensor(self, y):
        if self.squeeze:
            y = y[None, :, :]
        ind = np.tril_indices(self.N)
        return torch.stack([y_i[ind] for y_i in y])


class Log1pe:
    def __init__(self, lower=1e-6):
        self._lower = lower

    def forward(self, x):
        return np.logaddexp(0, x) + torch.Tensor([self._lower])

    def forward_tensor(self, x):
        return torch.nn.functional.softplus(x) + self._lower

    def backward_tensor(self, y):
        ys = torch.max(
            y - self._lower,
            torch.as_tensor(torch.finfo(settings.torch_float).eps,
                            dtype=settings.torch_float))
        return ys + torch.log(-torch.expm1(-ys))

    def backward(self, y):
        ys = np.maximum(y - self._lower, np.finfo(settings.numpy_float).eps)
        return ys + np.log(-np.expm1(-ys))

    def _inverse(self, y):
        return self.backward_tensor(y)

    def __call__(self, y):
        return self.forward_tensor(y)


def newgetter(self, name):
    if '_parameters' in self.__dict__:
        _parameters = self.__dict__['_parameters']
        if name in _parameters:
            if hasattr(self, 'transform'):
                # print("forwarded")
                return self.transform(_parameters[name])
            else:
                # print("no forward yet")
                return _parameters[name]
    if '_buffers' in self.__dict__:
        _buffers = self.__dict__['_buffers']
        if name in _buffers:
            return _buffers[name]
    if '_modules' in self.__dict__:
        modules = self.__dict__['_modules']
        if name in modules:
            # if hasattr(self, 'transform'):
            if hasattr(modules[name], 'transform'):
                # print("modules forwarded")
                return modules[name].transform(modules[name])
            else:
                # print("modules no forward yet")
                return modules[name]
    raise AttributeError("'{}' object has no attribute '{}'".format(
        type(self).__name__, name))


# MyModule.__getattr__ = newgetter


class Param(MyModule, nn.Parameter):
    # def __init__(self, var, transform=transforms.identity_transform()):
    def __init__(self, var, transform=Log1pe()):
        super(Param, self).__init__()
        self.transform = transform
        # self.var_ = var
        # self.var = nn.Parameter(self.transform._inverse(self.var_))
        self.var_ = nn.Parameter(var)
        self.var = self.transform(self.var_)
        # self.var.transform = transform

    def __call__(self):
        return self.var

    def __repr__(self):
        return 'Parameter containing:\n' + super(Param, self).__repr__()
