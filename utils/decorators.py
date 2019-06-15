import torch

def castargs_pytorch_to_numpy(func):
    print("Inside decorator")
    def wrapper_func(*args, **kwargs):
        new_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                new_args.append(arg.detach().numpy())
            else:
                new_args.append(arg)
        # don't do anything with keyword args yet
        # for key, val in kwargs:
        #     print(key, val)
        return func(*new_args, **kwargs)
    return wrapper_func