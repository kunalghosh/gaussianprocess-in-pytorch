import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from utils import castargs_pytorch_to_numpy

@castargs_pytorch_to_numpy
def visualize1d(x=None, y=None, y_noisy=None, x_test=None, predictive_mean=None, predictive_var=None, filename=None):
    """
    Visualize the data and the GP predictions.

    It is not ok to have x and x_test to be None.
    It is ok to have either y or y_noisy to be None but not both.
    It is ok to have posterior variance to be None, but not posterior mean.
    """
    assert x is not None, "Please provide the training locations x."
    assert x_test is not None, "Please provide the test locations x_test."
    assert any([y is not None, y_noisy is not None]), "Please provide either true function evaluations, y or the noisy ones y_noisy."
    assert predictive_mean is not None, "Please provide atleast the prediction mean for the visualization."
    assert filename is not None, "Please provide a filename where to save the plot."

    plt.figure()
    if y is not None:
        plt.plot(x, y, 'r', label="True function")
    if y_noisy is not None:
        plt.scatter(x, y_noisy, c='b', marker=".", label="Noisy evaluations of true function")

    # plot predictive mean
    plt.plot(x_test, predictive_mean, label="Predictive mean.")
    # pre-compute predictive standard deviation.
    print(predictive_var.shape)
    pred_std = np.sqrt(predictive_var).flatten()
    predictive_mean = predictive_mean.flatten()

    plt.fill_between(x_test.flatten(), predictive_mean - 2 * pred_std,
                     predictive_mean + 2 * pred_std, color="b", alpha=0.2,
                     label="Two standard deviations")
    plt.legend(loc="best") # loc=best prevents legend from overlapping fig.
    plt.savefig(filename)
    plt.clf()
    plt.close()
    print(f"Saved figure {filename}")
