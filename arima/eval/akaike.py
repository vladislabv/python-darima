import math 
import numpy as np

def aicc(n, k, data, data_mu, data_sigma):
    """
    Calculate the Corrected Akaike Information Criterion (AICC).

    Parameters:
    n (int): Sample size.
    k (int): Number of model parameters.
    log_likelihood (float): Log-likelihood of the model.

    Returns:
    float: AICC value.
    """
    likelihood = np.prod(1 / (data_sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((data - data_mu) / data_sigma)**2))
    return -2 * math.log(likelihood) + 2 * k + 2 * k * (k + 1) / (n - k - 1)


def aic(n, k, log_likelihood):
    pass


def bic(n, k, log_likelihood):
    pass