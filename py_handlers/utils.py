import math
import numpy as np

def ppf(x):
    """
    Approximates the probability density function of a normal continuous random variable
    Check: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
    :return: Numpy Array
    :rtype: np.ndarray
    """
    a = -9
    b = 9
    v2 = math.sqrt(2)
    while (b - a) > 1e-9:
        c = (a + b)/2
        r = 0.5 + 0.5 * math.erf(c/v2)
        if r > x:
            b = c
        else:
            a = c
    return c

def inv_box_cox(y: np.ndarray, lambda_, biasadj=False):
    """
    Helper function for inverse Box-Cox transformation
    :return: Numpy Array
    :rtype: np.ndarray
    """
    if lambda_ == 0:
        return np.exp(y)
    elif lambda_ == 1:
        return y
    else:
        if biasadj:
            return (y * lambda_ + 1) ** (1 / lambda_)
        else:
            return (y ** lambda_ - 1) / lambda_
        
def ar_to_ma(ar_coeffs, ma_length):
    """
    Converts an array of AR coefficients to an array of MA coefficients with the required length
    :return: Numpy Array
    :rtype: np.ndarray
    """
    p = len(ar_coeffs)
    q = ma_length

    ma_coeffs = np.zeros(q)

    for j in range(q):
        for k in range(j+1):
            if k < p:
                ma_coeffs[j] += ar_coeffs[k] * ma_coeffs[j - k]

    return ma_coeffs
