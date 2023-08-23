import math
import numpy as np


def kpss_test(x, null = ("Level", "Trend"), lshort = True, alpha = 0.05):
    if not isinstance(x, np.array):
        raise TypeError("x must be a numpy array")
    
    n = len(x)

    if null == "Trend":
        t = np.arange(1, n + 1)
        e = np.ndarray.astype(get_lm_residuals(x, t), dtype=np.float64)
        table = np.array([0.216, 0.176, 0.146, 0.119], dtype=np.float64)
    elif null == "Level":
        e = np.ndarray.astype(get_lm_residuals(x, 1), dtype=np.float64)
        table = np.array([0.739, 0.574, 0.463, 0.347], dtype=np.float64)
    else:
        raise ValueError("null must be either 'Level' or 'Trend'")

    tablep = np.array([0.01, 0.025, 0.05, 0.1], dtype=np.float64)
    s = np.cumsum(e, axis=0, dtype=np.float64)
    eta = sum(s^2)/(n^2)
    s2 = float(sum(e^2)/n)
    if lshort:
        l = math.ceil(4 * (n/100)^0.25)
    
    l = math.ceil(12 * (n/100)^0.25)
    

    s2 = tseries_pp_sum(e, n, l, s2)
    stat = eta/s2
    pval = approx(table, tablep, stat)

    return pval <= alpha 


def tseries_pp_sum(u, n, l, sum_value):
    tmp1 = 0.0
    for i in range(1, l + 1):
        tmp2 = 0.0
        for j in range(i, n):
            tmp2 += u[j] * u[j - i]
        tmp2 *= 1.0 - (i / (l + 1.0))
        tmp1 += tmp2
    tmp1 /= n
    tmp1 *= 2.0
    sum_value[0] += tmp1
    return sum_value


def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)
 
    # mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)
 
    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
 
    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
 
    return (b_0, b_1)


def calc_residuals(x, y, b):
    if not isinstance(x, np.array) and isinstance(y, np.array):
        raise TypeError("x and y must be numpy arrays")
    if not len(x) == len(y):
        raise ValueError("x and y must have the same length")
    return y - b[0] - b[1]*x


def get_lm_residuals(x, y):
    b = estimate_coef(x, y)
    return calc_residuals(x, y, b)


def approx(x, y, xout, method="linear", ties="mean"):
    """
    Custom implementation of linear interpolation for approximating values.

    Parameters:
    x (array-like): The x-coordinates of the data points.
    y (array-like): The y-coordinates of the data points.
    xout (array-like): The x-coordinates where values should be approximated.
    method (str, optional): The interpolation method ("linear" or "constant").
                           Defaults to "linear".
    ties (str, optional): How to handle ties in x values ("mean", "min", "max").
                          Defaults to "mean".

    Returns:
    array: The interpolated/approximated values at xout.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    xout = np.asarray(xout)
    
    if method == "linear":
        indices = np.argsort(x)
        x_sorted = x[indices]
        y_sorted = y[indices]
        
        if ties == "mean":
            unique_x = np.unique(x_sorted)
            interpolated_values = np.interp(xout, unique_x, np.interp(unique_x, x_sorted, y_sorted))
        elif ties == "min":
            interpolated_values = np.interp(xout, x_sorted, y_sorted, left=y_sorted[0])
        elif ties == "max":
            interpolated_values = np.interp(xout, x_sorted, y_sorted, right=y_sorted[-1])
        else:
            raise ValueError("Invalid 'ties' parameter value. Use 'mean', 'min', or 'max'.")
    elif method == "constant":
        interpolated_values = np.interp(xout, x, y, left=y[0], right=y[-1])
    else:
        raise ValueError("Invalid 'method' parameter value. Use 'linear' or 'constant'.")
    
    return interpolated_values
