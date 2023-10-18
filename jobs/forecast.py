import numpy as np
from scipy.stats import norm


def predict_ar(Theta, sigma2, x, n_ahead=1, se_fit=True):
    # Check arguments
    if n_ahead < 1:
        raise ValueError("'n_ahead' must be at least 1")
    if x is None:
        raise ValueError("Argument 'x' is missing")
    if not isinstance(x, np.ndarray):
        raise ValueError("'x' must be a NumPy array")

    h = n_ahead
    n = len(x)
    st = x.index[1]  # Assuming x is a pandas Series with DateTimeIndex
    coef = Theta
    p = len(coef) - 2  # AR order
    X = np.column_stack([np.ones(n), np.arange(1, n + 1)] + [x.shift(i) for i in range(1, p + 1)])

    # Fitted values
    fits = np.dot(X, coef).flatten()
    res = x - fits

    # Forecasts
    y = np.append(x, np.zeros(h))
    for i in range(h):
        y[n + i] = np.sum(coef * np.concatenate(([1, n + i], y[n + i - np.arange(1, p + 1)])))
    pred = y[n + np.arange(h)]

    # Standard errors
    if se_fit:
        psi = np.polymul(np.array([1]), coef[-p:])
        vars = np.cumsum(np.polymul(np.array([1]), psi ** 2))
        se = np.sqrt(sigma2 * vars[-h:])
        result = {"fitted": fits, "residuals": res, "pred": pred, "se": se}
    else:
        result = {"fitted": fits, "residuals": res, "pred": pred}

    return result


def forecast_darima(Theta, sigma2, x, period, h=1, level=(80, 95), fan=False, lambda_=None, biasadj=False):
    # Check and prepare data
    x = x.asfreq(freq=period)
    pred = predict_ar(Theta, sigma2, x, n_ahead=h)

    if fan:
        level = list(range(51, 100, 3))
    else:
        level = [level] if isinstance(level, int) else level

    lower = np.empty((h, len(level)))
    upper = np.empty((h, len(level)))
    for i, conf_level in enumerate(level):
        qq = norm.ppf(0.5 * (1 + conf_level / 100))
        lower[:, i] = pred["pred"] - qq * pred["se"]
        upper[:, i] = pred["pred"] + qq * pred["se"]

    # Convert the results
    result = {
        "level": level,
        "mean": pred["pred"],
        "se": pred["se"],
        "lower": lower,
        "upper": upper,
        "fitted": pred["fitted"],
        "residuals": pred["residuals"],
    }

    return result

