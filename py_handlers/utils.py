import math

def ppf(x):
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