import numpy as np

# Solves linear system given by Tridiagonal Matrix
# Helper for calculating cubic splines
def tri_diag_solve(A, B, C, F):
    n = B.size
    assert A.ndim == B.ndim == C.ndim == F.ndim == 1 and (
        A.size == B.size == C.size == F.size == n
    ) #, (A.shape, B.shape, C.shape, F.shape)
    Bs, Fs = np.zeros_like(B), np.zeros_like(F)
    Bs[0], Fs[0] = B[0], F[0]
    for i in range(1, n):
        Bs[i] = B[i] - A[i] / Bs[i - 1] * C[i - 1]
        Fs[i] = F[i] - A[i] / Bs[i - 1] * Fs[i - 1]
    x = np.zeros_like(B)
    x[-1] = Fs[-1] / Bs[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (Fs[i] - C[i] * x[i + 1]) / Bs[i]
    return x
    
# Calculate cubic spline params
def calc_spline_params(x, y):
    a = y
    h = np.diff(x)
    c = np.concatenate((np.zeros((1,), dtype = y.dtype),
        np.append(tri_diag_solve(h[:-1], (h[:-1] + h[1:]) * 2, h[1:],
        ((a[2:] - a[1:-1]) / h[1:] - (a[1:-1] - a[:-2]) / h[:-1]) * 3), 0)))
    d = np.diff(c) / (3 * h)
    b = (a[1:] - a[:-1]) / h + (2 * c[1:] + c[:-1]) / 3 * h
    return a[1:], b, c[1:], d
    
# Spline value calculating function, given params and "x"
def func_spline(x, ix, x0, a, b, c, d):
    dx = x - x0[1:][ix]
    return a[ix] + (b[ix] + (c[ix] + d[ix] * dx) * dx) * dx

def searchsorted_merge(a, b, sort_b):
    ix = np.zeros((len(b),), dtype = np.int64)
    if sort_b:
        ib = np.argsort(b)
    pa, pb = 0, 0
    while pb < len(b):
        if pa < len(a) and a[pa] < (b[ib[pb]] if sort_b else b[pb]):
            pa += 1
        else:
            ix[pb] = pa
            pb += 1
    return ix
    
# Compute piece-wise spline function for "x" out of sorted "x0" points
def piece_wise_spline(x, x0, a, b, c, d):
    xsh = x.shape
    x = x.ravel()
    #ix = np.searchsorted(x0[1 : -1], x)
    ix = searchsorted_merge(x0[1 : -1], x, False)
    y = func_spline(x, ix, x0, a, b, c, d)
    y = y.reshape(xsh)
    return y
    
def test():
    import matplotlib.pyplot as plt, scipy.interpolate
    #from timerit import Timerit
    #Timerit._default_asciimode = True
    np.random.seed(0)
    
    def f(n):
        x = np.sort(np.random.uniform(0., n / 5 * np.pi, (n,))).astype(np.float64)
        return x, (np.sin(x) * 5 + np.sin(1 + 2.5 * x) * 3 + np.sin(2 + 0.5 * x) * 2).astype(np.float64)
    def spline_numba(x0, y0):
        a, b, c, d = calc_spline_params(x0, y0)
        return lambda x: piece_wise_spline(x, x0, a, b, c, d)
    def spline_scipy(x0, y0):
        f = scipy.interpolate.CubicSpline(x0, y0, bc_type = 'natural')
        return lambda x: f(x)

    # x0, y0 = f(50)
    x0=[-0.83,0.14,-1.09,1.09,-0.54,2.03,3.0]
    y0=[-2.03,-2.06,0.71,1.49,2.06,2.43,3.0]
    # timings()

    sorted_pairs = sorted(zip(x0, y0))  # Pairs (x[i], y[i]) are sorted based on x
    x_sorted, y_sorted = zip(*sorted_pairs)  # Unzip the sorted pairs

    # Convert back to lists if needed
    x0 = list(x_sorted)
    y0 = list(y_sorted)

    x0 = np.array(x0)
    y0 = np.array(y0)

    shift = 3
    x = np.linspace(x0[0], x0[-1], 1000, dtype = np.float64)
    ys = spline_scipy(x0, y0)(x)
    yn = spline_numba(x0, y0)(x)
    assert np.allclose(ys, yn), np.absolute(ys - yn).max()
    plt.plot(x0, y0, label = 'orig')
    plt.plot(x, ys, label = 'spline_scipy')
    plt.plot(x, yn, '-.', label = 'spline_numba')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    test()