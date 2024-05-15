import numpy
from numpy import array, sqrt, nan


class ReScaler():

    '''
    Class to de-bias the observed prevalence by affine transformation. The day
    when the nonresprate will depend on the status this class will take this
    into account. Observation is meant to be a list, each entry being the rate of
    positive tests. If smoothin has been set, then obsetvation is a list of lists.

    If smoothing has been set, then the estimates are further manipulated
    the problem is, given arrays T[k],Y[k] for k in range(n),
    and given a function w then one wants to minimize the cost

    m0[n], m1[n] = argmin(sum(w[n,k] * (Y[k] - m0 - m1 * T[k])**2 for k in range(n + 1)))
    w[n,k] = obs[n,k] * (1 - obs[n,k]) * num[n,k] * smoothing(T[n] - T[k]).

    This corresponds to solve
        m0 * a + m1 * b = A
        m0 * b + m1 * c = B
    with
        a = sum(w(T[n] - T[k]) for k in range(n + 1))
        b = sum(w(T[n] - T[k]) * T[k] for k in range(n + 1))
        c = sum(w(T[n] - T[k]) * T[k]**2 for k in range(n + 1))
        A = sum(w(T[n] - T[k]) * Y[k] for k in range(n + 1))
        B = sum(w(T[n] - T[k]) * Y[k] * T[k] for k in range(n + 1))
    that is
        m0 = (A * c - B * b) / (a * c - b**2)
        m1 = (B * a - A * b) / (a * c - b**2).

    This is a weighted least square difference probrem where the solution
    is a line approximating the prevalence of the infection focusing
    on the more recent estimates, that are more reliable. However it
    is not clear which smoothing could be better for this purpose.
    As well the resulting estimate can be negative, which might be undesirable.

    '''

    def __init__(self, false_positive=0.0, false_negative=0.0, smoothing=None):
        if false_negative + false_positive == 1.0:
            raise Exception('useless test')

        self.false_positive = false_positive
        self.false_negative = false_negative
        self.smoothing, self.degree, self.times = smoothing
        if not self.degree in [0, 1, 2]:
            raise Exception('invalid deg.')

    def __call__(self, observation):
        # times is longer than observation
        if self.smoothing is None:
            pos, neg, _ = observation
            estimates = array(pos, dtype=numpy.double) / (array(pos, dtype=numpy.double) + array(neg, dtype=numpy.double))
            estimates -= self.false_positive
            estimates /= 1.0 - self.false_negative - self.false_positive
            return estimates
        elif self.times is None:
            raise Exception('please insert times of sampling')
        smooth_estimate = []
        times = array(self.times)
        for n, (pos, neg, _) in enumerate(observation):
            for time in range(times[n], times[n + 1]):
                try:
                    assert self.degree == 2
                    obs = array(pos) / (array(pos) + array(neg))
                    estimates = (obs - self.false_positive) / (1 - self.false_negative - self.false_positive)
                    temp = array([self.smoothing(time - times[k]) for k in range(n + 1)], dtype=numpy.double)
                    a = temp.sum()
                    b = (temp * times[0: n + 1]).sum()
                    c = (temp * (times[0: n + 1]**2)).sum()
                    d = (temp * (times[0: n + 1]**3)).sum()
                    e = (temp * (times[0: n + 1]**3)).sum()
                    a_0 = c * e - d**2
                    b_0 = c * d - b * e
                    c_0 = b * d - c**2
                    d_0 = b * c - a * d
                    e_0 = a * c - b**2
                    det = a * c * e + b* c * d + c * b * d - a * d**2 - e * b**2 - c**3
                    assert det != 0
                    A = (temp * estimates[0: n + 1]).sum()
                    B = (temp * times[0: n + 1] * estimates[0: n + 1]).sum()
                    C = (temp * times[0: n + 1]**2 * estimates[0: n + 1]).sum()
                    m0 = (a_0 * A + b_0 * B + c_0 * C) / det
                    m1 = (b_0 * A + c_0 * B + d_0 * C) / det
                    m2 = (c_0 * A + d_0 * B + e_0 * C) / det
                    y = m0 + m1 * time + m2 * time**2
                    assert (0.0 <= y <= 1.0)
                    smooth_estimate.append(y)
                except:
                    try:
                        assert self.degree >= 1
                        obs = array(pos, dtype=numpy.double) / (array(pos, dtype=numpy.double) + array(neg,  dtype=numpy.double))
                        estimates = (obs - self.false_positive) / (1.0 - self.false_negative - self.false_positive)
                        temp = array([self.smoothing(time - times[k]) for k in range(n + 1)], dtype=numpy.double)
                        a = temp.sum()
                        b = (temp * times[0: n + 1]).sum()
                        c = (temp * (times[0: n + 1]**2)).sum()
                        assert a * c != b**2
                        A = (temp * estimates[0: n + 1]).sum()
                        B = (temp * times[0: n + 1] * estimates[0: n + 1]).sum()
                        m0 = (A * c - B * b) / (a * c - b**2)
                        m1 = (B * a - A * b) / (a * c - b**2)
                        y = m0 + m1 * time
                        assert (0.0 <= y <= 1.0)
                        smooth_estimate.append(y)
                    except:
                        obs = array(pos, dtype=numpy.double) / (array(pos, dtype=numpy.double) + array(neg, dtype=numpy.double))
                        estimates = (obs - self.false_positive) / (1.0 - self.false_negative - self.false_positive)
                        temp = array([self.smoothing(time - times[k]) for k in range(n + 1)], dtype=numpy.double)
                        y = (temp * estimates[0: n + 1]).sum() / temp.sum()
                        try:
                            assert (0.0 <= y <= 1.0) # ASSERTION ERROR
                            smooth_estimate.append(y)
                        except:
                            for x in temp: assert x == 0
                            smooth_estimate.append(estimates[n])
            return array(smooth_estimate, dtype=numpy.double)