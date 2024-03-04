import numpy
from numpy import array


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
        m1 * b + m2 * c = B
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

    def __init__(self, false_positive=0, false_negative=0, smoothing=None):
        if false_negative + false_positive != 1:
            self.false_positive = false_positive
            self.false_negative = false_negative
            self.smoothing = smoothing
        else:
            raise Exception('useless test')

    def __call__(self, observation, times=None):
        if self.smoothing is None:
            pos, neg = observation
            estimates = (array(pos) / (array(pos) + array(neg)) - self.false_positive)
            estimates /= (1 - self.false_negative - self.false_positive)
            return estimates
        elif times is None:
            raise Exception('please insert times of sampling')
        else:
            smooth_estimate = []
            for n, (pos, neg) in enumerate(observation):
                obs = array(pos) / (array(pos) + array(neg))
                estimates = (obs - self.false_positive) / (1 - self.false_negative - self.false_positive)
                temp = array([self.smoothing(times[n] - times[k]) for k in range(n + 1)], dtype=numpy.double)
                temp *= obs * (1 - obs) * (array(pos) + array(neg))
                try:
                    a = temp.sum()
                    b = (temp * times[0: n + 1]).sum()
                    c = (temp * (times[0: n + 1]**2)).sum()
                    assert a * c != b**2
                    A = (temp * estimates[0: n + 1]).sum()
                    B = (temp * times[0: n + 1] * estimates[0: n + 1]).sum()
                    m0 = (A * c - B * b) / (a * c - b**2)
                    m1 = (B * a - A * b) / (a * c - b**2)
                    smooth_estimate.append(m0 + m1 * times[n])
                except AssertionError:
                    smooth_estimate.append(estimates[n])
            return array(smooth_estimate)
