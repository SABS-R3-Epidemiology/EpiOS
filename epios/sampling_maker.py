from numpy.random import binomial


class SamplingMaker():
    '''Class to return the results of sampling

    '''

    def __init__(self, nonresprate=0, keeptrack=False, TheData=None,
                 false_positive=0, false_negative=0, threshold=None):
        self.recognised = ['InfectASympt', 'InfectMild', 'InfectGP', 'InfectHosp', 'InfectICU', 'InfectICURecov']
        self.nonresprate = nonresprate
        self.keeptrack = keeptrack
        self.TheData = TheData
        self.false_positive = false_positive
        self.false_negative = false_negative
        self.threshold = threshold

    def testresult(self, load):
        if bool(binomial(1, self.nonresprate)):
            return 'NonResponder'
        if self.threshold is None:
            if load in self.recognised:
                p = 1 - self.false_negative
            else:
                p = self.false_positive
        else:
            if load > self.threshold:
                p = 1 - self.false_negative
            else:
                p = self.false_positive
        if bool(binomial(1, p)):
            return 'Positive'
        else:
            return 'Negative'

    def __call__(self, sampling_times, people):
        if self.keeptrack:
            STATUSES = self.TheData.loc[sampling_times, people]
            return STATUSES.apply(lambda x: list(map(self.testresult, x)))
        else:
            times_people = zip(sampling_times, people)
            STATUSES = map(lambda t: self.TheData.loc[[t[0]], t[1]], times_people)
            RESULTS = map(lambda x: x.apply(lambda x: list(map(self.testresult, x))), STATUSES)
            return list(RESULTS)
