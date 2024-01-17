from numpy.random import binomial


class SamplingMaker():
    '''Class to return the results of sampling

    Parameters:
    -----------

    nonresprate : float, between 0 and 1
        The probability of a person that do not respond
    keeptrack : bool
        Whether or not to change people sampled at each time point
    TheData : pandas.DataFrame
        The infection data of the population at different time points
    false_positive : float, between 0 and 1
        The possibility of a normal person to get a positive test result
    false_negative : float, between 0 and 1
        The possibility of a infected person to get a negative test result
    threshold : NoneType
        (Tbh, I also do not know what is this for)

    '''

    def __init__(self, nonresprate=0, keeptrack=False, TheData=None,
                 false_positive=0, false_negative=0, threshold=None):
        self.recognised = [3, 4, 5, 6, 7, 8]
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
        '''
        This will return the test result for samples provided

        Parameters:
        -----------

        sampling_times : list
            A list of time points to sample
        people : list
            A list of ID of people sampled

        '''
        if self.keeptrack:
            STATUSES = self.TheData.loc[sampling_times, people]
            return STATUSES.apply(lambda x: list(map(self.testresult, x)))
        else:
            times_people = zip(sampling_times, people)
            STATUSES = map(lambda t: self.TheData.loc[[t[0]], t[1]], times_people)
            RESULTS = map(lambda x: x.apply(lambda x: list(map(self.testresult, x))), STATUSES)
            return list(RESULTS)
