from numpy.random import binomial


class SamplingMaker():
    '''
    Class to return the results of sampling

    Parameters:
    -----------
        non_resp_rate : float, between 0 and 1
            The probability that the result of a test is 'NonResponder',
            despite infectious status and viral load.

            Default is zero.
        threshold : float or None
            If the viral load is higher then the threshold,
            then the result of the test will be positive,
            otherwise it will be negative.

            Default is None (see recognised below)
        false_positive : float, between 0 and 1
            If the result is supposed to be negative,
            then it will be positive with probability false_positive.

            Default is zero.
        false_negative : float, between 0 and 1
            If the result is supposed to be positive,
            then it will be positive with probability false_negative.

            Default is zero.
        keep_track : bool
            If this is True, the same group of people is tested at each timestep.
            Otherwise (default), at each timestep a new group of peaople is selected for testing.
        data : pandas.DataFrame
            index is the list of times the simulation ran.
            columns is the list of IDs of the entire populations.
            If threshold is None this contains the infectious statuses of the entire population.
            Otherwise this contains the viral loads of the entire population.
    '''

    def __init__(self, non_resp_rate=0, keep_track=False, data=None,
                 false_positive=0, false_negative=0, threshold=None):
        self.non_resp_rate = non_resp_rate
        self.recognised = [3, 4, 5, 6, 7, 8]
        self.threshold = threshold
        self.false_positive = false_positive
        self.false_negative = false_negative
        self.keep_track = keep_track
        self.data = data

    def __call__(self, sampling_times, people):

        '''
        Method to return the results for all the planned tests

        Inputs:
        -------
            sampling_times : list
                List of the planned times for tests in the same format as data.index.
            people : list
                If keep_track is True this is a list of IDs in the same format as data.columns.
                Otherwise it is a list of the same length as sampling_times.
                In this case each element is a list of IDs in the same format as data.columns.

        Output:
        -------
            A pandas.DataFrame if keep_track is True
            A list of pandas.DataFrame objects otherwise.

        '''

        if self.keep_track:
            STATUSES = self.data.loc[sampling_times, people]
            return STATUSES.apply(lambda x: list(map(self._testresult, x)))
        else:
            # STATUSES is an iterator that returns the loads of the next group of people selected for testing
            # SINGLETEST is a function that maps testresult on the loads of a group of people
            times_people = zip(sampling_times, people)
            STATUSES = map(lambda t: self.data.loc[[t[0]], t[1]], times_people)
            SINGLETEST = lambda x: x.apply(lambda x: list(map(self._testresult, x)))
            return list(map(SINGLETEST, STATUSES))

    def _testresult(self, load):
        '''
        Method to return the result for one test

        If threshold is None, then load is the infectiousness status of the testes person
        Otherwise, it is the viral load of the tested person.
        Possible outputs are 'NonResponder', 'Positive', 'Negative'.
        '''

        if bool(binomial(1, self.non_resp_rate)):
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
