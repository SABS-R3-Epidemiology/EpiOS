from numpy.random import binomial
import pandas as pd



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
                 false_positive=0, false_negative=0, threshold=None,
                 inf_data=None, model=None):
        self.non_resp_rate = non_resp_rate
        self.recognised = [3, 4, 5, 6, 7, 8]
        self.threshold = threshold
        self.false_positive = false_positive
        self.false_negative = false_negative
        self.keep_track = keep_track
        self.data = data
        self.inf_data = inf_data
        self.model = model

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
            VIRAL_LOADS = self.inf_data.loc[sampling_times, people]
            #return STATUSES.apply(lambda x: list(map(self._testresult, x, VIRAL_LOADS)))

            results = STATUSES.apply(lambda x: list(map(lambda s, v: self._testresult(s, v), x, VIRAL_LOADS[x.name])))
            return results
        else:
            # STATUSES is an iterator that returns the loads of the next group of people selected for testing
            # SINGLETEST is a function that maps testresult on the loads of a group of people
            times_people = zip(sampling_times, people)
            STATUSES = map(lambda t: self.data.loc[[t[0]], t[1]], times_people)
            VIRAL_LOADS = map(lambda t: self.inf_data.loc[[t[0]], t[1]], times_people)
            SINGLETEST = lambda x: x.apply(lambda x: list(map(self._testresult, x)))
            return list(map(SINGLETEST, STATUSES))
        


    def _testresult(self, load, infectiousness):
        '''
        Method to return the result for one test

        If threshold is None, then load is the infectiousness status of the testes person
        Otherwise, it is the viral load of the tested person.
        Possible outputs are 'NonResponder', 'Positive', 'Negative'.
        '''

        #print(infectiousness)

        false_positive = self.false_positive
        false_negative = self.false_negative
        
        if self.model is not None:

            false_positive, false_negative = self.model(false_positive, false_negative, infectiousness)

        #false_positive = (1 - infectiousness) * false_positive
        #false_negative = (1 + infectiousness) * false_negative


        if bool(binomial(1, self.non_resp_rate)):

            return 'NonResponder'
        
        if self.threshold is None:

            if load in self.recognised: #infected

                p = 1 - false_negative

                infected_flag = True

            else:

                p = false_positive # not infected

                infected_flag = False

        else:

            if load > self.threshold:

                p = 1 - false_negative

                infected_flag = True

            else:

                p = false_positive

                infected_flag = False

        if bool(binomial(1, p)):

            return f"[Positive, {infected_flag}]"
        
        else:

            return f"[Negative, {not infected_flag}]"
