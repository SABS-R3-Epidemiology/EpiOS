from numpy.random import binomial


class SamplingMaker():
    
    '''
    Class to return the results of sampling
    ---------------------------------------
    Parameters:
        nonresprate (float between 0 and 1):
            The probability that the result of a test is 'NonResponder' despite infectious status and viral load.
            Default is zero.
        threshold (float or None): 
            If the viral load is higher then the threshold, then the result of the test will be positive, otherwise it will be negative.
            Default is None (see recognised below)
        recognised:
            If threshold is None then the result is supposed to be positive if the infectious status is one of the recognised.
        false_positive (float between 0 and 1):
            If the result is supposed to be negative, then it will be positive with probability false_positive.
            Default is zero.
        false_negative (float between 0 and 1):
            If the result is supposed to be positive, then it will be positive with probability false_negative.
            Default is zero.
        keeptrack (boolean):
            If this is True, the same group of people is tested at each timestep.
            Otherwise (default), at each timestep a new group of peaople is selected for testing.
        TheData (pandas.DataFrame):
            index is the list of times the simulation ran.
            columns is the list of IDs of the entire populations.
            If threshold is None this contains the infectious statuses of the entire population through the all simulation
            Otherwise this contains the viral loads of the entire population through the all simulation
    Methods:
        __init__: fills the fields above.
        __call__: returns the results for all the planned tests.
        testresult: returns the result (stochastic) for one test.

    '''

    def __init__(self, nonresprate=0, keeptrack=False, TheData=None, false_positive=0, false_negative=0, threshold=None):
        self.nonresprate = nonresprate
        self.recognised = ['InfectASympt', 'InfectMild', 'InfectGP', 'InfectHosp', 'InfectICU', 'InfectICURecov']
        self.threshold = threshold
        self.false_positive = false_positive
        self.false_negative = false_negative
        self.keeptrack = keeptrack
        self.TheData = TheData


    def __call__(self, sampling_times, people):

        '''
        Method to return the results for all the planned tests
        ------------------------------------------------------
        Inputs:
            sampling_times(list): list of the planned times for tests in the same format as TheData.index.
            people(list):
                If keeptrack is True this is a list of IDs in the same format as TheData.columns.
                Otherwise it is a list of the same length as sampling_times.
                In this case each element is a list of IDs in the same format as TheData.columns.
        Output:
            A pandas.DataFrame if keeptrack is True
            A list of pandas.DataFrame objects otherwise.

        '''

        if self.keeptrack:
            STATUSES = self.TheData.loc[sampling_times,people]
            return STATUSES.apply(lambda x: list(map(self.testresult, x)))
        else:
            # STATUSES is an iterator that returns the loads of the next group of people selected for testing
            # SINGLETEST is a function that maps testresult on the loads of a group of people, returning the actial results
            times_people=zip(sampling_times, people)
            STATUSES = map(lambda t:self.TheData.loc[[t[0]],t[1]],times_people)
            SINGLETEST = lambda x: x.apply(lambda x: list(map(self.testresult, x)))
            return list(map(SINGLETEST,STATUSES))
        

    def testresult(self,load):

        '''
        Method to return the result for one test
        ----------------------------------------
        If threshold is None, then load is the infectiousness status of the testes person
        Otherwise, it is the viral load of the tested person.
        Possible outputs are 'NonResponder', 'Positive', 'Negative'.
        '''

        if bool(binomial(1, self.nonresprate)):
            return 'NonResponder'
        if self.threshold == None:
            if load in self.recognised:
                p = 1 - self.false_negative
            else:
                p = self.false_positive
        else:
            if load>self.threshold:
                p = 1 - self.false_negative
            else:
                p = self.false_positive
        if bool(binomial(1, p)):
            return 'Positive'
        else:
            return 'Negative'