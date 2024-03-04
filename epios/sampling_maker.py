from numpy.random import binomial
from numpy import array, nan

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

    def __init__(self,
                 non_resp_rate=None,
                 data=None,
                 false_positive=0,
                 false_negative=0,
                 threshold=None):

        if non_resp_rate is None:
            self.non_resp_rate = 0
        else:
            self.non_resp_rate = non_resp_rate
        self.recognised = [3, 4, 5, 6, 7, 8]
        self.threshold = threshold
        self.false_positive = false_positive
        self.false_negative = false_negative
        self.data = data

    def __call__(self, sampling_times, people, keep_track=False, post_proc=False, callback=None, output=None, stratify=None):

        '''
        Method to return the results for all the planned tests

        Inputs:
        -------
            sampling_times : list
                List of the planned times for tests in the same format as data.index.
            people : list
                If keep_track is True this is a list of IDs in the same
                format as columns. Otherwise this is a list of the same
                length as sampling_times. In this case all elements are
                lists of IDs in the same format as columns.

        Output:

        '''

        assert not (keep_track and post_proc)

        if stratify is None:
            count_positive = lambda x: x.value_counts().get('Positive', 0)
            count_negative = lambda x: x.value_counts().get('Negative', 0)
        else:
            classes = {stratify(id) for id in self.data.columns if id != 'time'}
            str_map = {strat_class: {id for id in self.data.columns if id != 'time' and stratify(id) == strat_class} for strat_class in classes}

            def count_positive(x):
                try:
                    obs = []
                    for strat_class in classes:
                        str_map_temp = [id for id in x.index if id != 'time' and stratify(id) == strat_class]
                        tested = x.loc[str_map_temp].value_counts().get('Positive', 0)
                        obs.append(tested * len(str_map[strat_class]) / len(str_map_temp))
                    return array(obs).sum() * len(x.index) / len(self.data.columns)
                except ZeroDivisionError:
                    return nan

            def count_negative(x):
                try:
                    obs = []
                    for strat_class in classes:
                        str_map_temp = [id for id in x.index if id != 'time' and stratify(id) == strat_class]
                        tested = x.loc[str_map_temp].value_counts().get('Negative', 0)
                        obs.append(tested * len(str_map[strat_class]) / len(str_map_temp))
                    return array(obs).sum() * len(x.index) / len(self.data.columns)
                except ZeroDivisionError:
                    return nan
        
        if keep_track:
            STATUSES = self.data.loc[sampling_times, people]
            result = STATUSES.apply(lambda x: x.apply(self._testresult))
            pos = result.apply(count_positive, axis=1)
            neg = result.apply(count_negative, axis=1)
            observ = (pos.to_list(), neg.to_list())  # result is a DataFrame, pos, neg are lists
        else:
            if callback is None:
                times_people = zip(sampling_times, people)
                STATUSES = map(lambda t: self.data.loc[t[0], t[1]], times_people)  # list of Series
                res = list(map(lambda x: x.apply(self._testresult), STATUSES))  # list of Series
            else:
                next_people = people
                res = []
                for sampling_time in sampling_times:
                    STATUSES = self.data.loc[sampling_time, next_people]  # Series
                    res.append(STATUSES.apply(self._testresult))  # list of Series
                    next_people = callback(res[-1])
            if post_proc:
                result = []
                observ = []
                temp = []
                for x in res:
                    for n, y in enumerate(temp):
                        temp[n] = y.drop(labels=x.index, errors='ignore')
                    temp.append(x)  # list of Series
                    result.append(temp.copy())  # list of list of Series
                    pos = list(map(count_positive, temp))
                    neg = list(map(count_negative, temp))
                    observ.append((pos, neg))
            else:
                result = res
                pos = list(map(count_positive, res))
                neg = list(map(count_negative, res))
                observ = (pos, neg)

        if output is None:
            return result
        elif output == 'nums_only':
            return observ
        elif output == 'also_nums':
            return result, observ
        else:
            raise Exception('no valid output, output can be nums_only or also_nums or None')

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
