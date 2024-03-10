from numpy.random import binomial
from numpy import array


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

    def __call__(self,
                 sampling_times,
                 people,
                 keep_track=False,
                 post_proc=False,
                 output=None,
                 callback=None,
                 stratify=None):

        '''
        Method to return the results for all the planned tests

        Inputs:
        -------
            sampling_times: list
                List of the planned times for tests in the same format as data.index.
            people: list
                If keep_track == True or callback != None, this is a list
                of IDs in the same format as columns. Otherwise this is a
                list of the same length as sampling_times. Any element is
                a list of IDs in the same format as columns.
            keep_track: bool,
            post_proc: bool,
            output: string,
            callback: function
                This update the sample each time, useful for additional sampling.
            stratify: function
                Takes one ID and returns its class in the stratification.
        Output:
            result if output == None
            result, observ if output == also_nums
            observ if output == nums_only (input you need for re_scaler)

            result is a DataFrame if keep_track == False
            in this case (observ = pos, neg) are lists
            is a list of Series if keep_track == True and post_proc == False
            in this case (observ = pos, neg) are lists
            is a list of lists Series if keep_track == True and post_proc == True
            in this case observ is a list of tuples of lists
            moreover if pos, neg = observ[k], then len(pos) == len(neg) == k

            If stratify is not None pos and neg are weighted averages depending on the stratification.
            They are multiplied by a factor to normalize their variance.
        '''

        assert not (keep_track and post_proc)

        # count_positive has to return the number of positive in a Series
        # count_negative has to return the number of negative in a Series
        if stratify is None:

            def count(x):
                positive = x.value_counts().get('Positive', 0)
                negative = x.value_counts().get('Negative', 0)
                variance = positive * negative / (positive + negative)
                # rescale the estimate to have unitary variance
                if positive + negative == 0:
                    return 0
                else:
                    return positive, negative, variance
        else:
            # in this case we want to approximate the number of positive/negative people into each class
            classes = {stratify(id) for id in self.data.columns if id != 'time'}
            str_map = {x: {id for id in self.data.columns if id != 'time' and stratify(id) == x} for x in classes}

            def count(x):
                pos, neg, var = [], [], []
                for strat_class in classes:
                    str_map_temp = [id for id in x.index if id != 'time' and stratify(id) == strat_class]
                    positive = x.loc[str_map_temp].value_counts().get('Positive', 0)
                    negative = x.loc[str_map_temp].value_counts().get('Negative', 0)
                    # compute the number of positive tests into a class and rescale it
                    if positive + negative > 0:
                        pos.append(positive * len(str_map[strat_class]) / (positive + negative))
                        # an estimate of the number of positive people into the same class
                        neg.append(negative * len(str_map[strat_class]) / (positive + negative))
                        # an estimate of the number of positive people into the same class
                        var.append(positive * negative * len(str_map[strat_class])**2 / (positive + negative)**3)
                        # an estimate of the variance of the computed value for this class
                return array(pos).sum(), array(neg).sum(), array(var).sum()
                # rescale the estimate to have unitary variance

        if keep_track:
            STATUSES = self.data.loc[sampling_times, people]
            result = STATUSES.apply(lambda x: x.apply(self._testresult))
            obs = result.apply(count, axis=1)
            pos = obs.apply(lambda x: x[0])
            neg = obs.apply(lambda x: x[1])
            var = obs.apply(lambda x: x[2])
            observ = (pos.to_list(), neg.to_list(), var.to_list())  # result is a DataFrame, pos, neg are lists
        else:
            if callback is None:
                times_people = zip(sampling_times, people)
                STATUSES = map(lambda t: self.data.loc[t[0], t[1]], times_people)  # list of Series
                res = list(map(lambda x: x.apply(self._testresult), STATUSES))  # list of Series
            else:
                # in this case you have to update the sample each time depending on res
                # this in order to deal with nonresponders and with additional sampling
                next_people = people  # list
                res = []
                for sampling_time in sampling_times:
                    STATUSES = self.data.loc[sampling_time, next_people]  # Series
                    res.append(STATUSES.apply(self._testresult))  # list of Series
                    next_people = callback(res[-1])  # list

            if post_proc:
                # in this case we need a list of Series for each
                # time, this list has to contain the information
                # from previous samples.
                result = []
                observ = []
                temp = []
                for x in res:
                    for n, y in enumerate(temp):
                        # dischard old tests that we updated to avoid redundancy
                        temp[n] = y.drop(labels=x.index, errors='ignore')
                    temp.append(x)  # list of n + 1 Series
                    result.append(temp.copy())  # list of lists of Series
                    obs = list(map(count, temp))
                    pos = list(map(lambda x: x[0], obs))  # list of n + 1 Series
                    neg = list(map(lambda x: x[1], obs))  # list of n + 1 Series
                    var = list(map(lambda x: x[2], obs))  # list of n + 1 Series
                    observ.append((pos, neg, var))  # list of lists of Series

            else:
                result = res
                obs = list(map(count, res))
                pos = list(map(lambda x: x[0], obs))
                neg = list(map(lambda x: x[1], obs))
                var = list(map(lambda x: x[2], obs))
                observ = (pos, neg, var)

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
