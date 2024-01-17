from numpy import array

class ReScaler():


    '''
    Class to de-bias the observed prevalence by affine transformation
    The day when the nonresprate will depend on the status this class will take this into account
    Observation is meant to be an array, each entry being the rate of positive tests at a certain time
    '''


    def __init__(self, false_positive=0, false_negative=0):
        if false_negative + false_positive != 1:
            self.false_positive = false_positive
            self.false_negative = false_negative
        else: raise Exception('useless test')

    def __call__(self, observation): return (array(observation) - self.false_positive)/( 1 - self.false_negative - self.false_positive)