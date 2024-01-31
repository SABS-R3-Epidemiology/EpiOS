from numpy import array

class ReScaler():


    '''
    Class to de-bias the observed prevalence by affine transformation
    The day when the nonresprate will depend on the status this class will take this into account
    Observation is meant to be an array, each entry being the rate of positive tests at a certain time

    If smoothing is set, then the estimates are further manipulated
    the problem is, given arrays T[k],Y[k] for k in range(n),
    and given a function w then one wants to minimize the cost
    m0[n],m1[n]=argmin(sum(w(T[n]-T[k])*(Y[k]-m0-m1*T[k])**2 for k in range(n+1)))
    this corresponds to solve
        m0*sum(w(T[n]-T[k]) for k in range(n+1))+m1*sum(w(T[n]-T[k])*T[k] for k in range(n+1))=sum(w(T[n]-T[k])*Y[k] for k in range(n+1))
        m0*sum(w(T[n]-T[k])*T[k] for k in range(n+1))+m1*sum(w(T[n]-T[k])*(T[k]**2) for k in range(n+1))=sum(w(T[n]-T[k])*Y[k]*T[k] for k in range(n+1))
    that can be written an
        m0*a+m1*b=A
        m1*b+m2*c=B
    that is
        m0=(A*c-B*b)/(a*c-b**2)
        m1=(B*a-A*b)/(a*c-b**2)
    '''


    def __init__(self, false_positive=0, false_negative=0, smoothing=None):
        if false_negative + false_positive != 1:
            self.false_positive = false_positive
            self.false_negative = false_negative
            self.smoothing = smoothing
        else: raise Exception('useless test')

    def __call__(self, observation, times=None):
        estimates = (array(observation) - self.false_positive)/( 1 - self.false_negative - self.false_positive)
        if self.smoothing == None:
            return estimates
        elif times==None: raise Exception('please insert times of sampling')
        else:
            smooth_estimate=[]
            for n in range(len(times)):
                temp=array([self.smoohting(times[n]-times[k]) for k in range(n+1)])
                a=temp.sum()
                b=(temp*times[0:n+1]).sum()
                c=(temp*(times[0:n+1]**2)).sum()
                A=(temp*estimates[0:n+1]).sum()
                B=(temp*times[0:n+1]*estimates[0:n+1]).sum()
                m0=(A*c-B*b)/(a*c-b**2)
                m1=(B*a-A*b)/(a*c-b**2)
                smooth_estimate.append(m0+m1*times[n])
            return array(smooth_estimate)
