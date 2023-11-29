
def person_allowed(sample, choice, threshold):
    """ function to see if the sampled person should be included in the generic sample

    Args:
        sample (list): list of people who have already been chosen
        choice (str): string id of the person being sampled
        threshold (int): the cap on the number of people sampled per household
    """
    
    # get the household of the person
    choice_household = choice[0:4]

    # list of samples only showing first three numbers, e.g. "0.0.0" or "0.2.1"
    sample = [s[0:4] for s in sample]

    # get number of times that household is in sample list
    sample_count = sample.count(choice) 

    # if adding this sample would exceed threshold then reject
    if sample_count > threshold:

        return False

    # otherwise, return true
    else:

        return True
    


