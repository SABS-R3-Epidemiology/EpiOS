
def person_allowed(sample, choice, threshold):
    """ function to see if the sampled person should be included in the generic sample

    Args:
        sample (list): list of people who have already been chosen
        choice (str): string id of the person being sampled
        threshold (int): the cap on the number of people sampled per household
    """

    # get the household of the person
    choice_household = '.'.join(choice.split('.')[:-1])

    # list of samples only showing first three numbers, e.g. "0.0.0" or "0.2.1"
    sample = ['.'.join(s.split('.')[:-1]) for s in sample]

    # get number of times that household is in sample list
    sample_count = sample.count(choice_household)

    # if adding this sample would exceed threshold then reject
    if sample_count >= threshold:

        return False

    # otherwise, return true
    else:

        return True

# Test


""" sample = ["0.0.0.0","0.0.0.1","0.0.0.2","0.0.0.3",
       "0.0.1.0","0.0.1.1",
       "0.0.2.0",
       "0.0.3.0","0.0.3.1","0.0.3.2","0.0.3.3","0.0.3.4",
       "0.1.0.0","0.1.0.1","0.1.0.2",
       "0.1.1.0","0.1.1.1",
       "2.0.0.0","2.0.0.1","2.0.0.2",
       "2.0.1.0","2.0.1.1",
       "2.1.0.0","2.1.0.1","2.1.0.2"]

choice = "0.0.0.4"
threshold = 4

boolean = person_allowed(sample, choice, threshold)

print(boolean) """
