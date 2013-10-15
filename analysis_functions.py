"""
A jump is a deflection in the value of a time-dependent variable of 
a fixed magnitude.
"""

test_array = [3,1,-3,-1,10,2,-10]
#pairs of jumps are:
#1. 3 @ 0 -> 2
#2. 1 @ 1 -> 4
#3. 10 @ at 4 -> 6

def detect_jumps(array,threshold):
    """
    TODO: implement threshold
    """
    pass

def pair_jumps(array,threshold):
    """
    Idea here is that pairs of deflections in data are detected,
    even if other pairs overlap.
    """
    pass


print detect_jumps(array,0.0)
