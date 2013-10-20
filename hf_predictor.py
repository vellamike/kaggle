''' Provides a signal using the HF data. Uses trained information such as device
signatures. '''

def total_on_time():
    ''' Used for the priors. '''
    pass

def period_liklihood(device,
                     start_stamp,
                     end_stamp):
    ''' What's the liklihood that the device was on between start_stamp and
    end_stamp? '''
    # Use intuition and total_on_time

def hf_emi_signatures(d,
                      device_signatures):
    ''' At each time and for each device, returns the probability, based on the
    signature alone, that the device goes on at that time. N.B: for nearly all
    slices, and for devices with a noticable signature, this will be 0. 
    Does the same for off times.
    
    1) Start at time d.start.
    2) Compute the difference in the power spectrum between this time and the
    last.
    3) What is the residual error of this difference when compared to each
    device signature?
    4) Move to next timeslice.
    '''




    pass

def hf_signal(d,
              device_signatures):
    ''' Returns, incorporating information from hf_emi_signatures and
    period_liklihood, the probability that a device is on at a time. '''
    pass

def hf_predictions(d,
                   device_signatures):
    ''' Predictions based solely on hf data. '''
    threshold = 0.5
    pass


