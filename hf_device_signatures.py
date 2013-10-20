''' Functions for computing and storing device signatures. '''


class Signature(Object):
    pass

class GaussianSignature(Signature):
    def __init__(self,
                 mean_frequency,
                 standard_deviation):
        self.mean_frequency = mean_frequency
        self.standard_deviation = standard_deviation
        return self

class EmpiricalSignature(Signature):
    def __init__(self, 
                 signature_data):
        self.signature_data = signature_data

def kitchen_lights_dimmer_21(d):
    pass
