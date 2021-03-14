class ParamScale(object):

    def __init__(self, config):
        #self.config = config
        self.z_range = config['particle']['z_p']
        self.a_range = config['particle']['a_p']
        self.n_range = config['particle']['n_p']
        self.ranges = [self.z_range, self.a_range, self.n_range]

    def normalize(self, params):
        scaled_params = []
        for param, minmax in list(zip(params, self.ranges)):
            pmin = minmax[0]
            pmax = minmax[1]
            prange = pmax - pmin
            scaled = (param - pmin)/prange
            scaled_params.append(scaled)
        return scaled_params

    def unnormalize(self, params):
        unscaled_params = []
        for param, minmax in list(zip(params, self.ranges)):
            pmin = minmax[0]
            pmax = minmax[1]
            prange = pmax - pmin
            unscaled = (param * prange) + pmin
            unscaled_params.append(unscaled)
        return unscaled_params
