# MusList :
#   List of geometrical mus & others
#   Their reference values & limits
#   Range to test
#
import numpy as np
import sympy as sp


# Todo : set assertions
class MusList :
    def __init__(self,basic_mus,geom_mus,geom_min,geom_max,geom_ref = None,ranges = None) :
        self.basic_mus = basic_mus.copy()
        self.geom_mus  = geom_mus.copy()
        self.geom_min  = geom_min.copy()
        self.geom_max  = geom_max.copy()
        self.geom_ref  = geom_ref.copy() if geom_ref is not None else [sp.Dummy(str(x) + "_bar") for x in geom_mus]
        self.ranges    = ranges.copy() if ranges is not None else None

    def make_ranges(self,grid_size = 10) :
        tab = [
            np.linspace(minimum,maximum,grid_size + 1,endpoint = False)[1:]
            for i,(mu,minimum,maximum) in enumerate(zip(self.geom_mus,self.geom_min,self.geom_max))
        ]

        for i in range(len(tab)) :
            for j in range(i) :
                tab[j] = np.expand_dims(tab[j],-1)
        self.ranges = np.broadcast_arrays(*tab)

    def numerical_results(self,eq,variables = None,values = None) :
        """ WARN : Numerical... """
        if variables is None : variables = []
        if values is None : values = []

        def func_max(*args) :
            return np.amax(np.stack(list(args)),axis = 0)
        def func_min(*args) :
            return 0  # np.amax(np.stack(args),axis = 0)

        args = tuple(list(self.geom_mus) + list(variables))
        f = sp.lambdify(args,eq,"numpy")

        result = np.atleast_1d(f(*self.ranges,*values))  # *variables, .. *values,
        # 'Max' : np.fmax

        # print(ranges)
        while len(result.shape) < len(self.ranges) :
            result = np.expand_dims(result,-1)
        return result



