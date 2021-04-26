import numpy as np
import scipy.special
import scipy.stats
import inspect
import glob
import corner
import matplotlib.pyplot as plt
import re
import ultranest as un
import chainconsumer as cc

# TO DO
# Improve the documentation
# Add possibility of using non-nested sampling. This means including alternative
# prior class and, log_likelihood function in LC_fit class.


class Prior_unit_cube():
    """Class to that acts as prior for parameters in nested sampling.

    Instances of this class holds function defining the transformation from
    unit hyper cube to prior space for any given single or set of parameters.

    Appart from the attributes listed below, each instance of this class will
    have attributes corresponding to each item in "prior_attributes".

    Attributes:
      name:
        Name of prior.
      linked_parameters:
        List of pointers to Parameter objects to which the prior pertains.
      prior_attributes:
        Dictionary of auxiliary values for prior (e.g. interval boundaries).

   """
    def __init__(self, name, prior_attributes, input_func=None):
        """Inits Prior_unit_cube class.

        Args:
          name:
            Name of prior as string. Certain names come with predefined
            transformations for some common priors; Uniform, Log_Uniform,
            Triangle, and Gaussian. These represent uniform, log uniform, \
            uniform sampling from a 2D triangle (useful for parameters a,b
            constrained such that min < a < b < max), and Gaussian priors,
            respectively.

          prior_attributes (dict):
            Dictionary with names and instances of auxiliary parameters for the
            prior. For uniform and log uniform this is lo and hi, for the lower
            and upper limit of the interval, respectively. For the Gaussian
            prior this is, mu and sigma. For the triangular prior this is P1
            and P2, corresponding to (lo,hi) and (hi,hi) in the case of
            lo < a < b < hi.

          input_func (func):
            Optional; Custom transformation between unit hyper cube and their
            desired prior.

        """
        self.name = name
        self.linked_parameters = []
        self.prior_attributes = prior_attributes
        for key in prior_attributes:
            setattr(self, key, prior_attributes[key])
        self._initiate_prior(input_func)

    def __call__(self, cube):
        return self.evaluate(cube)

    def _custom_func(self,c):
        """Custom function.
        """
        return self._input_func(c,**self.prior_attributes)

    def _Uniform_prior(self, c):
        """Uniform sampling over an interval [lo, hi].
        """
        return c * (self.hi - self.lo) + self.lo

    def _Log_Uniform_prior(self, c):
        """Log uniform sampling over an interval [lo, hi].
        """
        return 10**(c * (np.log10(self.hi) - np.log10(self.lo)) + np.log10(self.lo))

    def _Triangle_Prior(self, point):
        """Uniform sampling in a triangular space defined by vertices
        [(0,0), P1, P2].
        """
        try:
            r1,r2 = point
        except:
            try:
                r1,r2 = point.T
            except:
                print(sys.exc_info())

        s = np.sqrt(r1)
        x = self.P1[0]*(1.0 - r2)*s + self.P2[0]*r2*s
        y = self.P1[1]*(1.0 - r2)*s + self.P2[1]*r2*s
        return np.array([x, y]).T

    def _initiate_prior(self,input_func):
        if input_func is not None:
            self._input_func = input_func
            self.func = self._custom_func
        elif self.name == 'Uniform':
            self.func = self._Uniform_prior
        elif self.name == 'Log_Uniform':
            self.func = self._Log_Uniform_prior
        elif self.name == 'Triangle':
            self.func = self._Triangle_Prior
        elif self.name == 'Gaussian':
            self.func = scipy.stats.norm(self.mu, self.sigma).ppf

    def evaluate(self,cube):
        '''Evaluate prior given input from unit hyper cube.'''
        return self.func(cube)


class Parameter():
    def __init__(self, name, vectorised=False):
        self.name = name
        self.vectorised = vectorised
        self.free = True
        self.value = None
        self.owner_function = None

    @property
    def prior(self):
        return self._prior

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if self.vectorised:
            if not isinstance(value,(np.ndarray)):
                if value==None:
                    pass
                else:
                    value = np.array([value])
        else:
            assert isinstance(value,(int,float)) or value==None, 'The value needs to be an int, float or None'
        self._value = value

    def set_prior(self,prior):
        self._prior = prior
        self._prior.linked_parameters.append(self)


class data_in():
    def __init__(self, name, value=None):
        self.name = name
        self.free = False
        self.value = value

    @property
    def free(self):
        return self._free

    @free.setter
    def free(self, free):
        assert free==False, 'Time cannot be a free parameter'
        self._free = False

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        assert isinstance(value,np.ndarray) or value==None, 'The value needs to be an ndarray or None'
        self._value = value




class Pulse_shape():
    def __init__(self, func, vectorised = False):

        self.func = func
        self.name = func.__name__
        self.args = inspect.getfullargspec(self.func).args
        self.vectorised = vectorised

        for arg in self.args:
            if arg=='times':
                setattr(self, arg, data_in(arg))
            else:
                setattr(self, arg, Parameter(arg, vectorised = self.vectorised))
                self.__getattribute__(arg).owner_function = self

        self.parameters = {arg:self.__getattribute__(arg) for arg in self.args}

    def __call__(self, parvals):
        return self.evaluate(parvals)

    @property
    def parvals(self):
        self._parvals = {arg:self.__getattribute__(arg).value for arg in self.args}
        return self._parvals

    @property
    def free_parameters(self):
        self._free_parameters = [self.__getattribute__(arg).name for arg in self.args if ((self.__getattribute__(arg).free == True) and (self.__getattribute__(arg).owner_function == self))]
        return self._free_parameters

    @property
    def priors(self):
        self._priors = {arg:(self.__getattribute__(arg).prior) for arg in self.free_parameters} # if isinstance(self.__getattribute__(arg),Parameter) #self.__getattribute__(arg).free==True
        return self._priors

    def set_args(self, parvals):
        if parvals:
            assert isinstance(parvals, dict), 'You need to pass a dictionary of type {parameter_name:value}'

            for key,val in parvals.items():
                self.__getattribute__(key).value = val

    def evaluate(self, parvals=None):
        if parvals:
            self.set_args(parvals)
        return self.func(**self.parvals)

class LC_fit():
    """Class to fit analytic functions to binned time series data.

    Although no sampler is included in this class, the log_likelihood and
    prior_transform functions assume a nested sampler that requires a
    transformation from the unit hyper cube onto the prior distribution space.

    The class is specifically designed to fit pulse shapes to Fermi GRB
    light curves.
    """
    def __init__(self, x, y, bkg=None, bkg_poly_order=1, vectorised = False):
        self.x = x
        self.y = y
        self.pulse_component_name_list = []
        self.parameters = {}
        if not bkg is None:
            self.add_bkg(bkg, poly_order=bkg_poly_order)
        self.vectorised = vectorised

    @property
    def free_parameters(self):
        self._free_parameters = {arg:self.__getattribute__(arg).free_parameters for arg in self.pulse_component_name_list}
        # free_param_list
        return self._free_parameters

    @property
    def priors(self):
        self._priors = {arg:self.__getattribute__(arg).priors for arg in self.pulse_component_name_list}
        return self._priors

    @property
    def priors_list(self):
        priors = []
        for component,pars in self.free_parameters.items():
            for par in pars:
                priors.append(self.priors[component][par])
        self._priors = set(priors)
        return self._priors

    @property
    def indexdict(self):
        return self._indexdict

    @property
    def all_parameters(self):
        self._all_parameters = {arg:{par:val for par,val in self.__getattribute__(arg).parvals.items() if par!='times'} for arg in self.pulse_component_name_list}
        return self._all_parameters

    def _create_indexdict(self):
        self._indexdict = {}
        i=0
        for component in self.pulse_component_name_list:
            for par in self.free_parameters[component]:
                self._indexdict[self.parameters[component][par]] = i
                i+=1

    def add_bkg(self, bkg, poly_order=1):
        self.bkg = bkg
        self.bkg_pol_coeff = np.polyfit(self.x, self.bkg, poly_order)
        self.bkg_func = np.poly1d(self.bkg_pol_coeff)

    def add_pulse_component(self, name, function, initial_par_dict = None):
        """Adds component to function be fitted to data.

        Only designed for additative components. If you are in need of a e.g.
        multiplicative components, consider combining functions outside of
        this class instead.
        """
        assert (name not in self.pulse_component_name_list), f'{name} is already added as a pulse component'

        setattr(self, name, Pulse_shape(function, vectorised = self.vectorised))

        self.pulse_component_name_list.append(name)
        self.parameters[name] = self.__getattribute__(name).parameters
        self.__getattribute__(name).times = data_in('times',self.x)

    def prior_transform(self, cube):
        self._create_indexdict()
        params = cube.copy()

        for prior in self.priors_list:
            indices = []
            for linked_parameter in prior.linked_parameters:
                indices.append(self.indexdict[linked_parameter]) # self.indexlist contains a list that maps a specific parameter to an index that just enumerates parameters in order from first free parameter of first function to last free parameter of last function.
            if self.vectorised:
                params[:,indices] = prior(cube[:,indices])

            else:
                params[indices] = prior(cube[indices])

        return params


    def set_param_values(self, param_values):
        i = 0

        for name in self.pulse_component_name_list:
            free_parameters = self.__getattribute__(name).free_parameters
            if self.vectorised:
                param_dict = {key:val for key,val in zip(free_parameters,param_values[:,i:i+len(free_parameters)].T)}
            else:
                param_dict = {key:val for key,val in zip(free_parameters,param_values[i:i+len(free_parameters)])}
            i+=len(free_parameters)
            self.__getattribute__(name).set_args(param_dict)


    def get_rates(self,x):

        self.rates = np.zeros(x.shape)

        for name in self.pulse_component_name_list:
            self.rates+=self.__getattribute__(name).evaluate({'times':x})

        return self.rates

    def log_likelihood(self, param_values):
        if self.vectorised:
            x = np.array([self.x for i in range(param_values.shape[0])]).T
            y = np.array([self.y for i in range(param_values.shape[0])]).T

        else:
            x = self.x
            y = self.y

        if param_values is not None:
            self.set_param_values(param_values)

        total_rate = self.get_rates(x) + self.bkg_func(x)

        if np.any(total_rate == 0.):
            return -np.inf
        else:
            return np.sum(-total_rate + y * np.log(total_rate) - scipy.special.gammaln(y + 1),axis=0)


    def plot_corner(self, results, param_names, **kwargs):
        samples = np.array(results['weighted_samples']['points'])
        weights = np.array(results['weighted_samples']['weights'])
        cumsum_weights = np.cumsum(weights)

        mask = cumsum_weights > 1e-4

        if mask.sum() == 1:
            print('Posterior has a really poor spread. Something funny is going on.')

        fig = corner.corner(samples[mask,:], weights=weights[mask], show_titles=True, labels = param_names, **kwargs)

        return fig

    def plot_corner_cc(self, results, param_names, **kwargs):

        samples = np.array(results['weighted_samples']['points'])
        weights = np.array(results['weighted_samples']['weights'])
        cumsum_weights = np.cumsum(weights)

        mask = cumsum_weights > 1e-4

        if mask.sum() == 1:
            print('Posterior has a really poor spread. Something funny is going on.')

        c = cc.ChainConsumer()
        c.add_chain(samples[mask,:], weights=weights[mask], parameters=param_names)
        c.configure(summary = True)

        fig = c.plotter.plot(**kwargs)

        # fig = corner.corner(samples[mask,:], weights=weights[mask], show_titles=True, labels = param_names, **kwargs)

        return fig

    def plot_fit(self, param_values=None, results=None):
        """Plots data and model corresponding to current parameter values, given
        parameter values, or mean posterior (in case results is given)

        Returns:
          Figure.
        """
        if results is not None:
            param_values = results['posterior']['mean']
            if self.vectorised:
                param_values = np.array([param_values])

        if param_values is not None:
            self.set_param_values(param_values)

        rates = self.get_rates(self.x)

        fig, ax = plt.subplots()
        ax.plot(self.x,self.y)
        ax.plot(self.x,self.rates+self.bkg)

        return fig

    def load_results(self, chain_folder, x_dim, run='latest'):
        """Load results from a previous run of ultranest.

        Args:
          chain_folder:
            Path to directory that holds run{i}/ folder(s). These in turn hold
            chain/ and results/ folders, etc.
          x_dim:
            Number of dimensions of posterior.
          run:
            Optional; if 'latest' uses the run{i} folder with highest value
            of i. Else str or int corresponding to the run{i} folder you want
            to access.

        Returns:
          results; which is a list with two dictionaries of results. Note that
          results[1] corresponds to standard ultranest results. See ultranest
          documentation for details.
        """
        number = 0
        res_path = None
        for subdir in glob.glob(chain_folder+'/*'):
            rundir = re.match('.*run(\d+)',subdir,re.IGNORECASE)
            if rundir:
                number_new = int(rundir.group(1))
                if run=='latest':
                    if number_new > number:
                        number = number_new
                        res_path = subdir
                else:
                    if str(number_new)==str(run):
                        res_path = subdir

        results = un.read_file(res_path,x_dim=x_dim)

        return results


if __name__ == '__main__':
    pass
