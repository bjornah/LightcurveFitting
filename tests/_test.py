import pytest
from LightcurveFitting.PulseFitting import Prior_unit_cube, Pulse_shape, LC_fit
import numpy as np
import random
import ultranest as un

random.seed(42)

def test_Uniform():
    '''
    Test transformation between unit cube and uniform distribution
    '''
    uniform = Prior_unit_cube('Uniform', {'lo':0,'hi':10})
    print('uniform test')
    assert uniform.evaluate(1) == 10

def test_Gaussian():
    '''
    Test transformation between unit cube and gaussian distribution
    '''
    gauss = Prior_unit_cube('Gaussian', {'mu':0,'sigma':10})
    print('gaussian test')
    assert gauss.evaluate(0.5) == 0.0

def test_custom_prior():
    '''
    Test transformation between unit cube and custom distribution
    '''
    f = lambda x, a, b: x**2 + a*x + b
    prior = Prior_unit_cube('custom', {'a':1,'b':2}, f)
    print('custom test')
    assert prior.func(0.5) == 2.75

def func(times, phys_par):
    '''
    Test function with which to evaluate functions in Pulse_shape and LC_fit classes
    '''
    return np.array([np.exp(phys_par*t) for t in times])

def test_Pulse_shape():
    '''
    Test if Pulse_shape handles parameter value assignment and function evaluation
    '''
    f = func
    fc = Pulse_shape(f)
    parvals = {'times': np.linspace(-4,1,30), 'phys_par':3}
    fc.set_args(parvals)
    assert fc.evaluate().shape == (30,)
    assert abs(fc.evaluate().sum()-49.736) < 0.01

def test_Pulse_shape_vectorised():
    '''
    Test if Pulse_shape handles parameter value assignment and function evaluation
    '''
    f = func
    fc = Pulse_shape(f, vectorised = True)
    parvals = {'times': np.linspace(-4,1,30), 'phys_par':np.array([3,4,5])}
    fc.set_args(parvals)
    assert fc.evaluate().shape == (30,3)
    # assert abs(fc.evaluate().sum()-49.736) < 0.01

# This defines some dumy data generated for and from func
x = np.linspace(-4,1,30)
y = np.array([2.12895097e-01, 2.26181083e-01, 2.15608673e-01, 2.17583380e-01,
       9.16241708e-02, 2.96141743e-03, 9.69434403e-02, 6.15517076e-02,
       7.96586598e-02, 2.31241665e-01, 2.56803895e-01, 1.22606543e-01,
       1.76096040e-02, 2.21555900e-01, 6.99363131e-02, 1.52720130e-01,
       9.84855946e-02, 1.80101635e-01, 2.49811856e-01, 3.81563810e-01,
       2.39165917e-01, 4.90869660e-01, 7.36598168e-01, 9.55487552e-01,
       1.76646622e+00, 2.65510114e+00, 4.29533832e+00, 7.42010268e+00,
       1.20267523e+01, 2.03703016e+01])
bkg = np.array([2.61458853, 2.94562148, 2.06149918, 2.62913237, 2.74297672,
       2.35851923, 2.45966609, 2.08162856, 2.63356012, 2.22279754,
       2.88882858, 2.66716658, 2.36726423, 2.74418485, 2.9571311 ,
       2.65023233, 2.59898686, 2.84385433, 2.96460839, 2.90880321,
       2.94909141, 2.11138619, 2.46703657, 2.00616275, 2.4523899 ,
       2.89418772, 2.7648525 , 2.97153278, 2.43185833, 2.57714305])

# initiate LC_fit instance with which to assess functions related to likelihood
lc_fit = LC_fit(x,y,bkg)
lc_fit.add_pulse_component('exp1',func)
parvals = {'times': x, 'phys_par':3}
param_names = ['phys_par']
lc_fit.exp1.set_args(parvals)
lc_fit.exp1.phys_par.set_prior(Prior_unit_cube('Uniform',{'lo':1,'hi':5}))

def test_prior_transformation():
    '''
    Check that the prior transormation works as intended
    '''
    cube = np.array([0.1])
    assert lc_fit.prior_transform(cube)==1.4

def test_loglikelihood():
    '''
    Check that the likelihood can be evaluated
    '''
    cube = np.array([0.1])
    param_values = lc_fit.prior_transform(cube)
    lc_fit.log_likelihood(param_values)
    assert abs(lc_fit.log_likelihood(param_values)-(-82.544))<0.01

def test_sampling():
    '''
    We just need to initiate a sampler here, which we do by running un.ReactiveNestedSampler, since this automatically tests that the likelihood and prior transform etc works, thanks to the option num_test_samples.
    '''
    sampler = un.ReactiveNestedSampler(param_names, lc_fit.log_likelihood, lc_fit.prior_transform, resume='overwrite',
                                        log_dir=f'chains/',num_test_samples=10,vectorized=False,draw_multiple=True)

    # results = sampler.run()
    # assert abs(results['posterior']['mean'][0]-2.735)<0.01


# initiate LC_fit instance with which to assess functions related to likelihood, now for vectorised entities
lc_fit_vec = LC_fit(x,y,bkg,vectorised=True)
lc_fit_vec.add_pulse_component('exp1',func)
parvals_vec = {'times': x, 'phys_par':np.array([3,4,5])}
param_names_vec = ['phys_par']
lc_fit_vec.exp1.set_args(parvals_vec)
lc_fit_vec.exp1.phys_par.set_prior(Prior_unit_cube('Uniform',{'lo':1,'hi':5}))


def test_test_prior_transformation_vec():
    '''
    Check what now
    '''
    cube = np.array([[0.1],[0.5]])
    assert lc_fit_vec.prior_transform(cube).shape==(2,1) #np.array([[1.4,9.0]])
    assert lc_fit_vec.prior_transform(cube)[0,0] == np.array([1.4])
    assert lc_fit_vec.prior_transform(cube)[1,0] == np.array([3.])


# def test_loglikelihood_vec():
#     '''
#     Check that the likelihood can be evaluated
#     '''
#     cube = np.array([[0.1,2]])
#     param_values_vec = lc_fit_vec.prior_transform(cube)
#     lc_fit_vec.log_likelihood(param_values_vec)
    # assert abs(lc_fit_vec.log_likelihood(param_values_vec)-(-82.544))<0.01
#
# def test_sampling_vec():
#     '''
#     We just need to initiate a sampler here, which we do by running un.ReactiveNestedSampler, since this automatically tests that the likelihood and prior transform etc works, thanks to the option num_test_samples.
#     '''
#     sampler = un.ReactiveNestedSampler(param_names, lc_fit.log_likelihood, lc_fit.prior_transform, resume='overwrite',
                                        # log_dir=f'chains/',num_test_samples=10,vectorized=False,draw_multiple=True)


# if __name__ == "__main__":
#     print("Everything passed")
