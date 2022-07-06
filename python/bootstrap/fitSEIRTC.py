#!/usr/bin/env python
# coding: utf-8

import numpy as np
from importlib import reload
from matplotlib import pyplot as plt
import matplotlib
from scipy.integrate import odeint
from scipy.optimize import least_squares
from sys import exit
import pickle
import pandas as pd
import seaborn as sns
from numpy.random import uniform
import sys

sys.path.append('../')
import compartmental

colorpalette = sns.color_palette('colorblind')

# Load data, filter dates
austria_data = pd.read_csv("../austria_data.csv")
data_to_fit  = austria_data.loc[(austria_data['date'] >= '2020-09-01') & 
                                (austria_data['date'] <  '2020-11-03')]
day_four_cases =  data_to_fit['week_mean_cases'].to_numpy()[4]
max_tests = np.max(data_to_fit['week_mean_tests'].to_numpy())

# Model
# Some of the parameters below are taken from 
# Weitz et al. 2020, 
# Modeling shield immunity to reduce COVID-19 epidemic spread
# (Table S1)

parameters = {
    'N'      : 8894380,  # Population
    'beta'   : 0.59,     # avg cont. / pers. / day * prob(symp. trans.)
    'alpha'  : 11.3,     # avg cont. / pers. / day * prob(cont. trac.) * prob(no trans.)
    'kappa'  : 0.66,     # prob(testing exposed)
    'p'      : 0.2,      # Asymptomatic fraction
    'd'      : 0.49,     # prob(detecting symptomatic)
    'gamma_E': 4 ** -1,  # (days)^-1 Inverse mean exposed period  
    'gamma_T': 1 ** -1,  # (days)^-1 Inverse mean test turnout time
    'gamma_I': 5 ** -1,  # (days)^-1 Inverse mean infectious period 
    'gamma_s': 0.43,     # (days)^-1 Inverse mean infectious-to-isolation time
    'T_m'    : 17080,    # ~ Maximum number of contact-tracable cases
    'k'      : 1,        # Tangent hyperbolic steepness
            }

compartments = [
    ('S'    , {"layer" : 1}), 
    ('T_S'  , {"layer" : 1}),
    ('E'    , {"layer" : 2}), 
    ('T_E'  , {"layer" : 2}), 
    ('I_a'  , {"layer" : 3}), 
    ('I_s'  , {"layer" : 3}),  
    ('T_I'  , {"layer" : 3}), 
    ('R_u'  , {"layer" : 4}),
    ('C'    , {"layer" : 4}), 
    ('R_k'  , {"layer" : 4})
    ]

# Note: 0.5 + 0.5 * tanh(k * (x_0 - x)) ~ \Theta (x_0 - x)

rates = [
    ('S',   'E'  , {"label": 
                    "beta * S *  (0.35 * I_a + I_s) / N"}),
    ('S',   'T_S', {"label": 
                    "alpha * S * (C / N) "
                    "* (0.5 + 0.5 * tanh(k * (T_m - T_S - T_E)))"}),
    ('E',   'T_E', {"label": 
                    "gamma_E * E * kappa * (C / (C + I_s + I_a)) " 
                    "* (0.5 + 0.5 * tanh(k * (T_m - T_S - T_E)))"}),
    ('T_S', 'S'  , {"label": "gamma_T * T_S"}),
    ('T_E', 'C'  , {"label": "gamma_T * T_E"}),     
    ('E',   'I_a', {"label": 
                    "(1 - kappa * (C / (C + I_s + I_a)) "
                    "* (0.5 + 0.5 * tanh(k * (T_m - T_S - T_E)))) "
                    "* p * gamma_E * E"}), 
    ('E',   'I_s', {"label": 
                    "(1 - kappa * (C / (C + I_s + I_a)) "
                    "* (0.5 + 0.5 * tanh(k * (T_m - T_S - T_E)))) "
                    "* (1 - p) * gamma_E * E"}),
    ('I_a', 'R_u', {"label": "gamma_I * I_a"}),
    ('I_s', 'T_I', {"label": "d * gamma_s * I_s"}),
    ('I_s',   'R_u', {"label": "(1 - d) * gamma_I * I_s"}),
    ('T_I', 'C'  , {"label": "gamma_T * T_I"}),
    ('C'  , 'R_k' ,{"label": 
                    "(1 / ((1 / gamma_I) "
                        "- (T_E * (1 / gamma_T) "
                            "+ T_I * ((1 / gamma_s) + (1 / gamma_T))) "
                        "/ (T_E + T_I))) * C"}),
        ]

model = compartmental.Model()
model.set_compartments(compartments)
model.set_rates(rates)
model.set_parameters(list(parameters.keys()))
replacements=[
    ['$\\gamma_T \, T_E$', '                       $\\gamma_T \, T_E$'],
    ['tanh', '\\tanh'],
    ['(0.5 + 0.5 \, \\tanh(k \, (T_m - T_S - T_E)))', 
    '\\Theta (T_m - \Sigma T)'],
    ['(C / (C + I_s + I_a))', 'c'],
    ['(1 / ((1 / \\gamma_I) - (T_E \, (1 / \\gamma_T) '
    '+ T_I \, ((1 / \\gamma_s) + (1 / \\gamma_T))) / (T_E + T_I)))', 
    '\\gamma_C']
            ]

ode_latex, ode_symbolic, ode = model.generate_ode()

def initiate_population(E_zero, parameters=parameters):

    # Initiating a population:
    initial_population = dict(
                            zip(model.compartments, 
                                np.zeros(len(model.compartments)))
                        )
    initial_population['S'] = parameters['N']

    # Number of people in contact tracing
    initial_population['C']  = data_to_fit['week_mean_cases'].to_numpy()[0] * (
                               1 / parameters['gamma_I']  
                             - 1 / parameters['gamma_T'] 
                             )
    initial_population['S'] -= initial_population['C']

    # Number of initially exposed people is a fit parameter
    initial_population['E']  = E_zero
    initial_population['S'] -= initial_population['E']

    # Total number of known closed cases on 1st of September: 24570
    # Assumption: equal number of unkown cases 
    # -- has probably a negligible effect on the results
    initial_population['R_k'] = 24570
    initial_population['S']  -= initial_population['R_k']
    initial_population['R_u'] = 24570
    initial_population['S']  -= initial_population['R_u']    

    # The rest of the population
    # I_a (0) and I_s(0) are functions of themselves, thus are 
    # determined via fixed-point iteration
    def IaIs(IaIs):
        I_a_guess = IaIs[0]
        I_s_guess = IaIs[1]
        c = initial_population['C'] / (
            initial_population['C'] + I_a_guess + I_s_guess
        )
        I_a = (1 - parameters['kappa'] * c) * parameters['p'] \
            * parameters['gamma_E'] * E_zero / parameters['gamma_I']
        I_s = (1 - parameters['kappa'] * c) * (1 - parameters['p']) \
            * parameters['gamma_E'] * E_zero / parameters['gamma_s']            
        return np.array([I_a, I_s])
    
    IaIs_guess = np.array([
                       parameters['p'] * initial_population['C'],
                       (1 - parameters['p']) * initial_population['C'],
                          ])
    
    IaIs_new = IaIs(IaIs_guess)
    count = 0

    while np.linalg.norm(IaIs_new - IaIs_guess) > 1:
        IaIs_guess = IaIs_new.copy()
        IaIs_new = IaIs(IaIs_guess)
        count += 1
        if count > 100:
            exit("Fixed-point iteration failed " 
                 "-- I_{a,s}(0) cannot be determined.")


    initial_population['I_a'] = IaIs_new[0]
    initial_population['S']  -= initial_population['I_a']

    initial_population['I_s'] = IaIs_new[1]
    initial_population['S']  -= initial_population['I_s']

    # We can now estimate T_E(0) and T_I(0)

    c_zero = initial_population['C'] / (
               initial_population['C'] 
             + initial_population['I_s'] 
             + initial_population['I_a']
             )

    initial_population['T_E'] = parameters['gamma_E'] * parameters['kappa'] \
                              * initial_population['E'] * c_zero \
                              / parameters['gamma_T']
    initial_population['S']  -= initial_population['T_E']
    initial_population['T_I'] = initial_population['I_s'] \
                              * parameters['gamma_s'] /  parameters['gamma_T'] 
    initial_population['S']  -= initial_population['T_I']

    # Finally, estimate T_S(0) by fixed point iteration
    # Current number of susceptibles:
    S_current = initial_population['S'] # final number will be a function of T_S(0)
    fT_S = lambda T_S : (S_current - T_S) * parameters['alpha'] \
                       * initial_population['C'] / (
                                    parameters['N'] * parameters['gamma_T']
                                                    )

    T_S_guess = fT_S(0)
    T_S_new = fT_S(T_S_guess)
    
    # Fixed point iteration:
    count = 0
    while abs(T_S_new - T_S_guess) > 1e-3:
        T_S_guess = T_S_new 
        T_S_new = fT_S(T_S_guess) 
        count += 1
        if count > 100:
            exit("Fixed-point iteration failed "
                 "-- T_S(0) cannot be determined.")


    initial_population['T_S'] =  T_S_new
    initial_population['S']  -= initial_population['T_S'] 

    parameters_unlim = parameters.copy()
    parameters_unlim['T_m'] = parameters['N']

    initial_population = model.initiate_exponential(
        initial_population, parameters_unlim, np.arange(0, 10), 
        ['T_S', 'T_E', 'I_a', 'I_s', 'T_I', 'C'], 'S', 
        tol=1, verbose=False, scale=1, max_iterations=100,
    )

    return initial_population

def cost_function(fit_parameters, data_cases, data_tests):
    """
    Cost function for least-squares optimization.

    fit_parameters[0] = E(0), range: [0, week_mean_cases[1] * 10]
    fit_parameters[1] = alpha,   range: [0, 1]
    fit_parameters[2] = kappa,   range: [0, 1]
    fit_parameters[3] = T_m,     range: [0, N]
    fit_parameters[4] = beta,    range: [0, 0.6]
    fit_parameters[5] = gamma_s, range: [1/5, 1]
    fit_parameters[6] = d,       range: [0, 1]
    """

    simulation_parameters            = parameters.copy()
    simulation_parameters['alpha']   = fit_parameters[1]
    simulation_parameters['kappa']   = fit_parameters[2]
    simulation_parameters['T_m']     = fit_parameters[3]
    simulation_parameters['beta']    = fit_parameters[4]
    simulation_parameters['gamma_s'] = fit_parameters[5]
    simulation_parameters['d']       = fit_parameters[6]
    
    # Initiating a simulation:
    initial_population = initiate_population(
        fit_parameters[0], 
        parameters = simulation_parameters,
        )

    # Run a simulation
    simulation_time = np.arange(0, len(data_tests))

    population = odeint(
        ode, 
        np.array(list(initial_population.values())), 
        simulation_time, 
        args = (list(simulation_parameters.values()),)
        )
    
    iC   = np.argwhere(np.array(model.compartments) == 'C')[0][0]
    iR_k = np.argwhere(np.array(model.compartments) == 'R_k')[0][0]
    iT_S = np.argwhere(np.array(model.compartments) == 'T_S')[0][0]
    iT_E = np.argwhere(np.array(model.compartments) == 'T_E')[0][0]
    iT_I = np.argwhere(np.array(model.compartments) == 'T_I')[0][0]

    simulated_cases = population[:, iC] + population[:, iR_k]
    simulated_daily_cases = simulated_cases[1:] - simulated_cases[0:-1]

    simulated_daily_tests = population[:, iT_S] \
                          + population[:, iT_E] \
                          + population[:, iT_I]


    cost_cases = (data_cases - simulated_daily_cases) \
               / data_cases #  
    cost_tests = (data_tests - simulated_daily_tests) \
               / data_tests  # np.linalg.norm(week_mean_tests) #  

    return np.append(cost_cases, cost_tests)


ranges = (
    [0,                      0, 0,         0,   0, 0.2, 0.0,], 
    [day_four_cases  * 10, 100, 1, max_tests, 0.6, 1.0, 1.0,]
    )

scales = [(ranges[1][i] - ranges[0][i]) for i in range(len(ranges[0]))]
scales = np.array(scales)

def fit(fit_parameters_guess, data_cases, data_tests):

    result = least_squares(
        cost_function, 
        fit_parameters_guess, 
        bounds=ranges,
        x_scale=scales,
        args=(data_cases, data_tests,)
        )

    optimal_parameters = parameters.copy()

    optimal_parameters['alpha']   = result['x'][1] # ,   range: [0, 10]
    optimal_parameters['kappa']   = result['x'][2] # ,   range: [0, 1]
    optimal_parameters['T_m']     = result['x'][3]
    optimal_parameters['beta']    = result['x'][4]
    optimal_parameters['gamma_s'] = result['x'][5]
    optimal_parameters['d']       = result['x'][6]

    optimal_initial_population = initiate_population(
            result['x'][0],
            parameters=optimal_parameters
            )
    
    return optimal_initial_population, optimal_parameters


def results(solution, model):

    results = {}
 
    if len(model.compartments) == 4:
        
        iI = np.argwhere(np.array(model.compartments) == 'I')[0][0]
        iR = np.argwhere(np.array(model.compartments) == 'R')[0][0]

        cases = solution[:, iI] + solution[:, iR]
        daily_cases = cases[1:] - cases[0:-1]

        daily_tests = np.zeros(solution.shape[0])
        daily_infections = daily_cases
        
    else:

        iC = np.argwhere(np.array(model.compartments) == 'C')[0][0]
        iI_a = np.argwhere(np.array(model.compartments) == 'I_a')[0][0]
        iI_s = np.argwhere(np.array(model.compartments) == 'I_s')[0][0]
        iR_k = np.argwhere(np.array(model.compartments) == 'R_k')[0][0]
        iR_u = np.argwhere(np.array(model.compartments) == 'R_u')[0][0]
        iT_S = np.argwhere(np.array(model.compartments) == 'T_S')[0][0]
        iT_E = np.argwhere(np.array(model.compartments) == 'T_E')[0][0]
        iT_I = np.argwhere(np.array(model.compartments) == 'T_I')[0][0]

        cases = solution[:, iC] + solution[:, iR_k]
        daily_cases = cases[1:] - cases[:-1]
        daily_tests = solution[:, iT_S] + solution[:, iT_E] + solution[:, iT_I]
        
        infections = (solution[:, iC] + solution[:, iI_a] + solution[:, iI_s] 
                    + solution[:, iR_u] + solution[:, iR_k])
        daily_infections = infections[1:] - infections[:-1]

        cumul_tests = np.zeros(daily_tests.shape)

        for day, n_test in enumerate(daily_tests):
            # if day == 0:
            #     cumul_tests[day] = daily_tests[day]
            # else:
            #     cumul_tests[day] = cumul_tests[day - 1] + daily_tests[day]
            cumul_tests[day] = cumul_tests[day - 1] + daily_tests[day]

    results['cases'] = daily_cases
    results['tests'] = daily_tests
    results['infections'] = daily_infections
    results['cumul_cases'] = cases[1:] - cases[0]
    results['cumul_tests'] = cumul_tests
    
    return results