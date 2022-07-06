import numpy as np 
import matplotlib as mpl 
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
import pandas as pd
import seaborn as sns
import datetime as dt
import pickle 
from scipy.integrate import odeint
import sys
import matplotlib.transforms as mtransforms
from numpy.random import poisson
import fitSEIRTC
import h5py
from pathlib import Path
sys.path.append('../')
import compartmental

colorpalette = sns.color_palette('colorblind', n_colors=5)

austria_data = pd.read_csv("../austria_data.csv")
data = austria_data.loc[(austria_data['date'] >= '2020-09-01') & 
                        (austria_data['date'] <= '2020-12-06')]

days = [dt.datetime.strptime(date, '%Y-%m-%d').date() 
        for date in data['date'].to_numpy()]

datafit = austria_data.loc[(austria_data['date'] >= '2020-09-01') & 
                           (austria_data['date'] <  '2020-11-03')]

daysfit = [dt.datetime.strptime(date, '%Y-%m-%d').date() 
           for date in datafit['date'].to_numpy()]

simulation_days =  np.arange(0, len(daysfit))

SEIRTC = pickle.load(open("../model_SEIRTC.p", "rb"))

init_pop_SEIRTC   = pickle.load(open("../initial_population_SEIRTC.p", "rb"))
parameters_SEIRTC   = pickle.load(open("../parameters_SEIRTC.p", "rb"))

SEIRTC.set_parameters(list(parameters_SEIRTC))

_, _, ode_SEIRTC = SEIRTC.generate_ode()

datapath = Path("bootstrap.hdf5")
datafile = h5py.File(datapath, 'r')
num_bootstrap = 10 # datafile['meta']['n_bootstrap'][()]
datafile.close()

fig, axes = plt.subplots(nrows = 1, ncols = 4, figsize=(12, 3))
labels = ['A', 'B', 'C', 'D']

for label, ax in zip(labels, axes):

    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label+'.', transform=ax.transAxes + trans,
            fontsize='xx-large', va='bottom')

# original solution
solution_SEIRTC = odeint(
    ode_SEIRTC,
    np.array(list(init_pop_SEIRTC.values())),
    simulation_days,
    args = (list(parameters_SEIRTC.values()),)
    )

results_SEIRTC = fitSEIRTC.results(solution_SEIRTC, SEIRTC)
cases = results_SEIRTC['cases']
tests = results_SEIRTC['tests']

np.random.seed(1)

bootstrap_cases = np.zeros(cases.shape)
bootstrap_tests = np.zeros(tests.shape)

for day, n_case in enumerate(cases):
    bootstrap_cases[day] = poisson(n_case)

for day, n_test in enumerate(tests):
    bootstrap_tests[day] = poisson(n_test)

axes[0].plot(cases, color=colorpalette[0], lw=2.0, label='best-fit SEIRTC')
axes[0].plot(
    bootstrap_cases, color='black', lw=1.0, label='Poisson bootstrap'
    )

axes[1].plot(tests, color=colorpalette[3], lw=2.0, label='best-fit SEIRTC')
axes[1].plot(
    bootstrap_tests, color='black', lw=1.0, label='Poisson bootstrap'
    )

# Bootstraps 

init_pop = init_pop_SEIRTC.copy()
pars = parameters_SEIRTC.copy()

for n in range(1, num_bootstrap + 1):
    datafile = h5py.File(datapath, 'r')

    for key in datafile[f'{n}']['init_pop'].keys():
        init_pop[key] = datafile[f'{n}']['init_pop'][key][()]

    for key in datafile[f'{n}']['pars'].keys():
        pars[key] = datafile[f'{n}']['pars'][key][()]

    datafile.close()

    sol = odeint(
        ode_SEIRTC,
        np.array(list(init_pop.values())),
        simulation_days,
        args = (list(pars.values()),)
        )

    results = fitSEIRTC.results(sol, SEIRTC)
    cases = results['cases']
    tests = results['tests']    
    axes[2].plot(cases, color=colorpalette[0], alpha=0.2)
    axes[3].plot(tests, color=colorpalette[3], alpha=0.2)

# axes[1].set_yscale('log')
# axes[1].set_yscale('log')

for ax in axes:
    ax.set_xlim(0, 20)

for i in [0, 2]:
    axes[i].set_ylim(350, 800)
    axes[i].set_ylabel("Cases")

for i in [1, 3]:
    axes[i].set_ylim(9000, 18000)
    axes[i].set_ylabel("Tests")

axes[0].legend()
axes[1].legend()

plt.tight_layout()
fig.savefig("bootstrap.png", dpi=150)

# plt.show()
