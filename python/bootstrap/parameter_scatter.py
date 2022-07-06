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
from epyestim import estimate_r
from epyestim.distributions import discretise_gamma
import fitSEIRTC
import h5py
from pathlib import Path
sys.path.append('../')
import compartmental
from matplotlib import ticker

parameters_SEIRTC   = pickle.load(open("../parameters_SEIRTC.p", "rb"))

datapath = Path("bootstrap.hdf5")
datafile = h5py.File(datapath, 'r')
num_bootstrap = datafile['meta']['n_bootstrap'][()]
datafile.close()

pars = {}

for par in parameters_SEIRTC:
    pars[par] = []

pars['E'] = []

for n in range(1, num_bootstrap + 1):
    datafile = h5py.File(datapath, 'r')

    pars['E'].append(datafile[f'{n}']['init_pop']['E'][()])

    for key in datafile[f'{n}']['pars'].keys():
        pars[key].append(datafile[f'{n}']['pars'][key][()])

    datafile.close()

fit_pars = ['E', 'alpha', 'kappa', 'T_m', 'beta', 'gamma_s', 'd']
par_labels = [
    '$E(0)$', '$\\alpha$', '$\\kappa$', '$T_m$', '$\\beta$', '$\\gamma_s$', '$d$'
    ]

for par in fit_pars:
    pars[par] = np.array(pars[par])
    print(f"mean({par}) = {np.mean(pars[par]):05.05f}," 
          f"stddev({par}) = {np.std(pars[par]):05.05f}")

# Remove outliers
mean_E = np.mean(pars['E'])
std_E = np.std(pars['E'])
n_outliers = 0

for i, E in enumerate(pars['E']):
    if abs(E - mean_E) > 3 * std_E: 
        n_outliers += 1
        for par in pars: 
            pars[par] = np.delete(pars[par], i)

print(f"{n_outliers} elements ruled as outliers")

fig, axes = plt.subplots(7, 7, figsize=(8, 8))
# plt.ticklabel_format(style='sci', axis='x')



for i, par_i in enumerate(fit_pars):
    for j, par_j in enumerate(fit_pars):

        axes[i, j].plot(pars[par_j], pars[par_i], '.', ms=2, alpha=0.5)

        if i == 6:
            axes[i, j].set_xlabel(par_labels[j])
            axes[i, j].xaxis.set_major_locator(ticker.MaxNLocator(nbins = 2))
            axes[i, j].tick_params(axis="x", labelrotation=-60)
        else:
            axes[i, j].xaxis.set_ticklabels([])
        
        if j == 0:
            axes[i, j].set_ylabel(par_labels[i])
            axes[i, j].yaxis.set_major_locator(ticker.MaxNLocator(nbins = 2))
        else:
            axes[i, j].yaxis.set_ticklabels([])

# fig.tight_layout()
fig.savefig("parameter_scatter.png", dpi=150)

# plt.show()