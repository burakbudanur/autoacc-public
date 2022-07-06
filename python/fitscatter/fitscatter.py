from unittest import case
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
from scipy.optimize import curve_fit
from epyestim import estimate_r
from epyestim.distributions import discretise_gamma
from pandas.core.series import Series
from scipy.stats import kstest
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.ticker import NullFormatter

sys.path.append('../')
import compartmental

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

    results['cases'] = daily_cases
    results['tests'] = daily_tests
    results['infections'] = daily_infections
    
    return results

colorpalette = sns.color_palette('colorblind', n_colors=5)

fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize=(9, 3), sharey=True)
labels = ['A', 'B', 'C']

for label, ax in zip(labels, axes):
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label+'.', transform=ax.transAxes + trans,
            fontsize='xx-large', va='bottom')
    # ax.set_xlim()

austria_data = pd.read_csv("../austria_data.csv")
data = austria_data.loc[(austria_data['date'] >= '2020-09-01') & 
                        (austria_data['date'] <= '2020-12-06')]

days = [dt.datetime.strptime(date, '%Y-%m-%d').date() 
        for date in data['date'].to_numpy()]

datafit = austria_data.loc[(austria_data['date'] >= '2020-09-01') & 
                           (austria_data['date'] <  '2020-11-03')]

daysfit = [dt.datetime.strptime(date, '%Y-%m-%d').date() 
           for date in datafit['date'].to_numpy()]

SEIR   = pickle.load(open("../model_SEIR.p", "rb"))
SEIRTC = pickle.load(open("../model_SEIRTC.p", "rb"))

init_pop_SEIR_rel = pickle.load(open("../initial_population_SEIR_rel.p", "rb"))
init_pop_SEIR_abs = pickle.load(open("../initial_population_SEIR_abs.p", "rb"))
init_pop_SEIRTC   = pickle.load(open("../initial_population_SEIRTC.p", "rb"))

parameters_SEIR_rel = pickle.load(open("../parameters_SEIR_rel.p", "rb"))
parameters_SEIR_abs = pickle.load(open("../parameters_SEIR_abs.p", "rb"))
parameters_SEIRTC   = pickle.load(open("../parameters_SEIRTC.p", "rb"))

SEIR.set_parameters(list(parameters_SEIR_rel))
SEIRTC.set_parameters(list(parameters_SEIRTC))

_, _, ode_SEIR = SEIR.generate_ode()
_, _, ode_SEIRTC = SEIRTC.generate_ode()

simulation_days =  np.arange(0, len(daysfit))

solution_SEIR_rel = odeint(ode_SEIR,
                           np.array(list(init_pop_SEIR_rel.values())),
                           simulation_days,
                           args = (list(parameters_SEIR_rel.values()),))

solution_SEIR_abs = odeint(ode_SEIR,
                           np.array(list(init_pop_SEIR_abs.values())),
                           simulation_days,
                           args = (list(parameters_SEIR_abs.values()),))

solution_SEIRTC = odeint(ode_SEIRTC,
                           np.array(list(init_pop_SEIRTC.values())),
                           simulation_days,
                           args = (list(parameters_SEIRTC.values()),))

results_SEIR_abs = results(solution_SEIR_abs, SEIR)
results_SEIR_rel = results(solution_SEIR_rel, SEIR)
results_SEIRTC   = results(solution_SEIRTC, SEIRTC)

case_data = datafit['week_mean_cases'].to_numpy()[1:]

axes[0].scatter(
    results_SEIR_rel['cases'], 
    case_data,
    color=colorpalette[2],
    label = 'SEIR$_1$',
    alpha=0.25
    )

axes[0].set_xlabel("Cases (SEIR$_1$)")
axes[0].set_ylabel("Cases (recorded)")

perc_diff = 100 * np.abs(results_SEIR_rel['cases'] - case_data) / case_data
print(f"max. %diff =  {np.max(perc_diff):02.02f}")

i_max = np.argwhere(perc_diff == np.max(perc_diff))[0][0]

axes[0].annotate(text = '', 
    xytext = (case_data[i_max], case_data[i_max]),
    xy = (results_SEIR_rel['cases'][i_max], case_data[i_max]),
    arrowprops=dict(arrowstyle='<->', color='crimson') 
)

# ks_res = kstest(results_SEIR_rel['cases'], case_data)
# print("kstest-SEIR_2:")
# print(ks_res)

axes[1].scatter(
    results_SEIR_abs['cases'], 
    case_data,
    color = colorpalette[0], 
    label = 'SEIR$_2$',
    alpha=0.25
    )

axes[1].set_xlabel("Cases (SEIR$_2$)")

perc_diff = 100 * np.abs(results_SEIR_abs['cases'] - case_data) / case_data
print(f"max. %diff =  {np.max(perc_diff):02.02f}")

i_max = np.argwhere(perc_diff == np.max(perc_diff))[0][0]

axes[1].annotate(text = '', 
    xytext = (case_data[i_max], case_data[i_max]),
    xy = (results_SEIR_abs['cases'][i_max], case_data[i_max]),
    arrowprops=dict(arrowstyle='<->', color='crimson') 
)

# ks_res = kstest(results_SEIR_abs['cases'], case_data)
# print("kstest-SEIR_1:")
# print(ks_res)

axes[2].scatter(
    results_SEIRTC['cases'], 
    case_data,
    color=colorpalette[3],
    label = 'SEIRTC',
    alpha=0.25
    )

axes[2].set_xlabel("Cases (SEIRTC)")

perc_diff = 100 * np.abs(results_SEIRTC['cases'] - case_data) / case_data
print(f"max. %diff =  {np.max(perc_diff):02.02f}")

axes[2].annotate(text = '', 
    xytext = (case_data[i_max], case_data[i_max]),
    xy = (results_SEIRTC['cases'][i_max], case_data[i_max]),
    arrowprops=dict(arrowstyle='<->', color='crimson') 
)

axins = zoomed_inset_axes(axes[2], zoom=4, loc='lower right')
# sub region of the original image
x1, x2 = 0.8 * case_data[i_max], 1.1 * case_data[i_max]
y1, y2 = 0.8 * case_data[i_max], 1.1 * case_data[i_max]

axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

axins.scatter(
    results_SEIRTC['cases'], 
    case_data,
    color=colorpalette[3],
    label = 'SEIRTC',
    alpha=0.25
    )

axins.annotate(text = '', 
    xytext = (case_data[i_max], case_data[i_max]),
    xy = (results_SEIRTC['cases'][i_max], case_data[i_max]),
    arrowprops=dict(arrowstyle='<->', color='crimson') 
)

# ks_res = kstest(results_SEIRTC['cases'], case_data)
# print("kstest-SEIRTC:")
# print(ks_res)

for ax in axes:

    ax.plot(
        [0.9 * np.min(case_data), 1.1 * np.max(case_data)], 
        [0.9 * np.min(case_data), 1.1 * np.max(case_data)],
        color = 'black', 
        ls = '--'
        )    

    ax.set_xscale('log')
    ax.set_yscale('log')


axins.plot(
    [0.9 * np.min(case_data), 1.1 * np.max(case_data)], 
    [0.9 * np.min(case_data), 1.1 * np.max(case_data)],
    color = 'black', 
    ls = '--'
    )    

axins.set_xscale('log')
axins.set_yscale('log')


axins.xaxis.set_major_formatter(NullFormatter())
axins.xaxis.set_minor_formatter(NullFormatter())

axins.yaxis.set_major_formatter(NullFormatter())
axins.yaxis.set_minor_formatter(NullFormatter())

# fix the number of ticks on the inset axes
# axins.yaxis.get_major_locator().set_params(nbins=2) 
# axins.xaxis.get_major_locator().set_params(nbins=2)
# axins.tick_params(labelleft=False, labelbottom=False)

mark_inset(axes[2], axins, loc1=2, loc2=3, fc="none", ec="0.5")

# xticks = axins.get_xticks()
# axins.set_xticklabels(['' for xtick in xticks])

# yticks = axins.get_yticks()
# axins.set_yticklabels(['' for ytick in yticks])

plt.tight_layout()
plt.savefig("fitscatter.png", dpi=150)
plt.show()