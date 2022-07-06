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

axes = plt.figure(figsize=(9, 9)).subplot_mosaic([['A', 'B'], 
                                                  ['C', 'D'],
                                                  ['E', 'F']])
fig = plt.gcf()

for label, ax in axes.items():
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label+'.', transform=ax.transAxes + trans,
            fontsize='xx-large', va='bottom')
    ax.set_xlim()

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

print("SEIRTC parameters:")
print(parameters_SEIRTC)
print("SEIRTC initial population:")
print(init_pop_SEIRTC)

print("SEIR^2 parameters:")
print(parameters_SEIR_abs)
print("SEIR^2 initial population:")
print(init_pop_SEIR_abs)

print("SEIR^1 parameters:")
print(parameters_SEIR_rel)
print("SEIR^1 initial population:")
print(init_pop_SEIR_rel)



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

axes['A'].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
axes['A'].plot(
    daysfit, 
    datafit['week_mean_cases'].to_numpy(),
    color=colorpalette[0], 
    lw = 2,
    label='Data')

axes['A'].plot(
    daysfit[1:],
    results_SEIR_abs['cases'], 
    color=colorpalette[1],
    lw = 2,
    ls = '-.',
    label = 'SEIR$_2$')

axes['A'].plot(
    daysfit[1:],
    results_SEIR_rel['cases'], 
    color=colorpalette[2],
    lw = 2,
    ls = ':',
    label = 'SEIR$_1$')

axes['A'].plot(
    daysfit[1:],
    results_SEIRTC['cases'], 
    color=colorpalette[3],
    lw = 3,
    ls = '--',
    label = 'SEIRTC')

axes['A'].set_yscale('log')

axes['A'].set_ylabel("Cases", fontsize='large')
axes['A'].tick_params(labelsize='medium')
axes['A'].legend(fontsize='medium', ncol=2, columnspacing=0.5)

axes['B'].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
axes['B'].plot(
    daysfit, 
    datafit['week_mean_tests'].to_numpy(),
    color=colorpalette[0], 
    lw = 2,
    label='Data')

axes['B'].plot(
    daysfit, 
    results_SEIRTC['tests'],  
    color=colorpalette[3],
    lw = 3,
    ls = '--',
    label = 'SEIRTC')

axes['B'].set_ylabel("Tests", fontsize='large')
axes['B'].tick_params(labelsize='medium')
axes['B'].ticklabel_format(axis='y', style = 'sci', scilimits = (1,2) ,useMathText=True)
axes['B'].legend(fontsize='medium')

lstyles = ['-', '--', '-.', '--']
count_comp = 0

for i_comp, comp in enumerate(SEIRTC.compartments):
    
    if not(comp in ['E', 'I_s', 'I_a', 'C', 'T_S', 'T_E', 'T_I', 'R_u', 'R_k']):
        continue
    elif comp == 'C':
        sol_C = solution_SEIRTC[:, i_comp]
    elif comp == 'I_a':
        sol_I_a = solution_SEIRTC[:, i_comp]
    elif comp == 'I_s':
        sol_I_s = solution_SEIRTC[:, i_comp]

    axes['C'].plot(
        daysfit,
        solution_SEIRTC[:, i_comp],
        label='$'+comp+'$',
        color=colorpalette[count_comp % 5], 
        ls = lstyles[count_comp % 4]
    )
    count_comp += 1

axes['C'].set_ylabel("Compartment population", fontsize='large')
axes['C'].set_yscale('log')
axes['C'].set_ylim(10, 1.5e5)
axes['C'].tick_params(labelsize='medium')
axes['C'].legend(fontsize='medium', ncol=5, framealpha=1.0, columnspacing=0.5)

axes['D'].plot(
    daysfit[1:], 
    results_SEIRTC['infections']/results_SEIRTC['cases'], 
    color=colorpalette[3], 
    lw = 3, 
    ls = '--')

axes['D'].set_ylabel('Infections / Cases', fontsize='large')

axes['E'].plot(
    daysfit, 
    datafit['week_mean_cases'].to_numpy() / datafit['week_mean_tests'].to_numpy(),
    color = colorpalette[0],
    label='Data',
    linestyle='-'
)

axes['E'].plot(
    daysfit[1:], 
    results_SEIRTC['cases'] / results_SEIRTC['tests'][1:],  
    color=colorpalette[3],
    lw = 3,
    ls = '--',
    label = 'SEIRTC')

axes['E'].legend(fontsize='medium', ncol = 1, framealpha=1.0, loc=2, columnspacing=0.5)
axes['E'].set_ylabel('Cases / Tests', fontsize='large')

si_mean = 4.46     # Epidemiologische Parameter des COVID19 Ausbruchs 
si_std_dev = 2.63  # - Update 30.10.2021, Österreich, 2020/2021

si_scale = si_std_dev ** 2 / si_mean
si_shape = si_mean / si_scale

window_size = 7

si_dist = discretise_gamma(si_shape, si_scale)

def eff_repr(data):

    if not type(data) == Series:
        data = Series(data) 

    posteriors = estimate_r.estimate_r(
        data,
        si_dist,
        a_prior = 1,
        b_prior = 5,
        window_size = window_size
    )

    Rt = (posteriors['a_posterior'].to_numpy() 
        * posteriors['b_posterior'].to_numpy())

    return Rt

eff_repr_data = eff_repr(datafit['week_mean_cases'])
eff_repr_C = eff_repr(results_SEIRTC['cases'])
eff_repr_IC = eff_repr(results_SEIRTC['cases'] + results_SEIRTC['infections'])

axes['F'].plot(daysfit[6:], 
               eff_repr_data, 
               color=colorpalette[0], 
               label='$R_{t}$')


axes['F'].plot(daysfit[7:], 
               eff_repr_C, 
               color = colorpalette[1],
               label='$R_{t}^{(C)}$',
               linestyle='--')

axes['F'].plot(daysfit[7:], 
               eff_repr_IC, 
               color = colorpalette[2],
               label='$R_{t}^{(I+C)}$',
               linestyle='-.')

axes['F'].legend(fontsize='medium', ncol = 3, framealpha=1.0, loc=4, columnspacing=0.5)
axes['F'].set_ylabel('Effect. repr. num.', fontsize='large')

axes['F'].set_ylim(0.75, 1.65)

fig = plt.gcf()
fig.autofmt_xdate()

# xlims = axes['A'].get_xlim()
for label, ax in axes.items():
    ax.axes.set_xlim(dt.date(2020, 8, 30), dt.date(2020, 11, 5))


plt.tight_layout()
plt.savefig("fits.png", dpi=150)
plt.show()