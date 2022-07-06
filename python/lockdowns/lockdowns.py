import numpy as np
import matplotlib as mpl 
from matplotlib import pyplot as plt
import seaborn as sns
import pickle 
from scipy.integrate import odeint
from scipy.optimize import root_scalar
import sys
from copy import deepcopy
import pandas as pd
import matplotlib.transforms as mtransforms
import datetime as dt
from matplotlib import dates as mdates

colorpalette = sns.color_palette('colorblind', n_colors=5)

sys.path.append('../')
import compartmental

SEIR   = pickle.load(open("../model_SEIR.p", "rb"))
SEIRTC = pickle.load(open("../model_SEIRTC.p", "rb"))

init_pop_SEIR = pickle.load(open("../initial_population_SEIR_abs.p", "rb"))
init_pop_SEIRTC   = pickle.load(open("../initial_population_SEIRTC.p", "rb"))

parameters_SEIR = pickle.load(open("../parameters_SEIR_abs.p", "rb"))
parameters_SEIRTC   = pickle.load(open("../parameters_SEIRTC.p", "rb"))

_, _, ode_SEIR = SEIR.generate_ode()
_, _, ode_SEIRTC = SEIRTC.generate_ode()

austria_data = pd.read_csv("../austria_data.csv")
data = austria_data.loc[
    (austria_data['date'] >= '2020-09-01') & 
    (austria_data['date'] <= '2020-12-06')
                    ]
dates = data['date'].to_numpy()

def rotateTickLabels(ax, rotation, which, rotation_mode='anchor', ha='right'):
    """ Rotates the ticklabels of a matplotlib Axes
    Parameters
    ----------
    ax : matplotlib Axes
        The Axes object that will be modified.
    rotation : float
        The amount of rotation, in degrees, to be applied to the labels.
    which : string
        The axis whose ticklabels will be rotated. Valid values are 'x',
        'y', or 'both'.
    rotation_mode : string, optional
        The rotation point for the ticklabels. Highly recommended to use
        the default value ('anchor').
    ha : string
        The horizontal alignment of the ticks. Again, recommended to use
        the default ('right').
    Returns
    -------
    None
    """

    if which == 'both':
        rotateTickLabels(ax, rotation, 'x', rotation_mode=rotation_mode, ha=ha)
        rotateTickLabels(ax, rotation, 'y', rotation_mode=rotation_mode, ha=ha)
    else:
        if which == 'x':
            axis = ax.xaxis

        elif which == 'y':
            axis = ax.yaxis

        for t in axis.get_ticklabels():
            t.set_horizontalalignment(ha)
            t.set_rotation(rotation)
            t.set_rotation_mode(rotation_mode)


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


def seven_day_incidence(days, solution, model, population):
    last_week = np.argwhere(days - (days[-1] - 7) >= 0).reshape(-1)

    if len(model.compartments) == 4:
        # SEIR model
        
        iI   = np.argwhere(np.array(model.compartments) == 'I')[0][0]
        iR   = np.argwhere(np.array(model.compartments) == 'R')[0][0]

        cases = solution[:, iI] + solution[:, iR]
    
    else:
        # SEIRTC model

        iC   = np.argwhere(np.array(model.compartments) == 'C')[0][0]
        iR_k = np.argwhere(np.array(model.compartments) == 'R_k')[0][0]
        iR_u = np.argwhere(np.array(model.compartments) == 'R_u')[0][0]

        cases = solution[:, iC] + solution[:, iR_k]

    cases_last_week = cases[last_week[-1]] - cases[last_week[0]]
    
    return (cases_last_week / population) * 100e3


def f_lockdown_day(day, lockdown_incidence, model, initial_population, parameters):
    
    _, _, ode = model.generate_ode()
    days = np.array([0, day-7, day])
    solution = odeint(ode, 
                      np.array(list(initial_population.values())), 
                      days, 
                      args = (list(parameters.values()),))
    incidence = seven_day_incidence(days, solution, model, parameters['N'])

    return incidence - lockdown_incidence


def beta_lockdown(day, beta, lockdown_day, c_lockdown = 0.45, steepness = 1):

    bl  = beta
    bl -= (1 - c_lockdown) * beta * (
            0.5 + 0.5 * np.tanh(steepness * (day - lockdown_day))
        )

    return bl    


def f_reopen_day(day, lockdown_incidence, lockdown_day, model, 
                 initial_population, parameters):
    
    # beta_l = lambda day: beta_lockdown(day, parameters['beta'], lockdown_day)
    def beta_l(day):
        return beta_lockdown(day, parameters['beta'], lockdown_day)

    model_lockdown = deepcopy(model)
    model_lockdown.set_inputs(['beta_l',])

    if len(model_lockdown.compartments) == 4:
        # SEIR model_lockdown
        model_lockdown.edges[('S', 'E')]['label'] = 'beta_l (t) * S *  I / N'
    else:
        # SEIRTC model_lockdown
        model_lockdown.edges[('S', 'E')]['label'] = \
            'beta_l (t) * S *  (0.35 * I_a + I_s) / N'

    _, _, ode = model_lockdown.generate_ode()
    days = np.array([0, day-7, day])
    solution = odeint(ode, 
                      np.array(list(initial_population.values())), 
                      days, 
                      args = (list(parameters.values()), [beta_l]))
    incidence = seven_day_incidence(days, solution, model_lockdown, 
                                    parameters['N'])

    return incidence - (lockdown_incidence - 10)


lockdown_incidences = np.arange(35, 150, 1)
try: 
    lockdown_durations_SEIR = np.loadtxt("lockdown_durations_SEIR.dat")
    lockdown_durations_SEIRTC = np.loadtxt("lockdown_durations_SEIRTC.dat")
except:
    lockdown_durations_SEIR = []
    lockdown_durations_SEIRTC = []

there = (
    np.array(lockdown_durations_SEIR).shape == lockdown_incidences.shape 
    and np.array(lockdown_durations_SEIRTC).shape == lockdown_incidences.shape
    )

if not(there):
    for incidence_lockdown in lockdown_incidences:

        lockdown_day_SEIR = root_scalar(f_lockdown_day, 
                                        args=(incidence_lockdown, 
                                        SEIR, 
                                        init_pop_SEIR, 
                                        parameters_SEIR,), 
                                        x0=10, 
                                        x1=20).root
        print(lockdown_day_SEIR)

        reopen_day_SEIR = root_scalar(f_reopen_day,
                                    args=(incidence_lockdown, 
                                            lockdown_day_SEIR, 
                                            SEIR, 
                                            init_pop_SEIR, 
                                            parameters_SEIR,),
            bracket = [lockdown_day_SEIR, lockdown_day_SEIR + 2 * lockdown_day_SEIR]
                                    ).root

        print(reopen_day_SEIR)

        lockdown_durations_SEIR.append(reopen_day_SEIR - lockdown_day_SEIR)


        lockdown_day_SEIRTC = root_scalar(f_lockdown_day, 
                                        args=(incidence_lockdown, 
                                        SEIRTC, 
                                        init_pop_SEIRTC, 
                                        parameters_SEIRTC,), 
                                        x0=10, 
                                        x1=50).root
        print(lockdown_day_SEIRTC)

        reopen_day_SEIRTC = root_scalar(f_reopen_day,
                                    args=(incidence_lockdown, 
                                            lockdown_day_SEIRTC, 
                                            SEIRTC, 
                                            init_pop_SEIRTC, 
                                            parameters_SEIRTC,),
            bracket = [lockdown_day_SEIRTC, lockdown_day_SEIRTC + 2 * lockdown_day_SEIRTC]
                                    ).root

        print(reopen_day_SEIRTC)

        lockdown_durations_SEIRTC.append(reopen_day_SEIRTC - lockdown_day_SEIRTC)

    lockdown_durations_SEIR   = np.array(lockdown_durations_SEIR)
    np.savetxt("lockdown_durations_SEIR.dat", lockdown_durations_SEIR)

    lockdown_durations_SEIRTC = np.array(lockdown_durations_SEIRTC)    
    np.savetxt("lockdown_durations_SEIRTC.dat", lockdown_durations_SEIRTC)

fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(9, 3))
labels = ['A', 'B']

for label, ax in zip(labels, axes):
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label+'.', transform=ax.transAxes + trans,
            fontsize='xx-large', va='bottom')
    # ax.set_xlim()


axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

days = [
    dt.datetime.strptime(date, '%Y-%m-%d').date() 
    for date in data['date'].to_numpy()
    ]


model_lockdown = deepcopy(SEIRTC)
model_lockdown.set_inputs(['beta_l',])

model_lockdown.edges[('S', 'E')]['label'] = \
    'beta_l (t) * S *  (0.35 * I_a + I_s) / N'

lockdown_date = '2020-11-03'
lockdown_day = np.argwhere(dates == lockdown_date)[0][0]

# beta_l = lambda day: beta_lockdown(day, parameters['beta'], lockdown_day)
def beta_l(day):
    return beta_lockdown(day, parameters_SEIRTC['beta'], lockdown_day)

_, _, ode = model_lockdown.generate_ode()
simulation_days = np.arange(0, len(days))
solution = odeint(
    ode, 
    np.array(list(init_pop_SEIRTC.values())), 
    simulation_days, 
    args = (list(parameters_SEIRTC.values()), [beta_l])
    )

sim_results = results(solution, model=model_lockdown)

axes[0].plot(
    days, 
    data['week_mean_cases'].to_numpy(), 
    color=colorpalette[2], 
    label='Cases (7-day avg.)'
    )

axes[0].plot(
    days[1:], 
    sim_results['cases'], 
    color=colorpalette[4], 
    label='Sim. Cases'
    )

axes[0].plot(
    [days[lockdown_day], days[lockdown_day]],
    [0, 10000],
    '--',
    lw=2.0,
    color='black'
)

axes[0].legend(fontsize='medium')
axes[0].set_ylabel("Cases", fontsize='large')
axes[0].set_xlim(days[0], days[-1])    

axes[1].plot(
    lockdown_incidences, lockdown_durations_SEIR, 
    '-', label='SEIR$_2$', lw=2.0, color=colorpalette[0]
    )
axes[1].plot(
    lockdown_incidences, lockdown_durations_SEIRTC, 
    '--', label='SEIRTC', lw=2.0, color=colorpalette[3]
    )

axes[1].set_xlabel("Lockdown incidence", fontsize='large')
axes[1].set_ylabel("Lockdown duration", fontsize='large')
axes[1].legend(fontsize='medium')
axes[1].tick_params(labelsize='large')

rotateTickLabels(axes[0], 30, 'x', rotation_mode='anchor', ha='right')

plt.subplots_adjust(bottom=0.21)

# plt.subplots_adjust(left=0.1,
#                     bottom=0.1, 
#                     right=0.9, 
#                     top=0.9, 
#                     wspace=0.4, 
#                     hspace=0.4)

# plt.tight_layout()
plt.savefig("lockdowns.png", dpi=200)
plt.show()