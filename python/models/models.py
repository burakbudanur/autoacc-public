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

sys.path.append('../')
import compartmental

def results(solution, model):

    results = {}

    if len(model.compartments) == 4:
        
        iI   = np.argwhere(np.array(model.compartments) == 'I')[0][0]
        iR   = np.argwhere(np.array(model.compartments) == 'R')[0][0]

        cases = solution[:, iI] + solution[:, iR]
        daily_cases = cases[1:] - cases[0:-1]

        daily_tests = np.zeros(solution.shape[0])
        daily_infections = daily_cases
        
    else:

        iC   = np.argwhere(np.array(model.compartments) == 'C')[0][0]
        iR_k = np.argwhere(np.array(model.compartments) == 'R_k')[0][0]
        iR_u = np.argwhere(np.array(model.compartments) == 'R_u')[0][0]

        iT_S = np.argwhere(np.array(model.compartments) == 'T_S')[0][0]
        iT_E = np.argwhere(np.array(model.compartments) == 'T_E')[0][0]
        iT_I = np.argwhere(np.array(model.compartments) == 'T_I')[0][0]

        cases = solution[:, iC] + solution[:, iR_k]
        daily_cases = cases[1:] - cases[0:-1]

        daily_tests = solution[:, iT_S] + solution[:, iT_E] + solution[:, iT_I]
        daily_infections = solution[1:, iR_u] - solution[0:-1, iR_u] + daily_cases

    results['cases']      = daily_cases
    results['tests']      = daily_tests
    results['infections'] = daily_infections
    
    return results

colorpalette = sns.color_palette('colorblind', n_colors=4)

replacements=[
    ['0.35', '\\rho'],
    ['$\\gamma_T \, T_E$', '                       $\\gamma_T \, T_E$'],
    ['tanh', '\\tanh'],
    ['(0.5 + 0.5 \, \\tanh(k \, (T_m - T_S - T_E)))', '\\Theta (T_m - \Sigma T)'],
    ['(C / (C + I_s + I_a))', 'c'],
    ['\\kappa \, c \, \\Theta (T_m - \Sigma T)', 'g'],
    ['\\alpha \, S \, (C / N) \, \\Theta (T_m - \Sigma T)', 'f'],
    ['\\gamma_E \, E \, g', 'g \, \\gamma_E \, E'],
    ['(1 / ((1 / \\gamma_I) - (T_E \, (1 / \\gamma_T) + T_I \, ((1 / \\gamma_s) + (1 / \\gamma_T))) / (T_E + T_I)))', '\\gamma_C']
                          ]

axes = plt.figure(figsize=(8, 6)).subplot_mosaic(
    """
    AAAA
    BBBB
    BBBB
    """
)

fig = plt.gcf()

for label, ax in axes.items():
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label + '.', transform=ax.transAxes + trans,
            fontsize='xx-large', va='bottom')


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

SEIR.set_parameters(list(parameters_SEIR_rel.keys()))
SEIRTC.set_parameters(list(parameters_SEIRTC.keys()))

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

SEIR.visualize(ax=axes['A'], scale=0.85)
# axes['A'].text(-0.015, 0.015, 'A', fontsize=24,             
#                horizontalalignment="center", 
#                verticalalignment="center",)

SEIRTC.visualize(ax=axes['B'], replacements=replacements, scale=0.85)
# axes['B'].text(-0.015, 1.0, 'B', fontsize=24,             
#                horizontalalignment="center", 
#                verticalalignment="center",)

# correct overlaps:
arrows = []
positions = []

for child in axes['B'].get_children():
    if type(child) == mpl.patches.FancyArrowPatch:
        arrows.append(child)
        positions.append(child._posA_posB.copy())

for arrow, position in zip(arrows, positions):
    pos_A = position[0]
    pos_B = position[1]
    if [pos_B, pos_A] in positions:
        pos_vec = (np.array([pos_B[0], pos_B[1]])
                 - np.array([pos_A[0], pos_A[1]]))
        len_pos_vec = np.linalg.norm(pos_vec)
        rot_pos_vec = np.array([pos_vec[1], -pos_vec[0]]) # -90deg rotation
        shift_vec = rot_pos_vec / np.linalg.norm(rot_pos_vec) # make unit vec

        new_pos_A = (pos_A[0] + 0.0025 * len_pos_vec * shift_vec[0], 
                     pos_A[1] + 0.0025 * len_pos_vec * shift_vec[1])
        new_pos_B = (pos_B[0] + 0.0025 * len_pos_vec * shift_vec[0], 
                     pos_B[1] + 0.0025 * len_pos_vec * shift_vec[1])

        arrow.set_positions(new_pos_A, new_pos_B)

plt.tight_layout()
plt.savefig("models.png", dpi=150)
plt.show()

