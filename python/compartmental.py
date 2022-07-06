import numpy as np
import networkx as nx
import sympy as sp
import matplotlib.pyplot as plt 
import matplotlib as mpl
from sys import exit
from scipy.optimize import curve_fit
from scipy.integrate import odeint


class Model(nx.DiGraph):
    """
    Base class for compartmental models. 

    See also:
    ---------
    nx.DiGraph
    """


    def set_compartments(self, compartments):
        """
        Set model compartments.
        Wrapper around nx.DiGraph.add_nodes_from 

        Parameters
        ----------
        compartments: list of tuples such as 

        compartments = [
            ('S'    , {"layer" : 1}), 
            ('I'    , {"layer" : 2}), 
            ('R'    , {"layer" : 3})
        ]
        
        the layer attributes are only used for visualization 
        and have no effect on the model function
        """

        self.add_nodes_from(compartments)
        self.compartments = list(self.nodes)



    def set_parameters(self, parameters):
        """
        Set model parameters.

        Parameters
        ----------
        parameters: list of strings such as 

        parameters = [
            'beta',
            'gamma',
        ]

        Note: Most of the lower-case Greek letters are going to
        be translated to latex symbols in visualize(), see below.
        """

        self.parameters = parameters


    def set_inputs(self, inputs):

        """
        Set model inputs.

        Parameters
        ----------
        inputs: list of strings such as 

        inputs = [
            'f(t)',
            'g(t)',
        ]

        The functions f(t) and g(t) will be interpreted 
        as the model ODE's time-dependent inputs. For an example, see

        Example_SIR_seasonality.ipynb
        """


        self.inputs = inputs


    def set_rates(self, rates):
        """
        Set transition rates of the compartmental model.
        Wrapper around nx.DiGraph.add_edges_from 

        Parameters
        ----------
        rates: list of tuples, such as 

        rates = [
            ('S', 'I', {"label" : "beta * S * I / N"}), 
            ('I', 'R', {"label" : "gamma * I"})
        ]
        
        the label attributes of the edges should be sympy-interpretable strings,
        composed of model compartments and parameters 
        """

        self.add_edges_from(rates)


    def visualize(self, figsize=None, ax=None, replacements = [], scale = 1.0, show_rates=True):
        """
        Generate a network visualization of the compartmental model

        Parameters
        ----------
        figsize: see plt.figure
        replacements: list of strings to be replaced by short-hands such as 
        replacements=[
            ['(0.5 + 0.5 \, tanh(k \, (T_m - T_t)))', '(\Theta [T_m - T_t])'],
            [') \\, (', ')$\n$\\times(']
        ]
        """

        def label_to_latex(labelstr):
            """
            Convert labelstr to latex-interpretable form for visualization
            """
            # Small Greek letters:
            latex_label = '$'+labelstr.replace('*', '\,')+'$'
            latex_label = latex_label.replace('alpha', '\\alpha')
            latex_label = latex_label.replace('beta', '\\beta')
            latex_label = latex_label.replace('gamma', '\\gamma')
            latex_label = latex_label.replace('delta', '\\delta')
            latex_label = latex_label.replace('epsilon', '\\epsilon')
            latex_label = latex_label.replace('zeta', '\\zeta')
            latex_label = latex_label.replace(' eta', ' \\eta')
            latex_label = latex_label.replace('(eta', '(\\eta')
            latex_label = latex_label.replace('$eta', '$\\eta')
            latex_label = latex_label.replace('theta', '\\theta')
            latex_label = latex_label.replace('kappa', '\\kappa')
            latex_label = latex_label.replace('lambda', '\\lambda')
            latex_label = latex_label.replace('mu', '\\mu')
            latex_label = latex_label.replace('nu', '\\nu')
            latex_label = latex_label.replace('xi', '\\xi')
            latex_label = latex_label.replace('pi', '\\pi')
            latex_label = latex_label.replace('rho', '\\rho')
            latex_label = latex_label.replace('sigma', '\\sigma')
            latex_label = latex_label.replace('phi', '\\phi')
            latex_label = latex_label.replace('chi', '\\chi')
            latex_label = latex_label.replace('psi', '\\psi')
            latex_label = latex_label.replace('omega', '\\omega')

            # Additional replacements
            for replacement in replacements:
                latex_label = latex_label.replace(replacement[0], 
                                                  replacement[1])

            return latex_label


        node_labels    = {node:label_to_latex(node) 
                          for node in list(self.nodes)}
        node_layers    = {node:self.nodes[node]['layer'] 
                          for node in list(self.nodes)}
        node_positions = {}

        i_node_layer = 0 # Counter for the nodes in a layer
        for node in list(self.nodes):

            layer = node_layers[node]
            num_nodes_layer = list(node_layers.values()).count(layer)
            y_nodes = np.arange(
                0 - num_nodes_layer/2 + 0.5, 
                num_nodes_layer - num_nodes_layer/2 + 0.5
            )

            pos_x = (np.float(node_layers[node] - 1) 
                  / max(node_layers.values()))
            pos_y = y_nodes[i_node_layer]

            node_positions[node] = np.array([pos_x, pos_y])

            i_node_layer += 1
            if i_node_layer == num_nodes_layer: i_node_layer = 0

        edge_labels = {
            edge:"\n\n"+label_to_latex(self.edges[edge]['label']) 
            for edge in list(self.edges)
        }
        
        # Generate a figure instance, if necessary
        if ax == None:
            
            if figsize == None:
                fig = plt.figure(figsize=(6,6))
            else: 
                fig = plt.figure(figsize=figsize)
            
            ax = fig.gca()

        
        mpl.rcParams["scatter.edgecolors"] = 'black'

        nx.draw(
            self, 
            ax = ax,
            pos=node_positions, 
            labels=node_labels, 
            node_size=800 * scale,
            font_size=16 * scale,
            node_color='white'
        )

        if show_rates:
            nx.draw_networkx_edge_labels(
                self, 
                ax = ax,
                pos=node_positions,
                edge_labels = edge_labels,         
                font_size = 14 * scale,
                label_pos=0.5,
                bbox=dict(fc="w", ec="w", alpha=0, zorder=100),
                font_color='red'
            )            
        

        # correct overlaps:
        arrows = []
        positions = []

        for child in ax.get_children():
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

        return


    def generate_ode(self):
        """
        Generate the right hand side of the ordinary differential equation 
        described by the compartments and transition rates
        """
        
        # dict for sympify

        if hasattr(self, "inputs"):
            
            ns     = {
                sym : sp.Symbol(sym) for sym in 
                self.compartments + self.parameters
            }
            
            for fun in self.inputs:
                ns[fun] = sp.Function(fun)

            input_symbols = [
                sp.Function(fun) for fun in self.inputs
            ]

        else:

            ns     = {
                sym : sp.Symbol(sym) for sym in 
                self.compartments + self.parameters
            } 
        

        # Symbolic counterparts of the model compartments and parameters
        compartment_symbols = [
            sp.Symbol(sym) for sym in self.compartments
        ]
        parameter_symbols = [
            sp.Symbol(sym) for sym in self.parameters
        ]

        # Dummy time symbol, only for compatibility with scipy.odeint
        t_symbol = sp.Symbol('t')

        # Symbolic rhs from the edge labels
        dim = len(compartment_symbols)
        self.rhs_symbolic = sp.Matrix(sp.symbols("rhs:" + str(dim)))
        self.rhs_symbolic = self.rhs_symbolic - self.rhs_symbolic

        for k, edge in enumerate(list(self.edges)):
                
            edge_label = self.edges[edge]['label']
            term = sp.sympify(edge_label, locals=ns)
            
            iedge0 = np.argwhere(np.array(list(self.nodes)) == edge[0])[0][0]
            iedge1 = np.argwhere(np.array(list(self.nodes)) == edge[1])[0][0]
            
            self.rhs_symbolic[iedge0] -= term
            self.rhs_symbolic[iedge1] += term

        self.rhs_latex = sp.latex(self.rhs_symbolic)

        if hasattr(self, "inputs"):

            self.rhs_lambda = sp.lambdify([compartment_symbols, 
                                           t_symbol, 
                                           parameter_symbols,
                                           input_symbols], 
                                        self.rhs_symbolic)

            def rhs(pop, tim, pars, inpts):
                return self.rhs_lambda(pop, tim, pars, inpts).reshape(-1)


        else:

            self.rhs_lambda = sp.lambdify([compartment_symbols, 
                                           t_symbol, 
                                           parameter_symbols], 
                                        self.rhs_symbolic)

            def rhs(pop, tim, pars):
                return self.rhs_lambda(pop, tim, pars).reshape(-1)


        return self.rhs_latex, self.rhs_symbolic, rhs


    def plot_compartment(self, simulation_time, solution, compartment, ax=None, scale = 1.0):

        if ax == None:
            fig = plt.figure(figsize=(6,6))
            ax  = fig.gca()

        show_only = [compartment]

        for k, compartment in enumerate(self.compartments):
            if not compartment in show_only: # don't show these
                continue
            ax.plot(simulation_time, 
                    solution[:, k], 
                    label='$'+compartment+'$')

            ax.legend(fontsize = 16 * scale, framealpha=0.5)
            ax.grid(True)
            plt.tight_layout()

        if ax.get_xlabel() == '':
            ax.set_xlabel('Days', fontsize = 16 * scale)
        
        if ax.get_ylabel() == '':
            ax.set_ylabel('Compartment population', fontsize = 16 * scale)

        return plt.gcf()


    def initiate_exponential(
        self, population_guess, parameters, fit_horizon, vary, take_from, 
        tol = 1, max_iterations = 100, verbose=False, scale=1,
        ):
        """
        Generate an initial population such that the dynamics of the compartments
        in the list vary can be well-approximated by an exponential for the 
        fit_horizon. 

        Parameters
        ----------
        - population_guess: initial guess for the population a dictionary 
        {'compartment': population (float)}
        
        - fit_horizon: array of time points to which an exponential will be fit

        - parameters: dictionary of parameter values {'parameter': value (float)}
        
        - vary: list of compartments to be varied 

        - take_from: compartment to be adjusted as those in vary are varied. 
        This would be "susceptibles" if the compartmental model is modeling the 
        early stage of an epidemic. 

        - tol: tolerance for terminating the fixed point iteration (default = 1).

        Usage example 
        -------------
        
        initial_population_exp = model.initiate_exponential(
            initial_population, parameters, np.arange(0, 10), ['R'], 'S'
        )
        
        see SIR_init_exp.ipynb for details.

        """

        for compartment in vary:
            if not(compartment in self.compartments):
                exit(f"{compartment} is not a compartment.")
        
        if not(take_from in self.compartments):
            exit(f"{take_from} is not a compartment.")

        ode_latex, ode_symbolic, ode = self.generate_ode()
        
        def exponential(x, a, b):
            return a * np.exp(b * x)

        def initial_deviation_from_exp(xdata, ydata, plot=False):
            popt, pcov = curve_fit(exponential, xdata, ydata)
            fit = exponential(xdata, *popt)
            return fit[0] - ydata[0]    

        initial_population = population_guess.copy()

        simulation = odeint(ode, 
                            list(initial_population.values()), 
                            fit_horizon, 
                            args = (list(parameters.values()),)
                            )
        
        residuals    = np.zeros(len(self.compartments)) 

        for i, compartment in enumerate(self.compartments):

            if compartment in vary:
                residuals[i] = initial_deviation_from_exp(
                                fit_horizon, simulation[:, i]
                                                        )

        n_iterations = 0

        if verbose:
            print('Starting residuals:', residuals)

        while np.any(np.abs(residuals) > tol):

            if n_iterations > max_iterations:
                print("Exponential initiation did not converge")
                return initial_population

            new_initial_population = initial_population.copy()

            for i, compartment in enumerate(self.compartments):
                if compartment in vary:
                    new_initial_population[compartment] += residuals[i] * scale
                    new_initial_population[take_from]   -= residuals[i] * scale

            new_simulation = odeint(ode, 
                    list(new_initial_population.values()), 
                    fit_horizon, 
                    args = (list(parameters.values()),))
    
            for i, compartment in enumerate(self.compartments):

                if compartment in vary:
                    residuals[i] = initial_deviation_from_exp(
                                    fit_horizon, new_simulation[:, i]
                                                            )

            initial_population = new_initial_population.copy()
            n_iterations += 1

            if verbose:
                print(f'Iteration {n_iterations}, residuals:', residuals)
        
        return initial_population
                
                
            

