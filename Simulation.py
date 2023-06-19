""" 
Code for Simulation the process of the sytem

the deterministic system:
    dx = A sqrt(x(t)) + G u(t) + D d(t), d is deterministic
The Stochatstic system:
    dx = A sqrt(x(t)) + G u(t) + D d, d is stochastic but piecewise constant
The stochastic system:
    dx = A sqrt(x(t)) + G u(t) + D d(t) dWt, where dWt is the weiner process
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.stats as stats


from types import SimpleNamespace
from cycler import cycler

nord_colors = ['#BF616A', '#D08770', '#EBCB8B', '#A3BE8C']
nord_theme = cycler(color = nord_colors)

# %% functions

# noise generator for model 2
def stochastic_noise_gen(dt, rate, seed = 42):
    noise = dict()
    rng = np.random.RandomState(seed)

    eta = stats.expon(scale=1/rate)
    def stochastic_noise(t):
        k = np.floor(t / dt)
        if not k in noise:
            noise[k] = eta.rvs(2, random_state=rng).reshape(2, 1)
        return noise[k]
    return stochastic_noise

# system generator
# maybe implement as a class?
# would also be nice to run simulation in same class


class simulation_system:

    def __init__(self, params, u, d, obs_noise):

        # set parameters of the system
        for k, v in params.items():
            setattr(self, k, v)

        # define matrices

        self.A = np.array([[-np.sqrt(2*self.g*self.rho*self.a_1**2 / self.A_1), 0, np.sqrt(2*self.g*self.rho*self.a_3**2 / self.A_3), 0],
                           [0, -np.sqrt(2*self.g*self.rho*self.a_2**2 / self.A_2),
                            0, np.sqrt(2*self.g*self.rho*self.a_4**2 / self.A_4)],
                           [0, 0, -np.sqrt(2*self.g*self.rho *
                                           self.a_3**2 / self.A_3), 0],
                           [0, 0, 0, -np.sqrt(2*self.g*self.rho*self.a_4**2 / self.A_4)]])
        self.Gamma = np.array([[self.gamma_1, 0],
                               [0, self.gamma_2],
                               [0, 1-self.gamma_2],
                               [1-self.gamma_1, 0]])
        self.D = np.array([[0, 0],
                           [0, 0],
                           [1, 0],
                           [0, 1]])
        self.G = self.g*np.array([[1, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])
        self.C = np.array([[1/(self.rho * self.A_1), 0, 0, 0],
                           [0, 1/(self.rho * self.A_2), 0, 0],
                           [0, 0, 1/(self.rho * self.A_3), 0],
                           [0, 0, 0, 1/(self.rho * self.A_4)]])

        # set noise and inputs
        self.u = u
        self.d = d
        self.obs_noise = obs_noise

    def system_eq(self, t, x):
        return self.A @ np.nan_to_num(np.sqrt(x)) + self.Gamma @ self.u(t) + self.D @ self.d(t)

    def observation_eq(self, t, x):
        return self.G @ x + self.obs_noise(t)

    def output_eq(self, t, x):
        return self.C @ x

    def run_sim(self, t_span, x0):
        self.sim = integrate.solve_ivp(
            self.system_eq, t_span, x0,
            solver='RK45',
            vectorized=True,
            max_step=0.1)
        # make observations
        self.obs = np.zeros(self.sim.y.shape)
        for i in range(len(self.sim.t)):
            self.obs[:,i] = self.observation_eq(self.sim.t[i], self.sim.y[:,i])
        # make outputs
        self.out = np.zeros(self.sim.y.shape)
        for i in range(len(self.sim.t)):
            self.out[:,i] = self.output_eq(self.sim.t[i], self.sim.y[:,i])

        # assign t, and y for ease of plotting
        self.t = self.sim.t
        self.y = self.sim.y
        
# %%
Params_base_min = {
    "gamma_1": 0.58,  # split 1
    "gamma_2": 0.68,  # split 2
    "A_1": 1e-2,  # Area tank 1
    "A_2": 2e-2,  # Area tank 2
    "A_3": 3e-2,  # Area tank 3
    "A_4": 4e-2,  # Area tank 4
    "a_1": 2e-4,  # Area hole 1
    "a_2": 2.5e-4,  # Area hole 2
    "a_3": 3e-4,  # Area hole 3
    "a_4": 3.5e-4,  # Area hole 4
    "g": 9.8,  # gravity 9.8 m/s^2
    "rho": 1000,  # density 1000 kg/m^3
}

Params_base_nonmin = {
    "gamma_1": 0.58,  # split 1
    "gamma_2": 0.68,  # split 2
    "A_1": 1e-2,  # Area tank 1
    "A_2": 2e-2,  # Area tank 2
    "A_3": 3e-2,  # Area tank 3
    "A_4": 4e-2,  # Area tank 4
    "a_1": 2e-4,  # Area hole 1
    "a_2": 2.5e-4,  # Area hole 2
    "a_3": 3e-4,  # Area hole 3
    "a_4": 0,  # Area hole 4
    "g": 9.8,  # gravity 9.8 m/s^2
    "rho": 1000,  # density 1000 kg/m^3
}

# %% make systems
det_min = simulation_system(Params_base_min,
                            u=lambda t: np.array([0, 0]).reshape(2, 1),
                            d=lambda t: np.array(
                                [np.sin(t) + 1, np.cos(t) + 1]).reshape(2, 1),
                            obs_noise= lambda t: np.array([0, 0, 0, 0]))
det_nonmin = simulation_system(Params_base_nonmin,
                               u=lambda t: np.array([0, 0]).reshape(2, 1),
                               d=lambda t: np.array(
                                   [np.sin(t) + 1, np.cos(t) + 1]).reshape(2,  1),
                               obs_noise = lambda t: np.array([0, 0, 0, 0]))
stoch_min = simulation_system(Params_base_min,
                               u=lambda t: np.array([0, 0]).reshape(2, 1),
                               d=stochastic_noise_gen(10, 1),
                               obs_noise=lambda t: stats.norm(scale=1).rvs(4))
stoch_nonmin = simulation_system(Params_base_nonmin,
                               u=lambda t: np.array([0, 0]).reshape(2, 1),
                               d=stochastic_noise_gen(10, 1),
                               obs_noise=lambda t: stats.norm(scale=1).rvs(4))


#%% run and plot simulations

x0 = np.array([0, 0, 0, 0])
tspan = [0, 200]
det_min.run_sim(tspan, x0)
det_nonmin.run_sim(tspan, x0)
stoch_min.run_sim(tspan, x0)
stoch_nonmin.run_sim(tspan, x0)



# tmp function to reuse plottin, without copy paste
def plot_systems(min_sys, nonmin_sys, ylim = [0,2]):

    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.set_prop_cycle(nord_theme)
    plt.plot(min_sys.t, np.transpose(min_sys.out) , linestyle='-')
    plt.plot(nonmin_sys.t, np.transpose(nonmin_sys.out), linestyle='--')
    plt.plot(0, 0, linestyle='-', color = 'k', label = 'minimum system')
    plt.plot(0, 0, linestyle='--', color = 'k', label = 'non-minimum system')
    plt.ylim(ylim)
    legend_elements = [plt.Line2D([0], [0], color = nord_colors[0], label = 'Tank 1'),
                    plt.Line2D([0], [0], color = nord_colors[1], label = 'Tank 2'),
                    plt.Line2D([0], [0], color = nord_colors[2], label = 'Tank 3'),
                    plt.Line2D([0], [0], color = nord_colors[3], label = 'Tank 4'),
                    plt.Line2D([0], [0], color = 'k', linestyle = '-', label = 'Minimum system'),
                    plt.Line2D([0], [0], color = 'k', linestyle = '--', label = 'Non-minimum system')]
    plt.legend(handles = legend_elements)

plot_systems(det_min, det_nonmin)
plot_systems(stoch_min, stoch_nonmin)

# %%
