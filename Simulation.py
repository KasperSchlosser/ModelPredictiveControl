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
import scipy.stats as stats

from cycler import cycler

from functions import *

#for making nice plots
nord_colors = ['#BF616A', '#D08770', '#EBCB8B', '#A3BE8C']
nord_theme = cycler(color=nord_colors)

# for plotting the systems, not general
def plot_systems(min_sys, nonmin_sys, ylim=[0, 2]):

    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.set_prop_cycle(nord_theme)
    plt.plot(min_sys.t, np.transpose(min_sys.out), linestyle='-')
    plt.plot(nonmin_sys.t, np.transpose(nonmin_sys.out), linestyle='--')
    plt.plot(0, 0, linestyle='-', color='k', label='minimum system')
    plt.plot(0, 0, linestyle='--', color='k', label='non-minimum system')
    plt.ylim(ylim)
    legend_elements = [plt.Line2D([0], [0], color=nord_colors[0], label='Tank 1'),
                       plt.Line2D([0], [0], color=nord_colors[1],
                                  label='Tank 2'),
                       plt.Line2D([0], [0], color=nord_colors[2],
                                  label='Tank 3'),
                       plt.Line2D([0], [0], color=nord_colors[3],
                                  label='Tank 4'),
                       plt.Line2D([0], [0], color='k',
                                  linestyle='-', label='Minimum system'),
                       plt.Line2D([0], [0], color='k', linestyle='--', label='Non-minimum system')]
    plt.legend(handles=legend_elements)

# %% parameters for the system
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

# noise terms
det_noise = lambda t: np.array([1,1]).reshape(2, 1)
det_obs_noise = lambda t: np.array([0, 0, 0, 0])

stoch_noise = stochastic_noise_gen(10,1)
stoch_obs_noise = lambda t: stats.norm(scale=1).rvs(4)

det_input = lambda t: np.array([0, 0]).reshape(2, 1)
stoch_input = lambda t: np.array([0, 0]).reshape(2, 1)

# simulation parameters
x0 = np.array([0, 0, 0, 0])
tspan = [0, 600]

# %% Make systems, run and plot
# deterministic
det_min = simulation_system(Params_base_min,
                            u = det_input,
                            d = det_noise,
                            obs_noise = det_obs_noise)
det_min.run_sim(tspan, x0)

det_nonmin = simulation_system(Params_base_nonmin,
                               u = det_input,
                               d = det_noise,
                               obs_noise = det_obs_noise)
det_nonmin.run_sim(tspan, x0)

plot_systems(det_min, det_nonmin)
plt.savefig('Figures/Simulation/deterministic.png')


# stochastic
stoch_min = simulation_system(Params_base_min,
                              u = stoch_input,
                              d = stoch_noise,
                              obs_noise = stoch_obs_noise)
stoch_min.run_sim(tspan, x0)

stoch_nonmin = simulation_system(Params_base_nonmin,
                                 u=lambda t: np.array([0, 0]).reshape(2, 1),
                                 d=stochastic_noise_gen(10, 1),
                                 obs_noise=lambda t: stats.norm(scale=1).rvs(4))
stoch_nonmin.run_sim(tspan, x0)

plot_systems(stoch_min, stoch_nonmin)
plt.savefig('Figures/Simulation/stochastic.png')

# SDE system
""" sde_min = SDE_system(Params_base_min, 
                     u = lambda t: np.array([0, 0]).reshape(2, 1),
                     d = lambda t: 3*np.array([np.sin(t) + 1, np.cos(t) + 1]).reshape(2, 1),
                     obs_noise = lambda t: stats.norm(scale=1).rvs([4,1]))
sde_min.run_sim(tspan, x0, dt = 0.1)

sde_nonmin = SDE_system(Params_base_nonmin, 
                        u = lambda t: np.array([0, 0]).reshape(2, 1),
                        d = lambda t: 3*np.array([np.sin(t) + 1, np.cos(t) + 1]).reshape(2, 1),
                        obs_noise = lambda t: stats.norm(scale=1).rvs([4,1]))
sde_nonmin.run_sim(tspan, x0, dt = 0.1)

plot_systems(sde_min, sde_nonmin,ylim = [-2,2])
plt.savefig('Figures/Simulation/sde.png')
"""

# %% Get steady state value
print(det_min.y[:,-1])