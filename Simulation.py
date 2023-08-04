#%%
import matplotlib.pyplot as plt
import numpy as np

import Systems

simulation_params = {
                     "tspan": (0,200),
                     "dt": 0.1,
                     "x0": 10*np.ones(4),
                }

sde_params = {
               "tspan": (0,200),
               "dt": 0.01,
               "x0": np.array([10,10,10,10,1,1]),
          }

state_names = ["tank 1",
               "tank 2",
               "tank 3",
               "tank 4"]
observation_names = ["observation 1",
                     "observation 2",
                     "observation 3",
                     "observation 4"]
output_names = ["output 1",
                "output 2",
                "output 3",
                "output 4"]

# %% Deterministic min

det_min = Systems.nonlinear_system(**Systems.sys_det_min)
det_min.run_continous(d = Systems.det_noise ,**simulation_params)
fig, _ = det_min.plot_sim(state_names = state_names, observation_names= observation_names, output_names=output_names)
fig.savefig("Figures/Simulation/det_min.png")

# %% Det termininistic non minimum

det_nonmin = Systems.nonlinear_system(**Systems.sys_det_nonmin)
det_nonmin.run_continous(d = Systems.det_noise, **simulation_params) 
fig, _ = det_nonmin.plot_sim(state_names = state_names, observation_names= observation_names, output_names=output_names)
fig.savefig("Figures/Simulation/det_nonmin.png")

# %%
stoch_min = Systems.nonlinear_system(**Systems.sys_stoch_min)
stoch_min.run_continous(d = Systems.stoch_noise, **simulation_params) 
fig, _ = stoch_min.plot_sim(state_names = state_names, observation_names= observation_names, output_names=output_names)
fig.savefig("Figures/Simulation/stoch_min.png")

# %%
stoch_nonmin = Systems.nonlinear_system(**Systems.sys_stoch_nonmin)
stoch_nonmin.run_continous(d = Systems.stoch_noise, **simulation_params) 
fig, _ = stoch_nonmin.plot_sim(state_names = state_names, observation_names= observation_names, output_names=output_names)
fig.savefig("Figures/Simulation/stoch_nonmin.png")

# %%

sde_min = Systems.nonlinear_system(**Systems.sys_sde_min)
sde_min.run_continous(d = Systems.sde_noise, **sde_params)
fig, _ = sde_min.plot_sim(state_names = state_names, observation_names= observation_names, output_names=output_names)
fig.savefig("Figures/Simulation/sde_min.png")

# %%

sde_nonmin = Systems.nonlinear_system(**Systems.sys_sde_nonmin)
sde_nonmin.run_continous(d = Systems.sde_noise, **sde_params)
fig, _ = sde_nonmin.plot_sim(state_names = state_names, observation_names= observation_names, output_names=output_names)
fig.savefig("Figures/Simulation/sde_nonmin.png")