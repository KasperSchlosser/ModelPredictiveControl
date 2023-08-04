#%%
import matplotlib.pyplot as plt
import numpy as np

import Systems

simulation_params = {
                     "tspan": (0,100),
                     "dt": 0.1,
                     "x0": 10*np.ones(4)
                }

state_names = ["tank 1",
               "tank 2",
               "tank 3",
               "tank 4"]
output_names = ["output 1",
                "output 2",
                "output 3",
                "output 4"]

# %% Deterministic

det_min = Systems.nonlinear_system(**Systems.sys_det_min)
det_min.run_continous(**simulation_params)
det_min.plot_sim(state_names = state_names, output_names=output_names)
