""" 
Code for Simulation the process of the sytem

the deterministic system:
    dx = A sqrt(x(t)) + G u(t) + D d(t), d is deterministic
The Stochatstic system:
    dx = A sqrt(x(t)) + G u(t) + D d, d is stochastic but piecewise constant
The stochastic system:
    dx = A sqrt(x(t)) + G u(t) + D d(t) dWt, where dWt is the weiner process
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.stats as stats
import copy

from types import SimpleNamespace


#%% functions

# noise generator for model 2
def stochastic_noise_gen(dt, rate):
    noise = dict()

    def stochastic_noise(t):
        k = np.floor(t % dt)
        if not k in noise:
            noise[k] = stats.expon(scale = 1/rate).rvs(2).reshape(2,1)
        return noise[k]
    return stochastic_noise

# system generator
# maybe implement as a class?
# would also be nice to run simulation in same class
def deterministic_gen(params):
    g = params.g
    gamma_1, gamma_2 = params.gamma_1, params.gamma_2
    A_1, A_2, A_3, A_4 = params.A_1, params.A_2, params.A_3, params.A_4
    a_1, a_2, a_3, a_4 = params.a_1, params.a_2, params.a_3, params.a_4
    rho = params.rho

    u = params.u
    d = params.d

    obs_noise = params.obs_noise

    A = np.array([[-np.sqrt(2*g*rho*a_1**2 / A_1), 0, np.sqrt(2*g*rho*a_3**2 / A_3), 0],
                    [0, -np.sqrt(2*g*rho*a_2**2 / A_2), 0, np.sqrt(2*g*rho*a_4**2 / A_4)],
                    [0, 0, -np.sqrt(2*g*rho*a_3**2 / A_3), 0],
                    [0, 0, 0, -np.sqrt(2*g*rho*a_4**2 / A_4)]])
        
    Gamma = np.array([[gamma_1, 0],
                    [0, gamma_2],
                    [0, 1-gamma_2],
                    [1-gamma_1, 0]])
    
    D = np.array([[0, 0],
                [0, 0],
                [1, 0],
                [0, 1]])
    G = g*np.array([[1,0,0,0],
                [0,1,0,0],
                [0,0,1,0],
                [0,0,0,1]])
    C = np.array([[1/(rho * A_1),0,0,0],
                    [0,1/(rho * A_2),0,0],
                    [0,0,1/(rho * A_3),0],
                    [0,0,0,1/(rho * A_4)]])

    def system_eq(t, x):
        return A @ np.nan_to_num(np.sqrt(x)) + Gamma @ u(t) + D @ d(t)
    
    def observation_eq(t, x):
        return G @ x + obs_noise(t)
    
    def output_eq(t, x):
        return C @ x

    return system_eq, observation_eq, output_eq


#%%

Params_base = SimpleNamespace(**{
    "gamma_1" : 0.58, # split 1
    "gamma_2" : 0.68, # split 2
    "A_1" : 1e-2, #Area tank 1
    "A_2" : 2e-2, #Area tank 2
    "A_3" : 3e-2, #Area tank 3
    "A_4" : 4e-2, #Area tank 4
    "a_1" : 2e-4, # Area hole 1
    "a_2" : 2.5e-4, # Area hole 2
    "a_3" : 3e-4, # Area hole 3
    "a_4" : 3.5e-4, # Area hole 4
    "g" : 9.8, # gravity 9.8 m/s^2
    "rho" : 1000, # density 1000 kg/m^3
})

# Deterministic system
params_det_min = copy.deepcopy(Params_base)

params_det_min.u =  lambda t: np.array([0,0]).reshape(2,1),
params_det_min.d =  lambda t: 2 * np.array([np.sin(t)**2, np.cos(t)**2]).reshape(2,1)
params_det_min.obs_noise = lambda t: np.array([0,0]).reshape(2,1)
#"d" : lambda t: np.array([1,1]).reshape(2,1)

params_det_nonmin = copy.deepcopy(Params_base)
params_det_nonmin.a_4 = 0 
params_det_nonmin.u =  lambda t: np.array([0,0]).reshape(2,1),
params_det_nonmin.d =  lambda t: 2 * np.array([np.sin(t)**2, np.cos(t)**2]).reshape(2,1)
params_det_nonmin.obs_noise = lambda t: np.array([0,0]).reshape(2,1)
#"d" : lambda t: np.array([1,1]).reshape(2,1)

# Stochastic system
params_stoch_min = copy.deepcopy(Params_base)
params_stoch_min.u =  lambda t: np.array([0,0]).reshape(2,1)
params_stoch_min.d =  lambda t: stochastic_noise_gen(1, 1)
params_stoch_min.obs_noise = lambda t: stats.norm(scale = 1).rvs(2).reshape(2,1)

params_stoch_nonmin = copy.deepcopy(Params_base)
params_stoch_nonmin.a_4 = 0
params_stoch_nonmin.u =  lambda t: np.array([0,0]).reshape(2,1)
params_stoch_nonmin.d =  stochastic_noise_gen(1, 1)
params_stoch_nonmin.obs_noise = lambda t: stats.norm(scale = 1).rvs(2).reshape(2,1)



#%%
minimum_phase = deterministic_gen(params_minimum)
non_minimum_phase = deterministic_gen(params_non_minimum)

#print(non_minimum_phase(0, np.array([1,1,1,1]).reshape(4,1)))
sol = integrate.solve_ivp(non_minimum_phase,
                          [0, 200],
                          np.array([0,0,0,0]),
                          method = "LSODA",
                          vectorized=True)

C = np.array([[1/(params_minimum.rho * params_minimum.A_1),0,0,0],
                    [0,1/(params_minimum.rho * params_minimum.A_2),0,0],
                    [0,0,1/(params_minimum.rho * params_minimum.A_3),0],
                    [0,0,0,1/(params_minimum.rho * params_minimum.A_4)]])
fig = plt.figure(figsize = (14,8))
plt.plot(sol.t, np.transpose(C @ sol.y))
plt.legend(["Tank 1", "Tank 2", "Tank 3", "Tank 4"])
plt.savefig("4tank.png")


# %%
