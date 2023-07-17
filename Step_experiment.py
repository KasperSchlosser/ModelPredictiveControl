import numpy as np
import scipy.integrate as integrate
import scipy.stats as stats
import scipy.signal as sig
import numpy.fft as fft

from cycler import cycler
from collections import defaultdict
from scipy.optimize import minimize

from functions import *

nord_colors = ['#BF616A', '#D08770', '#EBCB8B', '#A3BE8C']
nord_theme = cycler(color=nord_colors)

#%% step responses
# we just start the simulation around the steady values
x0 = np.array([12.75510122 16.32652775 17.00680245 16.65972493])
tspan = [0, 1000]

#make step functions

def step_func(step_size):
    step = defaultdict(lambda : np.array([0.0,0.0]).reshape(2,1))
    for x in np.arange(10 ,600.1, 10):
        step[x] = step[x-10] + ((np.random.rand(2,1) > 0.5) * 2 - 1) * step_size
        step[x] = np.clip(step[x], 0, np.inf)
    return lambda x: step[np.round(x,-1)], step


#def step_func(step_size, step_time = 200):
#        return lambda t: np.ones([2,1]) * step_size if t > step_time else np.zeros([2,1])

u_10, step_10 = step_func(0.1)
u_25, step_25 = step_func(0.25)
u_50, step_50 = step_func(0.5)

"""
u_10 = step_func(0.1)
u_25 = step_func(0.25)
u_50 = step_func(0.5)
"""
# %% Deterministic step response
det_step_10 = simulation_system(Params_base_min,
                                #u=lambda t: np.array([0, 0]).reshape(2, 1),
                                u = u_10,
                                d = det_noise,
                                obs_noise = det_obs_noise)
det_step_25 = simulation_system(Params_base_min,
                                #u=lambda t: np.array([0, 0]).reshape(2, 1),
                                u = u_25,
                                d = det_noise,
                                obs_noise = det_obs_noise)
det_step_50 = simulation_system(Params_base_min,
                                #u=lambda t: np.array([0, 0]).reshape(2, 1),
                                u = u_50,
                                d = det_noise,
                                obs_noise = det_obs_noise)

#stochastoch step respone
stoch_step_10 = simulation_system(  Params_base_min,
                                    u = u_10,
                                    d = stochastic_noise_gen(10, 1),
                                    obs_noise = lambda t: stats.norm(scale=1).rvs(4))
stoch_step_25 = simulation_system(  Params_base_min,
                                    u = u_25,
                                    d = stochastic_noise_gen(10, 1),
                                    obs_noise = lambda t: stats.norm(scale=1).rvs(4))
stoch_step_50 = simulation_system(  Params_base_min,
                                    u = u_50,
                                    d = stochastic_noise_gen(10, 1),
                                    obs_noise = lambda t: stats.norm(scale=1).rvs(4))

# %%

def transfer_function_H(y,u, dt):
    U = fft.fft(u, axis = 0)
    Y = fft.fft(y)
    freq = fft.fftfreq(Y.size, d = dt)

    U_nonzero = np.all(np.abs(U) > 1e-4,axis = 1)
    U = U[U_nonzero,:]
    Y = Y[U_nonzero]
    freq = freq[U_nonzero]



    H =  Y[:,np.newaxis] / U

    return H, freq

def estimate_transfer(H, freq, nzeros, npoles, ninputs, x0 = None, l1 = 0):
    # estimate the transfer function from data using squared loss an l1 regularization
    # parameters is in the shape [zeros , poles]
    # also add constant term

    nterms = nzeros + npoles + 2
    if x0 is None:
        x0 = np.random.random([nterms * ninputs])
    
    zstop = nzeros + 1
    pstop = nzeros + 1 + npoles + 1
    def loss (x):
        vals = np.zeros([len(H), ninputs], dtype=np.complex64)
        for s, v in enumerate(1j*freq):
            for i in range(ninputs):
                vals[s,i] = np.polyval(x[nterms*i:nterms*i + zstop] , v) / np.polyval(x[nterms*i+zstop:nterms*i+pstop], s)

        l1_reg = l1*np.abs(x).sum()

        L = np.linalg.norm((vals - H), axis = 0).sum() + l1_reg
        return L
    
    res = minimize(loss, x0, tol = 1e-3, options = {"maxiter": 100, "disp":True}, method = "nelder-mead")
    systems = [sig.lti(res.x[nterms*i:nterms*i + zstop], res.x[nterms*i + zstop:nterms*i + pstop])for i in range(ninputs)]

    return systems, res

det_step_10.run_sim([0,1000],x0)
det_step_25.run_sim([0,1000],x0)
det_step_50.run_sim([0,1000],x0)

TF = [0,0,0,0]
for i in range(4):
    
    u_vals_10 = np.array([u_10(t) for t in det_step_10.t]).squeeze()
    u_vals_25 = np.array([u_25(t) for t in det_step_25.t]).squeeze()
    u_vals_50 = np.array([u_50(t) for t in det_step_50.t]).squeeze()

    H_10, freq10 = transfer_function_H(det_step_10.y[0], u_vals_10, det_step_10.t[1] - det_step_10.t[0])
    H_25, freq25 = transfer_function_H(det_step_25.y[0], u_vals_25, det_step_25.t[1] - det_step_25.t[0])
    H_50, freq50 = transfer_function_H(det_step_50.y[0], u_vals_50, det_step_50.t[1] - det_step_50.t[0])

    H = np.concatenate([H_10, H_25, H_50])
    freq = np.concatenate([freq10,freq25,freq50])

    TF[i] = estimate_transfer(H,freq, 1,1, ninputs = 2, l1 = 1)[0]

#%% print system
Ts = 0.1


def imp_response(A_d, B_d, C_d, D_d, n):
    if n == 0: return D_d
    else: return C_d @ A_d ** (n-1) @ B_d

for i in range(4):
    print(TF[i])
    plt.figure()
    for s in TF[i]:
        dis


# %% Linearization of systems

def discretize_system(A_c, B_c, Ts):
        
        n = A_c.shape[0] # number of states
        m = B_c.shape[1] # number of inputs

        pad = np.zeros([n, np.max([0, n - m ])])

        mat = np.block([[A_c, B_c, pad], [np.zeros(A_c.shape), np.zeros(B_c.shape), pad]])
        mat = expm(mat*Ts)
        
        return mat[:n,:n], mat[:n, n:n+m]
A_c = det_min.A
B_c = det_min.Gamma
C_c = det_min.G
Ts = 0.1

A_d, B_d = discretize_system(A_c, B_c, Ts)
C_d = C_c
D_d = np.zeros([4,2])

# %%

def imp_response(A_d, B_d, C_d, D_d, n):
    if n == 0: return D_d
    else: return C_d @ A_d ** (n-1) @ B_d


impulseresponse = np.moveaxis(np.dstack([imp_response(A_d,B_d,C_d,D_d,x) for x in range(200)]), [0,1,2], [1,2,0])

plt.figure()
plt.subplot(1,2,1)
plt.plot(impulseresponse[:,:,0])
plt.subplot(1,2,2)
plt.plot(impulseresponse[:,:,1])
# %%
