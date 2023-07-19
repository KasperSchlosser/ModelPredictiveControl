#%%
import numpy as np
import scipy.signal as sig
import numpy.fft as fft

from cycler import cycler
from collections import defaultdict
from scipy.optimize import minimize

from functions import *

nord_colors = ['#BF616A', '#D08770', '#EBCB8B', '#A3BE8C']
nord_theme = cycler(color=nord_colors)

# %%
#function to make step functions
def step_func(step_size):
    step = defaultdict(lambda : np.array([0.0,0.0]).reshape(2,1))
    for x in np.arange(10 ,600.1, 10):
        step[x] = step[x-10] + ((np.random.rand(2,1) > 0.5) * 2 - 1) * step_size
        step[x] = np.clip(step[x], 0, np.inf)
    return lambda x: step[np.round(x,-1)], step

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

#%%
#simulation parameters
#start at roughly steady state
x0 = np.array([12.75510122, 16.32652775, 17.00680245, 16.65972493])
tspan = [0, 100]

#steps to use
steps = [0.1, 0.25, 0.5]
us = [step_func(step)[0] for step in steps]

# time step to discretize to
Ts = 0.1


#create systems

det_systems = [simulation_system(Params_base_min, u, det_noise, det_obs_noise) for u in us]
stoch_systems = [simulation_system(Params_base_min, u, stoch_noise, stoch_obs_noise) for u in us]

#run systems
for sys in det_systems:
    sys.run_sim(tspan, x0)
for sys in stoch_systems:
    sys.run_sim(tspan, x0)

#%% estimate transfer functions
TF = [None, None, None, None]
for i in range(4):
    Hs = []
    freqs = []
    for j, sys in enumerate(det_systems):
        dt = sys.t[-1] - sys.t[-2]
        u_vals = np.array([us[j](t) for t in sys.t]).squeeze()
        H, freq = transfer_function_H(sys.out[i], u_vals, dt)
        Hs.append(H)
        freqs.append(freq)
    

    Hs = np.concatenate(Hs)
    freqs = np.concatenate(freqs)

    TF[i] = estimate_transfer(H,freq, 1,1, ninputs = 2, l1 = 1)[0]
#%% print system


def imp_response(A_d, B_d, C_d, D_d, n):
    if n == 0: return D_d
    else: return C_d @ A_d ** (n-1) @ B_d


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
