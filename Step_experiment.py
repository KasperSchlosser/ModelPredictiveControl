#%%
import numpy as np
import scipy.signal as sig
import numpy.fft as fft
import matplotlib.pyplot as plt

from cycler import cycler
from collections import defaultdict
from scipy.optimize import minimize
from scipy.linalg import expm

from functions import *

import matplotlib as mpl
mpl.rcParams['axes.prop_cycle'] = nord_theme

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

def estimate_transfer(H, freq, nzeros, npoles, ninputs, x0 = None, l2 = 0):
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

        l2_reg = l2*(x**2).sum()

        L = np.linalg.norm((vals - H), axis = 0).sum() + l2_reg
        return L
    
    res = minimize(loss, x0, tol = 1e-3, options = {"maxiter": 100, "disp":True})
    systems = [sig.lti(res.x[nterms*i:nterms*i + zstop], res.x[nterms*i + zstop:nterms*i + pstop])for i in range(ninputs)]

    return systems, res

def imp_response(A_d, B_d, C_d, D_d, n):
    if n == 0: return D_d
    else: return C_d @ A_d ** (n-1) @ B_d

def discretize_system(A_c, B_c, Ts):
        
        n = A_c.shape[0] # number of states
        m = B_c.shape[1] # number of inputs

        pad = np.zeros([n, np.max([0, n - m ])])

        mat = np.block([[A_c, B_c, pad], [np.zeros(A_c.shape), np.zeros(B_c.shape), pad]])
        mat = expm(mat*Ts)
        
        return mat[:n,:n], mat[:n, n:n+m]

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
TF = []
nzeros = 2
npoles = 2
l2 = 1
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

    TF.append(estimate_transfer(H,freq, nzeros, npoles, ninputs = 2, l2 = l2)[0])
# %% find MIMO impulse response

dt = 0.1 # sample time to use
K = 300 # numer of inpule response coefficients to include
H = np.zeros([K, 4, 2])

for i in range(4): # 4 states
    for j in range(2): # inputs
        # find discrete coeficients
        dsys = TF[i][j].to_ss().to_discrete(dt)
        
        H[0,i,j] = dsys.D
        for k in range(1,K):
            H[k,i,j] = dsys.C @ np.linalg.matrix_power(dsys.A, k-1) @ dsys.B

#plot impluse response coefficients

x_ax = dt * np.arange(K)
plt.figure(figsize=(20,10))
for i in range(2): # for each input

    plt.subplot(2,2,i+1)
    plt.plot(x_ax, H[:,:,i])
    plt.legend(["Tank 1", "Tank 2", "Tank 3", "Tank 4"])
plt.savefig("Figures/Step response/experimental.png")
# %% Linearization of systems, theoretical
# point to linearise around
x0 = np.array([12.75510122, 16.32652775, 17.00680245, 16.65972493]) # point to lin

#as im lazy this is how i get the matrices
sys = simulation_system(Params_base_min, None, None, None)
A_c = sys.A / (2 * np.sqrt(x0))
B_c = sys.Gamma

A_d, B_d = discretize_system(A_c, B_c, Ts)
C_d = sys.G
D_d = np.zeros([4,2])

#plot impulse response
impulseresponse = np.moveaxis(np.dstack([imp_response(A_d,B_d,C_d,D_d,x) for x in range(K)]), [0,1,2], [1,2,0])

plt.figure()
plt.subplot(1,2,1)
plt.plot(x_ax, impulseresponse[:,:,0])
plt.legend(["Tank 1", "Tank 2", "Tank 3", "Tank 4"])
plt.subplot(1,2,2)
plt.plot(x_ax, impulseresponse[:,:,1])
plt.legend(["Tank 1", "Tank 2", "Tank 3", "Tank 4"])

plt.savefig("Figures/Step response/Theoretical.png")
# %%

#print the discrete system
print(f'A:\n{A_d},\n B:\n{B_d},\n C:\n{C_d}')


# %%
