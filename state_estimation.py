
# %%
import numpy as np
import control as ct
import matplotlib.pyplot as plt

from functions import *
import matplotlib as mpl
mpl.rcParams['axes.prop_cycle'] = nord_theme
#%% 

class kalmanfilter:
    # for now no inputs
    def __init__(self, A , B , C , D, Sigma_noise, Sigma_obs):
        # x_k+1 = A x_k + B u_k + eta_k
        # y_k = C x_K + D u_k + nu_k 
        # eta and nu follows normal distribution and assumed independent

        self.A = A 
        self.B = B 
        self.C = C 
        self.D = D

        self.Sigma_noise = Sigma_noise
        self.Sigma_obs = Sigma_obs

        self.nstates = A.shape[0]
        self.nobs = C.shape[0]

    def run_filter(self, y, u, x0, P_0 = None):

        # run the filter
        # assume rows are time points, and columns are variables
        N = len(y) #number of time points
        self.x_est = np.zeros([N, self.nstates]) # estimated states
        self.x_pred = np.zeros([N-1, self.nstates]) # one step predictions
        self.y_est = np.zeros([N, self.nobs]) # estimated states
        self.y_pred = np.zeros([N-1, self.nobs])

        self.x_est[0] = x0

        

        if P_0 is None:
            # no initial covariance given, use static filter
            self.dynamic = False

            self.P_est = ct.dare(   self.A.T,
                                    self.C.T,
                                    self.Sigma_noise,
                                    self.Sigma_obs
                                )[0]
            self.P_pred = self.A @ self.P_est @ self.A.T + self.Sigma_noise
            self.P_obs = self.C @ self.P_pred @ self.C.T + self.Sigma_obs
            self.K = self.P_pred @ self.C.T @ np.linalg.pinv(self.P_obs)
        else:
            # dynamic filter precalculate convariances and kalman gain
            self.dynamic = True

            self.P_est = np.zeros([N, self.Sigma_noise.shape])
            self.P_pred = np.zeros([N-1, self.Sigma_noise.shape])
            self.P_obs = np.zeros([N-1, self.Sigma_noise.shape])
            self.K = np.zeros([N, self.nstates, self.nobs])


            self.P_est[0] = P_0
            for i in range(1,N-1):
                self.P_pred[i] = self.A @ self.P_est[i] @ self.A.T + self.Sigma_noise
                self.P_obs[i] = self.C @ self.P_pred[i] @ self.C.T + self.Sigma_obs
                self.K[i] = self.P_pred[i] @ self.C.T @ np.linalg.pinv(self.P_obs[i])
                self.P_est[i+1] = (np.identity(self.nstates) - self.K[i] @ self.C) @ self.P_pred[i]

        #filter
        for i in range(1,N):
            self.x_pred[i-1] = self.A @ self.x_est[i] + self.B @ u[i]
            self.y_pred[i-1] = self.C @ self.x_pred[i-1] + self.D @ u[i]

            e =  self.y_pred[i-1] - y[i] 

            if self.dynamic:
                self.x_est[i] = self.x_pred[i-1] - self.K[i] @ e

            else:
                self.x_est[i] = self.x_pred[i-1] - self.K @ e
            
            self.y_est[i] = self.C @ self.x_est[i] + self.D @ u[i]
                



# %% taken from step experiment 

N = 10000
TS = 0.1
T = np.arange(0,N) * TS
A = np.array([
    [0.99608767, 0.,         0.00292993, 0.        ],
    [0.,         0.99694218, 0.,         0.00299216],
    [0.,         0.,         0.99706432, 0.        ],
    [0.,         0.,         0.,         0.99700325]
    ])
B = np.array([
    [5.78864684e-02, 4.69325747e-05],
    [6.28990033e-05, 6.78959812e-02],
    [0.00000000e+00, 3.19530061e-02],
    [4.19370368e-02, 0.00000000e+00]
    ])
D = np.zeros([4,2])
#C = np.array([[0.5,0.5],[-2,1]])
C = np.array([
    [9.8, 0.,  0.,  0. ],
    [0.,  9.8, 0.,  0. ],
    [0.,  0.,  9.8, 0., ],
    [0.,  0.,  0.,  9.8]
    ])

Sigma_noise = TS*np.array([
    [0.001,0,0,0],
    [0,0.001,0,0],
    [0,0,0.1,0],
    [0,0,0,0.1]
    ])
Sigma_obs = np.array([
    [0.01,0,0,0],
    [0,0.01,0,0],
    [0,0,0.01,0],
    [0,0,0,0.01]
    ])

u = 0 * np.ones([N,2])
X = np.zeros([N,4])
Y = np.zeros([N,4])

#noise = np.clip(np.random.multivariate_normal(np.array([0,0,0,0]), Sigma_noise, N), 0, None)
#noise = np.abs(np.random.multivariate_normal(np.array([0,0,0,0]), Sigma_noise, N))
noise = np.random.multivariate_normal(np.array([0,0,0,0]), Sigma_noise, N)
obs_noise = np.random.multivariate_normal(np.array([0,0,0,0]),Sigma_obs, N)


X[0] = np.array([10,10,10,10])
Y[0] = C @ X[0] + obs_noise[0]
for i in range(1,N):
    X[i] = A @ X[i-1] + B @ u[i] + noise[i]
    Y[i] = C @ X[i] + obs_noise[i]
""" plt.figure()
plt.plot(T, X)
plt.figure()
plt.plot(T, Y) """



# %%

filt = kalmanfilter(A,B,C,D, Sigma_noise, Sigma_obs)
filt.run_filter(Y, u, np.array([1,1,1,1]))

plt.figure()
plt.plot(T, X)
plt.plot(T, filt.x_est, ls='--')
plt.legend(["tank " + str(i) for i in range(1,5) ] + ["tank " + str(i) + " est" for i in range(1,5)] )

plt.figure()
plt.plot(T, Y)
plt.plot(T, filt.y_est, ls='--')
plt.legend(["measurement " + str(i) for i in range(1,5) ] + ["measurement " + str(i) + " est" for i in range(1,5)] )
# %%
