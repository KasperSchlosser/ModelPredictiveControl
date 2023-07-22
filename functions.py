import numpy as np
import scipy.integrate as integrate
import scipy.stats as stats
import scipy.signal as sig
import numpy.fft as fft
import cvxopt

from cycler import cycler
from scipy.optimize import minimize

nord_colors = ['#BF616A', '#D08770', '#EBCB8B', '#A3BE8C']
nord_theme = cycler(color=nord_colors)


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
            self.obs[:, i] = self.observation_eq(
                self.sim.t[i], self.sim.y[:, i])
        # make outputs
        self.out = np.zeros(self.sim.y.shape)
        for i in range(len(self.sim.t)):
            self.out[:, i] = self.output_eq(self.sim.t[i], self.sim.y[:, i])

        # assign t, and y for ease of plotting
        self.t = self.sim.t
        self.y = self.sim.y

class SDE_system(simulation_system):

    def system_eq(self, t, x):
        return self.A @ np.nan_to_num(np.sqrt(x)) + self.Gamma @ self.u(t)

    def run_sim(self, t_span, x0, dt, seed=42):
        # set seed for reproducibility
        rng = np.random.RandomState(seed)

        # simulate using euler-maruyama method
        t = np.arange(t_span[0], t_span[1], dt)
        y = np.zeros([4, len(t)])
        obs = np.zeros(y.shape)
        out = np.zeros(y.shape)

        y[:,0] = x0
        for i in range(1, len(t)):
            #state update
            yn = y[:,i-1].reshape(4,1)
            
            yn = yn + self.system_eq(t[i], yn)*dt + self.D @ self.d(
                t[i]) * stats.norm(scale=dt).rvs(1, random_state=rng)
            
            y[:,i] = yn.reshape(4)
            # calc observation and output
            obs[:,i] = self.observation_eq(t[i], yn).reshape(4)
            out[:,i] = self.output_eq(t[i], yn).reshape(4)
        #save to object

        self.t = t
        self.y = y
        self.obs = obs
        self.out = out


# noise generator for model 2
def stochastic_noise_gen(dt, rate, seed=42):
    noise = dict()
    rng = np.random.RandomState(seed)

    eta = stats.expon(scale=1/rate)

    def stochastic_noise(t):
        k = np.floor(t / dt)
        if not k in noise:
            noise[k] = eta.rvs(2, random_state=rng).reshape(2, 1)
        return noise[k]
    return stochastic_noise


def tranfer_function_H(u,y, dt):
    U = fft.fft2(u)
    Y = fft.fft(y)
    freq = fft.fftfreq(Y.size, d = dt)
    
    U_nonzero = np.nonzero(U)

    Y = Y[U_nonzero]
    U = U[U_nonzero]
    freq = freq[U_nonzero]

    H =  np.linalg.solve(U, Y)

    return H, freq

def estimate_transfer(H, freq, nzeros, npoles, x0 = None, l1 = 0):
    # estimate the transfer function from data using squared loss an l1 regularization
    # parameters is in the shape [zeros , poles]
    # also add constant term
    if x0 is None:
        x0 = np.random.random(nzeros + npoles + 2)
    
    def loss (x):
        zeros = x[:nzeros+1]
        poles = x[nzeros+1:]

        vals  = np.array([np.polyval(zeros, s) / np.polyval(poles, s) for s in 1j*freq])

        l1_reg = l1*np.abs(x).sum()

        L = np.linalg.norm((vals - H).reshape(-1,1), axis = 1).sum() + l1_reg
        return L
    
    res = minimize(loss,x0)
    system = sig.lti(res.x[:nzeros+1], res.x[nzeros+1:])

    return system, res

def qpsolve(H,g,l,u,A,bl,bu, xinit):
    #Solve the program
    # min 1/2 x' H x + g' x
    # st.   l < x < u
    #       bl < A x < ul
    n = H.shape[0]
    m = A.shape[0]
    G1 = np.array([
        [np.indentity(n), np.zeros([n,n])],
        [np.zeros[n,n], -np.identity(n)]
        ])

    G2 = np.array([
        [A, np.zeros(A.shape)],
        [np.zeros(A.shape), -A]
    ])

    G = np.array([
        [G1, np.zeros([G1.shape[0], G2.shape[1]])],
        [np.zeros([G2.shape[0], G1.shape[1]]), G2]
        ])
    
    b = np.vstack((u, -l, bu, -bl ))

    return cvxopt.solvers.qp(H, g, G, b)

# %% parameters
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