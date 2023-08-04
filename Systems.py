import numpy as np
import matplotlib.pyplot as plt

class simulation_system:

    # dx = A f(x) + B u(t) + D d(t) G eta
    # y = C x + nu
    def __init__(self, A, B, C, D, noise_cov, obs_cov, output_matrix, G = None, dt = None):
        self.A = A # System matrix
        self.B = B # Input matrix 
        self.C = C # Observation matrix
        self.D = D # disturbance matrix
        self.output_matrix = output_matrix

        # for now assume system noise and observation noise are independent
        self.noise_cov = noise_cov # covariance matrix of system noise
        self.obs_cov = obs_cov # covariance matrix of observation matrix

        self.nstates = self.A.shape[0]
        self.nobs = self.C.shape[0]
        self.noutputs = self.output_matrix.shape[0]

        # G is the noise -> system translation matrix
        # if not given assume noise is directly put into system
        if G is None: G = np.identity(self.nstates)
        self.G = G

        #assume noise have zero mean
        self.noise_mean = np.zeros(self.G.shape[1]) 
        self.obs_mean = np.zeros(self.nobs)

        self.dt = dt # if not none then assume system is discrete

    def run_continous(self, tspan, dt, u = None, x0 = None, d = None):

        N = int(np.ceil((tspan[1] - tspan[0]) / dt))
        self.T = dt * np.arange(N)
        self.X = np.zeros([N, self.nstates])
        self.Y = np.zeros([N, self.nobs])
        self.Out = np.zeros([N, self.noutputs])

        if u is None:
            u = lambda y, t:  np.zeros(self.B.shape[1])
        if d is None:
            d = lambda t: np.zeros(self.D.shape[1])

        if not x0 is None:
            self.X[0] = x0
            self.Y[0] = self.obs_eq(self.X[0], 0)
            self.Out[0] = self.out_eq(self.X[0])

        for i in range(1,N):
            # x(t+dt) = x(t) + x'(t) * dt + eta_dt
            self.X[i] = self.X[i-1] + dt * ( self.system_eq(self.X[i-1], u(self.Y[i-1], self.T[i]), self.T[i]) + (self.D @ d(self.T[i])).T + (self.G @ np.random.multivariate_normal(self.noise_mean, self.noise_cov, 1).T).T )
            self.X[i] = np.clip(self.X[i], 0, None) # system should be physical

            self.Y[i] = self.obs_eq(self.X[i], self.T[i]) + np.random.multivariate_normal(self.obs_mean, self.obs_cov,1)
            self.Out[i] = self.out_eq(self.X[i])
    
    def obs_eq(self, x, t):
        return self.C @ x
    def out_eq(self, x):
        return self.output_matrix @ x

    def run_discrete(self, u, tspan, x0 = None):
        N = int(np.ceil((tspan[1] - tspan[0]) / self.dt))
        self.T = self.dt * np.arange(N)
        self.X = np.zeros([N, self.nstates])
        self.Y = np.zeros([N, self.nobs])
        self.Out = np.zeros([N, self.noutputs])

        if not x0 is None:
            self.X[0] = x0
            self.Y[0] = self.obs_eq(self.X[0], 0)
            self.Out[0] = self.out_eq(self.X[0], 0)
        
        for i in range(1,N):
            self.X[i] = self.system_eq(self.X[i-1], u(self.Y[i-1], self.T[i])) + self.G @ np.random.multivariate_normal(self.noise_mean, self.noise_cov, 1)
            self.Y[i] = self.obs_eq(self.X[i]) + np.random.multivariate_normal(self.obs_mean, self.obs_cov,1)
            self.Out[i] = self.out_eq(self.X[i], self.T[i])

    def plot_sim(self, fig = None, state_names = None, observation_names = None, output_names = None):
        if fig is None: fig = plt.figure(figsize=(20,10))

        axes = fig.subplots(1,3)

        axes[0].plot(self.T,self.X)
        axes[1].plot(self.T,self.Y)
        axes[2].plot(self.T,self.Out)

        axes[0].set_title("States")
        axes[1].set_title("Observations")
        axes[2].set_title("Outputs")

        if state_names is not None: axes[0].legend(state_names)
        if observation_names is not None: axes[1].legend(observation_names)
        if output_names is not None: axes[2].legend(output_names)
        
        return fig, axes

class nonlinear_system(simulation_system):
    def system_eq(self, x, u, t):
        return self.A @ np.nan_to_num(np.sqrt(x)) + self.B @ u

class Linear_system(simulation_system):
    def system_eq(self, x, u, t):
        return self.A @ x + self.B @ u

def stochastic_noise_gen(T_shift, seed=42):
    noise = dict()
    rng = np.random.RandomState(seed)

    def stochastic_noise(t):
        k = np.floor(t / T_shift)
        if not k in noise:
            noise[k] = rng.multivariate_normal(
                np.array([1,1]),
                np.array([[0.25,0],[0,0.25]]),
                1
            )

            noise[k] = np.clip(noise[k], 0, None).T
        return noise[k]
    return stochastic_noise

# define systems
gamma_1 =  0.58  # split 1
gamma_2 =  0.68  # split 2
A_1 =  1e-2  # Area tank 1
A_2 =  2e-2  # Area tank 2
A_3 = 3e-2  # Area tank 3
A_4 =  4e-2  # Area tank 4
a_1 =  2e-4  # Area hole 1
a_2 =  2.5e-4  # Area hole 2
a_3 =  3e-4  # Area hole 3
a_4 =  3.5e-4 # Area hole 4
g = 9.8  # gravity 9.8 m/s^2
rho =  1000  # density of water 1000 kg/m^3

# output variable of the systems, the water mass -> water level translation matrix
output_matrix = np.array([
        [1/(rho * A_1), 0, 0, 0],
        [0, 1/(rho * A_2), 0, 0],
        [0, 0, 1/(rho * A_3), 0],
        [0, 0, 0, 1/(rho * A_4)]
    ])
output_matrix_sde = np.array([
        [1/(rho * A_1), 0, 0, 0, 0, 0],
        [0, 1/(rho * A_2), 0, 0, 0, 0],
        [0, 0, 1/(rho * A_3), 0, 0, 0],
        [0, 0, 0, 1/(rho * A_4), 0, 0]
    ])

# system parameters
sys_det_min = {
    "A" : np.array([
            [-np.sqrt(2 * g * rho * a_1**2 / A_1), 0, np.sqrt(2 * g * rho * a_3**2 / A_3), 0],
            [0, -np.sqrt(2 * g * rho * a_2**2 / A_2), 0, np.sqrt(2 * g * rho * a_4**2 / A_4)],
            [0, 0, -np.sqrt(2 * g * rho * a_3**2 / A_3), 0],
            [0, 0, 0, -np.sqrt(2 * g * rho * a_4**2 / A_4)]
        ]),
    "B" : np.array([
            [gamma_1, 0],
            [0, gamma_2],
            [0, 1-gamma_2],
            [1-gamma_1, 0]
        ]),
    "C" : np.array([
            [g, 0, 0, 0],
            [0, g, 0, 0],
            [0, 0, g, 0],
            [0, 0, 0, g]
        ]),
    "D" : np.array([
            [0, 0],
            [0, 0],
            [1, 0],
            [0, 1]
        ]),
    "output_matrix" : output_matrix,
    "noise_cov" : np.zeros([4,4]),
    "obs_cov" : np.zeros([4,4])

}

sys_det_nonmin = {
    "A" : np.array([
            [-np.sqrt(2 * g * rho * a_1**2 / A_1), 0, np.sqrt(2 * g * rho * a_3**2 / A_3), 0],
            [0, -np.sqrt(2 * g * rho * a_2**2 / A_2), 0, 0],
            [0, 0, -np.sqrt(2 * g * rho * a_3**2 / A_3), 0],
            [0, 0, 0, 0]
        ]),
    "B" : np.array([
            [gamma_1, 0],
            [0, gamma_2],
            [0, 1-gamma_2],
            [1-gamma_1, 0]
        ]),
    "C" : np.array([
            [g, 0, 0, 0],
            [0, g, 0, 0],
            [0, 0, g, 0],
            [0, 0, 0, g]
        ]),
    "D" : np.array([
            [0, 0],
            [0, 0],
            [1, 0],
            [0, 1]
        ]),
    "output_matrix" : output_matrix,
    "noise_cov" : np.zeros([4,4]),
    "obs_cov" : np.zeros([4,4])

}

sys_stoch_min = {
    "A" : np.array([
            [-np.sqrt(2 * g * rho * a_1**2 / A_1), 0, np.sqrt(2 * g * rho * a_3**2 / A_3), 0],
            [0, -np.sqrt(2 * g * rho * a_2**2 / A_2), 0, np.sqrt(2 * g * rho * a_4**2 / A_4)],
            [0, 0, -np.sqrt(2 * g * rho * a_3**2 / A_3), 0],
            [0, 0, 0, -np.sqrt(2 * g * rho * a_4**2 / A_4)]
        ]),
    "B" : np.array([
            [gamma_1, 0],
            [0, gamma_2],
            [0, 1-gamma_2],
            [1-gamma_1, 0]
        ]),
    "C" : np.array([
            [g, 0, 0, 0],
            [0, g, 0, 0],
            [0, 0, g, 0],
            [0, 0, 0, g]
        ]),
    "D" : np.array([
            [0, 0],
            [0, 0],
            [1, 0],
            [0, 1]
        ]),
    "output_matrix" : output_matrix,
    "noise_cov" : np.zeros([4,4]),
    "obs_cov" : np.identity(4)*10

}

sys_stoch_nonmin = {
    "A" : np.array([
            [-np.sqrt(2 * g * rho * a_1**2 / A_1), 0, np.sqrt(2 * g * rho * a_3**2 / A_3), 0],
            [0, -np.sqrt(2 * g * rho * a_2**2 / A_2), 0, 0],
            [0, 0, -np.sqrt(2 * g * rho * a_3**2 / A_3), 0],
            [0, 0, 0, 0]
        ]),
    "B" : np.array([
            [gamma_1, 0],
            [0, gamma_2],
            [0, 1-gamma_2],
            [1-gamma_1, 0]
        ]),
    "C" : np.array([
            [g, 0, 0, 0],
            [0, g, 0, 0],
            [0, 0, g, 0],
            [0, 0, 0, g]
        ]),
    "D" : np.array([
            [0, 0],
            [0, 0],
            [1, 0],
            [0, 1]
        ]),
    "output_matrix" : output_matrix,
    "noise_cov" : np.zeros([4,4]),
    "obs_cov" : np.identity(4)*10

}

sys_sde_min = {
    "A" : np.array([
            [-np.sqrt(2 * g * rho * a_1**2 / A_1), 0, np.sqrt(2 * g * rho * a_3**2 / A_3), 0, 0, 0],
            [0, -np.sqrt(2 * g * rho * a_2**2 / A_2), 0, np.sqrt(2 * g * rho * a_4**2 / A_4), 0, 0],
            [0, 0, -np.sqrt(2 * g * rho * a_3**2 / A_3), 0, 1, 0],
            [0, 0, 0, -np.sqrt(2 * g * rho * a_4**2 / A_4), 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ]),
    "B" : np.array([
            [gamma_1, 0],
            [0, gamma_2],
            [0, 1-gamma_2],
            [1-gamma_1, 0],
            [0, 0],
            [0, 0]
        ]),
    "C" : np.array([
            [g, 0, 0, 0, 0, 0],
            [0, g, 0, 0, 0, 0],
            [0, 0, g, 0, 0, 0],
            [0, 0, 0, g, 0, 0]
        ]),
    "D" : np.array([[0]]),
    "output_matrix" : output_matrix_sde,
    "noise_cov" : np.diag([0, 0, 0, 0, 0.25, 0.25]),
    "obs_cov" : np.identity(4)
}

sys_sde_nonmin = {
    "A" : np.array([
            [-np.sqrt(2 * g * rho * a_1**2 / A_1), 0, np.sqrt(2 * g * rho * a_3**2 / A_3), 0, 0, 0],
            [0, -np.sqrt(2 * g * rho * a_2**2 / A_2), 0, 0, 0, 0],
            [0, 0, -np.sqrt(2 * g * rho * a_3**2 / A_3), 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ]),
    "B" : np.array([
            [gamma_1, 0],
            [0, gamma_2],
            [0, 1-gamma_2],
            [1-gamma_1, 0],
            [0, 0],
            [0, 0]
        ]),
    "C" : np.array([
            [g, 0, 0, 0, 0, 0],
            [0, g, 0, 0, 0, 0],
            [0, 0, g, 0, 0, 0],
            [0, 0, 0, g, 0, 0]
        ]),
    "D" : np.array([[0]]),
    "output_matrix" : output_matrix_sde,
    "noise_cov" : np.diag([0, 0, 0, 0, 0.1, 0.1]),
    "obs_cov" : np.identity(4)

}

# noise parameters
det_noise = lambda t: np.array([1,1])
stoch_noise = stochastic_noise_gen(10)
sde_noise = lambda t: np.array([[0]])

