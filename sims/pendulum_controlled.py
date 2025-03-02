import numpy as np
import cv2
import time
import control
import matplotlib.pyplot as plt
import scipy.signal as signal
from sims.InvertedPendulum import InvertedPendulum
from sims.sim_db import Database, SPSType
from types import SimpleNamespace
from scipy.signal import savgol_filter
from scipy.linalg import solve_discrete_are
import threading
class CartPendulumPlant:
    """
    CartPendulumPlant class models the dynamics of the inverted pendulum system.
    It includes the system parameters, LQR controller design, and dynamics simulation.
    """
    def __init__(self, dt=0.02, db: Database=None, sim=True):
        """
        Initializes the plant model with default parameters.
        
        Args:
            dt (float): Time step for numerical integration.
        """
        self.dt = dt
        self.sim = sim
        self.ss_event = threading.Event()
        self.initialised = False
        # sim params
        self.g = 9.8
        self.L = 1.5
        self.m = 1.0
        self.M = 5.0
        self.d1 = 1.0
        self.d2 = 0.5
        self.ss, self.armax, self.sig_ss = self.system_matrices()
        if db is not None:
            self.db = db
            self.db.subscribe("ss", self.ss_callback)

    def update_params(self, d1=1.0, d2=0.5):
        """
        Updates the damping parameters of the system.
        
        Args:
            d1 (float): Damping coefficient for the cart.
            d2 (float): Damping coefficient for the pendulum.
        """
        self.d1 = d1
        self.d2 = d2
        self.ss, self.armax, self.sig_ss = self.system_matrices()

    
    def system_matrices(self):
        """
        Returns the state-space matrices of the linearized system and G,H matrices.
        A, B, C, D matrices are used to define the state-space model.
        G, H  matrices are used to define the ARMAX model.
        Returns:
            tuple: State-space matrices (A, B, C, D).
        """
        _q = (self.m + self.M) * self.g / (self.M * self.L)
        A = np.array([
            [0, 1, 0, 0],
            [0, -self.d1, -self.m * self.g / self.M, 0],
            [0, 0, 0, 1],
            [0, self.d1 / self.L, _q, -self.d2]
        ])
        B = np.array([[0], [1 / self.M], [0], [-1 / (self.M * self.L)]])
        C = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        D = np.zeros((C.shape[0], B.shape[1]))

        ss_d = {
            "A": A,
            "B": B,
            "C": C,
            "D": D
        }
        ss_d = SimpleNamespace(**ss_d)
        sig_ss = signal.StateSpace(A, B, C, D, dt=self.dt)

        G = sig_ss.to_tf()
        H = np.array([1, 0])
        armax = {'G': G, 'H': H}
        armax = SimpleNamespace(**armax)
        return ss_d, armax, sig_ss
    
    def ss_callback(self, ss):
        self.ss = ss
        print("[Controller] State space matrices updated")
        if not self.initialised:
            self.ss_event.set()
    
    def write_data_to_db(self, y, u, r, sps_type: SPSType):
        data = {
            "y": y,
            "u": u,
            "r": r,
            "sps_type": sps_type
        }
        data = SimpleNamespace(data)
        if self.db:
            self.db.write_data(data=data)
        else:
            print("No database provided.")

    def dynamics(self, y, u):
        """
        Defines the dynamics of the inverted pendulum system.

        Args:
            y (ndarray): Current state vector.
            u (float): Applied input.

        Returns:
            ndarray: Derivatives of the state vector.
        """
        if self.sim:
            theta, theta_dot = y[2], y[3]
            x_ddot = (u - self.m * self.L * theta_dot**2 * np.cos(theta) + self.m * self.g * np.sin(theta) * np.cos(theta)) / (self.M + self.m - self.m * np.sin(theta)**2)
            theta_ddot = -self.g/self.L * np.cos(theta) - np.sin(theta)/self.L * x_ddot
            return np.array([y[1], x_ddot - self.d1 * y[1], y[3], theta_ddot - self.d2 * y[3]])
        else:
            x_dot = np.dot(self.ss.A, y) + np.dot(self.ss.B.T, u)
            return x_dot.reshape(-1)
    
    def clip_angle(self, angle):
        angle = angle % (2 * np.pi)
        if angle > np.pi:
            angle -= 2 * np.pi
        return angle

    def rk4_step(self, state, u):
        """
        Performs one step of numerical integration using the Runge-Kutta 4th order method.

        Args:
            u (float): Control input.
        """
        k1 = self.dt * self.dynamics(state, u)
        k2 = self.dt * self.dynamics(state + k1 / 2, u)
        k3 = self.dt * self.dynamics(state + k2 / 2, u)
        k4 = self.dt * self.dynamics(state + k3, u)
        state += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        state[2] = self.clip_angle(state[2])  # Angle wrapping
        return state


def dlqe(A, C, Q, R):
    """ Discrete Linear Quadratic Estimator (DLQE) """
    P = solve_discrete_are(A.T, C.T, Q, R)
    K = P @ C.T @ np.linalg.inv(C @ P @ C.T + R)
    L = A @ K
    return L, P, K
class Observer:
    """
    Extended Kalman Filter (EKF) Observer with Adaptive Noise Covariance Estimation
    and Joseph Stabilized Update.
    """

    def __init__(self, plant, window_size=100, epsilon=1e-6, sg_window=100, sg_order=4):
        self.plant = plant
        self.dt = plant.dt
        self.estimated_state = np.zeros(plant.ss.A.shape[0])
        self.P = np.diag([0.1, 0.1, 0.1, 0.1])
        self.Q = np.diag([0.01, 0.01, 0.01, 0.01])
        self.R = np.eye(self.plant.ss.C.shape[1]) * 0.1  # Measurement Noise Covariance
        self.C = np.identity(self.plant.ss.C.shape[1])
        self.L, _, _ = dlqe(plant.ss.A, self.C, self.Q, self.R)
 
        self.window_size = window_size
        self.epsilon = epsilon
        self.residuals = []
        self.prediction_errors = []
        self.last_state = np.array([0, np.pi/4])

        # Savitzky-Golay filter params
        self.sg_window = sg_window
        self.sg_order = sg_order

    def update(self, measured_state, u):
        """
        Update Step with Adaptive Q and R Estimation + Joseph Stabilized Update.
        """
        # Prediction Step
        predicted_state = self.plant.rk4_step(self.estimated_state.copy(), u)
        A_lin = self.plant.ss.A
        # Innovation
        measured_vel = (measured_state - self.last_state) / self.dt
        measured_s = np.array([measured_state[0], measured_vel[0], measured_state[1], measured_vel[1]])
        innovation = measured_s - self.C @ predicted_state
        self.residuals.append(innovation)
        self.prediction_errors.append(predicted_state - self.estimated_state)
        # Trim Buffers
        self.residuals = self.residuals[-self.window_size:]
        self.prediction_errors = self.prediction_errors[-self.window_size:]

        # Adaptive R
        if len(self.residuals) == self.window_size:
            self.R = self.estimate_measurement_covariance()

        # Adaptive Q
        if len(self.prediction_errors) == self.window_size:
            self.Q = self.estimate_process_covariance()

        self.P = solve_discrete_are(self.plant.ss.A.T, self.C.T, self.Q, self.R)
        # Innovation Covariance with Regularization
  
        S = self.C @ self.P @ self.C.T + self.R
        regularization = np.eye(S.shape[0]) * self.epsilon
        S += regularization
        try: 
            K = self.P @ self.C.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            print("[WARNING] Singular Matrix detected, applying larger regularization")
            S += np.eye(S.shape[0]) * 1e-3
            K = self.P @ self.C.T @ np.linalg.inv(S)

        # State Correction

        self.estimated_state = predicted_state + K @ innovation 

        self.estimated_state[0] = measured_state[0]
        self.estimated_state[2] = measured_state[1]

        # Joseph Stabilized Covariance Update
        # I = np.eye(self.P.shape[0])
        # self.P = (I - K @ self.p.C) @ self.P @ (I - K @ self.C).T + K @ self.R @ K.T
        self.last_state = measured_state

    def apply_smoothing(self, residuals):
        residual_matrix = np.vstack(residuals)
        smoothed_residual = np.apply_along_axis(
            lambda x: savgol_filter(x, window_length=self.sg_window, polyorder=self.sg_order),
            axis=0,
            arr=residual_matrix
        )
        return smoothed_residual

    def estimate_measurement_covariance(self):
        residual_matrix = self.apply_smoothing(self.residuals)
        normalized = residual_matrix / (np.max(np.abs(residual_matrix), axis=0) + self.epsilon)
        R_est = np.cov(normalized.T) + self.epsilon * np.eye(self.R.shape[0])
        return R_est

    def estimate_process_covariance(self):
        error_matrix = self.apply_smoothing(np.vstack(self.prediction_errors))
        normalized = error_matrix / (np.max(np.abs(error_matrix), axis=0) + self.epsilon)

        velocity_noise = np.var(normalized[:, 3])  # Velocity noise only
        Q_est = np.cov(normalized.T) + self.epsilon * np.eye(self.Q.shape[0])
        Q_est[3, 3] += velocity_noise * 10  # Boost process noise for velocity
        return Q_est
class Controller:
    """
    Controller class handles the control of the cart-pendulum system.
    """
    def __init__(self, plant, desired = np.array([0, 0, np.pi/2, 0])):
        """
        Initializes the controller with the plant model.

        Args:
            plant (CartPendulumPlant): Plant model instance.
        """
        self.plant = plant
        self.Q = np.diag([10, 0.1, 10, 0.1])
        self.R = np.array([[1]])
        self.armax, self.K = self.design_lqr()
        self.desired = desired
        
    def design_lqr(self):
        """
        Designs the LQR controller for the linearized system.

        Returns:
            tuple: State-space model and LQR gain matrix.
        """
        # LQR design
        K, _, _ = control.lqr(self.plant.ss.A, self.plant.ss.B, self.Q, self.R)
        # K = np.dot(self.plant.ss.C, K.T).reshape(K.shape[0], -1)
        # G, H, F, L 
        F = L  = K 
        armax = {'F': F, 'L': L}
        armax = SimpleNamespace(**armax)
        return armax, K
    
    def get_u(self, y):
        """
        Computes the LQR control input.

        Args:
            y (ndarray): Current state vector.

        Returns:
            ndarray: Control input vector.
        """

        u = self.K @ (self.desired - y)
        # u = np.clip(u, -100, 100)
        return u 



class CartPendulumSimulation:
    """
    CartPendulumSimulation class handles the simulation and visualization of the cart-pendulum system.
    """
    def __init__(self, plant: CartPendulumPlant, controller, observer, T=10, disturbance=50.0):
        """
        Initializes the simulation environment.

        Args:
            plant (CartPendulumPlant): Plant model instance.
            T (float): Total simulation time.
            disturbance (float): Magnitude of the disturbance force.
        """
        self.plant = plant
        self.controller = controller
        self.observer = observer
        self.T = T
        self.disturbance = disturbance
        self.sim_model = CartPendulumPlant()
        self.sim = InvertedPendulum()
        self.apply_disturbance = False
    
    def square_wave(self, T, f, fs):
        """
        Generate a square wave using numpy.

        Parameters:
            T : float
                Duration of the signal in seconds
            f : float
                Frequency of the square wave in Hz
            fs : float
                Sampling frequency in Hz

        Returns:
            t : numpy.ndarray
                Time vector
            y : numpy.ndarray
                Square wave signal
        """
        t = np.linspace(0, T, int(fs * T), endpoint=False)  # Time vector
        y =  signal.square(2 * np.pi * f * t)
        return t, y
    
    def impulse_wave(self, T, f, fs):
        """
        Generate an impulse wave.

        Parameters:
            T : float
                Duration of the signal in seconds
            f : float
                Frequency of impulses in Hz
            fs : float
                Sampling frequency in Hz

        Returns:
            t : numpy.ndarray
                Time vector
            y : numpy.ndarray
                Impulse wave signal
        """
        t = np.linspace(0, T, int(fs * T), endpoint=False)  # Time vector
        y = np.zeros_like(t)                                # Initialize signal with zeros
        
        impulse_period = int(fs / f)                        # Samples between impulses
        y[::impulse_period] = 1                             # Set impulses at intervals
        
        return t, y

    def sim_model_response(self, T, f, input_type="square_wave"):
        def _input():
            input_funcs = {"impulse_wave": self.impulse_wave,
                           "square_wave": self.square_wave}
            if input_type not in input_funcs:
                raise ValueError(f"Invalid input type: {input_type}. Must be one of {list(input_funcs.keys())}")
            input_func = input_funcs[input_type]
            inputs = input_func(T=T, f=f, fs=1/self.sim_model.dt)
            return inputs
        t, u = _input()
        initial_state = np.array([0.0, 0.0, np.pi/4 + 0.1, 0.0])
        states = []
        states.append(initial_state)
        for _u in u:
            states.append(self.sim_model.rk4_step(states[-1].copy(), _u))
        states.pop()
        states = np.array(states)
        return states, u, t
    
    def initialise_plant(self, T=5, f=1, input_type="square_wave", timeout=5, max_retries=3):
        """
        Whack the system with square wave or impulses, send the data to DB,
        and wait for an SS update with retries before raising TimeoutError.

        Args:
            T (float): Duration of signal.
            f (float): Frequency of signal.
            input_type (str): Type of input signal ("square_wave" or "impulse").
            timeout (float): Timeout duration in seconds.
            max_retries (int): Maximum number of retries if update is not received.

        Raises:
            TimeoutError: If no SS update is received within the timeout after retries.
        """
        if self.plant.initialised:
            print("[Init] Already initialized, skipping whacking")
            return

        retries = 0
        # Send input to DB
        y, u, t = self.sim_model_response(T=T, f=f, input_type=input_type)
        self.plant.write_data_to_db(y=y, u=u, r=None, sps_type=SPSType.OPEN_LOOP)

        while retries <= max_retries:
            print(f"[Init] Whacking attempt {retries + 1}/{max_retries + 1}")
            
            # Wait for SS update
            self.plant.ss_event.clear()
            print("[Init] Waiting for SS update...")

            update_received = self.ss_event.wait(timeout)

            if update_received:
                print("[Init] Initial SS update received!")
                self.plant.initialised = True
                print("[Init] Latest SS:", self.plant.ss)
                return  # Exit the loop if update is received

            print("[Init] Timeout! No update received.")
            retries += 1

        print("[Init] Max retries reached. No SS update received.")
        raise TimeoutError("[Init] Failed to receive SS update after retries.")




    def run(self):
        """
        Runs the cart-pendulum simulation.
        """
        history = []
        observer_history = []
        input_history = []
        start = time.time()
        db_t = 0
        state = np.array([0.0, 0.0, np.pi/4 + 0.1, 0.0])
        measured_state = np.dot(self.plant.ss.C, state) # Initial state measurement
        u = 0.0
        state = self.sim_model.rk4_step(state, u)
        self.observer.estimated_state = state.copy()
        self.observer.last_state = measured_state.copy()
        for _ in range(int(self.T / self.plant.dt)):
            estimated_state = self.observer.estimated_state
            u = self.controller.get_u(estimated_state)[0]
            if self.apply_disturbance:
                print("Exciting System!")
                u += self.disturbance
            
            state = self.sim_model.rk4_step(state, u)
            # state[2] = self.observer.clip_angle(state[2])
            measured_state = np.dot(self.plant.ss.C, state)
            # clip measured_state angle

            self.observer.update(measured_state.copy(), u) 
            estimated_state = self.observer.estimated_state
            
            rendered = self.sim.step(state, _ * self.plant.dt)
            cv2.imshow('Cart-Pendulum', rendered)
            history.append(state.copy())
            observer_history.append(self.observer.estimated_state.copy())
            input_history.append(u)
            key = cv2.waitKey(1)
            curr = time.time()

            if key == ord('q'):
                break
            elif key == ord('d'):
                self.apply_disturbance = True
                db_t = curr

            if self.apply_disturbance and curr - db_t > 0.5:
                self.apply_disturbance = False

            elapsed = curr - start
            if elapsed < self.plant.dt:
                time.sleep(self.plant.dt - elapsed)
            start = time.time()

        cv2.destroyAllWindows()
        self.plot_history(np.array(history), np.array(observer_history), np.array(input_history))

    def plot_history(self, history, observer_history, input_history):
        """
        Plots the system state over time.

        Args:
            history (ndarray): History of system states.
        """
        t = np.linspace(0, len(history) * self.plant.dt, len(history))
        plt.figure(figsize=(10, 6))
        plt.subplot(5, 1, 1)
        plt.plot(t, history[:, 0], label='Cart Position')
        plt.plot(t, np.zeros_like(t), 'r--', label='Reference Position')
        plt.plot(t, observer_history[:, 0], label='Estimated Cart Position')
        plt.legend()
        plt.grid()
        plt.subplot(5, 1, 2)
        plt.plot(t, history[:, 2], label='Pendulum Angle')
        plt.plot(t, np.pi/2 * np.ones_like(t), 'r--', label='Reference Angle')
        plt.plot(t, observer_history[:, 2], label='Estimated Pendulum Angle')
        plt.legend()
        plt.grid()
        plt.subplot(5, 1, 3)
        plt.plot(t, history[:, 1], label='Cart Velocity')
        plt.plot(t, np.zeros_like(t), 'r--', label='Reference Velocity')
        plt.plot(t, observer_history[:, 1], label='Estimated Cart Velocity')
        plt.legend()
        plt.grid()
        plt.subplot(5, 1, 4)
        plt.plot(t, history[:, 3], label='Pendulum Angular vel')
        plt.plot(t, np.zeros_like(t), 'r--', label='Reference Vel')
        plt.plot(t, observer_history[:, 3], label='Estimated Pendulum Vel')
        plt.legend()
        plt.grid()
        plt.suptitle('Cart-Pendulum State Over Time')
        plt.subplot(5, 1, 5)
        plt.plot(t, input_history, label='input Force')
        plt.legend()
        plt.grid()
        plt.suptitle('Controller Input Over Time')
        plt.show()

import numpy as np
from numpy.linalg import matrix_rank

def observability_matrix(A, C):
    n = A.shape[0]  # Number of states
    O = C
    for i in range(1, n):
        O = np.vstack([O, C @ np.linalg.matrix_power(A, i)])
    return O

# Example check

if __name__ == '__main__':
    plant = CartPendulumPlant(dt=0.02, sim=False)

    O = observability_matrix(plant.ss.A, plant.ss.C)
    obs_rank = matrix_rank(O)

    if obs_rank == plant.ss.A.shape[0]:
        print(f"System is Observable (Rank = {obs_rank})")
    else:
        print(f"System is NOT Observable (Rank = {obs_rank})")

    
    G, H  = plant.armax.G, plant.armax.H

    print("G: ", G)
    print("H: ", H)
    # ss_c = control.ss(plant.ss.A, plant.ss.B.T, plant.ss.C, plant.ss.D)

    # print("ss_c: ", ss_c)
    # print("canonical form: ", control.canonical_form(ss_c, form='observable'))
    # plant.ss = ss_g
    desired = np.array([0, 0, np.pi/2, 0])
    # desired = np.dot(plant.ss.C, desired)
    controller = Controller(plant, desired=desired)
    # controller.K = np.dot(controller.K, plant.ss.C.T)
    F, L = controller.armax.F, controller.armax.L

    observer = Observer(plant)
    sim = CartPendulumSimulation(plant, controller, observer, T=20, disturbance=50)
    sim.run()
