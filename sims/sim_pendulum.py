import numpy as np
import cv2
import time
import control
import matplotlib.pyplot as plt
import scipy.signal as signal
from sims.InvertedPendulum import Pendulum
from sims.sim_db import Database, SPSType
from types import SimpleNamespace
from scipy.signal import savgol_filter
from scipy.linalg import solve_discrete_are
import threading

class PendulumPlant:
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
        self.g = 9.81
        self.L = 1.1
        self.m =1.0
        self.d = 0.5
        self.inertia = self.m * (self.L ** 2)
        self.ss, self.armax, self.sig_ss = self.system_matrices()
        if db is not None:
            self.db = db
            self.db.subscribe("ss", self.ss_callback)

    def update_params(self,d=0.5):
        """
        Updates the damping parameters of the system.
        
        Args:
            d (float): Damping coefficient for the pendulum.
        """
        self.d = d
        self.ss, self.armax, self.sig_ss = self.system_matrices()

    
    def system_matrices(self):
        """
        Returns the state-space matrices of the linearized system and G,H matrices.
        A, B, C, D matrices are used to define the state-space model.
        G, H  matrices are used to define the ARMAX model.
        Returns:
            tuple: State-space matrices (A, B, C, D).
        """
        A = np.array([[0, 1],
                      [-self.g / self.L, -self.d / self.inertia]])
        B = np.array([[0],
                      [1 / self.inertia]])
        C = np.array([[1, 0]])
        D = np.array([[0]])
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
            theta, theta_dot = y[0], y[1]
            # theta_ddot = -self.g/self.L * np.cos(theta) - np.sin(theta)/self.L * x_ddot
            theta_ddot = (u - self.d * theta_dot - self.m * self.g * self.L * np.cos(theta))/self.inertia
            return np.array([theta_dot, theta_ddot])
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
        # state[0] = self.clip_angle(state[0])  # Angle wrapping
        return state

def plot_history(history, dt, labels=['Pendulum Angle', 'Pendulum Angular vel']):
    """
    Plots the system state over time.

    Args:
        history (ndarray): History of system states.
    """
    t = np.linspace(0, len(history) * dt, len(history))
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, history[:, 0], label='Pendulum Angle')
    plt.plot(t, np.pi/2 * np.ones_like(t), 'r--', label='Reference Angle')
    plt.legend()
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(t, history[:, 1], label='Pendulum Angular vel')
    plt.plot(t, np.zeros_like(t), 'r--', label='Reference Vel')
    plt.legend()
    plt.grid()
    plt.suptitle('Sim Input Over Time')
    plt.show()





if __name__ == '__main__':
    history = []
    T=20
    dt=0.02
    plant = PendulumPlant(dt,sim=True)
    sim = Pendulum()
    state = np.array([np.pi/4, 0.0])
    for _ in range(int(T/plant.dt)):
        state = plant.rk4_step(state, u=0)
        rendered = sim.draw(state, _ * plant.dt)
        cv2.imshow('Cart-Pendulum', rendered)
        history.append(state.copy())
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        
    cv2.destroyAllWindows()
    plot_history(np.array(history), dt)