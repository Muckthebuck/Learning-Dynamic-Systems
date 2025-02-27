import numpy as np
import cv2
import time
import control
import matplotlib.pyplot as plt
from InvertedPendulum import InvertedPendulum
import scipy.signal as signal

class CartPendulumPlant:
    """
    CartPendulumPlant class models the dynamics of the inverted pendulum system.
    It includes the system parameters, LQR controller design, and dynamics simulation.
    """
    def __init__(self, dt=0.02):
        """
        Initializes the plant model with default parameters.
        
        Args:
            dt (float): Time step for numerical integration.
        """
        self.dt = dt
        self.g = 9.8
        self.L = 1.5
        self.m = 1.0
        self.M = 5.0
        self.d1 = 1.0
        self.d2 = 0.5
        self.ss, self.armax = self.system_matrices()
    
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
        C = np.eye(4)
        D = np.zeros((4, 1))
        ss = signal.StateSpace(A, B, C, D, dt=self.dt)

        G = ss.to_tf()
        H = np.array([1, 0])
        armax = {'G': G, 'H': H}
        return ss, armax

    def dynamics(self, y, u):
        """
        Defines the dynamics of the inverted pendulum system.

        Args:
            y (ndarray): Current state vector.
            u (float): Applied input.

        Returns:
            ndarray: Derivatives of the state vector.
        """
        theta, theta_dot = y[2], y[3]
        x_ddot = (u - self.m * self.L * theta_dot**2 * np.cos(theta) + self.m * self.g * np.sin(theta) * np.cos(theta)) / (self.M + self.m - self.m * np.sin(theta)**2)
        theta_ddot = (-self.g * np.cos(theta) - np.sin(theta) * x_ddot) / self.L
        return np.array([y[1], x_ddot - self.d1 * y[1], y[3], theta_ddot - self.d2 * y[3]])

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
        return state

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
        self.Q = np.diag([200, 1, 10, 1])
        self.R = np.array([[1]])
        self.ss_armax, self.K = self.design_lqr()
        self.desired = desired
        
    def design_lqr(self):
        """
        Designs the LQR controller for the linearized system.

        Returns:
            tuple: State-space model and LQR gain matrix.
        """
        # LQR design
        K, _, _ = control.lqr(self.plant.ss.A, self.plant.ss.B, self.Q, self.R)
    
        # G, H, F, L 
        F = L  = K 
        ss_armax = {'F': F, 'L': L}

        return ss_armax, K
    
    def get_u(self, y):
        """
        Computes the LQR control input.

        Args:
            y (ndarray): Current state vector.

        Returns:
            ndarray: Control input vector.
        """
        return self.K @ (self.desired - y)



class CartPendulumSimulation:
    """
    CartPendulumSimulation class handles the simulation and visualization of the cart-pendulum system.
    """
    def __init__(self, plant, controller, T=10, disturbance=50.0):
        """
        Initializes the simulation environment.

        Args:
            plant (CartPendulumPlant): Plant model instance.
            T (float): Total simulation time.
            disturbance (float): Magnitude of the disturbance force.
        """
        self.plant = plant
        self.controller = controller
        self.T = T
        self.disturbance = disturbance
        self.sim_model = CartPendulumPlant()
        self.sim = InvertedPendulum()
        self.apply_disturbance = False

    def run(self):
        """
        Runs the cart-pendulum simulation.
        """
        history = []
        start = time.time()
        db_t = 0
        state = np.array([0.0, 0.0, np.pi/4 + 0.1, 0.0])
        for _ in range(int(self.T / self.plant.dt)):
            u = self.controller.get_u(state)
            if self.apply_disturbance:
                print("Exciting System!")
                u[0] += self.disturbance
            state = self.sim_model.rk4_step(state, u[0])
            rendered = self.sim.step(state, _ * self.plant.dt)
            cv2.imshow('Cart-Pendulum', rendered)
            history.append(state.copy())
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
        self.plot_history(np.array(history))

    def plot_history(self, history):
        """
        Plots the system state over time.

        Args:
            history (ndarray): History of system states.
        """
        t = np.linspace(0, len(history) * self.plant.dt, len(history))
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(t, history[:, 0], label='Cart Position')
        plt.plot(t, np.zeros_like(t), 'r--', label='Reference Position')
        plt.legend()
        plt.grid()
        plt.subplot(2, 1, 2)
        plt.plot(t, history[:, 2], label='Pendulum Angle')
        plt.plot(t, np.pi/2 * np.ones_like(t), 'r--', label='Reference Angle')
        plt.legend()
        plt.grid()
        plt.suptitle('Cart-Pendulum State Over Time')
        plt.show()


if __name__ == '__main__':
    plant = CartPendulumPlant(dt=0.02)
    controller = Controller(plant)
    sim = CartPendulumSimulation(plant, controller, T=20, disturbance=50)
    sim.run()
