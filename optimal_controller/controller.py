import numpy as np
from dB.sim_db import Database
from optimal_controller.optimal_controls import get_optimal_controller
from types import SimpleNamespace
import threading

class Plant:
    """
    Plant class keeps a set of estimated system dynamics.
    The class subscribes to the database for updates on the system dynamics.
    The class is used to design and implement controllers and observers.
    """
    def __init__(self, dt=0.02, db: Database=None):
        """
        Initializes the plant model with default parameters.
        
        Args:
            dt (float): Time step for numerical integration.
        """
        self.dt = dt
        self.initialised = False
        # sim params
        self.ss = None
        self.new_update = False
        
        if db is not None:
            self.db = db
        else:
            # default "sim.db" database
            self.db = Database()
        
        self.db.subscribe("ss", self.ss_callback)
        self.initialised_event = threading.Event()

    def ss_callback(self, ss):
        """
        self.ss is a simplenamespace object with attributes
        A: list of A ss matrices in OCF form
        B: list of B ss matrices in OCF form
        """
        self.ss = ss
        self.new_update = True
        print("[Controller] State space matrices updated")
        if not self.initialised:
            self.initialised = True
            self.initialised_event.set()
    
    def dynamics(self, x, u):
        """
        Defines the dynamics of the inverted pendulum system.

        Args:
            x (ndarray): Current state vector.
            u (float): Applied input.

        Returns:
            ndarray: Derivatives of the state vector.
        """
        x_dot = np.mean(np.dot(self.ss.A, x) + np.dot(self.ss.B, u), axis=0)
        return x_dot.reshape(-1)
    
    def clip_angle(self, angle):
        angle = angle % (2 * np.pi)
        if angle > np.pi:
            angle -= 2 * np.pi
        return angle


class Controller:
    """
    Controller class handles the control of the cart-pendulum system.
    """
    def __init__(self, plant: Plant, type: str = "lqr", n: tuple = (2, 1)):
        """
        Initializes the controller with the plant model.

        Args:
            plant (CartPendulumPlant): Plant model instance.
        """
        self.plant = plant
        self.Q = 0.4 * np.eye(n[0])
        self.R = np.eye(n[1])
        if self.plant.initialised:
            self.K = self.design_lqr()
        
    def design_lqr(self):
        """
        Designs the LQR controller for the linearized system.

        Returns:
            tuple: State-space model and LQR gain matrix.
        """
        # LQR design

        K = get_optimal_controller(self.plant.ss.A, self.plant.ss.B, self.Q, self.R)
        # K = np.dot(self.plant.ss.C, K.T).reshape(K.shape[0], -1)
        # G, H, F, L 
        

        # @c-hars TODO: update the F,L matrices to reflect the integrator tracking
        F = L  = K 
        armax = {'F': F, 'L': L}
        armax = SimpleNamespace(**armax)
        self.plant.db.write_controller(controller=armax)
        return K
    
    def get_u(self, y, r: np.ndarray):
        """
        Computes the LQR control input.

        Args:
            y (ndarray): Current state vector.

        Returns:
            ndarray: Control input vector.
        """
        if self.plant.new_update:
            self.K = self.design_lqr()
            self.plant.new_update = False

        # @c-hars TODO: update the input version to reflect the integrator tracking
        u = self.K @ (r - y)
        # u = np.clip(u, -100, 100)
        return u 


