import numpy as np
from dB.sim_db import Database
from optimal_controller.optimal_controls import get_optimal_controller
from types import SimpleNamespace
import threading
import logging
import cv2
class Plant:
    """
    Plant class keeps a set of estimated system dynamics.
    The class subscribes to the database for updates on the system dynamics.
    The class is used to design and implement controllers and observers.
    """
    def __init__(self, dt=0.02, db: str = "sim.db"):
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
        self.db = Database(db_name=db)
          
        
        self.db.subscribe("ss", self.ss_callback)
        self.initialised_event = threading.Event()
        self.initialised_event.clear()

    def ss_callback(self, ss):
        """
        self.ss is a simplenamespace object with attributes
        A: list of A ss matrices in OCF form
        B: list of B ss matrices in OCF form
        """
        logging.info("[Controller] State space matrices updated")
        self.ss = self.db._deserialize(ss)
        self.new_update = True

        if not self.initialised:
            self.initialised = True
            self.initialised_event.set()
    
    def dynamics(self, x, u):
        """
        Defines the dynamics of the average system.

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
    def __init__(self, plant: Plant, L: np.ndarray, type: str = "lqr", n: tuple = (2, 1), 
                 Q: np.ndarray= None, R: np.ndarray = None):
        """
        Initializes the controller with the plant model.

        Args:
            plant (CartPendulumPlant): Plant model instance.
        """
        self.plant = plant
        if Q is None:
            self.Q = 0.4 * np.eye(n[0])
        else:
            self.Q = Q
        if R is None:
            self.R = 1.0 * np.eye(n[1])
        else:
            self.R = R
        self.L = L
        if self.plant.initialised:
            self.K = self.design_lqr()
        self.logger = logging.getLogger(__name__)
        self.heard_back = False
        
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
        self.logger.info(f"[Controller] New controller K: {K}")
        # @c-hars TODO: update the F,L matrices to reflect the integrator tracking
        if K is not None:
            self.F = K
            armax = {'F': K, 'L': self.L}
            armax = SimpleNamespace(**armax)
            self.plant.db.write_controller(controller=armax)
        else:
            self.logger.warning(f"[Controller] Failed to get new controller")
    
    def get_u(self, y: np.ndarray, r: np.ndarray):
        """
        Computes the LQR control input.

        Args:
            y (ndarray): Current state vector.

        Returns:
            ndarray: Control input vector.
        """
        if self.plant.new_update:
            self.design_lqr()
            self.heard_back = True
            self.plant.new_update = False

        r = r.reshape(-1,1)
        y = y.reshape(-1,1)
        # @c-hars TODO: update the input version to reflect the integrator tracking
        u = np.dot(self.L,r) - np.dot(self.F,y)
        # u = np.clip(u, -100, 100)
        return u.flatten()


import cv2
import numpy as np
import threading
import ast

class LTuner:
    def __init__(self, controller):
        self.controller = controller
        self.input_str = ""
        self.running = False

    def start(self):
        if self.running:
            print("[LTuner] Already running.")
            return
        self.running = True
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()

    def _run(self):
        cv2.namedWindow("L Input")
        display = np.ones((100, 600), dtype=np.uint8) * 255

        print("[LTuner] Enter L as a list (e.g., [1.0, -0.5]) and press ENTER. ESC to quit.")

        while self.running:
            display[:] = 255
            cv2.putText(display, f"Enter L: {self.input_str}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0), 2)
            cv2.imshow("L Input", display)

            key = cv2.waitKey(100) & 0xFF

            if key == 27:  # ESC
                self.running = False
                break
            elif key in (13, 10):  # ENTER
                try:
                    # Safely parse the list-like string
                    values = ast.literal_eval(self.input_str)
                    arr = np.array(values, dtype=np.float64)

                    if arr.size != self.controller.L.size:
                        print(f"[LTuner] Expected {self.controller.L.size} values, got {arr.size}.")
                    else:
                        self.controller.L = arr.reshape(self.controller.L.shape)
                        print(f"[LTuner] Updated L:\n{self.controller.L}")
                except Exception as e:
                    print(f"[LTuner] Invalid input: {e}")
                self.input_str = ""
            elif key == 8:  # Backspace
                self.input_str = self.input_str[:-1]
            elif 32 <= key <= 126:  # Printable ASCII
                self.input_str += chr(key)

        cv2.destroyWindow("L Input")
        print("[LTuner] Closed tuning window.")
