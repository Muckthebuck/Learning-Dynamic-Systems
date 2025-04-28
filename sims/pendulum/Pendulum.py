import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union
from sims.pendulum.PendulumSimBase import PendulumSimBase


class Pendulum(PendulumSimBase):
    """
    Simulates a simple pendulum using numerical integration and visualizes the motion.
    
    Attributes:
        dt (float): Time step for simulation.
        g (float): Acceleration due to gravity.
        L (float): Length of the pendulum rod.
        m (float): Mass of the pendulum bob.
        d (float): Damping coefficient.
        inertia (float): Moment of inertia of the pendulum.
        state (np.ndarray): Current state [theta, theta_dot].
        C (np.ndarray): Output matrix for measuring the system state.
    """
    def __init__(self, dt: float = 0.02, initial_state: Optional[np.ndarray] = np.array([np.pi/4, 0.0]), 
                 plot_system: bool = False, history_limit: int = 200) -> None:
        """
        Initializes the pendulum simulation.

        Args:
            dt (float): Simulation time step.
            initial_state (Optional[np.ndarray]): Initial state [theta, theta_dot].
        """

        super().__init__(initial_state=initial_state, C=np.array([[1, 0]]), 
                         labels=["Pendulum Angle", "Pendulum Angular Velocity", "Input"], 
                         sim_title="Pendulum Simulation", dt=dt, 
                         plot_system=plot_system, history_limit=history_limit)

        self.g: float = 9.81
        self.L: float = 1.1
        self.m: float = 1.0
        self.d: float = 0.5
        self.inertia: float = self.m * (self.L ** 2)


    def dynamics(self, y: np.ndarray, u: Union[float, np.ndarray]) -> np.ndarray:
        """
        Computes the system dynamics.

        Args:
            y (np.ndarray): Current state [theta, theta_dot].
            u (float): Control input (torque).

        Returns:
            np.ndarray: State derivative [theta_dot, theta_ddot].
        """
        # if u is np array, convert to float
        if isinstance(u, np.ndarray):
            u = u[0]
        theta, theta_dot = y
        theta_ddot = (u - self.d * theta_dot - self.m * self.g * self.L * np.sin(theta)) / self.inertia # theta_ddot = u - d*theta_dot - mgL*sin(theta)/J
        
        return np.array([theta_dot, theta_ddot])

    def draw(self, state_vec: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        """
        Draws the pendulum state.

        Args:
            state_vec (np.ndarray): Current state.
            t (Optional[float]): Time stamp.

        Returns:
            np.ndarray: Image array for OpenCV display.
        """
        BOB_ANG = state_vec[0] * 180.0 / np.pi
        IM = np.zeros((512, 512, 3), dtype='uint8')
        center_x, center_y = IM.shape[1] // 2, 100
        pendulum_bob_x = int(center_x + self.L * 100 * np.cos(state_vec[0]))
        pendulum_bob_y = int(center_y - self.L * 100 * np.sin(state_vec[0]))

        cv2.line(IM, (center_x, center_y), (pendulum_bob_x, pendulum_bob_y), (255, 255, 255), 3)
        cv2.circle(IM, (pendulum_bob_x, pendulum_bob_y), 10, (255, 255, 255), -1)
        cv2.putText(IM, f"theta={BOB_ANG:.2f} deg", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 250), 1)
        if t is not None:
            cv2.putText(IM, f"t={t:.2f} sec", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 250), 1)
        return IM


if __name__ == '__main__':
    T = 10
    dt = 0.02
    state = np.array([np.pi/4, 0.0])
    sim = Pendulum(dt=dt, initial_state=state, plot_system=True)
    
    for _ in range(int(T / dt)):
        __, done = sim.step(u=0,t=_*dt)
        if done:
            break

    sim.show_final_plot()