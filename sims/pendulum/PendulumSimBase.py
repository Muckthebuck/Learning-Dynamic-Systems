import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union
from abc import ABC, abstractmethod


class PendulumSimBase(ABC):
    """
    Simulates a simple pendulum using numerical integration and visualizes the motion.
    
    Attributes:
        dt (float): Time step for simulation.
        state (np.ndarray): Current state.
        C (np.ndarray): Output matrix for measuring the system state.
        history (list[np.ndarray]): Stores past states for real-time plotting.
        fig (plt.Figure): Matplotlib figure for live plots.
        axs (np.ndarray): Axes for live plots.
        lines (List[plt.Line2D]): Line objects for states.
    """
    def __init__(self, initial_state: np.ndarray, C: np.ndarray, 
                 labels: Optional[list[str]] = None, sim_title: str = "Sim", dt: float = 0.02, 
                 plot_system: bool = False, history_limit: float = 2, noise_std: float = 0.002) -> None:
        """
        Initializes the pendulum simulation.

        Args:
            dt (float): Simulation time step.
            initial_state (Optional[np.ndarray]): Initial state [theta, theta_dot].
        """
        self.dt: float = dt
        self.history_limit=int(history_limit/dt)
        self.state: np.ndarray = initial_state
        self.C: np.ndarray = C
        self.plot_system: bool = plot_system
        self.labels = labels
        self.sim_title = sim_title
        self.noise_std = noise_std
        # initialise plotting
        if plot_system:
            self._initialise_plotting()
    def set_initial_state(self, state):
        self.state = state
    def _initialise_plotting(self):
        # History for live plotting
        self.history: list[np.ndarray] = []
        n = len(self.labels)
        self.fig, self.axs = plt.subplots(n, 1, figsize=(8, 6))
        self.lines = []
        for i in range(n):
            line, = self.axs[i].plot([], [], label=self.labels[i])
            self.lines.append(line)
        for ax in self.axs:
            ax.legend()
            ax.grid()
        plt.ion()
        plt.show()
    
    def reset(self) -> None:
        """
        Resets the simulation state and history.
        """
        self.state = np.zeros_like(self.state)
        self.history = []
        if self.plot_system:
            for line in self.lines:
                line.set_data([], [])
            for ax in self.axs:
                ax.relim()
                ax.autoscale_view()
            plt.draw()
            plt.pause(0.001)
    
    @abstractmethod
    def dynamics(self, y: np.ndarray, u: float) -> np.ndarray:
        """
        Computes the system dynamics.

        Args:
            y (np.ndarray): Current state.
            u (float): Control input.

        Returns:
            np.ndarray: State derivative.
        """
        raise NotImplementedError("Subclasses must implement 'compute_dynamics'.")

    def full_state_to_obs_y(self, state):
        return np.dot(self.C, state)
    
    def step(self, u: float, t, full_state: bool = False, r: Optional[float] = None) -> Union[Tuple[np.ndarray, bool], Tuple[np.ndarray, bool, np.ndarray]]:
        """
        Advances the simulation by one time step using RK4 integration.

        Args:
            u (float): Control input (torque).
            t (float): current time.
            full_state (bool): Whether to return full state as well

        Returns:
            np.ndarray: Measured output [theta].
            bool: Whether simulation is now stopped.
            np.ndarray: Also returns fully observed state if full_state arg is true
        """
        done = False
        self.state = self._rk4_step(self.state, u)
        if self.plot_system:
            self.history.append(np.append(self.state,u))
            if len(self.history) > self.history_limit:
                self.history = self.history[-self.history_limit:]  # Keep only the last N points
            self.update_plot(t)
            done = not self.render(t) # if render returns False, simulation is manually stopped
        if not full_state:
            return np.dot(self.C, self.state), done
        if full_state:
            return np.dot(self.C, self.state), done, self.state

    def _rk4_step(self, state: np.ndarray, u: float) -> np.ndarray:
        """
        Performs one step of RK4 integration.

        Args:
            state (np.ndarray): Current state.
            u (float): Control input.

        Returns:
            np.ndarray: Updated state.
        """
        k1 = self.dt * self.dynamics(state, u)
        k2 = self.dt * self.dynamics(state + k1 / 2, u)
        k3 = self.dt * self.dynamics(state + k2 / 2, u)
        k4 = self.dt * self.dynamics(state + k3, u)
        state += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return state

    def _clip_angle(self, angle: float) -> float:
        """
        Wraps angle within [-pi, pi].

        Args:
            angle (float): Input angle in radians.

        Returns:
            float: Wrapped angle in range [-pi, pi].
        """
        angle = angle % (2 * np.pi)
        return angle - 2 * np.pi if angle > np.pi else angle

    def render(self, t) -> bool:
        """
        Renders the pendulum visualization.

        Returns:
            bool: False if user quits, True otherwise.
        """
        rendered = self.draw(self.state, t)
        cv2.imshow(self.sim_title, rendered)
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            plt.ioff()
            return False
        return True

    @abstractmethod
    def draw(self, state_vec: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        """
        Draws the simulation image

        Args:
            state_vec (np.ndarray): Current state.
            t (Optional[float]): Time stamp.

        Returns:
            np.ndarray: Image array for OpenCV display.
        """
        raise NotImplementedError("Subclasses must implement 'draw'.")


    def update_plot(self, t) -> None:
        """
        Updates the live plot.
        """
        history_array = np.array(self.history)
        min_time = np.max([0, t-self.history_limit*self.dt])
        _t = np.linspace(min_time, t, len(history_array))
        for i in range(len(self.labels)):
            self.lines[i].set_data(_t, history_array[:, i])
        for ax in self.axs:
            if t < (self.history_limit*self.dt):
                ax.set_xlim(0, self.history_limit*self.dt)
            else:
                ax.set_xlim(min_time, t)
            ax.relim()
            ax.autoscale_view()
        plt.draw()
        plt.pause(0.001)
    
    def show_final_plot(self) -> None:
        """
        Displays the final plot after the simulation ends.
        """
        cv2.destroyAllWindows()
        if self.plot_system:
            plt.ioff()
            plt.show()

