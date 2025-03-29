from types import SimpleNamespace
from scipy import signal
import math
import sys
import argparse
from dB.sim_db import Database, SPSType
from sims.pendulum import Pendulum, CartPendulum
from sims.car import CarlaSps
from optimal_controller.controller import Plant, Controller
import numpy as np
from typing import Union

SIM_CLASS_MAP = {
    "Pendulum": (Pendulum, np.array([np.pi/4, 0.0]), (2,1)),
    "Cart-Pendulum": (CartPendulum, np.array([0, 0, np.pi/4, 0.0]), (4,1)),
    "Carla": (CarlaSps, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), (11,1))
}

class Sim:
    """
    CartPendulumSimulation class handles the simulation and visualization of the cart-pendulum system.
    """
    def __init__(self, disturbance=50.0):
        """
        Initializes the simulation environment.

        Args:
            plant (CartPendulumPlant): Plant model instance.
            T (float): Total simulation time.
            disturbance (float): Magnitude of the disturbance force.
        """
        self.parse_arguments()
        self.apply_disturbance = False

    def parse_arguments(self):
        parser = argparse.ArgumentParser(description="Process arguments inside a class.")
        parser.add_argument("--T", type=float, default=10, help="Total simulation time")
        parser.add_argument("--dt", type=float, default=0.02, help="Simulation time step")
        parser.add_argument("--sim", type=str, required=True, 
                            choices=["Pendulum", "Cart-Pendulum", "Carla"], 
                            help=f"Your simulation (must be one of {list(SIM_CLASS_MAP.keys())})")
        parser.add_argument("--plot_system", action=argparse.BooleanOptionalAction, default=True, help="Plot the system")
        parser.add_argument("--history_limit", type=int, default=200, help="Limit of history for plotting")
        parser.add_argument("--dB", type=str, default="sim.db", help="Database file")
        parser.add_argument("--disturbance", type=float, default=50.0, help="Magnitude of disturbance force")
        parser.add_argument("--controller", type=str, default="lqr", help="Controller type")
        args = parser.parse_args()
        self.T: float = args.T
        self.dt: float = args.dt
        sim, state, self.n = SIM_CLASS_MAP[args.sim]
        self.initial_state: np.ndarray = state
        self.sim: Union[Pendulum, CartPendulum, CarlaSps] = sim(dt=self.dt, 
                                                                initial_state=state, 
                                                                plot_system=args.plot_system, 
                                                                history_limit=args.history_limit)
        self.db = Database(args.dB)
        self.controller_plant = Plant(dt=self.dt, db=self.db)
        self.controller = Controller(plant=self.controller_plant, type=args.controller, n=self.n)
    
    def write_data_to_db(self, y: np.ndarray, u: np.ndarray, r: np.ndarray, sps_type: SPSType):
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
        y = np.tile(y, (self.n[1], 1)).T
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
        y = np.tile(y, (self.n[1], 1)).T
        return t, y

    def sim_model_response(self, T, f, input_type="square_wave"):
        def _input():
            input_funcs = {"impulse_wave": self.impulse_wave,
                           "square_wave": self.square_wave}
            if input_type not in input_funcs:
                raise ValueError(f"Invalid input type: {input_type}. Must be one of {list(input_funcs.keys())}")
            input_func = input_funcs[input_type]
            inputs = input_func(T=T, f=f, fs=1/self.dt)
            return inputs
        t, u = _input()
        self.sim.state = self.initial_state
        states = []
        states.append(self.sim.state.copy())
        for _u, t in zip(u, t):
            states.append(self.sim.step(u=_u, t=t)[0])
        states.pop()
        states = np.array(states)
        return states, u, t
    
    def initialise_plant(self, T=5, f=1, input_type="square_wave", timeout=10, max_retries=3):
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
        if self.controller_plant.initialised:
            print("[Init] Already initialized, skipping whacking")
            return

        # Send input to DB
        y, u, t = self.sim_model_response(T=T, f=f, input_type=input_type)
        self.write_data_to_db(y=y, u=u, r=None, sps_type=SPSType.OPEN_LOOP)
        self.controller_plant.initialised_event.wait(timeout=timeout)
        print("[Init] Plant initialized")


    def run(self):
        """
        Runs the cart-pendulum simulation.
        """
        state = self.sim.state
        for _ in range(int(self.T / self.dt)):
            u = self.controller.get_u(state)
            state, done = self.sim.step(u=u, t=_ * self.dt)
            self.db
            if done:
                break

        self.sim.show_final_plot()

 

if __name__ == '__main__':
    # Placeholder for controller (you may need to replace this with an actual controller instance)
    controller = None  

    # Create the simulation instance
    simulation = Sim(controller=controller, disturbance=50.0)

    # initialise the plant
    simulation.initialise_plant(T=5, f=1, input_type="square_wave", timeout=10, max_retries=3)
    # Run the simulation
    simulation.run()
