from types import SimpleNamespace
from scipy import signal
import math
import sys
import argparse
from dB.sim_db import Database, SPSType
from sims.pendulum import Pendulum, CartPendulum
from sims.car import CarlaSps
from sims.armax import ARMAX
from optimal_controller.controller import Plant, Controller
import numpy as np
from typing import Union, List
import logging
import time

SIM_CLASS_MAP = {
    "Pendulum": (Pendulum, np.array([np.pi/4, 0.0]), (2,1)),
    "Cart-Pendulum": (CartPendulum, np.array([0, 0, np.pi/4, 0.0]), (4,1)),
    "Carla": (CarlaSps, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), (11,1)), 
    "2d_armax": (ARMAX, np.array([0]), (1,1))
}

class Sim:
    """
    CartPendulumSimulation class handles the simulation and visualization of the cart-pendulum system.
    """
    def __init__(self, raw_args: List[str]=None, db: Database=None, args: argparse.Namespace=None, logger: logging.Logger=None):
        """
        Initializes the simulation environment.

        Args:
            raw_args (list): Command line arguments to override default values.
            db (Database): Database object for storing simulation data.
            args (argparse.Namespace): Parsed command line arguments.
        """
        self.logger = logger if logger else logging.getLogger(__name__)
        self.parse_arguments(raw_args=raw_args, db=db, args=args)
    
    def _parse_args(self, raw_args: List[str]=None, args: argparse.Namespace=None) -> argparse.Namespace:
        if args is not None:
            return args
        parser = argparse.ArgumentParser(description="Process arguments inside a class.")
        parser.add_argument("--T", type=float, default=10, help="Total simulation time")
        parser.add_argument("--dt", type=float, default=0.02, help="Simulation time step")
        parser.add_argument("--sim", type=str, required=True, 
                            choices=["Pendulum", "Cart-Pendulum", "Carla"], 
                            help=f"Your simulation (must be one of {list(SIM_CLASS_MAP.keys())})")
        parser.add_argument("--plot_system", action="store_true", help="Enable plotting")
        parser.add_argument("--history_limit", type=float, default=2, help="Limit of history for plotting")
        parser.add_argument("--dB", type=str, default="sim_data.db", help="Database file")
        parser.add_argument("--disturbance", type=float, default=50.0, help="Magnitude of disturbance force")
        parser.add_argument("--apply_disturbance", action="store_true", help="apply disturbance")
        parser.add_argument("--controller", type=str, default="lqr", help="Controller type")
        args = parser.parse_args(raw_args)
        return args


    def parse_arguments(self, raw_args=None, db:Database=None, args: argparse.Namespace=None):
        args = self._parse_args(raw_args=raw_args, args=args)
        self.T: float = args.T
        self.dt: float = args.dt
        sim, state, self.n = SIM_CLASS_MAP[args.sim]
        self.initial_state: np.ndarray = state
        self.apply_disturbance = args.apply_disturbance
        self.disturbance = args.disturbance

        self.sim: Union[Pendulum, CartPendulum, CarlaSps, Armax] = sim(dt=self.dt, 
                                                                initial_state=state, 
                                                                plot_system=args.plot_system, 
                                                                history_limit=args.history_limit)
        
        if db is not None:
            self.db = db
            db_name = db.db_name
        else:
            self.db = Database(args.dB)
            db_name = args.dB
        self.controller_plant = Plant(dt=self.dt, db=db_name)
        self.controller = Controller(plant=self.controller_plant, type=args.controller, n=self.n)
    
    def write_data_to_db(self, y: np.ndarray, u: np.ndarray, r: np.ndarray, sps_type: SPSType):
        data = {
            "y": y,
            "u": u,
            "r": r,
            "sps_type": sps_type
        }
        data = SimpleNamespace(**data)
        if self.db:
            self.db.write_data(data=data)
        else:
            self.logger.warning("No database provided.")
    
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
    
    def impulse_wave(self, T, f, fs, A=40):
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
        y[::impulse_period] = A                             # Set impulses at intervals
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
        states.append(np.dot(self.sim.C, self.sim.state))
        for _u, t in zip(u, t):
            states.append(self.sim.step(u=_u, t=t)[0])
        states.pop()
        states = np.array(states)
        return states, u, t
    
    def initialise_plant(self, T=5, f=1, input_type="square_wave"):
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
            self.logger.info("[Init] Already initialized, skipping whacking")
            return

        # Send input to DB
        y, u, t = self.sim_model_response(T=T, f=f, input_type=input_type)
        self.write_data_to_db(y=y, u=u, r=None, sps_type=SPSType.OPEN_LOOP)
        curr_t = t
        i=0
        n_u = u.shape[0]
        self.logger.debug(f"[SIMS] Sending {n_u} inputs to DB")
        self.logger.info("[Init] Waiting for SS update...")
        while self.controller_plant.initialised_event.is_set() is False:
            # continue running the simulation with the same input recurrently
            self.sim.step(u=u[i%n_u], t=curr_t)
            i += 1
            curr_t += self.dt

            
        self.logger.info("[Init] Plant initialized")


    def run(self):
        """
        Runs the cart-pendulum simulation.
        """
        state = self.sim.state
        y = np.dot(self.sim.C, state)
        r = np.pi/2
        buffer_len = 1000
        history_y = []
        history_u = []
        for _ in range(int(self.T / self.dt)):
            # get the controller output
            u = self.controller.get_u(state, r=r)

            # history management and write to DB
            if len(history_y) < buffer_len:
                history_y.append(y.copy())
                history_u.append(u.copy())
            if len(history_y) == buffer_len:
                self.write_data_to_db(y=np.array(history_y).reshape(self.n[0],buffer_len), 
                                      u=np.array(history_u).reshape(self.n[1],buffer_len), 
                                      r=np.array([r]*buffer_len), 
                                      sps_type=SPSType.CLOSED_LOOP)
                history_y = []
                history_u = []

            # update the state
            y, done, state = self.sim.step(u=u, t=_ * self.dt, full_state=True)
            if done:
                break

        self.sim.show_final_plot()

def run_simulation(raw_args=None, db:Database=None, args: argparse.Namespace=None, logger: logging.Logger=None):
    """
    Run the simulation with optional command line arguments.

    Args:
        raw_args (list): Command line arguments to override default values.
    """
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logger if logger else logging.getLogger(__name__)
    # Create the simulation instance
    simulation = Sim(raw_args=raw_args, db=db, args=args, logger=logger)
    logger.info("[Init] Simulation instance created")

    # initialise the plant
    simulation.initialise_plant(T=2, f=5, input_type="impulse_wave")
    # Run the simulation
    simulation.run()
 

if __name__ == '__main__':
    # Create the simulation instance
    run_simulation()