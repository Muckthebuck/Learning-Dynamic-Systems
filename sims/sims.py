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
import ast

SIM_CLASS_MAP = {
    "Pendulum": (Pendulum, np.array([np.pi/4, 0.0]), (2,1)),
    "Cart-Pendulum": (CartPendulum, np.array([0, 0, np.pi/4, 0.0]), (4,1)),
    "Carla": (CarlaSps, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), (11,1)), 
    "armax": (ARMAX, np.array([0]), (1,1))
}


class Sim:
    """
    CartPendulumSimulation class handles the simulation and visualization of the cart-pendulum system.
    """
    def __init__(self,
                 T: float,
                 dt: float,
                 sim: str,
                 plot_system: bool,
                 history_limit: float,
                 dB: str,
                 disturbance: float,
                 apply_disturbance: bool,
                 controller: str,
                 reference: List[str],
                 r_a: np.ndarray,
                 r_f: np.ndarray,
                 A: np.ndarray=None,
                 B: np.ndarray=None,
                 C: np.ndarray=None,
                 L = None,
                 db: Database = None,
                 logger: logging.Logger = None):
        self.logger = logger if logger else logging.getLogger(__name__)
        self.T = T
        self.dt = dt
        self.apply_disturbance = apply_disturbance
        self.disturbance = disturbance
        self.L = L
        sim_type = sim
        sim, state, self.n = SIM_CLASS_MAP[sim]
        self.initial_state = state
        if sim_type == "ARMAX":
            self.sim: ARMAX = sim(A=A, B=B, C=C, dt=self.dt, 
                                initial_state=state, 
                                plot_system=plot_system, 
                                history_limit=history_limit)
        else: 
            self.sim: Union[Pendulum, CartPendulum, CarlaSps] = sim(dt=self.dt, 
                                                                    initial_state=state, 
                                                                    plot_system=plot_system, 
                                                                    history_limit=history_limit)
        self.db = db if db else Database(dB)
        db_name = self.db.db_name
        self.controller_plant = Plant(dt=dt, db=db_name)
        self.controller = Controller(plant=self.controller_plant, type=controller, n=self.n, L=L)
        self.reference = reference
        self.r_a=r_a
        self.r_f=r_f
    

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



    def get_r(self, i: int) -> Union[float, np.ndarray]:
        def _get_r(r_type, f, a):
            if r_type == "constant":
                return a
            elif r_type == "sin":
                return a * np.sin(2 * np.pi * f * i * self.dt)
            elif r_type == "square":
                return a * np.sign(np.sin(2 * np.pi * f * i * self.dt)) 

        n_r = len(self.reference)
        if n_r==1:
            f = self.r_f[0]
            a = self.r_a[0]
            return _get_r(self.reference[0], f, a)
        else:
            r = np.zeros(n_r)
            for i in range(n_r):
                f = self.r_f[i]
                a = self.r_a[i]
                r_type = self.reference[i]
                r[i] = _get_r(r_type, f, a)
            return r



    def run(self):
        """
        Runs the cart-pendulum simulation.
        """
        state = self.sim.state
        y = np.dot(self.sim.C, state)

        buffer_len = 1000
        history_y = []
        history_u = []
        i=0
        n_iters = int(self.T/self.dt) if self.T>0 else np.inf
        while True:
            # get the controller output
            r = self.get_r(i)
            u = self.controller.get_u(state, r=r)
            if self.n[1] == 1 and type(u)==np.ndarray:
                u = u[0]
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
            y, done, state = self.sim.step(u=u, t=i* self.dt, full_state=True)
            i+=1
            if done or i>=n_iters:
                break

        self.sim.show_final_plot()

# Function to parse the input string into a NumPy array
def parse_array(input_string):
    try:
        # Safely evaluate the string into a list using ast.literal_eval
        parsed_list = ast.literal_eval(input_string)
        # Convert the list into a NumPy array
        return np.array(parsed_list)
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError("Input must be a valid list-like string (e.g. [1, 2, 3])")
def parse_ref_list(input_string):
    try:
        choices = ["square", "sin", "constant"]
        # Safely evaluate the string into a list using ast.literal_eval
        parsed_list = ast.literal_eval(input_string)
        
        if not isinstance(parsed_list, list):
            raise ValueError(f"Invalid reference type must be a list of allowed values: {choices}")

        for item in parsed_list:
            if item not in choices:
                raise ValueError(f"Invalid reference type: {item}. Allowed values are: {choices}")
        
        return parsed_list

    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError("Input must be a valid list-like string with allowed values (e.g. ['square', 'constant'])")


def parse_sim_args(raw_args: List[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process arguments for the simulation.")
    parser.add_argument("--T", type=float, default=10, help="Total simulation time")
    parser.add_argument("--dt", type=float, default=0.02, help="Simulation time step")
    parser.add_argument("--sim", type=str, required=True,
                        choices=["Pendulum", "Cart-Pendulum", "Carla", "armax"],
                        help="Simulation type")
    parser.add_argument("--plot_system", action="store_true", help="Enable plotting")
    parser.add_argument("--history_limit", type=float, default=2, help="Limit of history for plotting")
    parser.add_argument("--dB", type=str, default="sim_data.db", help="Database file")
    parser.add_argument("--disturbance", type=float, default=50.0, help="Disturbance magnitude")
    parser.add_argument("--apply_disturbance", action="store_true", help="Apply disturbance")
    parser.add_argument("--controller", type=str, default="lqr", help="Controller type")
    parser.add_argument("--L", type=parse_array, required=True, help="Reference gain")
    parser.add_argument("--reference", type=parse_ref_list, required=True, help="reference wave, if constant must give --r_a is used as the value")
    parser.add_argument("--r_a", type=parse_array, required=True, help="Amplitude of the reference signal")
    parser.add_argument("--r_f", type=parse_array, required=True, help="Frequency of the reference signal")
    parser.add_argument("--A",type=parse_array, help="ARMAX sim A polynomial")
    parser.add_argument("--B",type=parse_array, help="ARMAX sim B polynomial")
    parser.add_argument("--C",type=parse_array, help="ARMAX sim C polynomial")
    return parser.parse_args(raw_args)


def run_simulation(raw_args=None, db: Database = None, logger: logging.Logger = None):
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logger if logger else logging.getLogger(__name__)

    # Parse and unpack arguments
    args = parse_sim_args(raw_args)
    if args.sim  == "armax" and not all([args.A, args.B, args.C]):
        logger.warning("Please provide A,B,C polynomials if ARMAX sim")
        return
        

    simulation = Sim(
        T=args.T,
        dt=args.dt,
        sim=args.sim,
        plot_system=args.plot_system,
        history_limit=args.history_limit,
        dB=args.dB,
        disturbance=args.disturbance,
        apply_disturbance=args.apply_disturbance,
        controller=args.controller,
        L=args.L,
        db=db,
        logger=logger, 
        reference=args.reference,
        r_a = args.r_a,
        r_f = args.r_f,
        A=args.A,
        B=args.B,
        C=args.C
    )

    logger.info("[Init] Simulation instance created")

    simulation.initialise_plant(T=2, f=5, input_type="square_wave")
    simulation.run()

 

if __name__ == '__main__':
    # Create the simulation instance
    run_simulation()