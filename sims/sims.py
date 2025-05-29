from types import SimpleNamespace
from scipy import signal
import math
import sys
import argparse
from dB.sim_db import Database, SPSType
from sims.pendulum import Pendulum, CartPendulum
from sims.water_tank import WaterTank
from sims.carla import CarlaSps
from sims.armax import ARMAX
from sims.car_sim import CarSim
from optimal_controller.controller import Plant, Controller, LTuner
import numpy as np
from typing import Union, List
import logging
import time
import ast
import sympy as sp
import cv2
import keyboard

SIM_CLASS_MAP = {
    "Pendulum": (Pendulum, np.array([np.pi/4, 0.0]), (2,1)),
    "Cart-Pendulum": (CartPendulum, np.array([0, 0, np.pi/4, 0.0]), (4,1)),
    "Carla": (CarlaSps, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), (11,1)), 
    "armax": (ARMAX, np.array([0]), (1,1)),
    "water_tank": (WaterTank, np.array([0]), (1,1)),
    "car_sim": (CarSim, np.array([1]), (1,1))
}


class Sim:
    """
    CartPendulumSimulation class handles the simulation and visualization of the cart-pendulum system.
    """
    def __init__(self,
                 T: float,
                 T_updates: float,
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
                 r_c: np.ndarray,
                 A: np.ndarray=None,
                 B: np.ndarray=None,
                 C: np.ndarray=None,
                 L = None,
                 Q: np.ndarray = None,
                 R: np.ndarray = None,
                 db: Database = None,
                 buffer_delay: int = 20,
                 logger: logging.Logger = None):
        self.logger = logger if logger else logging.getLogger(__name__)
        self.T = T
        self.T_updates = T_updates
        self.dt = dt
        self.apply_disturbance = apply_disturbance
        self.disturbance = disturbance
        self.L = L
        n_r,n_o = self.L.shape
        self.sim_type = sim
        sim, state, self.n = SIM_CLASS_MAP[sim]
        n_o,n_i = self.n
        self.n = (n_o,n_i,n_r)
        self.initial_state = state
        self.buffer_delay = buffer_delay

        self.is_paused = False

        match self.sim_type:
            case "armax":
                self.initial_state = np.zeros(max(len(A)-1, len(B)-1))
                self.sim: ARMAX = sim(A=A, B=B, C=C, dt=self.dt, 
                                    initial_state=self.initial_state, 
                                    plot_system=plot_system, 
                                    history_limit=history_limit)
            case "Pendulum" | "Cart-Pendulum":
                self.sim: Union[Pendulum, CartPendulum] = sim(dt=self.dt, 
                                                                    initial_state=state, 
                                                                    plot_system=plot_system, 
                                                                    history_limit=history_limit)
            case "water_tank":
                self.sim: WaterTank = sim(plot_system=True, visual=True)
            case "car_sim":
                road_length = int(self.T/self.dt) if self.T>0 else np.inf
                self.sim: CarSim = sim(dt=dt, road_length=road_length)
            case _:
                raise NotImplementedError
        
        self.db = db if db else Database(dB)
        db_name = self.db.db_name
        self.controller_plant = Plant(dt=dt, db=db_name)
        self.controller = Controller(plant=self.controller_plant, type=controller, n=self.n, L=L, Q=Q, R=R)
        self.reference = reference
        self.r_a=r_a
        self.r_f=r_f
        self.r_c =r_c
        self.i=0

        keyboard.add_hotkey("p", self.toggle_pause  )

        # tuner = LTuner(self.controller)
        # tuner.start()  # opens the slider window


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
    
    def square_wave(self, T, f, fs, c=0,A=1):
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
        y =  A*signal.square(2 * np.pi * f * t) + c
        y = np.tile(y, (self.n[1], 1)).T
        return t, y
    
    def impulse_wave(self, T, f, fs, c=0, A=40):
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
        y = np.zeros_like(t)+c                                # Initialize signal with zeros
        
        impulse_period = int(fs / f)                        # Samples between impulses
        y[::impulse_period] = A+c                             # Set impulses at intervals
        y = np.tile(y, (self.n[1], 1)).T
        return t, y

    def safe_input(self,u):
        """ 
        Ensure input is clipped if neeeded per simulation
        """
        if self.sim.input_limit is not None:
            u = np.clip(u, -self.sim.input_limit[0], self.sim.input_limit[1])
        return u

    def sim_model_response(self, T, f, A, c, input_type="square_wave"):
        def _input():
            if self.sim_type == "car_sim":
                input_func = self.sim.open_loop_input_sequence
            else:
                input_funcs = {"impulse_wave": self.impulse_wave,
                            "square_wave": self.square_wave}
                if input_type not in input_funcs:
                    raise ValueError(f"Invalid input type: {input_type}. Must be one of {list(input_funcs.keys())}")
                input_func = input_funcs[input_type]
            inputs = input_func(T=T, f=f, fs=1/self.dt,A=A,c=c)
            return inputs
        t, u = _input()
        u = self.safe_input(u)
        
        self.sim.set_initial_state(self.initial_state)
        y = []
        y.append(self.sim.full_state_to_obs_y(state=self.sim.state))
        for _u, t in zip(u, t):
            y.append(self.sim.step(u=_u, t=t)[0])
        y.pop()
        y = np.array(y)
        return y.reshape(self.n[0],-1), u.reshape(self.n[1],-1), t
    
    def initialise_plant(self, T=5, f=1, A=1, c=0, input_type="square_wave"):
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
        y, u, t = self.sim_model_response(T=T, f=f, A=A, c=c, input_type=input_type)
        self.write_data_to_db(y=y, u=u, r=None, sps_type=SPSType.OPEN_LOOP)
        curr_t = t
        u = u.T # time by input
        n_u = u.shape[0]
        self.i = n_u
        self.logger.debug(f"[SIMS] Sending {n_u} inputs to DB")
        self.logger.info("[Init] Waiting for SS update...")
        while self.controller_plant.initialised_event.is_set() is False:
            # continue running the simulation with the same input recurrently
            if self.is_paused:
                continue

            self.sim.step(u=u[self.i%n_u], t=curr_t)
            self.i += 1
            curr_t += self.dt

            
        self.logger.info("[Init] Plant initialized")



    def get_r(self, i: int) -> Union[float, np.ndarray]:
        def _get_r(r_type, f, a, c=0):
            if self.sim_type == "car_sim":
                return self.sim.get_reference_x()
            if r_type == "constant":
                return a
            elif r_type == "sin":
                return a * np.sin(2 * np.pi * f * i * self.dt) +c
            elif r_type == "square":
                return a * np.sign(np.sin(2 * np.pi * f * i * self.dt)) +c

        n_r = len(self.reference)
        if n_r==1:
            f = self.r_f[0]
            a = self.r_a[0]
            c = self.r_c[0]
            return _get_r(self.reference[0], f, a,c)
        else:
            r = np.zeros(n_r)
            for i in range(n_r):
                f = self.r_f[i]
                a = self.r_a[i]
                c = self.r_c[i]
                r_type = self.reference[i]
                r[i] = _get_r(r_type, f, a, c)
            return r

    def toggle_pause(self):
        print("Pause pressed")
        self.is_paused = not self.is_paused

    def run(self):
        """
        Runs the cart-pendulum simulation.
        """
        state = self.sim.state
        y = self.sim.full_state_to_obs_y(state=state)
        buffer_len = int(self.T_updates/self.dt)
        history_y = []
        history_u = []
        history_r = []
        n_iters = int(self.T/self.dt) if self.T>0 else np.inf
        start_i = self.i
        buffer_delay = self.buffer_delay
        while True:
            if self.is_paused:
                continue

            # get the controller output
            r = self.get_r(self.i)
            u = self.controller.get_u(state, r=r)
            u = self.safe_input(u)
            if self.n[1] == 1 and type(u)==np.ndarray:
                u = u[0]
            # history management and write to DB
            # maintain moving window for history
            if len(history_y) >= buffer_len:
                history_y.pop(0)
                history_u.pop(0)
                history_r.pop(0)

            history_y.append(y.copy())
            history_u.append(u.copy())
            history_r.append(r.copy())
            if self.controller.heard_back and buffer_delay>0:
                buffer_delay -= 1
            if self.controller.heard_back and buffer_delay<=0 and len(history_y)==buffer_len:
                self.write_data_to_db(y=np.array(history_y).reshape(self.n[0],buffer_len), 
                                      u=np.array(history_u).reshape(self.n[1],buffer_len), 
                                      r=np.array(history_r).reshape(self.n[2],buffer_len), 
                                      sps_type=SPSType.CLOSED_LOOP)
                self.controller.heard_back = False
                buffer_delay = self.buffer_delay
                
                history_y = []
                history_u = []
                history_r = []

            # update the state
            y, done, state = self.sim.step(u=u, t=self.i* self.dt, full_state=True,r=r)
            self.i+=1
            if done or self.i-start_i>=n_iters:
                cv2.destroyAllWindows()
                self.logger.info("[Sim] Simulation finished")
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

def parse_array_sym(input_string):
    try:
        expr_list = sp.sympify(input_string)
        return np.array([float(val.evalf()) for val in expr_list])
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid input: {e}")

def parse_sim_args(raw_args: List[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process arguments for the simulation.")
    parser.add_argument("--T", type=float, default=10, help="Total simulation time")
    parser.add_argument("--T_updates", type=float, default=10, help="Time between each update. Please ensure there is enough time.")
    parser.add_argument("--dt", type=float, default=0.02, help="Simulation time step")
    parser.add_argument("--sim", type=str, required=True,
                        choices=["Pendulum", "Cart-Pendulum", "Carla", "armax", "water_tank", "car_sim"],
                        help="Simulation type")
    parser.add_argument("--plot_system", action="store_true", help="Enable plotting")
    parser.add_argument("--history_limit", type=float, default=2, help="Limit of history for plotting")
    parser.add_argument("--dB", type=str, default="sim_data.db", help="Database file")
    parser.add_argument("--disturbance", type=float, default=50.0, help="Disturbance magnitude")
    parser.add_argument("--apply_disturbance", action="store_true", help="Apply disturbance")
    parser.add_argument("--controller", type=str, default="lqr", help="Controller type")
    parser.add_argument("--L", type=parse_array, required=True, help="Reference gain")
    parser.add_argument("--Q", type=parse_array, help="Q matrix for LQR")
    parser.add_argument("--R", type=parse_array, help="R matrix for LQR")
    parser.add_argument("--reference", type=parse_ref_list, required=True, help="reference wave, if constant must give --r_a is used as the value")
    parser.add_argument("--r_a", type=parse_array_sym, required=True, help="Amplitude of the reference signal")
    parser.add_argument("--r_f", type=parse_array, required=True, help="Frequency of the reference signal")
    parser.add_argument("--r_c", type=parse_array, required=True, help="vertical shift of reference signal")
    parser.add_argument("--A",type=parse_array, help="ARMAX sim A polynomial")
    parser.add_argument("--B",type=parse_array, help="ARMAX sim B polynomial")
    parser.add_argument("--C",type=parse_array, help="ARMAX sim C polynomial")
    parser.add_argument("--N",type=int, help="SPS number of points")
    return parser.parse_args(raw_args)


def run_simulation(raw_args=None, db: Database = None, logger: logging.Logger = None):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logger if logger else logging.getLogger(__name__)

    # Parse and unpack arguments
    args = parse_sim_args(raw_args)

    if args.sim == "armax":
        if any(arr is None or arr.size == 0 for arr in (args.A, args.B, args.C)):
            logger.warning("Please provide non-empty A, B, C polynomials if ARMAX sim")
            return


    simulation = Sim(
        T=args.T,
        T_updates=args.T_updates,
        dt=args.dt,
        sim=args.sim,
        plot_system=args.plot_system,
        history_limit=args.history_limit,
        dB=args.dB,
        disturbance=args.disturbance,
        apply_disturbance=args.apply_disturbance,
        controller=args.controller,
        L=args.L,
        Q=args.Q,
        R=args.R,
        db=db,
        logger=logger, 
        reference=args.reference,
        r_a = args.r_a,
        r_f = args.r_f,
        r_c = args.r_c,
        A=args.A,
        B=args.B,
        C=args.C
    )

    logger.info("[Init] Simulation instance created")
    T = (args.N+100)*args.dt
    if args.sim == "water_tank":
        simulation.initialise_plant(T=T, f=2.0, A=1.0, input_type="square_wave")
    else:
        simulation.initialise_plant(T=T, f=args.r_f[0], A=args.r_a[0], input_type="square_wave")
    simulation.run()

 

if __name__ == '__main__':
    # Create the simulation instance
    run_simulation()