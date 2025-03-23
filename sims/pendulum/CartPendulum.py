import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from sims.pendulum.PendulumSimBase import PendulumSimBase

class CartPendulum(PendulumSimBase):
    """
    Simulates a simple pendulum using numerical integration and visualizes the motion.
    
    Attributes:
        dt (float): Time step for simulation.
        g (float): Acceleration due to gravity.
        L (float): Length of the pendulum rod.
        m (float): Mass of the pendulum bob.
        M (float): Mass of the cart. 
        d1 (float): damping coefficient
        d2 (float): pendulum damping coefficient
        state (np.ndarray): Current state [position, vel, theta, theta_dot].
        C (np.ndarray): Output matrix for measuring the system state.
    """
    def __init__(self, dt: float = 0.02, initial_state: Optional[np.ndarray] = np.array([0, 0, np.pi/4, 0.0]), 
                 plot_system: bool = False, history_limit: int = 200) -> None:
        """
        Initializes the pendulum simulation.

        Args:
            dt (float): Simulation time step.
            initial_state (Optional[np.ndarray]): Initial state [theta, theta_dot].
        """
        super().__init__(initial_state=initial_state, C=np.array([[1, 0, 0, 0], [0, 0, 1, 0]]),
                         labels=["Cart Position", "Cart Velocity", "Pendulum Angle", "Pendulum Angular Velocity"],
                         sim_title="Cart Pendulum Simulation", dt=dt,
                         plot_system=plot_system, history_limit=history_limit)
        self.g: float = 9.81
        self.L: float = 1.1
        self.m: float = 1.0
        self.M: float = 5.0
        self.d1: float = 0.5
        self.d2: float = 0.5
        self.inertia: float = self.m * (self.L ** 2)


    def dynamics(self, y: np.ndarray, u: float) -> np.ndarray:
        """
        Computes the system dynamics.

        Args:
            y (np.ndarray): Current state [theta, theta_dot].
            u (float): Control input (torque).

        Returns:
            np.ndarray: State derivative [theta_dot, theta_ddot].
        """
        theta, theta_dot = y[2], y[3]
        x_ddot = (u - self.m * self.L * theta_dot**2 * np.cos(theta) + self.m * self.g * np.sin(theta) * np.cos(theta)) / (self.M + self.m - self.m * np.sin(theta)**2)
        theta_ddot = -self.g/self.L * np.cos(theta) - np.sin(theta)/self.L * x_ddot
        return np.array([y[1], x_ddot - self.d1 * y[1], y[3], theta_ddot - self.d2 * y[3]])

    def draw(self, state_vec: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        """
        Draws the pendulum state.

        Args:
            state_vec (np.ndarray): Current state.
                x0 : position of the cart
                x1 : veclocity of the cart
                x2 : angle of pendulum. In ref frame with x as forward of the cart and y as up. Angile with respect to ground plane
                x3 : angular velocity of the pendulum
            t (Optional[float]): Time stamp.

        Returns:
            np.ndarray: Image array for OpenCV display.
        """
        CART_POS = state_vec[0]
        BOB_ANG  = state_vec[2]*180. / np.pi # degrees for displaying only

        IM = np.zeros( (512, 512,3), dtype='uint8' )

        # Ground line
        cv2.line(IM, (0, 450), (IM.shape[1], 450), (19,69,139), 4 )


        # Mark ground line
        XSTART = -5.
        XEND = 5.
        for xd in np.linspace( XSTART, XEND, 11 ):
            x = int(   (xd - XSTART) / (XEND - XSTART) * IM.shape[0]   )

            cv2.circle( IM, (x, 450), 5, (0,255,0), -1 )

            cv2.putText(IM, str(xd), (x-15,450+15), 
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (200,200,250), 1)


        # Draw Wheels of the cart
        wheel_1_pos = int(   (CART_POS - 1.2 - XSTART) / (XEND - XSTART) * IM.shape[0]   )
        wheel_2_pos = int(   (CART_POS + 1.2 - XSTART) / (XEND - XSTART) * IM.shape[0]   )

        cv2.circle( IM, (wheel_1_pos, 415), 25, (255,255,255), 6 )
        cv2.circle( IM, (wheel_2_pos, 415), 25, (255,255,255), 6 )
        cv2.circle( IM, (wheel_1_pos, 415), 2, (255,255,255), -1 )
        cv2.circle( IM, (wheel_2_pos, 415), 2, (255,255,255), -1 )

        # Cart base
        cart_base_start = int(   (CART_POS - 2.5 - XSTART) / (XEND - XSTART) * IM.shape[0]   )
        cart_base_end   = int(   (CART_POS + 2.5 - XSTART) / (XEND - XSTART) * IM.shape[0]   )

        cv2.line( IM, (cart_base_start, 380), (cart_base_end, 380), (255,255,255), 6 )

        # Pendulum hinge
        pendulum_hinge_x = int(   (CART_POS - XSTART) / (XEND - XSTART) * IM.shape[0]   )
        pendulum_hinge_y = 380
        cv2.circle( IM, (pendulum_hinge_x, pendulum_hinge_y), 10, (255,255,255), -1 )


        # Pendulum
        pendulum_bob_x = int( self.L * 100 * np.cos(state_vec[2]) )
        pendulum_bob_y = int( self.L * 100 * np.sin(state_vec[2]) )
        cv2.circle( IM, (pendulum_hinge_x+pendulum_bob_x, pendulum_hinge_y-pendulum_bob_y), 
                   10, (255,255,255), -1 )
        cv2.line( IM, (pendulum_hinge_x, pendulum_hinge_y), 
                 (pendulum_hinge_x+pendulum_bob_x, pendulum_hinge_y-pendulum_bob_y), 
                 (255,255,255), 3 )

        # Mark the current angle
        angle_display = BOB_ANG % 360
        if( angle_display > 180 ):
            angle_display = -360+angle_display
        cv2.putText(IM, "theta="+str( np.round(angle_display,4) )+" deg", 
                    (pendulum_hinge_x-15, pendulum_hinge_y-15), 
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (200,200,250), 1)


        # Display on top
        if t is not None:
            cv2.putText(IM, "t="+str(np.round(t,4))+"sec", (15, 15), 
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (200,200,250), 1)
            cv2.putText(IM, "ANG="+str(np.round(BOB_ANG,4))+" degrees", (15, 35), 
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (200,200,250), 1)
            cv2.putText(IM, "POS="+str(np.round(CART_POS,4))+" m", (15, 55), 
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (200,200,250), 1)

        return IM

if __name__ == '__main__':
    T = 10
    dt = 0.02
    state = np.array([0, 0, np.pi/4, 0.0])
    sim = CartPendulum(dt=dt, initial_state=state, plot_system=True)
    
    for _ in range(int(T / dt)):
        __, done = sim.step(u=0,t=_*dt)
        if done:
            break
    sim.show_final_plot()