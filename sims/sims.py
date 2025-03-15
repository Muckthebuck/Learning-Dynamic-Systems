from sims.pendulum_controlled import *
from sims.sim_db import Database
from scipy import signal
import math
import sys
if __name__ == '__main__':
    db = Database()
    plant = CartPendulumPlant(dt=0.02, db=db)

    desired = np.array([0, 0, np.pi/2, 0])
    controller = Controller(plant, desired=desired)

    observer = Observer(plant)
    sim = CartPendulumSimulation(plant, controller, observer, T=20, disturbance=50)

    # whack the system and get some sample data to initialise the plant model (open loop sps)
    try:
        sim.initialise_plant(T=5, f=1, input_type="square_wave", timeout=5, max_retries=3)
    except TimeoutError as e:
        print(e)
        sys.exit(1)
    
    # 
    # labels = ['Cart Position', 'Cart Velocity', 'Pendulum Angle', 'Angular Velocity']
    # plt.figure(figsize=(10, 6))
    # for i in range(1, 5):
    #     plt.subplot(5, 1, i)
    #     plt.plot(t, y[:, i-1], label=labels[i-1])
    #     plt.legend()
    #     plt.grid()
    #     plt.suptitle('Cart-Pendulum State and input Over Time')
    
  
    # plt.subplot(5, 1, 5)
    # plt.plot(t, u, label='Input Force')
    # plt.legend()
    # plt.grid()
    

    plt.show()

