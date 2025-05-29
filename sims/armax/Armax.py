import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union
from optimal_controller.state_space import tf_to_ocf

class ARMAX:
    def __init__(self, A, B, C, dt=0.01, history_limit=10, noise_std=0.02,
                 plot_system=False, initial_state:np.ndarray=np.array([0])):
        self.A = np.array(A)
        self.B = np.array(B)
        self.C = np.array(C)
        self.dt = dt
        self.noise_std = noise_std
        self.plot_system = plot_system

        self.buffer_length = max(len(self.A) - 1, len(self.B), len(self.C))
        self.full_state_length = max(len(self.A), len(self.B))-1
        self.y_buffer = np.zeros(self.buffer_length)
        self.u_buffer = np.zeros(self.buffer_length)
        self.n_buffer = np.zeros(self.buffer_length)
        self.idx = 0
        self.input_limit = None
        self.history = []
        self.history_limit = int(history_limit / dt)
        self.current_time = 0.0
        self.done = False
        self.state = np.zeros(self.full_state_length).reshape(1,-1).T
        if plot_system:
            self._init_plot()
        self.A_mat_ocf, self.B_mat_ocf, self.C_mat_ocf = tf_to_ocf(self.B, self.A)

    def _init_plot(self):
        self.fig, self.axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        
        self.line_y0, = self.axs[0].plot([], [], label="Output Y")
        self.line_r, = self.axs[0].plot([], [], label="Reference R", linestyle='--')
        self.axs[0].set_ylabel("Output / Reference")
        self.axs[0].legend(loc="upper right")  # Legend positioned at top right
        self.axs[0].grid()

        self.line_u, = self.axs[1].plot([], [], label="Input U")
        self.axs[1].set_ylabel("Input (U)")
        self.axs[1].legend(loc="upper right")  # Legend positioned at top right
        self.axs[1].grid()

        self.line_n, = self.axs[2].plot([], [], label="Noise N")
        self.axs[2].set_ylabel("Noise (N)")
        self.axs[2].set_xlabel("Time [s]")
        self.axs[2].legend(loc="upper right")  # Legend positioned at top right
        self.axs[2].grid()

        self.fig.canvas.mpl_connect('close_event', self._on_close)
        plt.ion()
        plt.show()

    def set_initial_state(self, state):
        return 

    def _on_close(self, event):
        self.done = True

    def step(self, u: Union[float, np.ndarray], 
             r: Optional[float] = None, 
             t: Optional[float] = None, 
             full_state : Optional[bool] = False) -> Union[Tuple[np.ndarray, bool], Tuple[np.ndarray, bool, np.ndarray]]:
        if self.done:
            if full_state:
                return np.array([0.0]), True, np.array([0.0])
            else:
                return np.array([0.0]), True
        noise = np.random.normal(0, self.noise_std)

        if type(u) == np.ndarray:
            u = u[0]

        def get_lags(buffer, lag_indices):
            return np.array([
                buffer[(self.idx - i) % self.buffer_length] if i > 0 else 0.0
                for i in lag_indices
            ])

        # Indices: A[1:] applies to Y[t-1], Y[t-2], ...
        y_lags = get_lags(self.y_buffer, range(1, len(self.A)))

        # B applies to U[t-1], U[t-2], ... for B[1:]
        u_lags = get_lags(self.u_buffer, range(1, len(self.B)))

        # C applies to N[t], N[t-1], ...
        n_lags = get_lags(self.n_buffer, range(1, len(self.C)))
        n_lags = np.concatenate(([noise], n_lags))
        
        y0 = (-np.dot(self.A[1:], y_lags)
              + np.dot(self.B[1:], u_lags)
              + np.dot(self.C, n_lags))

        self.y_buffer[self.idx] = y0
        self.u_buffer[self.idx] = u
        self.n_buffer[self.idx] = noise
        self.idx = (self.idx + 1) % self.buffer_length

        self.history.append([y0, u, noise, r if r is not None else 0.0])
        if len(self.history) > self.history_limit:
            self.history = self.history[-self.history_limit:]

        if t is None:
            self.current_time += self.dt
        else:
            self.current_time = t

        if self.plot_system:
            self._update_plot()

        done = False
        output = np.array([y0])

        # self.state = self.curr_full_state()
        self.state = self.A_mat_ocf @ self.state + self.B_mat_ocf * u

        if full_state:
            return output, done, self.state
        else:
            return output, done
        
    def curr_full_state(self):
        return self.state
        # full_state = np.array(self.history[-self.full_state_length:])[:,0]
        # full_state = np.flip(full_state)
        # # padd zero if nto full length 
        # # Pad with zeros if not full length
        # if len(full_state) < self.full_state_length:
        #     padding = np.zeros(self.full_state_length - len(full_state))
        #     full_state = np.concatenate((full_state, padding))
                
    #     return full_state.reshape(-1,1)
    def full_state_to_obs_y(self, state):
        return np.array([state.flatten()[-1]])
    
    def _update_plot(self):
        history_arr = np.array(self.history)
        if len(history_arr) < 2:
            return  # not enough data to plot
        t_vals = np.linspace(self.current_time - len(history_arr) * self.dt, self.current_time, len(history_arr))

        self.line_y0.set_data(t_vals, history_arr[:, 0])
        self.line_r.set_data(t_vals, history_arr[:, 3])
        self.line_u.set_data(t_vals, history_arr[:, 1])
        self.line_n.set_data(t_vals, history_arr[:, 2])

        for ax in self.axs:
            ax.relim()
            ax.autoscale_view()
            if len(t_vals) >= 2:
                ax.set_xlim(t_vals[0], t_vals[-1])
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def show_final_plot(self):
        if not self.plot_system:
            return

        if len(self.history) == 0:
            print("No data to plot.")
            return

        # Create a new figure with final data
        plt.ioff()
        fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

        history_arr = np.array(self.history)
        t_vals = np.linspace(0, self.current_time, len(history_arr))

        axs[0].plot(t_vals, history_arr[:,0], label="Output Y[0]")
        axs[0].plot(t_vals, history_arr[:,3], linestyle='--', label="Reference R")
        axs[0].set_ylabel("Output / Reference")
        axs[0].legend()
        axs[0].grid()

        axs[1].plot(t_vals, history_arr[:,1], label="Input U")
        axs[1].set_ylabel("Input (U)")
        axs[1].legend()
        axs[1].grid()

        axs[2].plot(t_vals, history_arr[:,2], label="Noise (N)")
        axs[2].set_ylabel("Noise (N)")
        axs[2].set_xlabel("Time [s]")
        axs[2].legend()
        axs[2].grid()

        plt.show()


def test_armax_online():
    A = [1, -0.7]
    B = [0, -0.4]
    C = [1]
    F = 0.16
    L = -0.6
    dt = 0.01
    t_sim = 10
    n_steps = int(t_sim / dt)

    freq = 0.2
    time_array = np.arange(n_steps) * dt
    R = 2 * np.sign(np.sin(2 * np.pi * freq * time_array))
    model = ARMAX(A, B, C, dt=dt, history_limit=5, noise_std=0.1, plot_system=True)
    

    outputs = []
    u = r = 0
    y = np.array([0.0])
    for i in range(n_steps):
        r = R[i]
        u = L * r - F * y[0]
        y, done = model.step(u, r)
        outputs.append(y)
        if done:
            break

    model.show_final_plot()

def test_armax_online_forever():
    # A = [1, -0.7]
    # B = [0, -0.4]
    # C = [1]
    A= [1.0, -0.7, -0.1]
    B=[0.0, -0.4, 0.2]
    C= [1.0]
    F = -0.440
    L = F*2.7
    dt = 0.01
    # t_sim and n_steps removed

    freq = 0.2
    model = ARMAX(A, B, C, dt=dt, history_limit=5, noise_std=0.2, plot_system=True)

    outputs = []
    u = r = 0
    y = np.array([0.0])
    i = 0
    while True:
        r = 2 * np.sin(2 * np.pi * freq * i * dt)  # dynamic reference
        u = L * r - F * y[0]
        y, done, full_state = model.step(u, r, full_state=True)
        print(full_state)
        outputs.append(y)
        i += 1
        if done:
            break

    model.show_final_plot()


if __name__ == "__main__":
    test_armax_online_forever()