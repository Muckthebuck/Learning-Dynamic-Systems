import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Tuple, Optional, Union
from plot_helpers.plot_helpers import move_figure

np.random.seed(42)  # For reproducibility
class WaterTank:
    def __init__(self, dt=0.02, history_limit=10, noise_std=0.05, plot_system=True, visual=True,
                 vis_width=800, vis_height=600, padding_left=100, padding_right=100,
                 padding_top=80, padding_bottom=80, tank_height_m=2.0, tank_width_m=1.0, 
                 a=5.0, b=1.0):

        self.dt = dt
        self.noise_std = noise_std
        self.plot_system = plot_system
        self.visual = visual

        self.state = np.array([0.0])
        self.level = 0.0  # normalized level (0 to 1)
        self.holes = []
        self.history = []
        self.history_limit = int(history_limit / dt)
        self.current_time = 0.0
        self.inflow = 0.0
        self.input_limit = np.array([0.0, 10.0])
        self.done = False
        # dh/dt = a * self.inflow - b * self.level  # simplified dynamics
        self.a = a
        self.b = b
        # Real dimensions in meters
        self.tank_height_m = tank_height_m
        self.tank_width_m = tank_width_m

        # Visualization parameters
        self.vis_width = vis_width
        self.vis_height = vis_height
        self.padding_left = padding_left
        self.padding_right = padding_right
        self.padding_top = padding_top
        self.padding_bottom = padding_bottom

        # Calculated pixel-to-meter conversion
        self.pixels_per_meter = (self.vis_height - self.padding_top - self.padding_bottom) / self.tank_height_m

        self.tank_img = np.ones((self.vis_height, self.vis_width, 3), dtype=np.uint8) * 255
        self.leak_trails = []

        self.punch_hole()

        if plot_system:
            self._init_plot()
        if visual:
            self._init_visual()

    def set_initial_state(self, state):
        self.state = state

    def full_state_to_obs_y(self, state):
        if isinstance(state, np.ndarray):
            return state
        elif isinstance(state, (float, np.floating)):
            return np.array([state])
        
    

    def _init_plot(self):
        self.fig, self.axs = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
        self.line_level, = self.axs[0].plot([], [], label="Water Level")
        self.line_ref, = self.axs[0].plot([], [], label="Reference", linestyle='--')
        self.axs[0].set_ylim(0, self.tank_height_m)
        self.axs[0].legend()
        self.axs[0].grid()
        self.line_inflow, = self.axs[1].plot([], [], label="Inflow U")
        self.axs[1].set_ylim(*self.input_limit)
        self.axs[1].legend()
        self.axs[1].grid()
        self.fig.canvas.mpl_connect('close_event', self._on_close)
        plt.ion()
        move_figure(self.fig, 1000, 0)
        plt.show()


    def _init_visual(self):
        cv2.namedWindow("Water Tank")
        cv2.moveWindow("Water Tank", 1000, 800)
        cv2.setMouseCallback("Water Tank", self._mouse_callback)

    def _on_close(self, event):
        self.done = True

    def _mouse_callback(self, event, x, y, flags, param):
        tank_top = self.padding_top
        tank_bottom = self.vis_height - self.padding_bottom
        tank_left = self.padding_left
        tank_right = self.vis_width - self.padding_right

        if event == cv2.EVENT_LBUTTONDOWN:
            if tank_left <= x <= tank_right and tank_top <= y <= tank_bottom:
                level_y = 1 - ((y - tank_top) / (tank_bottom - tank_top))
                level_y = np.clip(level_y, 0, 1)
                hole_y = int(tank_bottom - level_y * (tank_bottom - tank_top))
                self.punch_hole(y=hole_y)

        elif event == cv2.EVENT_RBUTTONDOWN:
            click_pos = np.array([x, y])
            if self.holes:
                hole_x = self._get_hole_x()
                distances = [np.linalg.norm(click_pos - np.array([hole_x, h['y']])) for h in self.holes]
                min_idx = np.argmin(distances)
                if distances[min_idx] < 15:
                    self.holes.pop(min_idx)
                    self.leak_trails.pop(min_idx)

    def punch_hole(self, y=None, y_m=None):
        tank_bottom = self.vis_height - self.padding_bottom
        tank_top = self.padding_top

        if y_m is not None:
            y = int(tank_bottom - y_m * self.pixels_per_meter)
        elif y is None:
            y = int(tank_bottom - self.level * (tank_bottom - tank_top))

        height_m = (tank_bottom - y) / self.pixels_per_meter  # convert pixel to meters
        self.holes.append({'rate': self.a, 'time': self.current_time, 'y': y, 'height_m': height_m})
        self.leak_trails.append([])

    def step(self, u: Union[float, np.ndarray], 
             r: Optional[float] = None, 
             t: Optional[float] = None, 
             full_state : Optional[bool] = False) -> Union[Tuple[np.ndarray, bool], Tuple[np.ndarray, bool, np.ndarray]]:
        
        if self.done:
            if full_state:
                return np.array([0.0]), True, np.array([0.0])
            else:
                return np.array([0.0]), True

        noise = np.random.normal(0, self.noise_std) if self.noise_std > 0 else 0.0

        if isinstance(u, np.ndarray):
            u = u[0]
        u = np.clip(u, *self.input_limit)

        tank_bottom = self.vis_height - self.padding_bottom
        tank_top = self.padding_top

        leak = 0
        for hole in self.holes:
            hole_height_m = hole['height_m']
            if self.level > hole_height_m:
                h = self.level - hole_height_m
                leak += (1- hole['rate'] * self.dt)
        self.level = leak * self.level + self.b * u * self.dt
        self.level = np.clip(self.level, 0.0, self.tank_height_m)


        self.inflow = max(0.0, u)

        self.history.append([self.level+noise, u, r if r is not None else 0.0])
        if len(self.history) > self.history_limit:
            self.history = self.history[-self.history_limit:]

        self.current_time += self.dt

        if self.plot_system:
            self._update_plot()
        if self.visual:
            self._update_visual()

        self.state = self.curr_full_state()
        if full_state:
            return np.array([self.level + noise]), False, self.state + noise
        else:
            return np.array([self.level + noise]), False

    def curr_full_state(self):
        return np.array([self.level])
   
    def _get_hole_x(self):
        return self.vis_width - self.padding_right + 10

    def _update_visual(self):
        img = self.tank_img.copy()

        tank_left = self.padding_left
        tank_right = self.vis_width - self.padding_right
        tank_top = self.padding_top
        tank_bottom = self.vis_height - self.padding_bottom
        water_top = int(tank_bottom - self.level * self.pixels_per_meter)

        if np.abs(self.level)>0.01:
            # Draw water level
            self._draw_rounded_gradient_rect(img, (tank_left+2, water_top), (tank_right-2, tank_bottom-1),
                                    top_color=(255, 230, 180), bottom_color=(255, 200, 100), radius=20)

        # Friendly outline color
        self._draw_rounded_rect(img, (tank_left, tank_top), (tank_right, tank_bottom), (120, 120, 120), thickness=3, radius=25)

        # Inflow arrow - soft blue
        inflow_x = (tank_left + tank_right) // 2
        cv2.arrowedLine(img, (inflow_x, self.padding_top // 2), (inflow_x, self.padding_top // 2 + 20), (90, 160, 250), 3, tipLength=0.3)

        # Holes and leak drops - soft purple/red
        hole_x = self._get_hole_x()
        for i, hole in enumerate(self.holes):
            y = hole['y']
            cv2.circle(img, (hole_x+2, y+2), 7, (120, 60, 60), -1)      # Shadow
            cv2.circle(img, (hole_x, y), 6, (180, 80, 80), -1)          # Main

            hole_height_norm = (tank_bottom - y) / (tank_bottom - tank_top)
            if self.level > hole_height_norm and np.random.rand() < hole['rate']:
                self.leak_trails[i].append([hole_x, y])

            new_trail = []
            for drop in self.leak_trails[i]:
                drop[1] += 4
                if drop[1] < tank_bottom:
                    cv2.circle(img, tuple(drop), 4, (255, 230, 180), -1)
                    overlay = img.copy()
                    cv2.circle(overlay, tuple(drop), 6, (255, 230, 180), -1)
                    alpha = 0.3
                    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
                    new_trail.append(drop)
            self.leak_trails[i] = new_trail

        # Friendly labels
        cv2.putText(img, f"Water Level: {self.level:.2f} m", (tank_left + 10, tank_bottom + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.putText(img, f"Inflow: {self.inflow:.2f}", (tank_left + 10, tank_top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.putText(img, f"Holes: {len(self.holes)} (L-click add, R-click remove)", (10, self.vis_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow("Water Tank", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or self.done:
            self.done = True
            cv2.destroyAllWindows()

    def _draw_rounded_gradient_rect(self, img, top_left, bottom_right, top_color, bottom_color, radius=20):
        x1, y1 = top_left
        x2, y2 = bottom_right
        height = y2 - y1
        width = x2 - x1

        # Create a temporary image for gradient
        grad_img = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw vertical gradient
        for i in range(height):
            alpha = i / height
            color = (
                int(top_color[0] * (1 - alpha) + bottom_color[0] * alpha),
                int(top_color[1] * (1 - alpha) + bottom_color[1] * alpha),
                int(top_color[2] * (1 - alpha) + bottom_color[2] * alpha),
            )
            cv2.line(grad_img, (0, i), (width - 1, i), color, 1)

        # Create a mask for rounded rectangle
        mask = np.zeros((height, width), dtype=np.uint8)

        # Draw filled rounded rectangle on mask (white inside)
        self._draw_filled_rounded_rect(mask, (0, 0), (width, height), 255, radius)

        # Apply mask to gradient image
        for c in range(3):
            grad_img[:, :, c] = cv2.bitwise_and(grad_img[:, :, c], mask)

        # Copy only pixels inside rounded rectangle from grad_img to img
        mask_bool = mask.astype(bool)
        for c in range(3):
            img[y1:y2, x1:x2, c][mask_bool] = grad_img[:, :, c][mask_bool]


    def _draw_filled_rounded_rect(self, img, top_left, bottom_right, color, radius):
        x1, y1 = top_left
        x2, y2 = bottom_right

        # Draw filled rectangle without corners
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness=-1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness=-1)

        # Draw 4 filled circles at corners
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness=-1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness=-1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness=-1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness=-1)

    def _draw_rounded_rect(self, img, top_left, bottom_right, color, thickness=2, radius=20):
        x1, y1 = top_left
        x2, y2 = bottom_right
        cv2.line(img, (x1+radius, y1), (x2-radius, y1), color, thickness)
        cv2.line(img, (x1+radius, y2), (x2-radius, y2), color, thickness)
        cv2.line(img, (x1, y1+radius), (x1, y2-radius), color, thickness)
        cv2.line(img, (x2, y1+radius), (x2, y2-radius), color, thickness)
        cv2.ellipse(img, (x1+radius, y1+radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2-radius, y1+radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1+radius, y2-radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2-radius, y2-radius), (radius, radius), 0, 0, 90, color, thickness)

    def _draw_gradient_rect(self, img, top_left, bottom_right, top_color, bottom_color):
        x1, y1 = top_left
        x2, y2 = bottom_right
        height = y2 - y1
        for i in range(height):
            alpha = i / height
            color = (
                int(top_color[0] * (1-alpha) + bottom_color[0] * alpha),
                int(top_color[1] * (1-alpha) + bottom_color[1] * alpha),
                int(top_color[2] * (1-alpha) + bottom_color[2] * alpha),
            )
            cv2.line(img, (x1, y1 + i), (x2, y1 + i), color, 1)

    def _put_text_shadow(self, img, text, pos, font, scale, color, thickness):
        x, y = pos
        cv2.putText(img, text, (x+2, y+2), font, scale, (0,0,0), thickness+2, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    def _update_plot(self):
        if len(self.history) == 0:
            return
        data = np.array(self.history)
        t_vals = np.linspace(self.current_time - self.dt * len(data), self.current_time, len(data))
        self.line_level.set_data(t_vals, data[:, 0])
        self.line_ref.set_data(t_vals, data[:, 2])
        self.line_inflow.set_data(t_vals, data[:, 1])
        self.axs[0].relim()
        self.axs[0].autoscale_view()
        self.axs[1].relim()
        self.axs[1].autoscale_view()
        plt.pause(0.001)

    def show_final_plot(self):
        if len(self.history) == 0:
            print("No data to plot.")
            return
        plt.ioff()
        fig, axs = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
        data = np.array(self.history)
        t_vals = np.linspace(0, self.current_time, len(data))
        axs[0].plot(t_vals, data[:, 0], label="Water Level")
        axs[0].plot(t_vals, data[:, 2], label="Reference", linestyle='--')
        axs[0].set_ylim(0, 1)
        axs[0].legend()
        axs[0].grid()
        axs[1].plot(t_vals, data[:, 1], label="Inflow U")
        axs[1].set_ylim(*self.input_limit)
        axs[1].legend()
        axs[1].grid()
        plt.show()



def test_water_tank_controller_forever():
    tank = WaterTank(noise_std=0.05, plot_system=True, visual=True)  # noise off for stability
    tank.punch_hole()

    # Controller gains (tweak as needed)
    F = 1.5
    L = 1.6

    dt = tank.dt
    i = 0
    while True:
        # Dynamic reference signal between 0.3 and 0.8
        r = 0.55 + 0.25 * np.sin(2 * np.pi * 0.05 * i * dt)

        y = tank.level
        u = L * r - F * y
        u = np.clip(u, *tank.input_limit)

        _, done = tank.step(u, r=r)
        if done:
            break
        i += 1


if __name__ == "__main__":
    test_water_tank_controller_forever()
