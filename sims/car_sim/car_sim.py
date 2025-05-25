import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from typing import Union, Optional, Tuple
import pygame
import time
import os

np.random.seed(42)

class CarSim:
    def __init__(self, lanes=3):
        self.road_length = 100  # total road length units
        self.lanes = lanes
        # Changed lane centers to integer indices for simpler consistent mapping
        self.lane_centers = [i for i in range(self.lanes)]
        self.car_pos = [1, 0]  # lateral position, longitudinal position
        self.prev_car_pos = self.car_pos.copy()
        self.state = np.array(self.car_pos)
        self.done = False
        self._init_plot()
        self._init_visual()
        self.obstacles = self._generate_traffic()
        self.ref_path = self._compute_reference_path()

        self.view_length = 30  # how much road length units visible ahead
        self.car_screen_y = 500  # fixed vertical pixel pos of car on screen

    def _generate_traffic(self):
        obs = []
        density = 0.15
        num_obstacles = int(self.road_length * density)
        buffer = 8  # minimum gap between cars in the same lane

        segment_length = self.road_length / num_obstacles

        for i in range(num_obstacles):
            # Random y within the segment
            y = np.random.uniform(i * segment_length, (i + 1) * segment_length)
            lane = np.random.randint(0, self.lanes)

            # Offset y if overlapping with existing obs in the same lane
            while any(abs(y - oy) < buffer and lane == ol for oy, ol in obs):
                y += buffer
                if y > self.road_length:
                    y = self.road_length
                    break

            obs.append((y, lane))

        obs.sort(key=lambda x: x[0])
        return obs

    def _compute_reference_path(self, safety_dist=30, lane_change_length=10):
        y_points = []
        x_points = []

        current_lane = 1  # Start center lane
        lane_change_start_y = None
        lane_change_end_y = None
        start_x = self.lane_centers[current_lane]
        target_lane = current_lane

        for y in range(self.road_length):
            # Check if we are in a lane change segment
            in_lane_change = lane_change_start_y is not None and lane_change_start_y <= y <= lane_change_end_y

            if not in_lane_change:
                # Find obstacles near this y
                close_obstacles = [(oy, ol) for oy, ol in self.obstacles if 0 <= oy - y < safety_dist]
                lanes_blocked = [ol for oy, ol in close_obstacles]

                if current_lane in lanes_blocked:
                    # Try to shift lanes
                    left_free = current_lane > 0 and (current_lane - 1) not in lanes_blocked
                    right_free = current_lane < self.lanes - 1 and (current_lane + 1) not in lanes_blocked

                    if left_free:
                        target_lane = current_lane - 1
                    elif right_free:
                        target_lane = current_lane + 1
                    else:
                        target_lane = current_lane

                    if target_lane != current_lane:
                        # Start lane change
                        lane_change_start_y = y
                        lane_change_end_y = y + lane_change_length
                        start_x = self.lane_centers[current_lane]
                        end_x = self.lane_centers[target_lane]

            if in_lane_change:
                # Compute normalized progress along lane change [0,1]
                t = (y - lane_change_start_y) / lane_change_length
                # Use a smooth polynomial to interpolate lateral position
                # Let's use a simple cubic polynomial for smoothness (ease in/out)
                # x(t) = start_x + (end_x - start_x) * (3t^2 - 2t^3)
                x = start_x + (end_x - start_x) * (3 * t ** 2 - 2 * t ** 3)

                if y == lane_change_end_y:
                    # End lane change
                    current_lane = target_lane
                    lane_change_start_y = None
                    lane_change_end_y = None
            else:
                # Just keep lateral position in current lane
                x = self.lane_centers[current_lane]

            y_points.append(y)
            x_points.append(x)

        # Fit a spline for smoothing (optional, your path is already smooth)
        spline = CubicSpline(y_points, x_points)
        smooth_y = np.linspace(0, self.road_length - 1, 500)
        smooth_x = spline(smooth_y)

        self.spline = spline
        self.smooth_ref_path = list(zip(smooth_y, smooth_x))
        return self.smooth_ref_path



    def _init_plot(self):
        self.fig, self.axs = plt.subplots(2, 1)
        self.track_plot = self.axs[0].plot([], [], label="Car")[0]
        self.ref_plot = self.axs[0].plot([], [], label="Reference")[0]
        self.axs[0].legend()
        self.axs[0].set_xlim(0, self.road_length)
        self.axs[0].set_ylim(-1, self.lanes + 1)
        self.axs[0].set_title("Tracking Reference Path")

        self.input_plot = self.axs[1].plot([], [], label="u (lateral vel)")[0]
        self.axs[1].set_xlim(0, self.road_length)
        self.axs[1].set_ylim(-1, 1)
        self.axs[1].set_title("Control Input")
        plt.tight_layout()
        self.fig.canvas.mpl_connect('close_event', self._on_close)

        self.plot_data = {"y": [], "x": [], "ref_x": [], "u": []}


    def _init_visual(self):
        pygame.init()
        self.screen_width = 800
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Car Simulation")

        self.road_color = (50, 50, 50)
        self.lane_color = (200, 200, 200)
        self.ref_color = (0, 255, 0)

        self.road_left = 100
        self.road_width = 600
        self.lane_width = self.road_width / self.lanes

        assets_dir = os.path.join(os.path.dirname(__file__), "assets")
        self.car_image = pygame.image.load(os.path.join(assets_dir, "green_car.png")).convert_alpha()
        # self.car_image = pygame.transform.scale(self.car_image, (160, 80))
        self.obs_image = pygame.image.load(os.path.join(assets_dir, "red_car.png")).convert_alpha()
        # self.obs_image = pygame.transform.scale(self.obs_image, (40, 80))

    

    def _on_close(self, event):
        self.done = True

    def set_initial_state(self, state):
        self.state = np.array(state)
        self.car_pos = list(state)

    def full_state_to_obs_y(self, state):
        return np.array([state[1]])
    



    def step(self, u: Union[float, np.ndarray],
             r: Optional[float] = None,
             t: Optional[float] = None,
             full_state: Optional[bool] = False) -> Union[Tuple[np.ndarray, bool], Tuple[np.ndarray, bool, np.ndarray]]:

        if self.done or self.car_pos[1] >= self.road_length:
            self.done = True
            return np.array([self.car_pos[1]]), True

        # Update lateral position (clamped)
        x = self.car_pos[0] + u
        x = np.clip(x, 0, self.lanes - 1)
        self.car_pos[0] = x

        # Advance forward
        self.car_pos[1] += 0.5  # speed in road units per step
        self.state = np.array(self.car_pos)

        # Calculate heading based on movement vector
        dx = self.car_pos[0] - self.prev_car_pos[0]
        dy = self.car_pos[1] - self.prev_car_pos[1]
        if dx != 0 or dy != 0:
            self.car_heading = np.arctan2(dy, dx)
        
        # Update previous position
        self.prev_car_pos = self.car_pos.copy()

        noise = np.random.normal(0, 0.01, size=1)

        sim._update_visual()
        sim._update_plot(u)

        if full_state:
            return np.array([self.car_pos[1] + noise]), False, self.state + noise
        else:
            return np.array([self.car_pos[1] + noise]), False

    def curr_full_state(self):
        return self.state

    def _lane_x_at_y(self, lane_pos, y_rel):
        """
        Calculate the horizontal pixel position of lane center at a relative distance y_rel from the car.
        Applies a perspective curve effect: lanes narrow and converge at horizon.
        y_rel=0 is car position (bottom), y_rel=view_length is top (horizon)
        lane_pos: can be fractional lateral lane position, with 0 = leftmost lane center.
        """
        # perspective scaling: lanes narrower near horizon
        perspective_scale = np.interp(y_rel, [0, self.view_length], [1.0, 0.3])
        horizon_shift = np.interp(y_rel, [0, self.view_length], [0, self.road_width * 0.1])

        # Remove the previous half lane width offset so lane_pos is directly lane center pos
        lane_center_px = self.road_left + horizon_shift + lane_pos * self.lane_width * perspective_scale
        return lane_center_px

    def _y_screen_pos(self, y_rel):
        """
        Convert relative y distance from car into pixel y position on screen (car fixed near bottom)
        y_rel=0 -> car_screen_y
        y_rel=view_length -> top (0)
        """
        return self.car_screen_y - (y_rel / self.view_length) * self.car_screen_y
    
    def _get_car_dimensions(self, y_rel):
        """
        Calculate car width and height in pixels at a relative distance y_rel from the ego vehicle.
        y_rel=0 means ego vehicle itself (bottom of screen).
        """
        aspect_ratio = 4.5 / 2.0  # typical car length/width ratio
        
        # Lane width in pixels (constant)
        lane_width_px = self.lane_width

        # For ego car, use slightly smaller width so it fits nicely inside the lane:
        base_car_width_px = lane_width_px * 0.6  # 85% of lane width

        # Calculate base car height based on aspect ratio
        base_car_height_px = base_car_width_px * aspect_ratio

        # Perspective scale: from 1.0 (ego car) to 0.3 (far away)
        min_scale = 0.2
        max_scale = 0.6
        scale = max_scale - (y_rel / self.view_length) * (max_scale - min_scale)
        scale = max(min_scale, min(max_scale, scale))

        width_scaled = int(base_car_width_px * scale)
        height_scaled = int(base_car_height_px * scale)

        return width_scaled, height_scaled

    def _get_lane_angle(self, lane_pos, y_rel, delta=5.0):
        """
        Estimate the angle (in degrees) of a lane line at a given relative y-position
        for correct car alignment. Accounts for perspective.
        """
        # Sample two points further apart for better angle stability
        y1 = y_rel
        y2 = min(self.view_length, y_rel + delta)

        x1 = self._lane_x_at_y(lane_pos, y1)
        x2 = self._lane_x_at_y(lane_pos, y2)

        y_screen1 = self._y_screen_pos(y1)
        y_screen2 = self._y_screen_pos(y2)

        # Vector in screen space (x right, y down in pygame)
        dx = x2 - x1
        dy = y_screen2 - y_screen1

        # Flip dy for orientation (cars go "up" visually)
        angle_rad = np.arctan2(-dy, dx) - np.pi/2
        angle_deg = np.degrees(angle_rad)

        return angle_deg




    def _update_visual(self):
        pygame.event.pump()
        self.screen.fill((30, 30, 30))

        # Road polygon
        top_left = (self._lane_x_at_y(-0.5, self.view_length), 0)
        top_right = (self._lane_x_at_y(self.lanes - 0.5, self.view_length), 0)
        bottom_right = (self._lane_x_at_y(self.lanes - 0.5, 0), self.car_screen_y)
        bottom_left = (self._lane_x_at_y(-0.5, 0), self.car_screen_y)
        pygame.draw.polygon(self.screen, self.road_color, [top_left, top_right, bottom_right, bottom_left])

        # Road boundaries
        left_boundary_points = [
            (self._lane_x_at_y(-0.5, y_rel), self._y_screen_pos(y_rel)) for y_rel in np.linspace(0, self.view_length, 30)
        ]
        right_boundary_points = [
            (self._lane_x_at_y(self.lanes - 0.5, y_rel), self._y_screen_pos(y_rel)) for y_rel in np.linspace(0, self.view_length, 30)
        ]
        pygame.draw.lines(self.screen, (255, 255, 255), False, left_boundary_points, 6)
        pygame.draw.lines(self.screen, (255, 255, 255), False, right_boundary_points, 6)

        # Lane lines
        for i in range(1, self.lanes):
            points = []
            for y_rel in np.linspace(0, self.view_length, 100):
                x = self._lane_x_at_y(i - 0.5, y_rel)
                y = self._y_screen_pos(y_rel)
                points.append((x, y))
            self.draw_dashed_line(self.screen, self.lane_color, points, dash_length=20, gap_length=15, width=4)

        # Obstacles
        for oy, ol in self.obstacles:
            y_rel = oy - self.car_pos[1]
            if 0 <= y_rel <= self.view_length:
                w, h = self._get_car_dimensions(y_rel)
                angle = self._get_lane_angle(ol, y_rel)
                obs_img_scaled = pygame.transform.smoothscale(self.obs_image, (w, h))
                obs_img_rotated = pygame.transform.rotate(obs_img_scaled, angle)
                obs_x = self._lane_x_at_y(ol, y_rel) - obs_img_rotated.get_width() / 2
                obs_y = self._y_screen_pos(y_rel) - obs_img_rotated.get_height() / 2
                self.screen.blit(obs_img_rotated, (obs_x, obs_y))

        # Reference path
        ref_points = []
        for y, lane_pos in self.smooth_ref_path:
            y_rel = y - self.car_pos[1]
            if 0 <= y_rel <= self.view_length:
                px = self._lane_x_at_y(lane_pos, y_rel)
                py = self._y_screen_pos(y_rel)
                ref_points.append((int(px), int(py)))
        if len(ref_points) > 1:
            pygame.draw.lines(self.screen, self.ref_color, False, ref_points, 3)

        # Ego car
        # Get scaled dimensions for ego car (y_rel = 0 since it's at the bottom)
        car_w, car_h = self._get_car_dimensions(0)
        scaled_ego = pygame.transform.smoothscale(self.car_image, (car_w, car_h))

        # Get angle and rotate
        car_angle = np.degrees(self.car_heading) -90
        rotated_ego = pygame.transform.rotate(scaled_ego, car_angle)

        # Center the rotated image at the ego position
        
        ego_center_x = self._lane_x_at_y(self.car_pos[0], 0)
        ego_center_y = self.car_screen_y
        rotated_rect = rotated_ego.get_rect(center=(ego_center_x, ego_center_y))

        self.screen.blit(rotated_ego, rotated_rect.topleft)


        pygame.display.flip()


    def draw_dashed_line(self, surface, color, points, dash_length=15, gap_length=10, width=4):
        """Draw dashed line on pygame surface connecting given points."""

        if len(points) < 2:
            return

        # Flatten points into segments
        total_len = 0
        seg_lengths = []
        for i in range(len(points) - 1):
            start = np.array(points[i])
            end = np.array(points[i + 1])
            seg_len = np.linalg.norm(end - start)
            seg_lengths.append((start, end, seg_len))
            total_len += seg_len

        dist_drawn = 0
        draw = True
        dash_remain = dash_length
        gap_remain = gap_length

        i = 0
        while i < len(seg_lengths):
            start, end, seg_len = seg_lengths[i]
            vec = (end - start) / seg_len  # unit vector
            pos = start.copy()

            while seg_len > 0:
                if draw:
                    step = min(dash_remain, seg_len)
                    end_pos = pos + vec * step
                    pygame.draw.line(surface, color, pos, end_pos, width)
                    pos = end_pos
                    seg_len -= step
                    dash_remain -= step
                    if dash_remain <= 0:
                        draw = False
                        gap_remain = gap_length
                else:
                    step = min(gap_remain, seg_len)
                    pos += vec * step
                    seg_len -= step
                    gap_remain -= step
                    if gap_remain <= 0:
                        draw = True
                        dash_remain = dash_length
            i += 1


    def _update_plot(self, u):
        x, y = self.car_pos
        ref_x = self.spline(y) if y < self.road_length else self.lane_centers[1]

        self.plot_data["y"].append(y)
        self.plot_data["x"].append(x)
        self.plot_data["ref_x"].append(ref_x)
        self.plot_data["u"].append(u)

        self.track_plot.set_data(self.plot_data["y"], self.plot_data["x"])
        self.ref_plot.set_data(self.plot_data["y"], self.plot_data["ref_x"])
        self.input_plot.set_data(self.plot_data["y"], self.plot_data["u"])

        for ax in self.axs:
            ax.relim()
            ax.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def show_final_plot(self):
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    sim = CarSim(lanes=4)
    plt.ion()

    def simple_lateral_controller(state, spline):
        x, y = state
        if y >= sim.road_length:
            return 0
        ref_x = spline(y)
        error = ref_x - x
        return 0.1 * error

    sim.set_initial_state([1.5, 0])

    while not sim.done:
        state = sim.curr_full_state()
        u = simple_lateral_controller(state, sim.spline)

        sim.step(u)


    sim.show_final_plot()
    pygame.quit()
