import numpy as np

class LQR_Controller:
    def __init__(self) -> None:
        self.is_reuse_noise = True  # Set flag to true if the noise samples should be retained across multiple calculations of output
                                    # This should help with the issue where random noise in the controller output could not be captured when calculating SPS
        self.t = 0
        self.Nt = [] # Vector of noise values

        self.K = 0.5

        self.sigma = 0.0001 # Variance of noise generated

    # Generates t length vector of noise values
    def regenerate_noise(self):
        self.Nt = [ np.random.normal(self.sigma) for x in range(self.t)]

    # TODO: Define the logic for the online controller
    def update_gain(self):
        pass

    # This function can be passed into the ARMAX Generator instance
    def generate_output(self, y_vals: float, t: int) -> np.ndarray:
        noise = 0.0
        prev_y = y_vals[-1]

        if self.is_reuse_noise:
            if len(self.Nt) < t+1:
                raise "Error: noise samples not initialised"
            noise = self.Nt[t]
        else:
            noise = np.random.normal(self.sigma)
            if len(self.Nt) < t:
                self.Nt = np.concatenate([self.Nt, np.zeros(t - len(self.Nt))])
            self.Nt[t] = noise
        
        return self.K * prev_y + noise
