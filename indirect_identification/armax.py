import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class ARMAX:
    def __init__(self, A, B, C, F, L):
        self.A = np.array(A)
        self.B = np.array(B)
        self.C = np.array(C)
        self.F = np.array(F)
        self.L = np.array(L)
    
    def simulate(self, n_samples, R=None, noise_std=0.1):
        Y = np.zeros(n_samples)
        U = np.zeros(n_samples)
        N = np.random.normal(0, noise_std, n_samples)
        
        if R is None:
            R = np.zeros(n_samples)
        
        max_order = max(len(self.A), len(self.B), len(self.C), len(self.F), len(self.L))
        
        for t in range(max_order, n_samples):
            Y[t] = (- np.dot(self.A[1:], Y[t-1:t-len(self.A):-1]) 
                    + np.dot(self.B, U[t-1:t-len(self.B)-1:-1])
                    + np.dot(self.C, N[t:t-len(self.C):-1]))
            
            U[t] = (np.dot(self.L, R[t:t-len(self.L):-1]) 
                    - np.dot(self.F, Y[t:t-len(self.F):-1]))
        
        return Y, U, N, R
    
    def plot_results(self, Y, U, N, R):
        fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        
        axs[0].plot(Y)
        axs[0].set_ylabel('Output (Y)')
        axs[0].set_title('ARMAX Closed-Loop Simulation Results')
        
        axs[1].plot(U)
        axs[1].set_ylabel('Input (U)')
        
        axs[2].plot(N)
        axs[2].set_ylabel('Noise (N)')
        
        axs[3].plot(R)
        axs[3].set_ylabel('Reference (R)')
        axs[3].set_xlabel('Time')
        
        plt.tight_layout()
        plt.show()

# Example usage
A = [1, -0.33]  # A(z^-1) = 1 - 0.33z^-1
B = [0.22]      # B(z^-1) = 0.22z^-1
C = [1, 0.15]   # C(z^-1) = 1 + 0.15z^-1
F = [0.31, 0.23] # F(z^-1) = 0.31 + 0.23z^-1
L = [1]        # L(z^-1) = 1

armax_model = ARMAX(A, B, C, F, L)

n_samples = 100
# square wave reference signal
R = signal.square(np.linspace(0, 10*np.pi, n_samples))

Y, U, N, R = armax_model.simulate(n_samples, R, noise_std=0.2)
armax_model.plot_results(Y, U, N, R)
