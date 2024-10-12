import numpy as np
import scipy

from LqrController import LQR_Controller
from ArmaxGenerator import ARMAX_Generator

class SPS:
    def __init__(self, p:float=(1-5/40), m:int=40, q:int=5, n:int=0, T:int=1) -> None:
        self.p = p # Confidence probability
        self.m = m # Integer m such that p = 1 − q/m
        self.q = q # Integer q such that p = 1 − q/m
        self.n = n # Number of data points

        self.T = T # block SPS window size
        self.T_repetitions_left: int = 0
        
        self.alpha = None # Random signs
        self.pi = None    # Random permutation

    # Process one time step
    # This is the main function for the real-time SPS implementation
    def update(self) -> None:
        self.n += 1
        self.append_alpha_column()
        self.generate_random_perm()

    # Generate a column of alpha values
    # This should be called whenever we add another data sample (n -> n+1)
    def append_alpha_column(self) -> None:
        new_col = None

        if self.T > 1 and self.T_repetitions_left > 0:
            self.T_repetitions_left -= 1 # Decrement repetitions

            # Repeat the last column in the array
            new_col = self.alpha[:, [-1]]

        else:
            # Check if we need to reset T
            if self.T > 1:
                self.T_repetitions_left = self.T - 1

            # Generate new column of random values
            options = [-1, 1]
            new_col = np.transpose(np.array([[np.random.choice(options) for _ in range(self.m)]]))

        # Update the state of the class instance
        if self.alpha is None:
            self.alpha = new_col
        else:
            self.alpha = np.concatenate([self.alpha, new_col], axis=1)

    # Generates matrix of integers for ordering SPS indicators
    def generate_random_perm(self) -> None:
        self.pi = np.random.permutation(range(self.m-1)) 

    # TODO Test this code
    def least_square_estimator(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        theta_hat = np.linalg.lstsq(np.matmul(np.transpose(X), X), np.matmul(np.transpose(X), y))
        
        return theta_hat
    
    def create_feature_matrix(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, y: np.ndarray, u: np.ndarray) -> np.ndarray:
        N = len(y)
        matrix_length = len(A) - 1 + len(B)
        
        X = np.zeros((N, matrix_length))

        for t in range(1, N):
            phi = self.get_phi(A, B, C, y, u, t)
            X[t,:] = phi

        return X

    def get_phi(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, y: np.ndarray, u: np.ndarray, t: int) -> np.ndarray:
        # This vector will consist of negative A parameters (except A0) and all B parameters
        # The noise parameter matrix C will be ignored

        num_y_samples = len(A) - 1 # Ignore first A parameter
        num_u_samples = len(B)

        # Calculate how many zeros need to be padded to output
        num_y_zeros = max(num_y_samples - t, 0)
        num_u_zeros = max(num_u_samples - t-1, 0)

        y_temp = y
        u_temp = u

        if num_y_zeros > 0:
            # Pad zeros to the left of the input if necessary
            zeros = num_y_zeros * [0]
            y_temp = np.concatenate([zeros, y])
    
        if num_u_zeros > 0:
            zeros = num_u_zeros * [0]
            u_temp = np.concatenate([zeros, u])        

        phi_y = y_temp[t+num_y_zeros-num_y_samples:t+num_y_zeros][::-1]
        
        # phi_u = u_temp[t:t-num_u_samples:-1]
        phi_u = u_temp[t+num_u_zeros-num_u_samples+1:t+num_u_zeros+1][::-1]

        return np.concatenate([phi_y, phi_u])
    
    def sps_indicator(self,
            y: np.ndarray, # Actual past outputs of the system
            u: np.ndarray, # Actual past inputs of the system
            A_star: np.ndarray, # Test A matrix
            B_star: np.ndarray, # Test B matrix
            C_star: np.ndarray, # Test C matrix
            D_star: int # Constant term
        ) -> bool:
        
        # Since we are using the ARMAX format, we are assuming that the system is closed-loop
        A_yt = self.basic_filter(A_star, y) # Compute A(q)y_t
        B_ut = self.basic_filter(B_star, u) # Compute B(q)u_t

        diff_AB = A_yt - B_ut

        # Reconstruct the noise C^-1(A(q)y_t - B(q)u_t)
        nt_theta = self.filter([1], C_star, diff_AB)
        sp_errors = np.multiply(self.alpha, nt_theta) # Regenerate sign perturbed noise
        
        # Initialise controller
        lqr = LQR_Controller()
        lqr.t = self.n
        lqr.regenerate_noise() # Generate noise samples once for re-use inside this loop
        lqr.is_reuse_noise = True 
        
        S = np.zeros(self.m-1)
        
        # Calculate baseline S0
        X = self.create_feature_matrix(A_star, B_star, C_star, y, u)
        R = np.matmul(np.transpose(X), X) / self.n
        R_root = scipy.linalg.cholesky(R)
        R_inv_root = scipy.linalg.inv(R_root)
        
        S[0] =  scipy.linalg.norm( np.matmul( R_inv_root, np.matmul(np.transpose(X), nt_theta/ self.n) ))
        
        # Calculate SPS 
        for i in range(1, self.m - 1):
            armax_bar = ARMAX_Generator(A_star, B_star, C_star, D_star, lqr.generate_output)
            # Set the noise for the armax generator
            armax_bar.e = sp_errors[i,:]
            
            # Generate the ARMAX data
            for _ in range(self.n):
                armax_bar.generate_datapoint()

            X_bar = self.create_feature_matrix(A_star, B_star, C_star, armax_bar.y, armax_bar.u)
            R_bar = np.matmul(np.transpose(X_bar), X_bar) / self.n
            R_bar_inv_root = scipy.linalg.inv(scipy.linalg.cholesky(R_bar))
            temp = np.matmul(R_bar_inv_root, np.transpose(X_bar))
            temp = np.matmul(temp, np.transpose(sp_errors[i,:]))
            temp = temp / self.n
            S[i] = scipy.linalg.norm( temp )
            
        # Combine S and Pi into a matrix
        SP = np.array([S, self.pi])

        # First sort by 1st col then 0th col
        sort_indices = np.lexsort((SP[1,:], SP[0,:]))
        
        # Check that the SP0 
        return sort_indices[0]  <= self.m - self.q - 1
    


    # FIR Filter where the A matrix is [1]
    def basic_filter(self, B: np.ndarray, X: np.ndarray) -> np.ndarray:
        # return np.convolve(B, X)[:len(X)]
        return scipy.signal.lfilter(B, [1.0], X)
    
    def filter(self, B: np.ndarray, A: np.ndarray, X) -> np.ndarray:
        return scipy.signal.lfilter(B, A, X)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    theta_real = [0.44, 0.33, 0]
    m = 40
    q = 5
    p = 1-q/m
    T = 1
    n = 100

    sps = SPS(p, m, q, n=0, T=T)

    # Actual system parameters
    A = [ 1, theta_real[0] ]
    B = [ 0, theta_real[1] ]
    C = [ 1, theta_real[2] ]


    # Known Controller
    known_controller = LQR_Controller()
    known_controller.t = n
    known_controller.regenerate_noise()
    known_controller.is_reuse_noise = True

    # Generate the real data
    armax = ARMAX_Generator(A, B, C, 0, known_controller.generate_output)

    # Simulate N datapoint updates in the system
    for i in range(n):
        armax.generate_noise()
        armax.generate_datapoint()
        sps.update()

    plt.subplot(3, 1, 1)
    plt.plot(armax.u)
    plt.title('Input u(t)')

    plt.subplot(3, 1, 2)
    plt.plot(armax.e)
    plt.title('Noise e(t)')

    plt.subplot(3, 1, 3)
    plt.plot(armax.y)
    plt.title('Output y(t)')

    print(armax.y)
    print(armax.u)

    # Calculate SPS 
    a = np.linspace(0.2, 0.8, 10)
    b = np.linspace(0.1, 0.5, 10)

    confidence_region = []

    # Test true theta

    for test_a in a:
        print("Testing a =", test_a)
        for test_b in b:
            A_star = [1, -test_a]
            B_star = [0, test_b]
            C_star = [1, 0]
            D_star = 0

            if sps.sps_indicator(armax.y, armax.u, A_star, B_star, C_star, D_star):
                confidence_region.append( (test_a, test_b) )

    x_points = [x[0] for x in confidence_region]
    y_points = [x[1] for x in confidence_region]

    print(x_points)
    plt.plot(x_points, y_points, 'o')
    
    is_true_theta_in_conf_region = sps.sps_indicator(armax.y, armax.u, A, B, C, 0)


    plt.waitforbuttonpress()