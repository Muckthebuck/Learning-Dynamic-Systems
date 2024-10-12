import numpy as np

class ARMAX_Generator:
    def __init__(self, A: list, B: list, C: list, D: list, input_controller) -> None:
        # System Parameters
        self.A = A  # AR coefficients
        self.B = B  # X input coefficients
        self.C = C  # MA coefficients
        self.D = D  # Constant term

        self.order = max(len(A), len(B), len(C))
        self.t = 0

        # TODO delete this
        def temp_input_controller(y, u) -> float: 
            sigma = 0.1
            return 0.4 * y[-1] + np.random.normal(sigma)
        
        self.input_controller = input_controller

        # State
        # The below vectors should all be of the same length equal to the current time step
        self.y = []  # Output of the system
        self.u = []  # Input to the system
        self.e = []  # Noise input to the system

    # TODO Replace this with a user-provided noise function which can be any zero-mean symmetric iid noise source
    def generate_noise(self) -> float:
        sigma = 0.1
        self.e.append(np.random.normal(0, sigma))
    
    # Generate new datapoints for the next timestep and append it to the system's state
    def generate_datapoint(self) -> None: 
        # AR component
        overlap = min(len(self.A)-1, len(self.y)) # Calculate the overlap between A(2), A(3), A(...) and the past inputs
        ar_sum = np.matmul(self.A[1:overlap+1:][::-1], self.y[-overlap::]) # Reverse the order of the coefficients and multiply the matrices

        # Exogenous input component
        overlap = min(len(self.B), len(self.u))
        exo_sum = np.matmul(self.B[:overlap:][::-1], self.u[-overlap::])

        # MA component
        overlap = min(len(self.C), len(self.e))
        ma_sum = np.matmul(self.C[:overlap:][::-1], self.e[-overlap::])

        # Compute output
        self.y.append( -ar_sum + exo_sum + ma_sum + self.D / self.A[0])
        self.u.append( self.input_controller(self.y, self.t) )
        
        # Increment next time index
        self.t += 1