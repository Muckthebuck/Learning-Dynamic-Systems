import numpy as np
import matplotlib.pyplot as plt

class RadialSearch:
    def __init__(self,
                 n_dim: int = 2,
                 n_vectors: int = 30,
                 ):
        self.n_dim = n_dim
        self.n_vectors = n_vectors
        self.vectors = []

    def _generate_unit_vectors(self):
        basis_vectors = np.eye(self.n_dim)

        self.vectors.extend(basis_vectors)
        
        for _ in range(self.n_vectors):
            rand_vec = np.random.uniform(-1, 1, self.n_dim) * basis_vectors
            rand_vec = np.sum(rand_vec, axis=1)
            rand_vec /= np.linalg.norm(rand_vec)    # Normalise to create unit vector
            self.vectors.extend(np.array([rand_vec]))

        self.vectors = np.array(self.vectors)

    def plot_unit_vectors_2d(self):
        if self.n_dim == 2:
            fig, ax = plt.subplots()
            ax.set_title("Generated Unit Vectors")
            ax.set_xlabel("a")
            ax.set_ylabel("b")

            for i in range(len(self.vectors)):
                ax.plot([0, self.vectors[i, 0]], [0, self.vectors[i, 1]])


if __name__ == "__main__":
    search = RadialSearch()
    search._generate_unit_vectors()
    search.plot_unit_vectors_2d()
    plt.waitforbuttonpress()
