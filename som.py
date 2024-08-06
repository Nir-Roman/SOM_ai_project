# som_algorithm.py
import numpy as np

class SOM:
    def __init__(self, x, y, input_len, sigma=1.0, learning_rate=0.5):
        self.x = x
        self.y = y
        self.input_len = input_len
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.weights = np.random.rand(x, y, input_len)
        self.initial_learning_rate = learning_rate
        self.initial_sigma = sigma

    def train(self, data, num_iterations):
        for i in range(num_iterations):
            print(f"go for it{num_iterations}")
            count=0
            for sample in data:
                print(f"I am going bro{count}")
                bmu_idx = self._find_bmu(sample)
                self._update_weights(sample, bmu_idx, i, num_iterations)
                count=count+1

    def _find_bmu(self, sample):
        differences = self.weights - sample
        distances = np.linalg.norm(differences, axis=2)
        return np.unravel_index(np.argmin(distances), distances.shape)

    def _update_weights(self, sample, bmu_idx, iteration, num_iterations):
        learning_rate = self._decay_function(self.initial_learning_rate, iteration, num_iterations)
        sigma = self._decay_function(self.initial_sigma, iteration, num_iterations)
        bmu_x, bmu_y = bmu_idx

        for x in range(self.x):
            for y in range(self.y):
                distance_to_bmu = np.linalg.norm(np.array([bmu_x - x, bmu_y - y]))
                if distance_to_bmu <= sigma:
                    influence = np.exp(-distance_to_bmu**2 / (2 * sigma**2))
                    self.weights[x, y] += influence * learning_rate * (sample - self.weights[x, y])

    def _decay_function(self, initial_value, iteration, num_iterations):
        return initial_value * np.exp(-iteration / num_iterations)

    def map_data(self, data):
        return np.array([self._find_bmu(d) for d in data])

    def calculate_distances(self, data):
        return np.array([np.linalg.norm(d - self.weights[self._find_bmu(d)]) for d in data])

    def get_weights(self):
        return self.weights
