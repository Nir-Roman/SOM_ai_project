import numpy as np
import matplotlib.pyplot as plt

class SOM:
    def __init__(self, m, n, input_dim, alpha, num_iterations):
        self.m = m
        self.n = n
        self.input_dim = input_dim
        self.alpha = alpha
        self.num_iterations = num_iterations
        self.weights = np.zeros((input_dim, n))

    def find_winning_neuron(self, input):
        d = np.zeros(self.n)  # 2

        for j in range(self.n):  # 2
            error = 0
            for i in range(self.input_dim):  # 4
                error += ((self.weights[i, j] - input[i])**2)

            d[j] = error
        return np.argmin(d)

    def update_weights(self, bmu, input):
        self.weights[:, bmu] += self.alpha * (input - self.weights[:, bmu])
    
    def train(self, data):
        for iteration in range(self.num_iterations):
            for data_point in data:
                bmu = self.find_winning_neuron(data_point)
                self.update_weights(bmu, data_point)
    
    def map_vects(self, data):
        mapped = np.array([self.find_winning_neuron(data_point) for data_point in data])
        return mapped

    def plot_results(self, data, mapped_clusters):
        plt.figure(figsize=(8, 6))
        
        # Plot input vectors
        scatter = plt.scatter(data[:, 0], data[:, 1], c=mapped_clusters, cmap='viridis', s=100, edgecolor='k')
        
        # Plot weights (SOM neurons)
        for i in range(self.n):
            plt.scatter(self.weights[0, i], self.weights[1, i], c='red', marker='x', s=200, label=f'Neuron {i+1}')
        
        plt.title('SOM Clustering of 2D Vectors')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True)
        plt.show()

x1 = np.array([0, 0, 1, 1])
x2 = np.array([1, 0, 0, 0])
x3 = np.array([0, 1, 1, 0])
x4 = np.array([0, 0, 0, 1])


########################## For 2D points Clustering
# x1 = np.array([0, 0])
# x2 = np.array([0, 2])
# x3 = np.array([1, 0])
# x4 = np.array([1, 2])

data = np.array([x1, x2, x3, x4])

# create som grid of 1x2 grid for 2 clusters
som = SOM(m=1, n=2, input_dim=4, alpha=0.5, num_iterations=100)

# Train the SOM
som.train(data)

# Display the Final Weights after 100 iterations
print("Final Weights")
print(som.weights)

# Map the data
mapped_clusters = som.map_vects(data)
print("Mapped clusters:", mapped_clusters)

# Plot the results
som.plot_results(data, mapped_clusters)