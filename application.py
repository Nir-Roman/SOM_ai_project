import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from som import SOM

class NetworkAnomalyDetection:
    def __init__(self, x, y, input_len, sigma=1.0, learning_rate=0.5, threshold=1.0):
        self.som = SOM(x, y, input_len, sigma, learning_rate)
        self.threshold = threshold

    def train_som(self, data, num_iterations):
        self.som.train(data, num_iterations)

    def detect_anomalies(self, data):
        distances = self.som.calculate_distances(data)
        anomalies = distances > self.threshold
        return anomalies, distances

    def visualize_results(self, data, anomalies):
        mapped = self.som.map_data(data)
        plt.figure(figsize=(10, 10))
        plt.scatter(mapped[:, 0], mapped[:, 1], c='blue', label='Normal Data')
        plt.scatter(mapped[anomalies, 0], mapped[anomalies, 1], c='red', label='Anomalies')
        plt.legend()
        plt.title('SOM Network Anomaly Detection')
        plt.show()

    def plot_som(self, data, title='SOM'):
        mapped = self.som.map_data(data)
        plt.figure(figsize=(10, 10))
        plt.scatter(mapped[:, 0], mapped[:, 1], s=10)
        plt.title(title)
        plt.xlabel('SOM X')
        plt.ylabel('SOM Y')
        plt.grid()
        plt.show()

# Function to load preprocessed data from CSV and drop the last column
def load_preprocessed_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(data.columns[-1], axis=1).values  # Drop the last column
    y = data[data.columns[-1]].values               # Extract the last column as labels

    # Normalize labels: Convert all anomaly labels (1, 2, 3) to 1
    y = np.where(y > 0, 1, 0)
    
    return X, y

# Example usage
if __name__ == "__main__":
    file_path = 'web_attack.csv'
    

    # Load preprocessed data
    X, y = load_preprocessed_data(file_path)
    print("file loaded")
    

    # Separate normal and anomaly data
    normal_data = X[y == 0]
    anomaly_data = X[y == 1]
    combined_data = np.vstack((normal_data, anomaly_data))

    # Initialize and train the NetworkAnomalyDetection
    nad = NetworkAnomalyDetection(x=10, y=10, input_len=combined_data.shape[1], sigma=1.0, learning_rate=0.5, threshold=1.0)
    print("initialized")
    nad.train_som(normal_data, num_iterations=100)
    print("traninig started")

    # Detect anomalies
    anomalies_detected, distances = nad.detect_anomalies(combined_data)

    # Visualize results
    nad.visualize_results(combined_data, anomalies_detected)

    # Plot SOM map
    nad.plot_som(combined_data, title='SOM Map')
