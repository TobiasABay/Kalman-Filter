import numpy as np
import matplotlib.pyplot as plt
import os

class KalmanFilter(object):
    def __init__(self, F=None, B=None, H=None, Q=None, R=None, P=None, x0=None):
        if F is None or H is None:
            raise ValueError("Set proper system dynamics.")
        
        self.n = F.shape[1]
        self.m = H.shape[1]
        
        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.m) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u=0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
                         (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)

def example():
    dt = 1.0 / 60
    F = np.array([[1, dt, 0, 0, 0], [0, 1, dt, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, dt], [0, 0, 0, 0, 1]])
    H = np.array([[1, 0, 0, 0, 0], [0, 0, 0, 1, 0]])
    Q = np.eye(5) * 0.05
    R = np.eye(2) * 0.5

    time = np.linspace(0, 10, 100)
    x_positions = np.sin(time) * 10  # Simulating left to right movement
    y_positions = np.cos(time) * 5   # Simulating slight vertical movement
    measurements = np.vstack((x_positions, y_positions)) + np.random.normal(0, 0.5, (2, 100))

    kf = KalmanFilter(F=F, H=H, Q=Q, R=R)
    predictions = []

    for z in measurements.T:
        pred = np.dot(H, kf.predict()).flatten()
        predictions.append(pred)
        kf.update(z.reshape(2, 1))

    predictions = np.array(predictions)
    
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))
    
    axs[0].plot(measurements[0], measurements[1], 'bo', markersize=3, alpha=0.5, label='Measurements')
    axs[0].plot(predictions[:, 0], predictions[:, 1], 'r-', label='Kalman Filter Prediction')
    axs[0].scatter(predictions[0, 0], predictions[0, 1], color='green', s=100, label='Start')
    axs[0].scatter(predictions[-1, 0], predictions[-1, 1], color='red', s=100, label='End')
    axs[0].set_xlabel("X Position")
    axs[0].set_ylabel("Y Position")
    axs[0].legend()
    axs[0].set_title("Player Movement on Field")
    
    axs[1].plot(time, predictions[:, 0], 'g-', label='X Position Over Time')
    axs[1].scatter(time[0], predictions[0, 0], color='green', s=100, label='Start')
    axs[1].scatter(time[-1], predictions[-1, 0], color='red', s=100, label='End')
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("X Position")
    axs[1].legend()
    axs[1].set_title("X Position Over Time (Parabolic Movement)")
    
    plt.tight_layout()
    
    output_folder = "output_images"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    plt.savefig(os.path.join(output_folder, "kalman_filter_graph.png"))
    plt.show()

if __name__ == '__main__':
    example()