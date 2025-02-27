import numpy as np
import matplotlib.pyplot as plt
import os

class KalmanFilter:
    def __init__(self, F, H, Q, R, P=None, x0=None):
        self.n = F.shape[1]  # Number of state variables
        self.m = H.shape[1]  # Number of measurement variables

        self.F = F  # State transition model
        self.H = H  # Measurement model
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.P = np.eye(self.n) if P is None else P  # Initial covariance
        self.x = np.zeros((self.n, 1)) if x0 is None else x0  # Initial state

    def predict(self):
        self.x = np.dot(self.F, self.x)  # Predict state
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q  # Predict uncertainty
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)  # Measurement residual
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))  # Residual covariance
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Kalman Gain

        self.x = self.x + np.dot(K, y)  # Update state
        I = np.eye(self.n)
        self.P = (I - np.dot(K, self.H)) @ self.P  # Update uncertainty


def example():
    dt = 1.0 / 60  # Time step (60 FPS)
    
    # Extended State: [x, vx, ax, y, vy, ay] (Position, Velocity, Acceleration)
    F = np.array([
        [1, dt, 0.5 * dt**2, 0,  0,  0],  # X position update
        [0, 1,  dt,           0,  0,  0],  # X velocity update
        [0, 0,  1,            0,  0,  0],  # X acceleration update
        [0, 0,  0,            1, dt, 0.5 * dt**2],  # Y position update
        [0, 0,  0,            0, 1,  dt],  # Y velocity update
        [0, 0,  0,            0, 0,  1]   # Y acceleration update
    ])
    
    H = np.array([
        [1, 0, 0, 0, 0, 0],  # We measure only X position
        [0, 0, 0, 1, 0, 0]   # We measure only Y position
    ])

    Q = np.eye(6) * 0.1  # Process noise (tuned for smoothness)
    R = np.eye(2) * 0.5  # Measurement noise

    # Simulate a nonlinear movement (player accelerating, then turning)
    time = np.linspace(0, 20, 100)
    x_positions = np.sin(time) * 10 + 0.5 * time**2  # X: Sin wave + acceleration
    y_positions = np.cos(time) * 5 + 2 * time       # Y: Curve with upward motion
    measurements = np.vstack((x_positions, y_positions)) + np.random.normal(0, 0.5, (2, 100))

    kf = KalmanFilter(F=F, H=H, Q=Q, R=R)
    predictions = []

    for z in measurements.T:
        pred = np.dot(H, kf.predict()).flatten()  # Predict position
        predictions.append(pred)
        kf.update(z.reshape(2, 1))  # Update with measurement

    predictions = np.array(predictions)
    
    # Plot results
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))
    
    axs[0].plot(measurements[0], measurements[1], 'bo', markersize=3, alpha=0.5, label='Measurements')
    axs[0].plot(predictions[:, 0], predictions[:, 1], 'r-', label='Kalman Filter Prediction')
    axs[0].scatter(predictions[0, 0], predictions[0, 1], color='green', s=100, label='Start')
    axs[0].scatter(predictions[-1, 0], predictions[-1, 1], color='red', s=100, label='End')
    axs[0].set_xlabel("X Position")
    axs[0].set_ylabel("Y Position")
    axs[0].legend()
    axs[0].set_title("Player Movement on Field with Acceleration")

    axs[1].plot(time, predictions[:, 0], 'g-', label='X Position Over Time')
    axs[1].scatter(time[0], predictions[0, 0], color='green', s=100, label='Start')
    axs[1].scatter(time[-1], predictions[-1, 0], color='red', s=100, label='End')
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("X Position")
    axs[1].legend()
    axs[1].set_title("X Position Over Time (With Acceleration)")

    plt.tight_layout()
    
    output_folder = "output_images"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    plt.savefig(os.path.join(output_folder, "kalman_filter_acceleration.png"))
    plt.show()

if __name__ == '__main__':
    example()
