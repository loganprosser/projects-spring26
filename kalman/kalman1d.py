"""
Missile tracker (alpha–beta filter) in 2D.

State estimate: position r_est and velocity v_est
Prediction:
    r_pred = r_est + v_est * dt
Update (when measurement r_meas exists):
    residual = r_meas - r_pred
    r_est = r_pred + alpha * residual
    v_est = v_est + (beta / dt) * residual
"""

import numpy as np
import matplotlib.pyplot as plt

# ====== Constants =======
PROCESS_NOISE = 1.0
MEASUREMENT_NOISE = 2.0

x = 1
SLOPES = [x, x**2, x**3]
EPOCHS = 20
DILUTION = .1

# ====== random particle path ======

particle_actual = np.zeros((EPOCHS, 2), dtype=float)
rng = np.random.default_rng()
SIGNS = [-1, 1]

particle_actual[0] = [0, 0]

for i in range(1, EPOCHS):
    num = rng.integers(1, 5)
    power = rng.uniform(1, 2)
    sign = rng.choice(SIGNS)
    move = DILUTION * sign * num**power
    #particle_actual[i-1, 1] if i > 0 else 1
    particle_actual[i] = [i, move]

# ==== Kalman Filter ====
"""
Simple kalman filter: x_(K+1) = x_K + w_k
predicting estimate: x_k+1^p = kx_k^e
uncertianty updates: P_k+1 = P_k + Q


"""

particle_estimate = np.zeros((EPOCHS, 2), dtype=float)
initial_estimate = (0.0, 0.0)
particle_estimate[0] = initial_estimate
#initial uncertianty
uncertianty = PROCESS_NOISE + MEASUREMENT_NOISE


for i in range(1, EPOCHS):
    sensor_read = rng.normal(0, np.sqrt(MEASUREMENT_NOISE))
    
    #prediction
    _ , x_pred = particle_estimate[i-1]
    uncertianty = uncertianty + PROCESS_NOISE
    # Kalman gain
    K = uncertianty / (uncertianty + MEASUREMENT_NOISE)
    # Update estimate
    particle_estimate[i] = (i, particle_estimate[i-1][1] + K * (sensor_read - particle_estimate[i-1][1]))
    # Update uncertianty
    uncertianty = (1 - K) * uncertianty + PROCESS_NOISE

# ===== Difference =====

differences = np.zeros((EPOCHS,2), dtype=float)
error = 0.0
for i in range(EPOCHS):
    differences[i][1] = abs(particle_actual[i][1] - particle_estimate[i][1])
    differences[i][0] = i
    error += differences[i][1]**2
    print(f"Actual: {particle_actual[i][1]:.2f}, Estimate: {particle_estimate[i][1]:.2f}, Difference: {differences[i][1]:.2f}")
    
error = np.sqrt(error / EPOCHS)
print(f"RMS error: {error:.2f}")
    
# === Plotting ===

plt.figure()
plt.plot(particle_actual[:, 0], particle_actual[:, 1], label="True trajectory")
plt.plot(particle_estimate[:, 0], particle_estimate[:, 1], label="Estimated trajectory", color="orange")
#plt.scatter(r_meas_hist[meas_mask, 0], r_meas_hist[meas_mask, 1], s=18, label="Measurements")
plt.xlabel("x")
plt.ylabel("y")
plt.title("2D Tracking: Truth vs Measurements vs Estimate")
plt.legend()
plt.grid(True)
plt.show()


#print(f"Actual particle path: {particle_actual} \n      Estimated particle path: {particle_estimate}")