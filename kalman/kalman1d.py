"""
Missile tracker (alpha–beta filter) in 1D.

command + optioin + F to find and replace in file 

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
EPOCHS = 100
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
Kalman gain: K_k+1 = P_k+1 / (P_k+1 + R)
update estimate: x_k+1^e = x_k+1^p + K_k+1 * (z_k+1 - x_k+1^p)
update uncertianty: P_k+1 = (1 - K_k+1) * P_k+1
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

# ===== LS Path =====
"""
num = n * sum(t*x) - -(sum(t) * sum(x))
den = n * sum(t^2) - (sum(t))^2
slope = num / den
intercept = (sum(x) - slope * sum(t)) / n
prediction = slope * t + intercept
"""

particle_ls_estimate = np.zeros((EPOCHS, 2), dtype=float)

for i in range(EPOCHS):
    sum_t = np.sum(particle_actual[:i+1, 0])
    sum_x = np.sum(particle_actual[:i+1, 1])
    sum_tx = np.sum(particle_actual[:i+1, 0] * particle_actual[:i+1, 1])
    sum_t2 = np.sum(particle_actual[:i+1, 0] ** 2)
   

    if i > 1:
        num = i * sum_tx - sum_t * sum_x
        den = i * sum_t2 - sum_t ** 2
        slope = num / den if den != 0 else 0
        intercept = (sum_x - slope * sum_t) / i if i > 0 else 0
    else:
        slope = 0
        intercept = particle_actual[i][1] if i < EPOCHS else 0

    particle_ls_estimate[i] = [i, slope * i + intercept]



# ===== Difference =====

differences = np.zeros((EPOCHS,2), dtype=float)
error = 0.0
for i in range(EPOCHS):
    differences[i][1] = abs(particle_actual[i][1] - particle_estimate[i][1])
    differences[i][0] = i
    error += differences[i][1]**2
    print(f"Actual: {particle_actual[i][1]:.2f}, Estimate: {particle_estimate[i][1]:.2f}, LS Estimate: {particle_ls_estimate[i][1]:.2f}, KalmanDiff: {differences[i][1]:.2f}")
    
error = np.sqrt(error / EPOCHS)
print(f"RMS error: {error:.2f}")
    
# === Plotting ===

plt.figure()
plt.plot(particle_actual[:, 0], particle_actual[:, 1], label="True trajectory")
plt.plot(particle_estimate[:, 0], particle_estimate[:, 1], label="Estimated trajectory", color="orange")
plt.plot(particle_ls_estimate[:, 0], particle_ls_estimate[:, 1], label="LS Estimate", color="red")
#plt.scatter(r_meas_hist[meas_mask, 0], r_meas_hist[meas_mask, 1], s=18, label="Measurements")
plt.xlabel("x")
plt.ylabel("y")
plt.title("1D Tracking: Truth vs Kalman vs LS")
plt.legend()
plt.grid(True)
plt.show()

#print(f"Actual particle path: {particle_actual} \n      Estimated particle path: {particle_estimate}")