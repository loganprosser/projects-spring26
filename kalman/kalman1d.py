"""
Particle tracker (alpha–beta filter) in 1D.

#command + optioin + F to find and replace in file

# numpy uses rng.normal(mean, stddev, size) to generate gaussian noise
# need to pass sqrt(variance) as stddev to get correct noise level

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
PROCESS_NOISE = 1
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
    # random step with some nonlinearity and noise fits well for PROCESS_NOISE = 1.0
    step = DILUTION * rng.choice(SIGNS) * rng.integers(1, 5)**rng.uniform(1, 2)
    # gaussian noise with variance equal to process noise
    #step = DILUTION * rng.choice(SIGNS) * rng.integers(1, 5)**rng.normal(1.5,1.0)
    particle_actual[i] = [i, particle_actual[i-1, 1] + step]
    
    # num = rng.integers(1, 5)
    # power = rng.uniform(1, 2)
    # sign = rng.choice(SIGNS)
    # move = DILUTION * sign * num**power
    # #particle_actual[i-1, 1] if i > 0 else 1
    # particle_actual[i] = [i, move]

# ===== Explicit Kalman Filter =====

particle_kalman_estimate = np.zeros((EPOCHS, 2), dtype=float)
initial_estimate = (0.0, 0.0)
uncertianty = 0.0 # since initial state is exactly known we can start with 0 uncertianty but could also start with some initial uncertianty if we wanted to be more realistic
particle_kalman_estimate[0] = initial_estimate

# can defintely write to be way better space / memory but want to keep explicit representation of kalman math
for i in range(1, EPOCHS):
    # z_t = x_t + gaussian noise
    sensor_read = particle_actual[i][1] + rng.normal(0, np.sqrt(MEASUREMENT_NOISE))
    # liklihood P(z_k | x_k) = P(z_k - x_k)
    likelihood = (sensor_read, MEASUREMENT_NOISE)
    
    # prior P(x_k | z_{1:k-1})
    prior = (particle_kalman_estimate[i-1][1], uncertianty + PROCESS_NOISE) # need to add process noise Q here not sure why
    
    # posterior P(x_k | z_{1:k}) ~ P(z_k | x_k) * P(x_k | z_{1:k-1})
    # Posterior variance = (1/P_{k|k-1} + 1/R)^{-1}
    variance = 1.0 / (1.0/prior[1] + 1.0/likelihood[1])
    
    # posterior mean = (x^_{k|k-1}/P_{k|k-1} + z_k/R) * variance
    mean = (prior[0]/prior[1] + likelihood[0]/likelihood[1]) * variance
    
    particle_kalman_estimate[i] = (i, mean)
    # posterior goes to prior for next epoch
    uncertianty = variance


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
    differences[i][1] = abs(particle_actual[i][1] - particle_kalman_estimate[i][1])
    differences[i][0] = i
    error += differences[i][1]**2
    print(f"Actual: {particle_actual[i][1]:.2f}, Estimate: {particle_kalman_estimate[i][1]:.2f}, LS Estimate: {particle_ls_estimate[i][1]:.2f}, KalmanDiff: {differences[i][1]:.2f}")
    
error = np.sqrt(error / EPOCHS)
print(f"RMS error: {error:.2f}")
    
# === Plotting ===

plt.figure()
plt.plot(particle_actual[:, 0], particle_actual[:, 1], label="True trajectory")
plt.plot(particle_kalman_estimate[:, 0], particle_kalman_estimate[:, 1], label="Kalman Estimate", color="red")
plt.plot(particle_ls_estimate[:, 0], particle_ls_estimate[:, 1], label="LS Estimate", color="purple")
#plt.scatter(r_meas_hist[meas_mask, 0], r_meas_hist[meas_mask, 1], s=18, label="Measurements")
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"1D Tracking: Truth vs Kalman vs LS [RMS error: {error:.2f}]")
plt.legend()
plt.grid(True)
plt.show()

#print(f"Actual particle path: {particle_actual} \n      Estimated particle path: {particle_estimate}")