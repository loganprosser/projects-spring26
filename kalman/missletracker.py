"""
original
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

# Constants
TIME_STEP = 1.0
EPOCHS = 200
P_MEASURE = 0.55
ALPHA = 0.35
BETA = 0.05

# Truth model (constant velocity + random position drift)
r_true = np.array([0.0, 0.0], dtype=float)
v_true = np.array([2.2, 0.8], dtype=float)
sigma_process_pos = 0.8

# Sensor noise
sigma_measure = 2.0

# Initial position
r_est = np.array([0.0, 0.0], dtype=float)
v_est = np.array([0.0, 0.0], dtype=float)

r_true_hist = np.zeros((EPOCHS, 2), dtype=float)
r_meas_hist = np.full((EPOCHS, 2), np.nan, dtype=float)  # FIX
r_est_hist = np.zeros((EPOCHS, 2), dtype=float)
meas_mask = np.zeros(EPOCHS, dtype=bool)

rng = np.random.default_rng(7)

for k in range(EPOCHS):
    # True position update numpy random 
    r_true = r_true + v_true * TIME_STEP + rng.normal(0.0, sigma_process_pos, size=2)
    r_true_hist[k] = r_true

    # measurement with gaussinan error
    r_meas = None
    if rng.random() < P_MEASURE:
        r_meas = r_true + rng.normal(0.0, sigma_measure, size=2)
        r_meas_hist[k] = r_meas
        meas_mask[k] = True

    # --- Predict ---
    r_pred = r_est + v_est * TIME_STEP
    v_pred = v_est

    # --- Update ---
    if r_meas is not None:
        e = r_meas - r_pred
        r_est = r_pred + ALPHA * e
        v_est = v_pred + (BETA / TIME_STEP) * e
    else:
        r_est = r_pred
        v_est = v_pred

    r_est_hist[k] = r_est

# Error calc
pos_err = np.linalg.norm(r_est_hist - r_true_hist, axis=1)
rmse = float(np.sqrt(np.mean(pos_err**2)))
print(f"RMSE position error: {rmse:.3f}")
print(f"Measurement availability: {meas_mask.mean()*100:.1f}%")


#chat gpt plots

# Plots

#trajectory
plt.figure()
plt.plot(r_true_hist[:, 0], r_true_hist[:, 1], label="True trajectory")
plt.plot(r_est_hist[:, 0], r_est_hist[:, 1], label="Estimated trajectory")
plt.scatter(r_meas_hist[meas_mask, 0], r_meas_hist[meas_mask, 1], s=18, label="Measurements")
plt.xlabel("x")
plt.ylabel("y")
plt.title("2D Tracking: Truth vs Measurements vs Estimate")
plt.legend()
plt.grid(True)
plt.show()

#error over time
plt.figure()
plt.plot(pos_err)
plt.xlabel("time step k")
plt.ylabel("||position error||")
plt.title("Position Error Magnitude Over Time")
plt.grid(True)
plt.show()
