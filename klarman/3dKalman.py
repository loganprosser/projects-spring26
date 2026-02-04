"""
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
TIME_STEP = .25
EPOCHS = 200
P_MEASURE = 0.60 # probability of taking a measurement
ALPHA = 0.8
BETA = 0.05

# Actual position arrays (constant velocity + random position drift)
r_true = np.array([0.0, 0.0, 0.0], dtype=float)
v_true = np.array([2.2, 0.8, 0.35], dtype=float)
sigma_process_pos = 0.8

# Sensor noise (like uncertainty in measurement)
sigma_measure = 1.0

# Initial position arrays
r_est = np.array([0.0, 0.0, 0.0], dtype=float)
v_est = np.array([0.0, 0.0, 0.0], dtype=float)

r_true_hist = np.zeros((EPOCHS, 3), dtype=float)
r_meas_hist = np.full((EPOCHS, 3), np.nan, dtype=float)  # FIX
r_est_hist = np.zeros((EPOCHS, 3), dtype=float)
meas_mask = np.zeros(EPOCHS, dtype=bool)

rng = np.random.default_rng(7)

# run algo on random movement
for k in range(EPOCHS):
    # True position update numpy random 
    r_true = r_true + v_true * TIME_STEP + rng.normal(0.0, sigma_process_pos, size=3)
    r_true_hist[k] = r_true

    # measurement with gaussinan error
    r_meas = None
    if rng.random() < P_MEASURE:
        r_meas = r_true + rng.normal(0.0, sigma_measure, size=3)
        r_meas_hist[k] = r_meas
        meas_mask[k] = True

    # predictions
    r_pred = r_est + v_est * TIME_STEP
    v_pred = v_est

    # update setp
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

#chat gpt plots...

# Plots
fig = plt.figure(figsize=(15, 6), constrained_layout=True)
gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0])

# Left side 3D trajectory
ax3d = fig.add_subplot(gs[0, 0], projection="3d")
ax3d.plot(r_true_hist[:, 0], r_true_hist[:, 1], r_true_hist[:, 2], label="True trajectory")
ax3d.plot(r_est_hist[:, 0],  r_est_hist[:, 1],  r_est_hist[:, 2], label="Estimated trajectory")

ax3d.scatter(
    r_meas_hist[meas_mask, 0],
    r_meas_hist[meas_mask, 1],
    r_meas_hist[meas_mask, 2],
    s=18,
    label="Measurements"
)

ax3d.set_xlabel("x")
ax3d.set_ylabel("y")
ax3d.set_zlabel("z")
ax3d.set_title("3D Tracking: Truth vs Measurements vs Estimate")
ax3d.legend()

# Right siude error over time
axerr = fig.add_subplot(gs[0, 1])
axerr.plot(pos_err)
axerr.set_xlabel("time step k")
axerr.set_ylabel("||position error||")
axerr.set_title(f"Position Error Over Time (RMSE={rmse:.3f})")
axerr.grid(True)

plt.show()