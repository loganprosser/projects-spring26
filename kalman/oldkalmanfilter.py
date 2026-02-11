# ==== Kalman Filter ====
"""
Simple kalman filter: x_(K+1) = x_K + w_k
predicting estimate: x_k+1^p = kx_k^e
uncertianty updates: P_k+1 = P_k + Q
Kalman gain: K_k+1 = P_k+1 / (P_k+1 + R)
update estimate: x_k+1^e = x_k+1^p + K_k+1 * (z_k+1 - x_k+1^p)
update uncertianty: P_k+1 = (1 - K_k+1) * P_k+1
"""

# if OLD_KALMAN:
#     particle_estimate = np.zeros((EPOCHS, 2), dtype=float)
#     initial_estimate = (0.0, 0.0)
#     particle_estimate[0] = initial_estimate
#     #initial uncertianty
#     uncertianty = PROCESS_NOISE + MEASUREMENT_NOISE

#     for i in range(1, EPOCHS):
#         sensor_read = rng.normal(0, np.sqrt(MEASUREMENT_NOISE))
        
#         #prediction
#         _ , x_pred = particle_estimate[i-1]
#         uncertianty = uncertianty + PROCESS_NOISE
#         # Kalman gain
#         K = uncertianty / (uncertianty + MEASUREMENT_NOISE)
#         # Update estimate
#         particle_estimate[i] = (i, particle_estimate[i-1][1] + K * (sensor_read - particle_estimate[i-1][1]))
#         # Update uncertianty
#         uncertianty = (1 - K) * uncertianty + PROCESS_NOISE
        