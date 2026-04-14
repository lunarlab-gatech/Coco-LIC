import numpy as np

# Lidar to Camera calibration for FAST-VIO2 (their calibrated results)
Rl2c = np.array([
    [-0.006567, -0.999975,  0.002804],
    [ 0.436679, -0.005390, -0.899601],
    [ 0.899593, -0.004684,  0.436703]
])
Pl2c = np.array([-0.034700, -0.109044, 0.020869])

# Lidar to Camera calibration for FAST-VIO2 (our calibrated results)
# Rl2c = np.array([
#     [-0.008655, -0.999961,  0.001772],
#     [ 0.447172, -0.005455, -0.894432],
#     [ 0.894406, -0.006949,  0.447201]
# ])
# Pl2c = np.array([-0.051469, -0.127918, -0.010792])

# IMU to Lidar 
Ri2l = np.eye(3) 
Ti2l = np.array([ 0.011, 0.02329, -0.04412 ])

# FAST-LIVO2 Transformation Chain
Rl2i = Ri2l.T
Pl2i = -Ri2l.T @ Ti2l

# Step 1: Invert Rl2c to get Camera → LiDAR
Rc2l = Rl2c.T
Pc2l = -Rl2c.T @ Pl2c

# Step 2: Chain Camera → LiDAR → IMU
Rc2i = Rl2i @ Rc2l
Tc2i = (Rl2i @ Pc2l) + Pl2i

# Formatting the output
def format_output(R, T):
    rot_flat = R.flatten()
    print(f"    Trans: [{T[0]:.6f}, {T[1]:.6f}, {T[2]:.6f}]")
    print(f"    Rot: [{rot_flat[0]:.6f}, {rot_flat[1]:.6f}, {rot_flat[2]:.6f},")
    print(f"    {rot_flat[3]:.6f}, {rot_flat[4]:.6f}, {rot_flat[5]:.6f},")
    print(f"    {rot_flat[6]:.6f}, {rot_flat[7]:.6f}, {rot_flat[8]:.6f}]")


print("--- Camera to IMU Extrinsics ---")
format_output(Rc2i, Tc2i)
'''
Their calibrated results:
--- Camera to IMU Extrinsics ---
    Trans: [0.017616, -0.058479, -0.062992]
    Rot: [-0.006567, 0.436679, 0.899593,
    -0.999975, -0.005390, -0.004684,
    0.002804, -0.899601, 0.436703]

Our calibrated results:
--- Camera to IMU Extrinsics ---
    Trans: [0.055408, -0.075530, -0.065377]
    Rot: [-0.008655, 0.447172, 0.894406,
    -0.999961, -0.005455, -0.006949,
    0.001772, -0.894432, 0.447201]
'''