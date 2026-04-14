import numpy as np

#########################################################
# Lidar to Camera calibration by FAST-Calib
# VNav (Feb 19th, 2026)
# Rcl = np.array(
#     [-0.006567, -0.999975,  0.002804,
#        0.436679, -0.005390,  -0.899601,
#        0.899593, -0.004684,  0.436703]
# ).reshape(3, 3)
# Pcl = np.array([-0.034700, -0.109044, 0.020869])

# ours (Apr 12th, 2026)
Rcl = np.array(
     [ -0.008655, -0.999961,  0.001772,
       0.447172, -0.005455, -0.894432,
       0.894406, -0.006949,  0.447201]
).reshape(3, 3)
Pcl = np.array([-0.051469, -0.127918, -0.010792])
#########################################################

# Lidar to Camera
Rl2c = Rcl
Pl2c = Pcl

# IMU to Lidar (MID-360 datasheet)
Ri2l = np.eye(3) 
Ti2l = np.array([0.011, 0.02329, -0.04412])

# Camera to LiDAR
Rc2l = Rl2c.T
Pc2l = -Rc2l @ Pl2c

# LiDAR to IMU
Rl2i = Ri2l.T
Pl2i = -Rl2i @ Ti2l

# Camera to IMU (Chain: C -> L -> I)
Rc2i = Rl2i @ Rc2l
Pc2i = (Rl2i @ Pc2l) + Pl2i

# Formatting the output (camera to imu(in lidar))
def format_output(R, T):
    rot_flat = R.flatten()
    print(f"    Trans: [{T[0]:.6f}, {T[1]:.6f}, {T[2]:.6f}]")
    print(f"    Rot: [{rot_flat[0]:.6f}, {rot_flat[1]:.6f}, {rot_flat[2]:.6f},")
    print(f"          {rot_flat[3]:.6f}, {rot_flat[4]:.6f}, {rot_flat[5]:.6f},")
    print(f"          {rot_flat[6]:.6f}, {rot_flat[7]:.6f}, {rot_flat[8]:.6f}]")

print("--- Camera to IMU Extrinsics ---")
format_output(Rc2i, Pc2i)