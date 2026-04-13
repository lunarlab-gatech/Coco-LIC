import numpy as np

#########################################################
# Lidar to Camera calibration by FAST-Calib
Rcl = np.array(
    [-0.006567, -0.999975,  0.002804,
       0.436679, -0.005390,  -0.899601,
       0.899593, -0.004684,  0.436703]
).reshape(3, 3)
Pcl = np.array([-0.034700, -0.109044, 0.020869])
#########################################################

# IMU to Lidar in lidar.yaml (same as MID-360 datasheet)
R_il = np.eye(3) 
T_il = np.array([ 0.011, 0.02329, -0.04412 ])

# FAST-LIVO2 Transformation Chain
Rli = R_il.T
Pli = -R_il.T @ T_il

Rci = Rcl @ Rli
Pci = (Rcl @ Pli) + Pcl

# Calculate Camera to IMU (Inverse)
R_ic = Rci.T
P_ic = -Rci.T @ Pci

# Formatting the output (camera to imu(in lidar))
def format_output(R, T):
    rot_flat = R.flatten()
    print(f"    Trans: [{T[0]:.6f}, {T[1]:.6f}, {T[2]:.6f}]")
    print(f"    Rot: [{rot_flat[0]:.6f}, {rot_flat[1]:.6f}, {rot_flat[2]:.6f},")
    print(f"    {rot_flat[3]:.6f}, {rot_flat[4]:.6f}, {rot_flat[5]:.6f},")
    print(f"    {rot_flat[6]:.6f}, {rot_flat[7]:.6f}, {rot_flat[8]:.6f}]")


print("--- Camera to IMU Extrinsics ---")
format_output(R_ic, P_ic)