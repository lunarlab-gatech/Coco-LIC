import numpy as np

# Lidar to Camera calibration for FAST-VIO2
Rcl = np.array(
      [ -0.008381, -0.999963,  0.001992,
       0.445510, -0.005517, -0.895260,
       0.895238, -0.006616,  0.445540]
).reshape(3, 3).transpose()
Pcl = np.array([ -0.052774, -0.128613, -0.014642])

# IMU to Lidar in lidar.yaml (same as MID-360 datasheet)
R_il = np.eye(3) 
T_il = np.array([ -0.011, -0.02329, 0.04412 ])

# FAST-LIVO2 Transformation Chain
Rli = R_il.T
Pli = -R_il.T @ T_il

Rci = Rcl @ Rli
Pci = (Rcl @ Pli) + Pcl

# Calculate Camera to IMU (Inverse)
R_ic = Rci.T
P_ic = -Rci.T @ Pci

# Formatting the output
def format_output(R, T):
    rot_flat = R.flatten()
    print(f"    Trans: [{T[0]:.6f}, {T[1]:.6f}, {T[2]:.6f}]")
    print(f"    Rot: [{rot_flat[0]:.6f}, {rot_flat[1]:.6f}, {rot_flat[2]:.6f},")
    print(f"    {rot_flat[3]:.6f}, {rot_flat[4]:.6f}, {rot_flat[5]:.6f},")
    print(f"    {rot_flat[6]:.6f}, {rot_flat[7]:.6f}, {rot_flat[8]:.6f}]")


print("--- Camera to IMU Extrinsics ---")
format_output(R_ic, P_ic)