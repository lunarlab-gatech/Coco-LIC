import numpy as np

def quaternion_to_matrix(q):
    """
    Converts a quaternion [x, y, z, w] to a 3x3 rotation matrix.
    """
    x, y, z, w = q
    
    # First, normalize the quaternion to ensure it's a valid rotation
    norm = np.sqrt(w**2 + x**2 + y**2 + z**2)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm

    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2]
    ])

# Values from the 'T_lidar_camera' in your image [x, y, z, w]
q_image = [0.50704408000427983, -0.502229193422708, 0.49526714567233776, -0.49536107293114834]
matrix = quaternion_to_matrix(q_image)

print(matrix)