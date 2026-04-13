#!/usr/bin/env python3
"""Plot continuous (100Hz) and discrete (image-frame) trajectories from Coco-LIC output."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial.transform import Rotation
import rosbag


def load_tum_trajectory(path):
    """Load TUM-format trajectory: t x y z qx qy qz qw."""
    tum = np.loadtxt(path)
    return tum[:, 0], tum[:, 1], tum[:, 2], tum[:, 3]


def load_colmap_images(path):
    """Load camera positions from COLMAP images.txt (world-to-camera convention).

    Inverts each pose to get the camera center in world frame.
    Returns timestamps (seconds, relative) and world-frame positions.
    """
    t_list, x_list, y_list, z_list = [], [], [], []
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    for i in range(0, len(lines), 2):  # every other line is empty (2D points)
        parts = lines[i].split()
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])

        # Invert world-to-camera to get camera center in world frame
        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()  # scipy uses [x,y,z,w]
        t_vec = np.array([tx, ty, tz])
        cam_pos = -R.T @ t_vec

        # Extract relative timestamp from image name: frame_<ns>.png
        ns = int(parts[9].replace("frame_", "").replace(".png", ""))
        t_list.append(ns * 1e-9)
        x_list.append(cam_pos[0])
        y_list.append(cam_pos[1])
        z_list.append(cam_pos[2])
    return np.array(t_list), np.array(x_list), np.array(y_list), np.array(z_list)


def load_rosbag_image_times(bag_path, image_topic):
    """Extract image message timestamps from a rosbag."""
    bag = rosbag.Bag(bag_path)
    img_times = []
    for _, _, t in bag.read_messages(topics=[image_topic]):
        img_times.append(t.to_sec())
    bag.close()
    return np.array(img_times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="./data", help="Directory containing output files")
    parser.add_argument("--prefix", default="data_0", help="File prefix (e.g. data_0)")
    parser.add_argument("--mode", default="LICO", help="Trajectory mode suffix (e.g. LICO, LIO)")
    parser.add_argument("--bag", default=None, help="Path to rosbag (optional, for rosbag timestamps)")
    parser.add_argument("--image-topic", default="/left_camera/image/compressed",
                        help="Image topic in rosbag")
    parser.add_argument("--no-3d", action="store_true", help="Skip 3D plot")
    args = parser.parse_args()

    d = args.data_dir
    p = args.prefix

    # --- Continuous trajectory (TUM, 100Hz) ---
    tum_path = f"{d}/{p}_{args.mode}.txt"
    t_cont, x_cont, y_cont, z_cont = load_tum_trajectory(tum_path)
    print(f"Loaded continuous trajectory: {len(t_cont)} poses from {tum_path}")

    # --- Discrete trajectory at image-frame timestamps (COLMAP) ---
    colmap_path = f"{d}/{p}_images.txt"
    t_img, x_img, y_img, z_img = load_colmap_images(colmap_path)
    print(f"Loaded image-frame poses: {len(t_img)} frames from {colmap_path}")

    # --- Optional: discrete trajectory at rosbag timestamps ---
    x_bag, y_bag, z_bag = None, None, None
    if args.bag:
        bag_times = load_rosbag_image_times(args.bag, args.image_topic)
        x_bag = np.interp(bag_times, t_cont, x_cont)
        y_bag = np.interp(bag_times, t_cont, y_cont)
        z_bag = np.interp(bag_times, t_cont, z_cont)
        print(f"Interpolated rosbag image timestamps: {len(bag_times)} frames")

    # --- 3D plot ---
    if not args.no_3d:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(x_cont, y_cont, z_cont, "b-", linewidth=1, label="Continuous (IMU frame)")
        ax.scatter(x_img, y_img, z_img, marker="*", c="r", s=50, label="Image frames (camera frame)")
        if x_bag is not None:
            ax.scatter(x_bag, y_bag, z_bag, marker="o", c="g", s=30, label="Rosbag timestamps")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend()
        ax.set_title("Coco-LIC Trajectory (3D)")
        plt.tight_layout()
        plt.savefig(f"{d}/trajectory_3d.png", dpi=150)
        print(f"Saved {d}/trajectory_3d.png")

    # --- 2D plot (top-down XY) ---
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(x_cont, y_cont, "b-", linewidth=1, label="Continuous (IMU frame)")
    ax.plot(x_img, y_img, "r*", markersize=8, label="Image frames (camera frame)")
    if x_bag is not None:
        ax.plot(x_bag, y_bag, "go", markersize=4, label="Rosbag timestamps")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend()
    ax.set_title("Coco-LIC Trajectory (top-down)")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(f"{d}/trajectory_2d.png", dpi=150)
    print(f"Saved {d}/trajectory_2d.png")

    plt.show()


if __name__ == "__main__":
    main()
