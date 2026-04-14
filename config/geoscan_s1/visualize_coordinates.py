import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ---------------------------------------------------------------------------
# Sensor extrinsics (all relative to IMU)
# ---------------------------------------------------------------------------

# Lidar to Camera calibration for FAST-VIO2
Rcl = np.array([
[-0.006567, -0.999975, 0.002804],
[ 0.436679, -0.005390, -0.899601],
[ 0.899593, -0.004684, 0.436703]
])

Pcl = np.array([-0.034700, -0.109044, 0.020869])

# IMU to Lidar
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

# ---------------------------------------------------------------------------
# Sensor extrinsics  (all relative to IMU)
# ---------------------------------------------------------------------------

# IMU — reference frame at origin
R_imu = np.eye(3)
t_imu = np.zeros(3)


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t
    return T


# Then update your sensors list:
sensors = [
    {"name": "IMU",    "T": make_T(R_imu, t_imu), "marker": "o", "mcolor": "black"},
    {"name": "LiDAR",  "T": make_T(Rli, Pli),                 "marker": "s", "mcolor": "purple"},
    {"name": "Camera", "T": make_T(R_ic, P_ic),             "marker": "^", "mcolor": "darkorange"},
]


AXIS_COLORS = ["red", "limegreen", "royalblue"]   # X, Y, Z
AXIS_NAMES  = ["X",   "Y",         "Z"]

# ---------------------------------------------------------------------------
# Determine a common axis scale so arrows are proportional to sensor spread
# ---------------------------------------------------------------------------
origins = np.array([s["T"][:3, 3] for s in sensors])
spread  = max(np.ptp(origins, axis=0).max(), 1e-3)
ARROW_LEN = spread * 0.55          # arrows ~55 % of sensor spread


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_frame_3d(ax, T, name, arrow_len):
    origin = T[:3, 3]
    for i, (col, lbl) in enumerate(zip(AXIS_COLORS, AXIS_NAMES)):
        d = T[:3, i]                       # i-th column = axis direction
        ax.quiver(*origin, *d,
                  length=arrow_len, normalize=True,
                  color=col, linewidth=2.5, arrow_length_ratio=0.2)
        tip = origin + arrow_len * 1.12 * d
        ax.text(*tip, lbl, color=col, fontsize=7, ha="center", va="center",
                fontweight="bold")
    ax.scatter(*origin, s=60, zorder=5, color="white", edgecolors="black", linewidths=1.5)


def draw_frame_2d(ax, T, name, arrow_len, xi, yi, xlbl, ylbl):
    """Project the 3-D frame onto a 2-D plane by picking axis indices xi, yi."""
    origin = T[:3, 3][[xi, yi]]
    for i, (col, lbl) in enumerate(zip(AXIS_COLORS, AXIS_NAMES)):
        d = T[:3, i][[xi, yi]]
        if np.linalg.norm(d) < 1e-9:
            continue
        d_n = d / np.linalg.norm(d)
        ax.annotate(
            "", xy=origin + arrow_len * d_n,
            xytext=origin,
            arrowprops=dict(arrowstyle="-|>", color=col,
                            lw=2.0, mutation_scale=14),
        )
        tip = origin + arrow_len * 1.18 * d_n
        ax.text(*tip, f"{lbl}", color=col, fontsize=7.5,
                ha="center", va="center", fontweight="bold")
    ax.plot(*origin, marker="o", ms=6, color="black", zorder=5)


def sensor_label_offset(name):
    """Small per-sensor text offset so labels don't overlap origins."""
    return {"IMU": (0.005, 0.005), "LiDAR": (-0.012, 0.005), "Camera": (0.005, -0.012)}.get(name, (0.005, 0.005))


# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(13, 10))
fig.suptitle("Geoscan S1 — Sensor Coordinate Frames  (reference: IMU)\n"
             "Red = X   Green = Y   Blue = Z", fontsize=12, fontweight="bold")

ax3d  = fig.add_subplot(2, 2, 1, projection="3d")
ax_xy = fig.add_subplot(2, 2, 2)   # top   view  XY
ax_xz = fig.add_subplot(2, 2, 3)   # front view  XZ
ax_yz = fig.add_subplot(2, 2, 4)   # side  view  YZ

# ---------------------------------------------------------------------------
# 3-D panel
# ---------------------------------------------------------------------------
for s in sensors:
    draw_frame_3d(ax3d, s["T"], s["name"], ARROW_LEN)

# Sensor name labels
for s in sensors:
    o = s["T"][:3, 3]
    ax3d.text(o[0], o[1], o[2] + ARROW_LEN * 0.25, s["name"],
              fontsize=9, fontweight="bold", ha="center",
              bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="grey", alpha=0.8))

# Axis limits
cx, cy, cz = origins.mean(axis=0)
half = spread / 2 + ARROW_LEN * 1.5
ax3d.set_xlim(cx - half, cx + half)
ax3d.set_ylim(cy - half, cy + half)
ax3d.set_zlim(cz - half, cz + half)
ax3d.set_xlabel("X (m)"); ax3d.set_ylabel("Y (m)"); ax3d.set_zlabel("Z (m)")
ax3d.set_title("3D Perspective", fontsize=10)
ax3d.view_init(elev=25, azim=-55)

# ---------------------------------------------------------------------------
# 2-D projection panels
# ---------------------------------------------------------------------------
proj_panels = [
    (ax_xy, 0, 1, "X (m)", "Y (m)", "Top view  (XY)"),
    (ax_xz, 0, 2, "X (m)", "Z (m)", "Front view (XZ)"),
    (ax_yz, 1, 2, "Y (m)", "Z (m)", "Side view  (YZ)"),
]

for ax2d, xi, yi, xlbl, ylbl, title in proj_panels:
    for s in sensors:
        draw_frame_2d(ax2d, s["T"], s["name"], ARROW_LEN, xi, yi, xlbl, ylbl)

    # Sensor name labels
    for s in sensors:
        o2d = s["T"][:3, 3][[xi, yi]]
        dx, dy = sensor_label_offset(s["name"])
        ax2d.text(o2d[0] + dx, o2d[1] + dy, s["name"],
                  fontsize=8, fontweight="bold", ha="left",
                  bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="grey", alpha=0.8))

    # Equal aspect + limits
    c2 = origins[:, [xi, yi]].mean(axis=0)
    h2 = spread / 2 + ARROW_LEN * 1.5
    ax2d.set_xlim(c2[0] - h2, c2[0] + h2)
    ax2d.set_ylim(c2[1] - h2, c2[1] + h2)
    ax2d.set_aspect("equal")
    ax2d.set_xlabel(xlbl); ax2d.set_ylabel(ylbl)
    ax2d.set_title(title, fontsize=10)
    ax2d.grid(True, linestyle="--", alpha=0.4)
    ax2d.axhline(0, color="grey", lw=0.5, alpha=0.5)
    ax2d.axvline(0, color="grey", lw=0.5, alpha=0.5)

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
legend_handles = [
    mpatches.Patch(color=c, label=f"{n} axis")
    for c, n in zip(AXIS_COLORS, AXIS_NAMES)
] + [
    mpatches.Patch(color=s["mcolor"], label=s["name"])
    for s in sensors
]
fig.legend(handles=legend_handles, loc="lower center", ncol=6,
           fontsize=9, framealpha=0.9)

plt.tight_layout(rect=[0, 0.05, 1, 1])
out = "sensor_frames_geoscan_s1.png"
plt.savefig(out, dpi=150)
print(f"Saved: {out}")
plt.show()
