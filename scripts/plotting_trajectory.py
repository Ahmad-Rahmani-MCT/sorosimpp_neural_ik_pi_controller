#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle, Patch
import matplotlib
import numpy as np

# =========================== CONFIGURATION =========================== #
REQUIRED_FRAMES = ["base"] + [f"cs{i}" for i in range(1, 39)] + ["tip"]

# --- SHARED VISUALIZATION SETTINGS ---
MIN_MOVEMENT_THRESHOLD = 0.005  # 5mm movement required to leave a ghost trace
WORKSPACE_RADIUS = 0.1
WORKSPACE_HEIGHT_START = 0.0
WORKSPACE_HEIGHT_END = -0.4
WORKSPACE_ALPHA = 0.05
GOAL_POSITION = [-0.07, -0.07, -0.39]  # Set to None if no goal

# --- ANIMATION SETTINGS ---
VIDEO_DURATION_SEC = 10   # Duration for both Trace and Turntable videos
FPS = 30                  # Frames per second
TOTAL_FRAMES = VIDEO_DURATION_SEC * FPS
# ===================================================================== #

def load_data():
    abs_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(abs_path)
    file_path = os.path.join(script_dir, "logged_data_csv", "ros_data_logged.csv")

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None, None
    
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    return df, script_dir

def plot_cylinder(ax, radius, z_start, z_end, center=(0,0), color='blue', alpha=0.1):
    """ Helper to plot the workspace cylinder """
    z = np.linspace(z_start, z_end, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid) + center[0]
    y_grid = radius * np.sin(theta_grid) + center[1]
    
    ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=alpha, shade=False)
    
    # Caps
    p = Circle(center, radius, color=color, alpha=alpha/2)
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z=z_end, zdir="z")
    
    p2 = Circle(center, radius, color=color, alpha=alpha/2)
    ax.add_patch(p2)
    art3d.pathpatch_2d_to_3d(p2, z=z_start, zdir="z")

def setup_axes(ax):
    """ Common axes setup for all visualizers """
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    
    # FIXED SYMMETRIC AXES
    ax.set_xlim(-0.15, 0.15)
    ax.set_ylim(-0.15, 0.15)
    ax.set_zlim(WORKSPACE_HEIGHT_END - 0.05, 0.05) 

    # Remove the gray background panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.grid(False)

def get_filename_prefix():
    if GOAL_POSITION:
        return f"{GOAL_POSITION[0]}_{GOAL_POSITION[1]}_{GOAL_POSITION[2]}"
    return "trajectory"

# =========================== 1. STATIC PLOT GENERATION ===========================
def generate_static_plot(df, script_dir):
    print("\n[1/3] Generating Static Plot...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Soft Robot Bending Evolution", fontsize=16)
    
    setup_axes(ax)
    colormap = matplotlib.colormaps['jet']
    
    # --- Plot Trajectory (Ghosts) ---
    last_plotted_tip = np.array([np.inf, np.inf, np.inf]) 
    num_steps = len(df)

    for i in range(num_steps):
        current_tip = np.array([df.iloc[i]['tip_pos_x'], df.iloc[i]['tip_pos_y'], df.iloc[i]['tip_pos_z']])
        dist = np.linalg.norm(current_tip - last_plotted_tip)
        
        if dist > MIN_MOVEMENT_THRESHOLD or i == 0:
            xs = [df.iloc[i][f'{frame}_pos_x'] for frame in REQUIRED_FRAMES]
            ys = [df.iloc[i][f'{frame}_pos_y'] for frame in REQUIRED_FRAMES]
            zs = [df.iloc[i][f'{frame}_pos_z'] for frame in REQUIRED_FRAMES]
            
            progress = i / num_steps
            color = colormap(progress)
            ax.plot(xs, ys, zs, color=color, alpha=0.3, linewidth=1.5)
            last_plotted_tip = current_tip

    # --- Plot Final Elements ---
    final_xs = [df.iloc[-1][f'{frame}_pos_x'] for frame in REQUIRED_FRAMES]
    final_ys = [df.iloc[-1][f'{frame}_pos_y'] for frame in REQUIRED_FRAMES]
    final_zs = [df.iloc[-1][f'{frame}_pos_z'] for frame in REQUIRED_FRAMES]

    final_color = colormap(1.0) 
    final_line, = ax.plot(final_xs, final_ys, final_zs, color=final_color, linewidth=2.5, label='Final Shape')
    base_scatter = ax.scatter(final_xs[0], final_ys[0], final_zs[0], color='gray', s=100, marker='s', label='Base')
    tip_scatter = ax.scatter(final_xs[-1], final_ys[-1], final_zs[-1], color='black', s=50, label='Tip')

    goal_scatter = None
    if GOAL_POSITION:
        goal_scatter = ax.scatter(GOAL_POSITION[0], GOAL_POSITION[1], GOAL_POSITION[2], color='green', s=150, marker='*', label='Goal')

    plot_cylinder(ax, radius=WORKSPACE_RADIUS, z_start=WORKSPACE_HEIGHT_START, z_end=WORKSPACE_HEIGHT_END, alpha=WORKSPACE_ALPHA)

    # --- Legend ---
    handles = [final_line, base_scatter, tip_scatter]
    if goal_scatter: handles.append(goal_scatter)
    handles.append(Patch(color='blue', alpha=WORKSPACE_ALPHA, label='Static Workspace'))
    ax.legend(handles=handles)

    # --- Save ---
    save_folder = os.path.join(script_dir, "point_reaching_plots")
    os.makedirs(save_folder, exist_ok=True)
    filename = f"{get_filename_prefix()}.png"
    plt.savefig(os.path.join(save_folder, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"      Saved to: point_reaching_plots/{filename}")

# =========================== 2. TRACE ANIMATION (NO TIME) ===========================
def generate_trace_animation(df, script_dir):
    print("\n[2/3] Generating Trace Animation (Robot Moving)...")
    
    # Downsampling
    data_len = len(df)
    step_size = max(1, int(data_len / TOTAL_FRAMES))
    indices_to_plot = list(range(0, data_len, step_size))
    if indices_to_plot[-1] != data_len - 1: indices_to_plot.append(data_len - 1)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    setup_axes(ax)
    ax.set_title("Soft Robot Trajectory Evolution")

    # Static Background
    plot_cylinder(ax, radius=WORKSPACE_RADIUS, z_start=WORKSPACE_HEIGHT_START, z_end=WORKSPACE_HEIGHT_END, alpha=WORKSPACE_ALPHA)
    
    goal_scatter = None
    if GOAL_POSITION:
        goal_scatter = ax.scatter(GOAL_POSITION[0], GOAL_POSITION[1], GOAL_POSITION[2], color='green', s=150, marker='*', label='Goal')
        
    ax.scatter(df.iloc[0]['base_pos_x'], df.iloc[0]['base_pos_y'], df.iloc[0]['base_pos_z'], color='gray', s=100, marker='s', label='Base')

    # Dynamic Elements
    current_arm_line, = ax.plot([], [], [], color='black', linewidth=2.5, label='Current Shape')
    tip_point, = ax.plot([], [], [], 'o', color='black', markersize=5, label='Tip')

    # --- Legend (FIXED) ---
    cylinder_proxy = Patch(color='blue', alpha=WORKSPACE_ALPHA, label='Static Workspace')
    
    handles = [current_arm_line, tip_point, cylinder_proxy]
    if goal_scatter:
        handles.append(goal_scatter)
        
    ax.legend(handles=handles, loc='upper right')

    last_ghost_tip = np.array([np.inf, np.inf, np.inf])
    colormap = matplotlib.colormaps['jet']

    def update(frame_idx):
        nonlocal last_ghost_tip
        idx = indices_to_plot[frame_idx]
        
        xs = [df.iloc[idx][f'{frame}_pos_x'] for frame in REQUIRED_FRAMES]
        ys = [df.iloc[idx][f'{frame}_pos_y'] for frame in REQUIRED_FRAMES]
        zs = [df.iloc[idx][f'{frame}_pos_z'] for frame in REQUIRED_FRAMES]
        
        current_arm_line.set_data(xs, ys)
        current_arm_line.set_3d_properties(zs)
        tip_point.set_data([xs[-1]], [ys[-1]])
        tip_point.set_3d_properties([zs[-1]])

        # Ghost Logic
        current_tip = np.array([xs[-1], ys[-1], zs[-1]])
        dist = np.linalg.norm(current_tip - last_ghost_tip)
        
        if dist > MIN_MOVEMENT_THRESHOLD:
            progress = idx / data_len
            color = colormap(progress)
            ax.plot(xs, ys, zs, color=color, alpha=0.3, linewidth=1.5)
            last_ghost_tip = current_tip
            
        return current_arm_line, tip_point

    ani = animation.FuncAnimation(fig, update, frames=len(indices_to_plot), interval=1000/FPS, blit=False)
    
    save_folder = os.path.join(script_dir, "point_reaching_animations")
    os.makedirs(save_folder, exist_ok=True)
    filename = f"{get_filename_prefix()}_trace.mp4"
    
    writer = animation.FFMpegWriter(fps=FPS, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(os.path.join(save_folder, filename), writer=writer)
    plt.close(fig)
    print(f"      Saved to: point_reaching_animations/{filename}")

# =========================== 3. TURNTABLE ANIMATION ===========================
def generate_turntable_animation(df, script_dir):
    print("\n[3/3] Generating Turntable Animation (Camera Rotation)...")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    setup_axes(ax)
    ax.set_title("Soft Robot Final Configuration (360° View)")

    # 1. Plot EVERYTHING statically (Ghosts + Final Shape + Environment)
    colormap = matplotlib.colormaps['jet']
    last_plotted_tip = np.array([np.inf, np.inf, np.inf])
    num_steps = len(df)

    # Plot Ghosts
    for i in range(num_steps):
        current_tip = np.array([df.iloc[i]['tip_pos_x'], df.iloc[i]['tip_pos_y'], df.iloc[i]['tip_pos_z']])
        dist = np.linalg.norm(current_tip - last_plotted_tip)
        if dist > MIN_MOVEMENT_THRESHOLD or i == 0:
            xs = [df.iloc[i][f'{frame}_pos_x'] for frame in REQUIRED_FRAMES]
            ys = [df.iloc[i][f'{frame}_pos_y'] for frame in REQUIRED_FRAMES]
            zs = [df.iloc[i][f'{frame}_pos_z'] for frame in REQUIRED_FRAMES]
            ax.plot(xs, ys, zs, color=colormap(i / num_steps), alpha=0.3, linewidth=1.5)
            last_plotted_tip = current_tip

    # Plot Final Shape
    final_xs = [df.iloc[-1][f'{frame}_pos_x'] for frame in REQUIRED_FRAMES]
    final_ys = [df.iloc[-1][f'{frame}_pos_y'] for frame in REQUIRED_FRAMES]
    final_zs = [df.iloc[-1][f'{frame}_pos_z'] for frame in REQUIRED_FRAMES]
    
    final_line, = ax.plot(final_xs, final_ys, final_zs, color=colormap(1.0), linewidth=2.5, label='Final Shape')
    base_scatter = ax.scatter(final_xs[0], final_ys[0], final_zs[0], color='gray', s=100, marker='s', label='Base')
    tip_scatter = ax.scatter(final_xs[-1], final_ys[-1], final_zs[-1], color='black', s=50, label='Tip')

    # Environment
    goal_scatter = None
    if GOAL_POSITION:
        goal_scatter = ax.scatter(GOAL_POSITION[0], GOAL_POSITION[1], GOAL_POSITION[2], color='green', s=150, marker='*', label='Goal')
    plot_cylinder(ax, radius=WORKSPACE_RADIUS, z_start=WORKSPACE_HEIGHT_START, z_end=WORKSPACE_HEIGHT_END, alpha=WORKSPACE_ALPHA)

    # Legend
    handles = [final_line, base_scatter, tip_scatter]
    if goal_scatter: handles.append(goal_scatter)
    handles.append(Patch(color='blue', alpha=WORKSPACE_ALPHA, label='Static Workspace'))
    ax.legend(handles=handles, loc='upper right')

    # 2. Animation Loop: Just Rotate Camera
    def update(frame):
        # Rotate azimuth angle from 0 to 360
        angle = frame * (360 / TOTAL_FRAMES)
        ax.view_init(elev=30, azim=angle)
        return fig,

    ani = animation.FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=1000/FPS, blit=False)
    
    save_folder = os.path.join(script_dir, "point_reaching_animations")
    os.makedirs(save_folder, exist_ok=True)
    filename = f"{get_filename_prefix()}_turntable.mp4"
    
    writer = animation.FFMpegWriter(fps=FPS, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(os.path.join(save_folder, filename), writer=writer)
    plt.close(fig)
    print(f"      Saved to: point_reaching_animations/{filename}")

# =========================== MAIN ===========================
def main():
    df, script_dir = load_data()
    if df is not None:
        generate_static_plot(df, script_dir)
        generate_trace_animation(df, script_dir)
        generate_turntable_animation(df, script_dir)
        print("\nAll tasks completed successfully!")

if __name__ == '__main__':
    main()