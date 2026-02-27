#!/usr/bin/env python3
import csv
import os
import matplotlib.pyplot as plt

def plot_workspace_targets(filename="workspace_targets.csv"):
    # Get the directory of the current script to locate the CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)

    # Check if the file exists before trying to open it
    if not os.path.exists(file_path):
        print(f"Error: The file '{filename}' was not found in {script_dir}.")
        print("Please run the generation script first.")
        return

    x_vals = []
    y_vals = []

    # Read the data from the CSV file
    with open(file_path, mode='r') as file:
        # csv.DictReader automatically uses the first row as dictionary keys
        reader = csv.DictReader(file)
        for row in reader:
            x_vals.append(float(row['target_x']))
            y_vals.append(float(row['target_y']))

    # Initialize the plot
    plt.figure(figsize=(8, 8))
    
    # Create a scatter plot
    # s=30 controls the size of the dots, alpha adds slight transparency
    plt.scatter(x_vals, y_vals, color='royalblue', s=30, alpha=0.8, edgecolors='black', linewidth=0.5)

    # Formatting the plot
    plt.title(f'Workspace Targets ({len(x_vals)} points)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    
    # Crucial for circular distributions: 
    # Ensures the X and Y axes have the same scale so the circle doesn't look stretched
    plt.axis('equal')
    
    # Add a background grid
    plt.grid(True, linestyle='--', alpha=0.6)

    # Display the plot
    print(f"--- SUCCESS ---")
    print(f"Plotting {len(x_vals)} points from {filename}...")
    plt.show()

if __name__ == '__main__':
    plot_workspace_targets()