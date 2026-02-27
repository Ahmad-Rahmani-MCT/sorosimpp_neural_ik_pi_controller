#!/usr/bin/env python3
import random
import math
import csv
import os

def generate_workspace_targets(num_points=100, radius=0.1, filename="workspace_targets.csv"):
    targets = []
    
    # Set a seed so you can reproduce these exact same 100 points if needed
    random.seed(42) 

    for _ in range(num_points):
        # The square root ensures points are uniformly distributed by area
        r = radius * math.sqrt(random.random())
        theta = random.uniform(0, 2 * math.pi)
        
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        
        # Rounding to 5 decimal places for clean data
        targets.append([round(x, 5), round(y, 5)])

    # Get the directory of the current script to save the CSV alongside it
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)

    # Write the points to a CSV file
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['target_x', 'target_y']) # Header row
        writer.writerows(targets)

    print(f"--- SUCCESS ---")
    print(f"Generated {num_points} uniformly distributed points.")
    print(f"Saved to: {file_path}")

if __name__ == '__main__':
    generate_workspace_targets()