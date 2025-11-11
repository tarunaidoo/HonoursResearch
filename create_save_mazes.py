import json
import os
import numpy as np
from tqdm import tqdm
from mazeGenerator import MazeGenerator
from mazeEnvironment import Maze

def generate_and_save_mazes_json():
    """Generates a fixed set of Random (Density-based) mazes and saves them to a JSON file."""

    maze_generator = MazeGenerator()
    
    random_maze_parameters = [
        # (size, density, num_mazes_to_generate)
        # Total tests per density: 5 * 5 = 25
        # Total unique mazes: 5 sizes * 3 densities * 5 mazes = 75
        
        # ----------------------------------
        # 4x4 (Trivial)
        ((4, 4), 0.1, 3),
        ((4, 4), 0.2, 3),
        ((4, 4), 0.3, 3),
        
        # 8x8 (Small)
        ((8, 8), 0.1, 3),
        ((8, 8), 0.2, 3),
        ((8, 8), 0.3, 3),
        
        # 16x16 (Medium)
        ((16, 16), 0.1, 3),
        ((16, 16), 0.2, 3),
        ((16, 16), 0.3, 3),
        
        # 32x32 (Large - Performance Threshold)
        ((32, 32), 0.1, 3),
        ((32, 32), 0.2, 3),
        ((32, 32), 0.3, 3),
        
        # 64x64 (Extreme - Context/Failure Point)
        ((64, 64), 0.1, 3),
        ((64, 64), 0.2, 3),
        ((64, 64), 0.3, 3),
        
        # 128x128 (Massive - Beyond Typical Use Cases) - Using actual LLM on this. Mock LLM for anything larger
        ((128, 128), 0.1, 3),
        ((128, 128), 0.2, 3),
        ((128, 128), 0.3, 3),
    ]

    mazes_data = []
    mazes_obj = []

    print("Starting maze generation...")
    
    # 1. Generate Random Mazes (Density-based)
    for size, density, num in tqdm(random_maze_parameters, desc="Generating Random Mazes"):
        for _ in range(num):
            maze = maze_generator.generate_random_maze(size, density)
            mazes_obj.append(maze)
            
            # Convert maze object to a serializable dictionary
            maze_dict = {
                'grid': maze.grid.tolist(),
                'rows': maze.rows,
                'cols': maze.cols,
                'start': maze.start,
                'goal': maze.goal,
                'obstacle_density': density,
                'scenario': f"Random_{density:.1f}D"
            }
            mazes_data.append(maze_dict)

    print(f"\nGenerated a total of {len(mazes_data)} mazes.")

    json_file_path = "pregenerated_mazes.json"
    with open(json_file_path, "w") as f:
        json.dump(mazes_data, f, indent=4)
    print(f"Mazes successfully saved to {json_file_path}")

    # --- Section to save binary version to a text file (for quick inspection) ---
    txt_file_path = "pregenerated_mazes_binary.txt"
    print(f"Saving binary versions to {txt_file_path}...")
    with open(txt_file_path, "w") as f:
        for i, maze in enumerate(mazes_obj):
            # Write a header for each maze
            f.write(f"--- Maze {i+1} ---\n")
            f.write(f"Size: {maze.rows}x{maze.cols}\n")
            f.write(f"Start: {maze.start}, Goal: {maze.goal}\n")
            f.write(f"Scenario: {mazes_data[i]['scenario']}\n\n") 
            
            # Write the binary grid
            # Assuming maze.get_binary_grid() exists on the Maze object
            f.write(np.array2string(maze.get_binary_grid(), separator=', ', max_line_width=np.inf))
            f.write("\n\n")

    print("Binary maze representations successfully saved.")

if __name__ == "__main__":
    generate_and_save_mazes_json()
