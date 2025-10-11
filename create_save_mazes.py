# import pickle
# import os
# from mazeGenerator import MazeGenerator

# def generate_and_save_mazes():
#     """Generates a fixed set of mazes and saves them to a file."""
    
#     # Define your parameters
#     maze_generator = MazeGenerator()
#     maze_parameters = [
#         # (size, density, num_mazes_to_generate)
#         ((5, 5), 0.1, 5),
#         ((5, 5), 0.3, 5),
#         ((10, 10), 0.1, 5),
#         ((10, 10), 0.2, 5),
#         ((10, 10), 0.3, 5),
#         ((20, 20), 0.1, 5),
#         ((20, 20), 0.2, 5),
#         ((20, 20), 0.3, 5),
#         ((40, 40), 0.1, 5),
#         ((40, 40), 0.2, 5),
#         ((40, 40), 0.3, 5),
#         ((80, 80), 0.1, 5),
#         ((80, 80), 0.2, 5),
#         ((80, 80), 0.3, 5),
#         ((160, 160), 0.1, 5),
#         ((160, 160), 0.2, 5),
#         ((160, 160), 0.3, 5),
#         ((320, 320), 0.1, 5),
#         ((320, 320), 0.2, 5),
#         ((320, 320), 0.3, 5),
#         ((640, 640), 0.1, 5),
#         ((640, 640), 0.2, 5),
#         ((640, 640), 0.3, 5),
#         ((1024, 1024), 0.1, 5),
#         ((1024, 1024), 0.2, 5),
#         ((1024, 1024), 0.3, 5),
#     ]

#     mazes = []
    
#     print("Starting maze generation...")
#     for size, density, num in maze_parameters:
#         print(f"Generating {num} mazes of size {size} with density {density}...")
#         for _ in range(num):
#             maze = maze_generator.generate_random_maze(size, density)
#             mazes.append(maze)

#     print(f"\nGenerated a total of {len(mazes)} mazes.")
    
#     # Save the list of Maze objects to a file
#     file_path = "pregenerated_mazes.pkl"
#     with open(file_path, "wb") as f:
#         pickle.dump(mazes, f)
        
#     print(f"Mazes successfully saved to {file_path}")

# if __name__ == "__main__":
#     generate_and_save_mazes()

import json
import os
import numpy as np
from tqdm import tqdm
from mazeGenerator import MazeGenerator

def generate_and_save_mazes_json():
    """Generates a fixed set of mazes (Random & Backtracking) and saves them to a JSON file."""

    maze_generator = MazeGenerator()
    
    # --- Random Maze Parameters (Density-based) ---
    # Used for testing the impact of obstacle density and size.
    random_maze_parameters = [
        # (size, density, num_mazes_to_generate)
        ((5, 5), 0.1, 3), 
        ((5, 5), 0.3, 3), 
        ((10, 10), 0.1, 3), 
        ((10, 10), 0.2, 3), 
        ((10, 10), 0.3, 3), 
        ((20, 20), 0.1, 3), 
        ((20, 20), 0.2, 3), 
        ((20, 20), 0.3, 3), 
        ((40, 40), 0.1, 3), 
        ((40, 40), 0.2, 3), 
        ((40, 40), 0.3, 3), 
    ]
    
    # --- Backtracking Maze Parameters (Algorithm-based) ---
    # Used for testing the impact of path tortuosity, independent of density.
    backtracking_maze_parameters = [
        # (size, num_mazes_to_generate)
        ((5, 5), 3),
        ((11, 11), 3), # Note: Generator adjusts size to be odd (10x10 -> 11x11, roughly)
        ((21, 21), 3), # (20x20 -> 21x21, roughly)
        ((41, 41), 3), # (40x40 -> 41x41, roughly)
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

    # 2. Generate Backtracking Mazes (Algorithm-based)
    for size, num in tqdm(backtracking_maze_parameters, desc="Generating Backtracking Mazes"):
        for _ in range(num):
            maze = maze_generator.generate_backtracking_maze(size)
            mazes_obj.append(maze)
            
            # Recalculate density for storage
            obstacle_count = np.sum(maze.grid == 1)
            total_cells = maze.rows * maze.cols
            density = obstacle_count / total_cells
            
            # Convert maze object to a serializable dictionary
            maze_dict = {
                'grid': maze.grid.tolist(),
                'rows': maze.rows,
                'cols': maze.cols,
                'start': maze.start,
                'goal': maze.goal,
                'obstacle_density': density,
                'scenario': "Backtracking_Alg"
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
            f.write(np.array2string(maze.get_binary_grid(), separator=', ', max_line_width=np.inf))
            f.write("\n\n")

    print("Binary maze representations successfully saved.")

if __name__ == "__main__":
    generate_and_save_mazes_json()