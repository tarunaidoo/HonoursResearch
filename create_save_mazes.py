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
    """Generates a fixed set of mazes and saves them to both a JSON and a text file."""

    maze_generator = MazeGenerator()
    maze_parameters = [
        ((5, 5), 0.1, 5),
        ((5, 5), 0.3, 5),
        ((10, 10), 0.1, 5),
        ((10, 10), 0.2, 5),
        ((10, 10), 0.3, 5),
        ((20, 20), 0.1, 5),
        ((20, 20), 0.2, 5),
        ((20, 20), 0.3, 5),
        ((40, 40), 0.1, 5),
        ((40, 40), 0.2, 5),
        ((40, 40), 0.3, 5),
        ((80, 80), 0.1, 5),
        ((80, 80), 0.2, 5),
        ((80, 80), 0.3, 5),
        ((160, 160), 0.1, 5),
        ((160, 160), 0.2, 5),
        ((160, 160), 0.3, 5),
        ((320, 320), 0.1, 5),
        ((320, 320), 0.2, 5),
        ((320, 320), 0.3, 5),
        ((640, 640), 0.1, 5),
        ((640, 640), 0.2, 5),
        ((640, 640), 0.3, 5),
        ((1024, 1024), 0.1, 5),
        ((1024, 1024), 0.2, 5),
        ((1024, 1024), 0.3, 5),
    ]

    mazes_data = []
    mazes_obj = []

    print("Starting maze generation...")
    for size, density, num in tqdm(maze_parameters, desc="Generating Maze Sets"):
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
                'obstacle_density': density
            }
            mazes_data.append(maze_dict)

    print(f"\nGenerated a total of {len(mazes_data)} mazes.")

    json_file_path = "pregenerated_mazes.json"
    with open(json_file_path, "w") as f:
        json.dump(mazes_data, f, indent=4)
    print(f"Mazes successfully saved to {json_file_path}")

    # --- New section to save binary version to a text file ---
    txt_file_path = "pregenerated_mazes_binary.txt"
    print(f"Saving binary versions to {txt_file_path}...")
    with open(txt_file_path, "w") as f:
        for i, maze in enumerate(mazes_obj):
            # Write a header for each maze
            f.write(f"--- Maze {i+1} ---\n")
            f.write(f"Size: {maze.rows}x{maze.cols}\n")
            f.write(f"Start: {maze.start}, Goal: {maze.goal}\n\n")

            # Write the binary grid
            # `np.array2string` is used to get a clean string representation of the numpy array
            f.write(np.array2string(maze.get_binary_grid(), separator=', ', max_line_width=np.inf))
            f.write("\n\n")

    print(f"Binary maze representations successfully saved.")

if __name__ == "__main__":
    generate_and_save_mazes_json()