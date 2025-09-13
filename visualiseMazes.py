import os
import json
import numpy as np

# Make sure this import matches your file structure
from mazeEnvironment import Maze

def load_mazes(file_path="pregenerated_mazes.json"):
    """
    Loads maze data from a JSON file and reconstructs Maze objects.

    Args:
        file_path (str): The path to the JSON file containing the mazes.

    Returns:
        list: A list of Maze objects.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Maze file not found at {file_path}. Please run generate_mazes.py first.")

    print(f"Loading mazes from {file_path}...")
    with open(file_path, "r") as f:
        mazes_data = json.load(f)

    # Reconstruct Maze objects from the loaded data
    mazes = []
    for maze_dict in mazes_data:
        # Assuming your Maze class can be initialized from a grid, start, and goal
        reconstructed_maze = Maze(
            grid=np.array(maze_dict['grid']), # Convert list of lists back to NumPy array
            start=tuple(maze_dict['start']), # Ensure start is a tuple
            goal=tuple(maze_dict['goal']) # Ensure goal is a tuple
        )
        mazes.append(reconstructed_maze)
    return mazes

def visualize_all_mazes_as_binary():
    """Loads and visualizes all mazes from the pre-generated file as a binary grid."""
    try:
        # Now uses the updated load_mazes function which reads from JSON
        mazes = load_mazes()
    except FileNotFoundError as e:
        print(e)
        return

    print(f"Successfully loaded {len(mazes)} mazes. Displaying them one by one...")
    print("--------------------------------------------------")

    for i, maze in enumerate(mazes):
        print(f"--- Maze {i+1} ---")
        print(f"Size: {maze.rows}x{maze.cols}")

        # The get_binary_grid() method still works as it's part of the Maze class
        binary_grid = maze.get_binary_grid()

        # Print the NumPy array directly
        print(binary_grid)

        # Optional: Display with start and goal for clarity
        print(f"Start: {maze.start}, Goal: {maze.goal}")
        print("\n")

if __name__ == "__main__":
    visualize_all_mazes_as_binary()