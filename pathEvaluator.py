import numpy as np
import re # <--- ADDED: Needed for robust parsing
from mazeEnvironment import Maze 

class PathEvaluator:
    def __init__(self, maze: Maze):
        self.maze = maze

    def calculate_path_length(self, path):
        return len(path) - 1 if path else 0 # Number of steps

    def is_valid_path(self, path):
        """Checks if a path is valid (no obstacles, reaches goal)."""
        if not path or path[0] != self.maze.start:
            return False

        current_pos = path[0]
        for i in range(1, len(path)):
            next_pos = path[i]
            # Check if move is valid (adjacent and not an obstacle)
            if not self.maze._is_valid_position(next_pos) or \
               self.maze.grid[next_pos[0], next_pos[1]] == 1:
                return False

            # Check if it's an adjacent move
            if abs(current_pos[0] - next_pos[0]) + abs(current_pos[1] - next_pos[1]) != 1:
                return False
            current_pos = next_pos
            
        return current_pos == self.maze.goal

    def calculate_optimality_ratio(self, actual_path_length, optimal_path_length):
        if optimal_path_length == 0:
            return float('inf') if actual_path_length > 0 else 1.0
        return actual_path_length / optimal_path_length

    def calculate_success_rate(self, is_valid):
        return 1.0 if is_valid else 0.0

    def calculate_maze_complexity_score(self, maze: Maze):
        """
        A simple heuristic for maze complexity.
        """
        obstacle_count = np.sum(maze.grid == 1)
        total_cells = maze.rows * maze.cols
        density = obstacle_count / total_cells
        return density * 100 

    def robust_parse_coordinates(self, raw_output: str) -> list[tuple]:
        """
        Uses RegEx to extract coordinates from LLM output (e.g., [(1, 2), (3, 4)]) 
        while ignoring any conversational text.
        """
        start_pos = self.maze.start
        
        # RegEx to find all occurrences of (digit, digit) tuples, including optional spaces
        coordinate_strings = re.findall(r'\(\s*\d+\s*,\s*\d+\s*\)', raw_output)
        
        parsed_path = []
        for match in coordinate_strings:
            try:
                # Extract the numbers from the string tuple
                r, c = map(int, re.findall(r'\d+', match))
                parsed_path.append((r, c))
            except ValueError:
                continue

        # Critical Fix: Ensure the path starts with the required start position
        if parsed_path and parsed_path[0] != start_pos:
            # If the path is non-empty but doesn't start at the start, prepend it
            parsed_path.insert(0, start_pos)
        elif not parsed_path and raw_output:
            # If nothing was parsed, the LLM failed to format, but include start for consistency
             return [start_pos]
        elif not parsed_path and not raw_output:
            # Truly empty output
             return []
            
        return parsed_path

    def evaluate_path(self, path, optimal_path_length, computation_time=None, token_usage=None):
        
        # Pylance Fix: Calculate variables before the final dictionary is created.
        is_valid = self.is_valid_path(path)
        path_length = self.calculate_path_length(path) if is_valid else float('inf')
        
        # Calculate these variables BEFORE using them in the dictionary
        optimality_ratio = self.calculate_optimality_ratio(path_length, optimal_path_length) if is_valid else float('inf')
        success_rate = self.calculate_success_rate(is_valid)

        metrics = {
            "path_length": path_length,
            "optimality_ratio": optimality_ratio, 
            "success_rate": success_rate,         
            "computation_time": computation_time,
            "token_usage": token_usage,
            "maze_complexity_score": self.calculate_maze_complexity_score(self.maze),
            "path_validity_check": is_valid       # Added this for clarity in results
        }
        return metrics