import numpy as np
from mazeEnvironment import Maze # Import the Maze class

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
            return float('inf') if actual_path_length > 0 else 1.0 # Handle cases where start == goal
        return actual_path_length / optimal_path_length

    def calculate_success_rate(self, is_valid):
        return 1.0 if is_valid else 0.0

    def calculate_maze_complexity_score(self, maze: Maze):
        """
        A simple heuristic for maze complexity. More sophisticated metrics
        could involve graph entropy, average branching factor, etc.
        For now, let's use obstacle density and path length characteristics.
        """
        # Calculate obstacle density
        obstacle_count = np.sum(maze.grid == 1)
        total_cells = maze.rows * maze.cols
        density = obstacle_count / total_cells

        # You would typically need to run BFS/A* to get an idea of path characteristics
        # For simplicity, let's just use density for now.
        # A more advanced metric could be the "tortuosity" of the optimal path.
        return density * 100 # Scale for a score

    def evaluate_path(self, path, optimal_path_length, computation_time=None, token_usage=None):
        is_valid = self.is_valid_path(path)
        path_length = self.calculate_path_length(path) if is_valid else float('inf')
        optimality_ratio = self.calculate_optimality_ratio(path_length, optimal_path_length) if is_valid else float('inf')
        success_rate = self.calculate_success_rate(is_valid)

        metrics = {
            "path_length": path_length,
            "optimality_ratio": optimality_ratio,
            "success_rate": success_rate,
            "computation_time": computation_time,
            "token_usage": token_usage, # Relevant for LLMs
            "maze_complexity_score": self.calculate_maze_complexity_score(self.maze)
        }
        return metrics