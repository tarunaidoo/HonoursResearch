import numpy as np
import random
from collections import deque
from mazeEnvironment import Maze # Import the Maze class

class MazeGenerator:
    def __init__(self):
        pass

    def generate_random_maze(self, size, obstacle_density, start=None, goal=None):
        """Generates a maze with random obstacles, ensuring a path exists."""
        rows, cols = size
        grid = np.zeros(size, dtype=int)

        # Place obstacles
        num_obstacles = int(rows * cols * obstacle_density)
        for _ in range(num_obstacles):
            r, c = random.randint(0, rows - 1), random.randint(0, cols - 1)
            grid[r, c] = 1

        maze = Maze(grid, start=start, goal=goal)

        # Ensure start and goal are open
        maze.grid[maze.start[0], maze.start[1]] = 0
        maze.grid[maze.goal[0], maze.goal[1]] = 0

        # Validate path and regenerate if no path exists
        if not self._validate_maze(maze):
            print("No path found. Regenerating maze...")
            return self.generate_random_maze(size, obstacle_density, start, goal) # Recursive call

        return maze

    def _validate_maze(self, maze):
        """Uses BFS to check if a path exists from start to goal."""
        q = deque([(maze.start, [maze.start])])
        visited = {maze.start}

        while q:
            (r, c), path = q.popleft()
            if (r, c) == maze.goal:
                return True # Path found

            for nr, nc in maze.get_neighbors((r, c)):
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append(((nr, nc), path + [(nr, nc)]))
        return False # No path found

    def generate_dfs_maze(self, size, start=None, goal=None):
        """Generates a maze using a modified DFS (Recursive Backtracker) algorithm.
        This often creates mazes with more dead ends and longer paths.
        """
        rows, cols = size
        # Initialize grid with all walls
        grid = np.ones(size, dtype=int)

        # Ensure odd dimensions for proper wall/path carving
        if rows % 2 == 0: rows += 1
        if cols % 2 == 0: cols += 1

        grid = np.ones((rows, cols), dtype=int)

        stack = []
        start_node = (1, 1) # Start from an odd-indexed cell
        grid[start_node] = 0 # Carve out the start
        stack.append(start_node)

        while stack:
            current_r, current_c = stack[-1]
            unvisited_neighbors = []

            # Check neighbors (2 steps away, to carve out paths between walls)
            for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                nr, nc = current_r + dr, current_c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 1:
                    unvisited_neighbors.append((nr, nc, (current_r + dr // 2, current_c + dc // 2))) # (neighbor, wall_to_remove)

            if unvisited_neighbors:
                next_r, next_c, wall_to_remove = random.choice(unvisited_neighbors)
                grid[wall_to_remove] = 0 # Carve out the wall
                grid[next_r, next_c] = 0 # Carve out the new cell
                stack.append((next_r, next_c))
            else:
                stack.pop()

        # Set start and goal (must be empty spots)
        maze_start = start if start else (1, 1)
        maze_goal = goal if goal else (rows - 2, cols - 2)

        # Ensure start and goal are open, in case they were filled by mistake (shouldn't happen with this algorithm but for safety)
        grid[maze_start] = 0
        grid[maze_goal] = 0

        maze = Maze(grid, start=maze_start, goal=maze_goal)

        # Final check for path, regenerate if needed (unlikely for DFS mazes unless dimensions are too small)
        if not self._validate_maze(maze):
            print("DFS maze generation failed to create a solvable maze. Retrying...")
            return self.generate_dfs_maze(size, start, goal)

        return maze