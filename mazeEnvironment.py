import numpy as np
import random
from collections import deque

class Maze:
    def __init__(self, grid, start=None, goal=None):
        self.grid = np.array(grid, dtype=int)
        self.rows, self.cols = self.grid.shape
        self.start = start if start else self._find_random_empty_spot()
        self.goal = goal if goal else self._find_random_empty_spot(exclude=self.start)

        # Ensure start and goal are valid
        if not self._is_valid_position(self.start) or self.grid[self.start[0], self.start[1]] == 1:
            raise ValueError("Start position is invalid or an obstacle.")
        if not self._is_valid_position(self.goal) or self.grid[self.goal[0], self.goal[1]] == 1:
            raise ValueError("Goal position is invalid or an obstacle.")

    def _find_random_empty_spot(self, exclude=None):
        empty_spots = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r, c] == 0:
                    empty_spots.append((r, c))
        if exclude and exclude in empty_spots:
            empty_spots.remove(exclude)
        if not empty_spots:
            raise ValueError("No empty spots available in the maze.")
        return random.choice(empty_spots)

    def _is_valid_position(self, pos):
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def get_neighbors(self, pos):
        r, c = pos
        neighbors = []
        # Define possible moves: (dr, dc) for (up, down, left, right)
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if self._is_valid_position((nr, nc)) and self.grid[nr, nc] == 0:
                neighbors.append((nr, nc))
        return neighbors

    def is_goal(self, pos):
        return pos == self.goal

    def display(self, path=None, current_pos=None):
        maze_display = np.copy(self.grid).astype(str)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r, c] == 1:
                    maze_display[r, c] = '#' # Obstacle
                else:
                    maze_display[r, c] = '.' # Free space

        if path:
            for r, c in path:
                if (r, c) != self.start and (r, c) != self.goal:
                    maze_display[r, c] = '*' # Path

        if self.start:
            maze_display[self.start[0], self.start[1]] = 'S'
        if self.goal:
            maze_display[self.goal[0], self.goal[1]] = 'G'
        if current_pos and current_pos != self.start and current_pos != self.goal:
            maze_display[current_pos[0], current_pos[1]] = '@' # Current agent position

        for row in maze_display:
            print(" ".join(row))
        print("-" * (self.cols * 2))