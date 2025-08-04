from collections import deque
import heapq # For A*
import time
from mazeEnvironment import Maze # Import the Maze class

class MazeSolver:
    def __init__(self, maze: Maze):
        self.maze = maze

    def bfs(self):
        """Breadth-First Search for shortest path in unweighted graph."""
        start_time = time.time()
        q = deque([(self.maze.start, [self.maze.start])]) # (current_pos, path_so_far)
        visited = {self.maze.start}
        nodes_expanded = 0

        while q:
            (r, c), path = q.popleft()
            nodes_expanded += 1

            if (r, c) == self.maze.goal:
                end_time = time.time()
                return path, end_time - start_time, nodes_expanded

            for nr, nc in self.maze.get_neighbors((r, c)):
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append(((nr, nc), path + [(nr, nc)]))
        end_time = time.time()
        return None, end_time - start_time, nodes_expanded # No path found

    def dfs(self, depth_limit=None):
        """Depth-First Search (Depth-Limited for practical use)."""
        start_time = time.time()
        stack = [(self.maze.start, [self.maze.start], 0)] # (current_pos, path_so_far, current_depth)
        visited_at_depth = {} # To handle loops in DFS and keep track of visited nodes for each depth
        nodes_expanded = 0

        while stack:
            (r, c), path, depth = stack.pop()
            nodes_expanded += 1

            if (r, c) == self.maze.goal:
                end_time = time.time()
                return path, end_time - start_time, nodes_expanded

            if depth_limit is not None and depth >= depth_limit:
                continue

            # Mark visited for current depth to avoid infinite loops and re-exploring
            if (r, c) not in visited_at_depth or depth < visited_at_depth[(r, c)]:
                visited_at_depth[(r, c)] = depth
            else:
                continue

            # Iterate neighbors in reverse to simulate LIFO for a stack (e.g., exploring first neighbor found first)
            for nr, nc in reversed(self.maze.get_neighbors((r, c))):
                if (nr, nc) not in visited_at_depth or depth + 1 < visited_at_depth[(nr, nc)]:
                    stack.append(((nr, nc), path + [(nr, nc)], depth + 1))

        end_time = time.time()
        return None, end_time - start_time, nodes_expanded # No path found

    def a_star(self):
        """A* Search using Manhattan distance heuristic."""
        start_time = time.time()
        # (f_cost, g_cost, (r, c), path_so_far)
        # f_cost = g_cost + h_cost
        # g_cost = cost from start to current node
        # h_cost = heuristic cost from current node to goal
        open_list = [(0, 0, self.maze.start, [self.maze.start])] # Min-heap for efficiency
        came_from = {} # To reconstruct path
        g_score = {self.maze.start: 0} # Cost from start to current
        f_score = {self.maze.start: self._manhattan_distance(self.maze.start, self.maze.goal)} # Estimated total cost
        nodes_expanded = 0

        while open_list:
            current_f_cost, current_g_cost, current_pos, current_path = heapq.heappop(open_list)
            nodes_expanded += 1

            if current_pos == self.maze.goal:
                end_time = time.time()
                return current_path, end_time - start_time, nodes_expanded

            for neighbor in self.maze.get_neighbors(current_pos):
                tentative_g_score = g_score[current_pos] + 1 # Cost to move to neighbor is 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current_pos
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._manhattan_distance(neighbor, self.maze.goal)
                    heapq.heappush(open_list, (f_score[neighbor], tentative_g_score, neighbor, current_path + [neighbor]))

        end_time = time.time()
        return None, end_time - start_time, nodes_expanded # No path found

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])