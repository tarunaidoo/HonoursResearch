import os
import time
import re # For parsing LLM output
from openai import OpenAI # Assuming you have pip install openai
from mazeEnvironment import Maze # Corrected import based on file structure

class LLMInterface:
    def __init__(self, api_key, model_name="gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def format_maze_for_prompt(self, maze: Maze, format_type="ascii"):
        """
        Formats the maze into a string representation for the LLM.
        Options: "ascii", "coordinate_list", "flattened_array"
        """
        if format_type == "ascii":
            maze_str = ""
            for r in range(maze.rows):
                row_chars = []
                for c in range(maze.cols):
                    if (r, c) == maze.start:
                        row_chars.append('S')
                    elif (r, c) == maze.goal:
                        row_chars.append('G')
                    elif maze.grid[r, c] == 1:
                        row_chars.append('#')
                    else:
                        row_chars.append('.')
                maze_str += " ".join(row_chars) + "\n"
            return maze_str.strip()
        elif format_type == "coordinate_list":
            obstacles = []
            for r in range(maze.rows):
                for c in range(maze.cols):
                    if maze.grid[r, c] == 1:
                        obstacles.append(f"({r},{c})")
            return f"Maze size: ({maze.rows},{maze.cols})\nStart: ({maze.start[0]},{maze.start[1]})\nGoal: ({maze.goal[0]},{maze.goal[1]})\nObstacles: [{', '.join(obstacles)}]"
        elif format_type == "flattened_array":
            flattened_grid = maze.grid.flatten().tolist()
            # Replace 0 with '.' and 1 with '#' for better LLM understanding
            grid_chars = []
            for i, val in enumerate(flattened_grid):
                r, c = divmod(i, maze.cols)
                if (r, c) == maze.start:
                    grid_chars.append('S')
                elif (r, c) == maze.goal:
                    grid_chars.append('G')
                elif val == 1:
                    grid_chars.append('#')
                else:
                    grid_chars.append('.')
            return f"Maze grid (row-major, {maze.rows}x{maze.cols}): {''.join(grid_chars)}\nStart: {maze.start}\nGoal: {maze.goal}"
        else:
            raise ValueError("Unsupported prompt format type.")

    def construct_prompt(self, maze: Maze, prompt_type="zero_shot", maze_format="ascii", examples=None):
        maze_representation = self.format_maze_for_prompt(maze, maze_format)

        base_instruction = "You are an AI assistant designed to solve mazes. Your task is to find a path from the 'S' (start) to the 'G' (goal) in the provided maze. You can only move up, down, left, or right. Avoid '#' (obstacles). Output your path as a list of (row,column) coordinates, starting with 'S' and ending with 'G'. For example: [(0,0), (0,1), (1,1)]"

        if prompt_type == "zero_shot":
            prompt = f"{base_instruction}\n\nMaze:\n{maze_representation}\n\nPath:"
        elif prompt_type == "few_shot":
            if not examples:
                raise ValueError("Few-shot prompting requires examples.")
            example_str = ""
            for ex_maze, ex_path in examples:
                example_maze_rep = self.format_maze_for_prompt(ex_maze, maze_format)
                example_str += f"Maze:\n{example_maze_rep}\nPath: {ex_path}\n\n"
            prompt = f"{base_instruction}\n\n{example_str}Maze:\n{maze_representation}\n\nPath:"
        elif prompt_type == "chain_of_thought":
            prompt = f"{base_instruction}\n\nThink step-by-step to find the optimal path. First, analyze the maze, then describe your reasoning, and finally, provide the path.\n\nMaze:\n{maze_representation}\n\nThought Process:"
        else:
            raise ValueError("Unsupported prompt type.")
        return prompt

    def parse_llm_output(self, output_text):
        """
        Parses the LLM's output string into a list of (row, col) tuples.
        Handles variations in formatting (e.g., spaces, no spaces, different brackets).
        """
        try:
            # Look for common list-like patterns
            match = re.search(r'\[\s*(?:\(\d+,\s*\d+\),?\s*)+\s*\]', output_text)
            if match:
                path_str = match.group(0)
                # Use ast.literal_eval for safe parsing of list of tuples
                import ast
                path = ast.literal_eval(path_str)
                # Ensure each element is a tuple of two integers
                if all(isinstance(p, tuple) and len(p) == 2 and
                       isinstance(p[0], int) and isinstance(p[1], int) for p in path):
                    return path

            # If direct list parsing fails, try more robust regex for coordinates
            coordinates = re.findall(r'\((\d+),\s*(\d+)\)', output_text)
            if coordinates:
                path = [(int(r), int(c)) for r, c in coordinates]
                return path

            return [] # Return empty list if parsing fails
        except Exception as e:
            print(f"Error parsing LLM output: {e}\nOutput: {output_text}")
            return []

    def get_llm_plan(self, maze: Maze, prompt_type="zero_shot", maze_format="ascii", examples=None, max_tokens=256, temperature=0.7):
        prompt = self.construct_prompt(maze, prompt_type, maze_format, examples)
        messages = [{"role": "user", "content": prompt}]

        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            end_time = time.time()
            response_text = response.choices[0].message.content.strip()
            token_usage = response.usage.total_tokens if response.usage else None

            parsed_path = self.parse_llm_output(response_text)

            # If using chain-of-thought, the actual path might be at the end.
            if prompt_type == "chain_of_thought":
                # A more sophisticated parser might extract the last list of coordinates
                # For now, we assume the path is the primary thing we want to parse.
                pass # The generic parser should still work

            return parsed_path, end_time - start_time, token_usage, response_text
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return [], None, None, None

# Mock LLM Interface for development/testing without OpenAI API calls
class MockLLMInterface:
    def __init__(self, api_key=None, model_name=None):
        pass
    def format_maze_for_prompt(self, maze, format_type): return "Mock Maze Rep"
    def construct_prompt(self, maze, prompt_type, maze_format, examples): return "Mock Prompt"
    def parse_llm_output(self, output_text):
        # Simple mock path, replace with more realistic parsing for actual use
        return [(0,0), (0,1), (1,1)]

    # The change has been applied here:
    def get_llm_plan(self, maze, prompt_type="zero_shot", maze_format="ascii", examples=None, max_tokens=256, temperature=0.7):
        # Mock LLM response
        mock_path = None
        if maze.start == (0,0) and maze.goal == (maze.rows-1, maze.cols-1):
             # Try to simulate a straight path if possible
            mock_path = [(r, c) for r in range(maze.rows) for c in range(maze.cols) if r == c and maze.grid[r,c] == 0]
            if not mock_path: # Fallback if no diagonal
                 mock_path = [(0,0), (0,1), (1,1), (1,2)] # Just a sample

            # Ensure start and goal are in path
            if maze.start not in mock_path: mock_path.insert(0, maze.start)
            if maze.goal not in mock_path: mock_path.append(maze.goal)

        return mock_path, 0.5, 50, "Mock LLM Output Text"