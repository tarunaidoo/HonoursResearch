# import os
# import time
# import re # For parsing LLM output
# from openai import OpenAI
# from mazeEnvironment import Maze 
# import ast # Needed for safe parsing in parse_llm_output

# class LLMInterface:
#     def __init__(self, api_key, model_name="gpt-3.5-turbo"):
#         self.client = OpenAI(api_key=api_key)
#         self.model_name = model_name

#     def format_maze_for_prompt(self, maze: Maze, format_type="ascii"):
#         """
#         Formats the maze into a string representation for the LLM.
#         Options: "ascii", "coordinate_list", "flattened_array"
#         """
#         if format_type == "ascii":
#             maze_str = ""
#             for r in range(maze.rows):
#                 row_chars = []
#                 for c in range(maze.cols):
#                     if (r, c) == maze.start:
#                         row_chars.append('S')
#                     elif (r, c) == maze.goal:
#                         row_chars.append('G')
#                     elif maze.grid[r, c] == 1:
#                         row_chars.append('#')
#                     else:
#                         row_chars.append('.')
#                 # Use a space separator for ASCII grids for clarity
#                 maze_str += " ".join(row_chars) + "\n"
#             return maze_str.strip()
#         elif format_type == "coordinate_list":
#             obstacles = []
#             for r in range(maze.rows):
#                 for c in range(maze.cols):
#                     if maze.grid[r, c] == 1:
#                         obstacles.append(f"({r},{c})")
#             return f"Maze size: ({maze.rows},{maze.cols})\nStart: ({maze.start[0]},{maze.start[1]})\nGoal: ({maze.goal[0]},{maze.goal[1]})\nObstacles: [{', '.join(obstacles)}]"
#         elif format_type == "flattened_array":
#             flattened_grid = maze.grid.flatten().tolist()
#             # Replace 0 with '.' and 1 with '#' for better LLM understanding
#             grid_chars = []
#             for i, val in enumerate(flattened_grid):
#                 r, c = divmod(i, maze.cols)
#                 if (r, c) == maze.start:
#                     grid_chars.append('S')
#                 elif (r, c) == maze.goal:
#                     grid_chars.append('G')
#                 elif val == 1:
#                     grid_chars.append('#')
#                 else:
#                     grid_chars.append('.')
#             return f"Maze grid (row-major, {maze.rows}x{maze.cols}): {''.join(grid_chars)}\nStart: {maze.start}\nGoal: {maze.goal}"
#         else:
#             raise ValueError("Unsupported prompt format type.")

#     def construct_prompt(self, maze: Maze, prompt_type="zero_shot", maze_format="ascii", examples=None):
#         maze_representation = self.format_maze_for_prompt(maze, maze_format)

#         base_instruction = "You are an AI assistant designed to solve mazes. Your task is to find a path from the 'S' (start) to the 'G' (goal) in the provided maze. You can only move up, down, left, or right. Avoid '#' (obstacles). Output your path as a list of (row,column) coordinates, starting with 'S' and ending with 'G'. For example: [(0,0), (0,1), (1,1)]"

#         if prompt_type == "zero_shot":
#             prompt = f"{base_instruction}\n\nMaze:\n{maze_representation}\n\nPath:"
#         elif prompt_type == "few_shot":
#             if not examples:
#                 raise ValueError("Few-shot prompting requires examples.")
#             example_str = ""
#             for ex_maze, ex_path in examples:
#                 example_maze_rep = self.format_maze_for_prompt(ex_maze, maze_format)
#                 example_str += f"Maze:\n{example_maze_rep}\nPath: {ex_path}\n\n"
#             prompt = f"{base_instruction}\n\n{example_str}Maze:\n{maze_representation}\n\nPath:"
#         elif prompt_type == "chain_of_thought":
#             prompt = f"{base_instruction}\n\nThink step-by-step to find the optimal path. First, analyze the maze, then describe your reasoning, and finally, provide the path.\n\nMaze:\n{maze_representation}\n\nThought Process:"
#         else:
#             raise ValueError("Unsupported prompt type.")
#         return prompt

#     def parse_llm_output(self, output_text):
#         """
#         Parses the LLM's output string into a list of (row, col) tuples.
#         """
#         try:
#             # Look for common list-like patterns
#             match = re.search(r'\[\s*(?:\(\d+,\s*\d+\),?\s*)+\s*\]', output_text)
#             if match:
#                 path_str = match.group(0)
#                 path = ast.literal_eval(path_str)
#                 # Ensure each element is a tuple of two integers
#                 if all(isinstance(p, tuple) and len(p) == 2 and
#                        isinstance(p[0], int) and isinstance(p[1], int) for p in path):
#                     return path

#             # If direct list parsing fails, try more robust regex for coordinates
#             coordinates = re.findall(r'\((\d+),\s*(\d+)\)', output_text)
#             if coordinates:
#                 path = [(int(r), int(c)) for r, c in coordinates]
#                 return path

#             return [] # Return empty list if parsing fails
#         except Exception as e:
#             print(f"Error parsing LLM output: {e}\nOutput: {output_text}")
#             return []

#     # --- MODIFIED FUNCTION ---
#     def get_llm_plan(self, maze: Maze, model_name=None, prompt_type="zero_shot", maze_format="ascii", examples=None, max_tokens=256, temperature=0.7):
#         """
#         Calls the OpenAI API to get a path plan.
        
#         Note: The 'model_name' parameter overrides the model set in __init__ for this call.
#         """
#         final_model = model_name if model_name else self.model_name # Use the passed model name
        
#         prompt = self.construct_prompt(maze, prompt_type, maze_format, examples)
#         messages = [{"role": "user", "content": prompt}]

#         start_time = time.time()
#         try:
#             response = self.client.chat.completions.create(
#                 model=final_model, # <-- Use the determined model name
#                 messages=messages,
#                 max_tokens=max_tokens,
#                 temperature=temperature
#             )
#             end_time = time.time()
#             response_text = response.choices[0].message.content.strip()
#             token_usage = response.usage.total_tokens if response.usage else None

#             parsed_path = self.parse_llm_output(response_text)

#             # If using chain-of-thought, the actual path might be at the end.
#             if prompt_type == "chain_of_thought":
#                 pass 

#             return parsed_path, end_time - start_time, token_usage, response_text
#         except Exception as e:
#             print(f"Error calling OpenAI API: {e}")
#             return [], None, None, None

# # Mock LLM Interface for development/testing without OpenAI API calls
# class MockLLMInterface:
#     def __init__(self, api_key=None, model_name=None):
#         pass
        
#     def format_maze_for_prompt(self, maze, format_type):
#         # Replicated logic to calculate prompt length accurately
#         if format_type == "ascii":
#             maze_str = ""
#             for r in range(maze.rows):
#                 row_chars = []
#                 for c in range(maze.cols):
#                     if (r, c) == maze.start:
#                         row_chars.append('S')
#                     elif (r, c) == maze.goal:
#                         row_chars.append('G')
#                     elif maze.grid[r, c] == 1:
#                         row_chars.append('#')
#                     else:
#                         row_chars.append('.')
#                 maze_str += " ".join(row_chars) + "\n"
#             return maze_str.strip()
#         # Simple string for other formats (assuming 'ascii' is the slow one)
#         return f"Mock Maze Rep for {format_type}"

#     def construct_prompt(self, maze, prompt_type, maze_format, examples):
#         maze_representation = self.format_maze_for_prompt(maze, maze_format)
#         base_instruction = "You are an AI assistant designed to solve mazes. Your task is to find a path from the 'S' (start) to the 'G' (goal) in the provided maze. You can only move up, down, left, or right. Avoid '#' (obstacles). Output your path as a list of (row,column) coordinates, starting with 'S' and ending with 'G'. For example: [(0,0), (0,1), (1,1)]"
#         return f"{base_instruction}\n\nMaze:\n{maze_representation}\n\nPath:"

#     def parse_llm_output(self, output_text):
#         # This function isn't strictly needed for the mock, but we'll return a mock path
#         return [(0,0), (0,1), (1,1)]

#     # --- MODIFIED FUNCTION WITH TIMING AND MODEL SCALING ---
#     def get_llm_plan(self, maze, model_name=None, prompt_type="zero_shot", maze_format="ascii", examples=None, max_tokens=256, temperature=0.7):
#         # 1. Generate the prompt to determine its length
#         prompt = self.construct_prompt(maze, prompt_type, maze_format, examples)
        
#         # 2. Calculate Token Count and Simulated Time Delay
#         input_token_count = len(prompt) // 4 # Rough estimation
        
#         # Base timing parameters
#         BASE_DELAY = 1.0  # seconds fixed overhead
#         PER_TOKEN_DELAY = 0.00005 # 0.05ms per token
#         SCALER = 1.0
        
#         # Scaling based on model name (Simulates differences in processing power/latency)
#         if model_name in ["gpt-4-turbo"]:
#             SCALER *= 1.5 
#         elif model_name in ["gpt-4o"]:
#             SCALER *= 1.2 # Assume 4o is faster than 4-turbo for some tasks
            
#         # Scaling based on prompt type
#         if prompt_type == "chain_of_thought":
#              SCALER *= 1.5 

#         # Exponential scaling for very large context windows (the main time drain)
#         if input_token_count > 5000:
#              SCALER *= 2.0 
        
#         simulated_time = BASE_DELAY + (input_token_count * PER_TOKEN_DELAY * SCALER)
        
#         # Cap the time to prevent infinite waiting
#         MAX_MOCK_TIME = 120.0 # Cap at 2 minutes per maze
#         simulated_time = min(simulated_time, MAX_MOCK_TIME)
        
#         # 3. Simulate the wait time
#         time.sleep(simulated_time)

#         # 4. Generate Mock Output
#         # (Mock path generation remains simple for speed, ensuring start/goal are present)
#         mock_path = [(0,0)] 
#         if maze.rows > 1 and maze.cols > 1:
#              mock_path.append((1,1))
#         mock_path.append((maze.rows-1, maze.cols-1))

#         output_token_count = max_tokens 
#         total_token_usage = input_token_count + output_token_count
#         mock_output_text = f"Mock LLM Output Text (Model: {model_name}). Path: {mock_path}"

#         # Return the mock path, the SIMULATED time, total tokens, and mock raw output
#         return mock_path, simulated_time, total_token_usage, mock_output_text

import os
import time
import re
from openai import OpenAI
from mazeEnvironment import Maze 
import json # Changed from 'ast' to 'json' for efficient parsing
import numpy as np # Needed for MockLLMInterface simple path generation

class LLMInterface:
    def __init__(self, api_key, model_name="gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def format_maze_for_prompt(self, maze: Maze, format_type="ascii"):
        """
        Formats the maze into a string representation for the LLM.
        Optimized to use list comprehensions and single join operations.
        """
        if format_type == "ascii":
            grid_lines = []
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
                # EFFICIENCY: Use NO separator to reduce token count (faster/cheaper)
                grid_lines.append("".join(row_chars)) 
            return "\n".join(grid_lines).strip()
        
        elif format_type == "coordinate_list":
            # OPTIMIZATION: Use list comprehension for faster object creation
            obstacles = [
                f"({r},{c})" 
                for r in range(maze.rows) 
                for c in range(maze.cols) 
                if maze.grid[r, c] == 1
            ]
            return f"Maze size: ({maze.rows},{maze.cols})\nStart: ({maze.start[0]},{maze.start[1]})\nGoal: ({maze.goal[0]},{maze.goal[1]})\nObstacles: [{', '.join(obstacles)}]"
        
        elif format_type == "flattened_array":
            grid_chars = []
            for r in range(maze.rows):
                for c in range(maze.cols):
                    if (r, c) == maze.start:
                        grid_chars.append('S')
                    elif (r, c) == maze.goal:
                        grid_chars.append('G')
                    elif maze.grid[r, c] == 1:
                        grid_chars.append('#')
                    else:
                        grid_chars.append('.')
            return f"Maze grid (row-major, {maze.rows}x{maze.cols}): {''.join(grid_chars)}\nStart: {maze.start}\nGoal: {maze.goal}"
        
        else:
            raise ValueError("Unsupported prompt format type.")

    def construct_prompt(self, maze: Maze, prompt_type="zero_shot", maze_format="ascii", examples=None):
        maze_representation = self.format_maze_for_prompt(maze, maze_format)

        # EFFICIENCY: Instruct JSON output for reliable and fast parsing
        base_instruction = "You are an AI assistant designed to solve mazes. Your task is to find a path from the 'S' (start) to the 'G' (goal) in the provided maze. You can only move up, down, left, or right. Avoid '#' (obstacles). **Your final output MUST be a JSON object with a single key 'path' containing a list of [row, column] coordinates. Do not include any text outside the JSON.** Example JSON: {\"path\": [[0,0], [0,1], [1,1]]}"

        if prompt_type == "zero_shot":
            prompt = f"{base_instruction}\n\nMaze:\n{maze_representation}\n\nJSON Output:"
        elif prompt_type == "few_shot":
            if not examples:
                raise ValueError("Few-shot prompting requires examples.")
            example_str = ""
            for ex_maze, ex_path in examples:
                example_maze_rep = self.format_maze_for_prompt(ex_maze, maze_format)
                example_str += f"Maze:\n{example_maze_rep}\nJSON Output: {ex_path}\n\n"
            prompt = f"{base_instruction}\n\n{example_str}Maze:\n{maze_representation}\n\nJSON Output:"
        elif prompt_type == "chain_of_thought":
            # CoT adds overhead, but we instruct the final answer to be JSON
            prompt = f"{base_instruction}\n\nThink step-by-step to find the path. First, analyze the maze, then describe your reasoning, and finally, provide the JSON path.\n\nMaze:\n{maze_representation}\n\nThought Process:"
        else:
            raise ValueError("Unsupported prompt type.")
            
        return prompt

    def parse_llm_output(self, output_text):
        """
        Parses the LLM's JSON output string into a list of (row, col) tuples.
        EFFICIENCY: Uses standard json library for fast and robust parsing.
        """
        try:
            # Attempt to find and load the JSON block
            # For CoT, the LLM might wrap the JSON in text, so we look for the first '{' and last '}'
            match = re.search(r'\{.*\}', output_text, re.DOTALL)
            if not match:
                raise json.JSONDecodeError("No JSON structure found.", output_text, 0)
            
            data = json.loads(match.group(0))
            path_list = data.get("path", [])
            
            # Convert list of lists (from JSON) to list of tuples (expected format)
            # This uses a list comprehension for efficiency
            path = [tuple(p) for p in path_list]

            # Basic validation
            if all(isinstance(p, tuple) and len(p) == 2 for p in path):
                return path

            return []
        except json.JSONDecodeError as e:
            # print(f"Error decoding JSON from LLM output: {e}\nOutput: {output_text}")
            return []
        except Exception as e:
            # print(f"General error parsing LLM output: {e}\nOutput: {output_text}")
            return []

    # --- MODIFIED FUNCTION ---
    def get_llm_plan(self, maze: Maze, model_name=None, prompt_type="zero_shot", maze_format="ascii", examples=None, max_tokens=256, temperature=0.7):
        """
        Calls the OpenAI API to get a path plan.
        EFFICIENCY: Now includes 'response_format' to force JSON output.
        """
        final_model = model_name if model_name else self.model_name
        
        prompt = self.construct_prompt(maze, prompt_type, maze_format, examples)
        messages = [{"role": "user", "content": prompt}]

        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=final_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                # CRITICAL EFFICIENCY CHANGE: Force the model to output a JSON object
                response_format={"type": "json_object"} 
            )
            end_time = time.time()
            response_text = response.choices[0].message.content.strip()
            token_usage = response.usage.total_tokens if response.usage else None

            # Parsing is now much cleaner because we forced JSON output
            parsed_path = self.parse_llm_output(response_text)

            return parsed_path, end_time - start_time, token_usage, response_text
        except Exception as e:
            # print(f"Error calling OpenAI API: {e}")
            return [], None, None, f"API Error: {e}"

# Mock LLM Interface remains largely the same, but imports numpy for simplicity.
class MockLLMInterface:
    def __init__(self, api_key=None, model_name=None):
        pass
        
    def format_maze_for_prompt(self, maze, format_type):
        # Replicated logic for rough mock prompt generation/length
        if format_type == "ascii":
            grid_lines = []
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
                grid_lines.append("".join(row_chars))
            return "\n".join(grid_lines).strip()
        return f"Mock Maze Rep for {format_type}"

    def construct_prompt(self, maze, prompt_type, maze_format, examples):
        maze_representation = self.format_maze_for_prompt(maze, maze_format)
        base_instruction = "You are an AI assistant designed to solve mazes..."
        return f"{base_instruction}\n\nMaze:\n{maze_representation}\n\nPath:"

    def parse_llm_output(self, output_text):
        # Mock parsing simply returns a dummy path
        return [(0,0), (0,1), (1,1)]

    # --- MOCK FUNCTION WITH TIMING AND MODEL SCALING ---
    def get_llm_plan(self, maze, model_name=None, prompt_type="zero_shot", maze_format="ascii", examples=None, max_tokens=256, temperature=0.7):
        prompt = self.construct_prompt(maze, prompt_type, maze_format, examples)
        
        # Calculate Token Count and Simulated Time Delay
        input_token_count = len(prompt) // 4
        
        BASE_DELAY = 1.0
        PER_TOKEN_DELAY = 0.00005
        SCALER = 1.0
        
        if model_name in ["gpt-4-turbo"]:
            SCALER *= 1.5
        elif model_name in ["gpt-4o"]:
            SCALER *= 1.2
            
        if prompt_type == "chain_of_thought":
             SCALER *= 1.5 

        # Exponential scaling for very large context windows (the main time drain)
        if input_token_count > 5000:
             SCALER *= 2.0
            
        simulated_time = BASE_DELAY + (input_token_count * PER_TOKEN_DELAY * SCALER)
        
        MAX_MOCK_TIME = 120.0
        simulated_time = min(simulated_time, MAX_MOCK_TIME)
        
        # Simulate the wait time
        time.sleep(simulated_time)

        # Generate Mock Output
        mock_path = [maze.start]
        if maze.rows > 1 and maze.cols > 1:
              mock_path.append(tuple(np.add(maze.start, (1, 0)))) # Try to move one step
        mock_path.append(maze.goal)

        output_token_count = max_tokens
        total_token_usage = input_token_count + output_token_count
        mock_output_text = f"{{\"path\": {mock_path}}}" # Mocked JSON output for consistency

        return mock_path, simulated_time, total_token_usage, mock_output_text