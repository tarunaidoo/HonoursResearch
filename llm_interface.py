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
import json # Used for parsing tool call arguments
import numpy as np 
import ast 

class LLMInterface:
    def __init__(self, api_key, model_name="gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    # --- NEW: Define the tool schema for the LLM to use ---
    PATH_SUBMISSION_TOOL = [
        {
            "type": "function",
            "function": {
                "name": "submit_path",
                "description": "Submits the final solution path as a list of coordinates from start 'S' to goal 'G'.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "array",
                            "description": "The list of path coordinates, where each element is a list of two integers: [row, column]. Must start at S and end at G.",
                            "items": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "minItems": 2,
                                "maxItems": 2
                            }
                        }
                    },
                    "required": ["path"],
                },
            }
        }
    ]

    def format_maze_for_prompt(self, maze: Maze, format_type="ascii"):
        """
        Formats the maze into a string representation for the LLM.
        Optimized for string handling and token count (removed spaces in ASCII).
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
                # EFFICIENCY: Removed space separator to reduce tokens/latency
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

        # Base instruction is now focused on using the tool
        base_instruction = "You are an AI assistant designed to solve mazes. Your task is to find a path from 'S' to 'G' by only moving up, down, left, or right, avoiding '#'. **You MUST call the `submit_path` tool with the path coordinates to provide your final answer.**"

        if prompt_type == "zero_shot":
            prompt = f"{base_instruction}\n\nMaze:\n{maze_representation}"
        elif prompt_type == "few_shot":
            if not examples:
                raise ValueError("Few-shot prompting requires examples.")
            example_str = ""
            for ex_maze, ex_path in examples:
                example_maze_rep = self.format_maze_for_prompt(ex_maze, maze_format)
                # Note: Few-shot path should be the tool call arguments, e.g., '{"path": [[...]]}'
                example_str += f"Maze:\n{example_maze_rep}\nTool Call: {ex_path}\n\n"
            prompt = f"{base_instruction}\n\n{example_str}Maze:\n{maze_representation}"
        elif prompt_type == "chain_of_thought":
            prompt = f"{base_instruction}\n\nThink step-by-step to find the path. First, analyze the maze, then describe your reasoning, and finally, call the `submit_path` tool with the coordinates.\n\nMaze:\n{maze_representation}\n\nThought Process:"
        else:
            raise ValueError("Unsupported prompt type.")
            
        return prompt

    def parse_llm_output(self, response_message):
        """
        Parses the LLM's response message object to extract the path from the tool call.
        EFFICIENCY: Uses the structured tool call for robust and fast parsing.
        """
        if response_message.tool_calls:
            # We enforce a single tool call via tool_choice in get_llm_plan
            tool_call = response_message.tool_calls[0]
            if tool_call.function.name == "submit_path":
                try:
                    # Arguments are returned as a JSON string
                    arguments = json.loads(tool_call.function.arguments)
                    path_list = arguments.get("path", [])
                    
                    # Convert list of lists (from tool args) to list of tuples (expected by evaluator)
                    path = [tuple(p) for p in path_list]
                    
                    if all(isinstance(p, tuple) and len(p) == 2 for p in path):
                        return path
                except Exception as e:
                    # print(f"Error processing tool arguments: {e}")
                    return []
        
        # Fallback if the model failed to use the tool
        return []

    # --- MODIFIED FUNCTION ---
    def get_llm_plan(self, maze: Maze, model_name=None, prompt_type="zero_shot", maze_format="ascii", examples=None, max_tokens=256, temperature=0.7):
        """
        Calls the OpenAI API to get a path plan, leveraging Tool Use for structured output
        and reduced parsing latency.
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
                # EFFICIENCY 1: Define the available tools
                tools=self.PATH_SUBMISSION_TOOL,
                # EFFICIENCY 2: Force the model to use the tool for guaranteed, fast output structure
                tool_choice={"type": "function", "function": {"name": "submit_path"}}
            )
            end_time = time.time()
            
            # The response is a message object containing tool call data
            response_message = response.choices[0].message
            # Store the raw object string for logging (raw_output field)
            response_text = str(response_message.tool_calls[0].function.arguments) if response_message.tool_calls else str(response_message)
            token_usage = response.usage.total_tokens if response.usage else None

            # Parse the tool call result
            parsed_path = self.parse_llm_output(response_message)

            return parsed_path, end_time - start_time, token_usage, response_text
        except Exception as e:
            # print(f"Error calling OpenAI API: {e}")
            return [], None, None, f"API Error: {e}"

# Mock LLM Interface remains largely the same
class MockLLMInterface:
    def __init__(self, api_key=None, model_name=None):
        pass
        
    def format_maze_for_prompt(self, maze, format_type):
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

        if input_token_count > 5000:
             SCALER *= 2.0
            
        simulated_time = BASE_DELAY + (input_token_count * PER_TOKEN_DELAY * SCALER)
        
        MAX_MOCK_TIME = 120.0
        simulated_time = min(simulated_time, MAX_MOCK_TIME)
        
        time.sleep(simulated_time)

        # Generate Mock Output
        mock_path = [maze.start]
        if maze.rows > 1 and maze.cols > 1:
              mock_path.append(tuple(np.add(maze.start, (1, 0)))) 
        mock_path.append(maze.goal)

        output_token_count = max_tokens
        total_token_usage = input_token_count + output_token_count
        # Mocked tool call arguments for consistency
        mock_output_text = f"{{\"path\": {[[r, c] for r, c in mock_path]}}}" 

        return mock_path, simulated_time, total_token_usage, mock_output_text