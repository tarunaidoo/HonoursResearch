import os
import time
import re # For parsing LLM output
from openai import OpenAI
from mazeEnvironment import Maze 
import ast # Needed for safe parsing in parse_llm_output
import numpy as np # Needed for flattened_array in Mock

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
                # Use a space separator for ASCII grids for clarity
                maze_str += " ".join(row_chars) + "\n"
            return maze_str.strip()
        elif format_type == "coordinate_list":
            obstacles = []
            for r in range(maze.rows):
                for c in range(maze.cols):
                    # Only list obstacles and special points
                    if maze.grid[r, c] == 1:
                        obstacles.append(f"({r},{c})")
            return f"Maze size: ({maze.rows},{maze.cols})\nStart: ({maze.start[0]},{maze.start[1]})\nGoal: ({maze.goal[0]},{maze.goal[1]})\nObstacles: [{', '.join(obstacles)}]"
        elif format_type == "flattened_array":
            # --- MODIFIED FOR CLARITY ---
            grid_chars = []
            for r in range(maze.rows):
                for c in range(maze.cols):
                    if (r, c) == maze.start:
                        grid_chars.append('S')
                    elif (r, c) == maze.goal:
                        grid_chars.append('G')
                    elif maze.grid[r, c] == 1:
                        grid_chars.append('1') # Use '1' instead of '#' for array-like format
                    else:
                        grid_chars.append('0') # Use '0' instead of '.'
            # Return a simple comma-separated string
            return f"Maze grid (R:{maze.rows}, C:{maze.cols}): {','.join(grid_chars)}\nStart: {maze.start}\nGoal: {maze.goal}"
        else:
            raise ValueError("Unsupported prompt format type.")

    def construct_prompt(self, maze: Maze, prompt_type="zero_shot", maze_format="ascii", examples=None):
        maze_representation = self.format_maze_for_prompt(maze, maze_format)

        base_instruction = "You are an AI assistant designed to solve mazes. Your task is to find a path from the 'S' (start) to the 'G' (goal) in the provided maze. You can only move up, down, left, or right. Avoid obstacles. Output your path as a single Python list of (row,column) coordinates, starting with 'S' and ending with 'G'. Do not include any explanation or extra text outside of the final list. For example: [(0,0), (0,1), (1,1)]"

        if prompt_type == "zero_shot":
            prompt = f"{base_instruction}\n\nMaze:\n{maze_representation}\n\nPath:"
        elif prompt_type == "few_shot":
            # --- EXPANDED FEW-SHOT LOGIC ---
            example_str = ""
            if not examples:
                # Use hardcoded minimal examples if none are provided
                examples = [
                    (Maze(np.array([[0,1],[0,0]], dtype=int), (0,0), (1,1)), "[(0,0), (1,0), (1,1)]"),
                ]
            
            for i, (ex_maze, ex_path) in enumerate(examples):
                example_maze_rep = self.format_maze_for_prompt(ex_maze, maze_format)
                example_str += f"Example {i+1} Maze:\n{example_maze_rep}\nPath: {ex_path}\n\n"
            
            prompt = f"{base_instruction}\n\n{example_str}Current Maze:\n{maze_representation}\n\nPath:"
        elif prompt_type == "chain_of_thought":
            # --- REFINED CoT INSTRUCTION ---
            prompt = f"{base_instruction.replace('Do not include any explanation or extra text outside of the final list.', 'Provide your step-by-step reasoning before the final path.')}\n\nThink step-by-step to find the path. First, analyze the maze, then describe your reasoning, and finally, provide ONLY the Python list for the path.\n\nMaze:\n{maze_representation}\n\nThought Process:"
        else:
            raise ValueError("Unsupported prompt type.")
        return prompt

    def parse_llm_output(self, output_text):
        """
        Parses the LLM's output string into a list of (row, col) tuples.
        """
        try:
            # Look for common list-like patterns
            match = re.search(r'(\[\s*(?:\(\d+,\s*\d+\),?\s*)+\s*\])', output_text)
            if match:
                path_str = match.group(1)
                path = ast.literal_eval(path_str)
                
                # Basic validation
                if isinstance(path, list) and all(isinstance(p, tuple) and len(p) == 2 and 
                                                isinstance(p[0], int) and isinstance(p[1], int) for p in path):
                    return path

            # Fallback: If direct list parsing fails, try robust regex for coordinates
            coordinates = re.findall(r'\((\d+),\s*(\d+)\)', output_text)
            if coordinates:
                path = [(int(r), int(c)) for r, c in coordinates]
                return path

            return [] # Return empty list if parsing fails
        except Exception as e:
            # print(f"Error parsing LLM output: {e}\nOutput: {output_text}") # Uncomment for debugging
            return []

    # --- MODIFIED FUNCTION (Increased max_tokens) ---
    def get_llm_plan(self, maze: Maze, model_name=None, prompt_type="zero_shot", maze_format="ascii", examples=None, max_tokens=1024, temperature=0.7):
        """
        Calls the OpenAI API to get a path plan.
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
                temperature=temperature
            )
            end_time = time.time()
            response_text = response.choices[0].message.content.strip()
            token_usage = response.usage.total_tokens if response.usage else None

            parsed_path = self.parse_llm_output(response_text)

            return parsed_path, end_time - start_time, token_usage, response_text
        except Exception as e:
            # print(f"Error calling OpenAI API: {e}") # Uncomment for debugging
            return [], None, None, None

# Mock LLM Interface for development/testing without OpenAI API calls
class MockLLMInterface:
    def __init__(self, api_key=None, model_name=None):
        self.llm_interface = LLMInterface(api_key="mock", model_name="mock")
        
    def format_maze_for_prompt(self, maze, format_type):
        return self.llm_interface.format_maze_for_prompt(maze, format_type)

    def construct_prompt(self, maze, prompt_type, maze_format, examples):
        return self.llm_interface.construct_prompt(maze, prompt_type, maze_format, examples)

    # --- MODIFIED MOCK FUNCTION WITH TIMING AND MODEL SCALING ---
    def get_llm_plan(self, maze, model_name=None, prompt_type="zero_shot", maze_format="ascii", examples=None, max_tokens=1024, temperature=0.7):
        # 1. Generate the prompt to determine its length
        prompt = self.construct_prompt(maze, prompt_type, maze_format, examples)
        
        # 2. Calculate Token Count and Simulated Time Delay
        input_token_count = len(prompt) // 4 # Rough estimation
        
        # Base timing parameters
        BASE_DELAY = 0.5 
        PER_TOKEN_DELAY = 0.00008 
        SCALER = 1.0
        
        # Scaling based on model name
        if model_name in ["gpt-4-turbo"]:
            SCALER *= 1.8 
        elif model_name in ["gpt-4o"]:
            SCALER *= 1.2
            
        # Scaling based on prompt type (CoT is slower)
        if prompt_type == "chain_of_thought":
             SCALER *= 1.5 
             
        # Scaling based on format (ASCII is slightly longer/more verbose)
        if maze_format == "ascii":
            SCALER *= 1.1

        # Exponential scaling for very large context windows
        if input_token_count > 5000:
             SCALER *= 2.0 
        
        simulated_time = BASE_DELAY + (input_token_count * PER_TOKEN_DELAY * SCALER)
        
        # Cap the time to prevent infinite waiting
        MAX_MOCK_TIME = 60.0 
        simulated_time = min(simulated_time, MAX_MOCK_TIME)
        
        # 3. Simulate the wait time
        time.sleep(simulated_time)

        # 4. Generate Mock Output
        path_length_estimate = max(2, (maze.rows * maze.cols) // 50) 
        
        mock_path = [maze.start]
        # Simple path simulation
        for i in range(path_length_estimate):
            r, c = mock_path[-1]
            if r + 1 < maze.rows:
                mock_path.append((r + 1, c))
            elif c + 1 < maze.cols:
                mock_path.append((r, c + 1))
        
        if mock_path[-1] != maze.goal:
            mock_path.append(maze.goal)

        output_token_count = len(mock_path) * 3
        total_token_usage = input_token_count + output_token_count
        mock_output_text = f"Mock LLM Output Text (Model: {model_name}, Prompt: {prompt_type}, Format: {maze_format}). Path: {mock_path}"

        # Return the mock path, the SIMULATED time, total tokens, and mock raw output
        return mock_path, simulated_time, total_token_usage, mock_output_text