import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pickle
import numpy as np
import json
import gc # Import for garbage collection

# Import classes from their respective modules
from mazeEnvironment import Maze
from mazeGenerator import MazeGenerator
from mazeSolver import MazeSolver
from pathEvaluator import PathEvaluator
from llm_interface import LLMInterface, MockLLMInterface # Import both real and mock

# --- run_experiment function (MODIFIED: Removed DFS/Backtracking Code) ---
def run_experiment(mazes, llm_interface, llm_models, llm_prompt_types=["zero_shot", "few_shot", "chain_of_thought"], llm_maze_formats=["ascii", "coordinate_list", "flattened_array"]):
    """
    Runs an experiment on a pre-generated set of mazes, skipping classical solvers 
    for extremely large mazes and testing multiple LLM models.
    
    NOTE: DFS/Backtracking solvers have been intentionally removed from this function.
    """
    results = []

    # Define the size at which we stop running BFS/A* (e.g., 1024x1024 or smaller)
    CLASSICAL_SOLVER_LIMIT = 1024

    for maze in tqdm(mazes, desc="Processing pre-generated mazes"):
    # Initialize evaluator once per maze
        evaluator = PathEvaluator(maze)
        solver = MazeSolver(maze)
        # Determine maze properties from the loaded object
        rows, cols = size = (maze.rows, maze.cols)
        density = np.sum(maze.grid == 1) / (rows * cols)
        scenario = getattr(maze, 'scenario', 'N/A')

        # Check if the maze size exceeds the classical solver limit
        run_classical_solvers = rows <= CLASSICAL_SOLVER_LIMIT and cols <= CLASSICAL_SOLVER_LIMIT

         # 1. Determine the Optimal Path Length (for evaluation)
        optimal_path, optimal_time, nodes_expanded_bfs = (None, None, None)
        optimal_path_length = 0
        
        if run_classical_solvers:
            # Run BFS to find the optimal path length (Required for Optimality Ratio calculation)
            optimal_path, optimal_time, nodes_expanded_bfs = solver.bfs()
            optimal_path_length = evaluator.calculate_path_length(optimal_path) if optimal_path else 0

        # 2. Classical Algorithms
        if run_classical_solvers:
            algorithms = {
                "BFS": lambda: (optimal_path, optimal_time, nodes_expanded_bfs), # Use the already calculated BFS results
                "A_Star": solver.a_star,
                # DFS/Backtracking solvers removed here.
            }
            
             # The loop for depth_limits is removed entirely as it was only for DFS.

            for alg_name, alg_func in algorithms.items():
                path, comp_time, nodes_expanded = alg_func()
                # Ensure classical solver results are robust
                comp_time = comp_time if comp_time is not None else 0.0
                metrics = evaluator.evaluate_path(path, optimal_path_length, computation_time=comp_time)
                results.append({
                    "maze_size": size,
                    "obstacle_density": density,
                    "maze_complexity_score": metrics["maze_complexity_score"],
                    "algorithm": alg_name,
                    "path_length": metrics["path_length"],
                    "optimality_ratio": metrics["optimality_ratio"],
                    "success_rate": metrics["success_rate"],
                    "computation_time": metrics["computation_time"],
                    "nodes_expanded": nodes_expanded,
                    "token_usage": None,
                    "raw_output": None,
                    "path_validity_check": evaluator.is_valid_path(path),
                    "scenario": scenario # <-- Scenario is captured here
                })
        else:
            tqdm.write(f"Skipping classical solvers for large maze size: {size}. Only running LLM tests.")

         # 3. LLM Experiments (Run for ALL mazes)
        for model_name in llm_models:
            for prompt_type in llm_prompt_types:
                for maze_format in llm_maze_formats:
                    
                    # 1. Execute LLM and get raw output string
                    llm_path_placeholder, llm_time, llm_tokens, llm_raw_output = llm_interface.get_llm_plan(
                        maze,model_name=model_name,prompt_type=prompt_type, maze_format=maze_format
                    )
                    # --- CRITICAL FIX: Ensure output is a string before parsing ---
                    # We also ensure llm_time/llm_tokens are non-None before evaluation
                    if not isinstance(llm_raw_output, str):
                        llm_raw_output = ""
                        
                    llm_time = llm_time if llm_time is not None else 0.0
                    llm_tokens = llm_tokens if llm_tokens is not None else 0
                    
                    # 2. Robustly parse the raw output 
                    llm_path = evaluator.robust_parse_coordinates(llm_raw_output)
                    
                    # 3. Evaluate the robustly parsed path (llm_path)
                    llm_metrics = evaluator.evaluate_path(llm_path,optimal_path_length,computation_time=llm_time,token_usage=llm_tokens)
                    
                    # 4. Append results
                    results.append({
                        "maze_size": size,
                        "obstacle_density": density,
                        "maze_complexity_score": llm_metrics["maze_complexity_score"],
                        "algorithm": f"LLM_{model_name}_{prompt_type}_{maze_format}",
                        "path_length": llm_metrics["path_length"],
                        "optimality_ratio": llm_metrics["optimality_ratio"],
                        "success_rate": llm_metrics["success_rate"],
                        "computation_time": llm_metrics["computation_time"],
                        "nodes_expanded": None,
                        "token_usage": llm_metrics["token_usage"],
                        "raw_output": llm_raw_output, # Still save the raw output for debugging
                        "path_validity_check": evaluator.is_valid_path(llm_path),
                        "scenario": scenario
                    })
                    
        # NEW: Explicitly call garbage collector after processing each maze
        gc.collect()
        
    return pd.DataFrame(results)

# --- load_mazes function (unchanged) ---
def load_mazes(file_path="pregenerated_mazes.json"):
    """
    Loads maze data from a JSON file and reconstructs a list of Maze objects.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Maze file not found at {file_path}. Please run create_save_mazes.py first.")
    
    print(f"Loading mazes from {file_path}...")
    with open(file_path, "r") as f:
        mazes_data = json.load(f)
        
    reconstructed_mazes = []
    for maze_dict in mazes_data:
        grid_array = np.array(maze_dict['grid'], dtype=int)
        start_coords = tuple(maze_dict['start'])
        goal_coords = tuple(maze_dict['goal'])

        maze = Maze(
            grid_array,
            start=start_coords,
            goal=goal_coords
        )
        
        # Load the scenario tag onto the maze object
        maze.scenario = maze_dict.get('scenario', 'default_scenario')

        reconstructed_mazes.append(maze)

    return reconstructed_mazes

# --- MODIFIED analyze_and_visualize_results function (Saves instead of Shows) ---
def analyze_and_visualize_results(df, base_filename):
    """Generates and saves analysis plots from the experiment results, separated by solver type."""

    # Create an output directory for figures if it doesn't exist
    output_dir = "experiment_figures"
    os.makedirs(output_dir, exist_ok=True)

    # --- 0. Feature Engineering for Grouping ---
    df['maze_size_str'] = df['maze_size'].astype(str)
    # Categorize solvers into two groups
    df['solver_group'] = df['algorithm'].apply(lambda x: 'LLM' if x.startswith('LLM') else 'Classical')
    
    # Create a detailed label for LLMs
    llm_mask = df['solver_group'] == 'LLM'
    df.loc[llm_mask, 'full_label'] = df.loc[llm_mask, 'algorithm'].str.replace('LLM_', '', regex=False)
    df.loc[~llm_mask, 'full_label'] = df.loc[~llm_mask, 'algorithm']

    # --- Loop through groups and generate separate plots ---
    solver_groups = df['solver_group'].unique()

    for group in solver_groups:
        group_df = df[df['solver_group'] == group].copy()
        group_tag = group.replace(" ", "_") # 'Classical' or 'LLM'
        x_col = 'full_label' # Use the full label for the X-axis

        print(f"Generating plots for {group} solvers...")

        # 1. Success Rate by Algorithm and SCENARIO
        plt.figure(figsize=(16, 8) if group == 'LLM' else (10, 6))
        sns.barplot(data=group_df, x=x_col, y="success_rate", hue="scenario")
        plt.title(f"{group} Solvers: Success Rate by Configuration and Scenario")
        plt.ylabel("Success Rate")
        plt.ylim(0, 1.05)
        plt.xlabel(f"{group} Configuration")
        plt.xticks(rotation=90, ha='right', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{group_tag}_Success_Rate_{base_filename}.png"))
        plt.close()

        # 2. Optimality Ratio vs. Complexity (Successful Solvers Only) with SCENARIO
        # Filter for successful paths (where optimality ratio is valid and finite)
        plot_df_optimal = group_df[
            (group_df["success_rate"] == 1) & 
            (group_df["optimality_ratio"] > 0) & 
            (group_df["optimality_ratio"] < np.inf)
        ]
        
        # Only plot if there's data after filtering
        if not plot_df_optimal.empty:
            plt.figure(figsize=(16, 8) if group == 'LLM' else (10, 6))
            sns.lineplot(data=plot_df_optimal,
                         x="maze_complexity_score", y="optimality_ratio", hue=x_col,
                         style="scenario", marker='o')
            plt.title(f"{group} Solvers: Optimality Ratio vs. Maze Complexity (Successful Solvers)")
            plt.ylabel("Optimality Ratio (Actual Path Length / Optimal)")
            plt.xlabel("Maze Complexity Score")
            plt.grid(True)
            plt.legend(title=f"{group} Configuration", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout(rect=(0, 0, 0.85, 1)) # Adjust layout for external legend
            plt.savefig(os.path.join(output_dir, f"{group_tag}_Optimality_Ratio_{base_filename}.png"))
            plt.close()

        # 3. Computation Time vs. Maze Size with SCENARIO
        plt.figure(figsize=(16, 8) if group == 'LLM' else (10, 6))
        
        # Adjust Y-axis label based on the group
        y_label = "Time (seconds)"
        if group == 'Classical':
            # Use microseconds for Classical to show meaningful variation
            group_df.loc[:, 'computation_time_micro'] = group_df['computation_time'] * 1e6
            y_col = 'computation_time_micro'
            y_label = "Time (microseconds)"
        else:
            y_col = 'computation_time'
            y_label = "Time (seconds)"
            
        sns.lineplot(data=group_df, x="maze_size_str", y=y_col, hue=x_col, style="scenario", marker='o', sort=False)
        plt.title(f"{group} Solvers: Computation Time vs. Maze Size and Scenario")
        plt.ylabel(y_label)
        plt.xlabel("Maze Size (N x N)")
        plt.grid(True)
        plt.legend(title=f"{group} Configuration", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=(0, 0, 0.85, 1)) # Adjust layout for external legend
        plt.savefig(os.path.join(output_dir, f"{group_tag}_Computation_Time_{base_filename}.png"))
        plt.close()

    # 4. LLM specific metrics (Token Usage by Scenario) - KEEP THIS AS IS (LLM-ONLY)
    llm_df = df[df["algorithm"].str.startswith("LLM")].copy()
    if not llm_df.empty:
        plt.figure(figsize=(16, 8))
        # Use the already created 'full_label' for the x-axis
        sns.barplot(data=llm_df, x='full_label', y="token_usage", hue="scenario")
        plt.title("LLM Token Usage by Model/Prompt Type and Scenario")
        plt.ylabel("Total Tokens")
        plt.xticks(rotation=90, ha='right', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"Token_Usage_{base_filename}.png"))
        plt.close()


if __name__ == "__main__":
    # --- Main Experiment Flow ---

    # 1. Load the pre-generated mazes from the file.
    try:
        pregenerated_mazes = load_mazes("pregenerated_mazes.json")
    except FileNotFoundError as e:
        print(e)
        print("Exiting. Please run the 'create_save_mazes.py' script first to create the dataset.")
        exit()

    # 2. Initialize the LLM client and define models.

    # Define the models to test (e.g., small, medium, large)
    LLM_MODELS_TO_TEST = [ 
                          "gpt-3.5-turbo",# Smallest/Fastest
                          "gpt-4-turbo",       # Medium/Balanced
                          "gpt-4o"  # Latest/Powerful
    ]
    
    # --- PHASE 3 CHANGE: Define full range of prompt variations and formats ---
    LLM_PROMPT_TYPES = ["zero_shot", "few_shot", "chain_of_thought"]
    LLM_MAZE_FORMATS = ["ascii", "coordinate_list", "flattened_array"]
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\nWARNING: OPENAI_API_KEY environment variable not set. Using MockLLMInterface.")
        llm_client = MockLLMInterface()
    else:
        print("\nUsing Real LLMInterface (Ensure you have credits and are prepared for API costs/rate limits).")
        llm_client = LLMInterface(api_key=api_key)
        
    # 3. Run the experiment on the loaded mazes.
    print("\nRunning full LLM and Classical experiment (excluding Backtracking/DFS)...")
    # --- MODIFIED CALL: Removed the depth_limits argument ---
    test_results_df = run_experiment(
        pregenerated_mazes,
        llm_client,
        llm_models=LLM_MODELS_TO_TEST, # <-- Full list
        llm_prompt_types=LLM_PROMPT_TYPES, # <-- Full list
        llm_maze_formats=LLM_MAZE_FORMATS # <-- Full list
    )
    
    # 4. Save and analyze the results.
    print("\nExperiment complete. Saving results...")
    current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"experiment_results_{current_time_str}.csv"
    test_results_df.to_csv(file_name, index=False)
    print(f"Results saved to {file_name}")
    
    # Generate the base file name for figures (without extension)
    base_figure_filename = f"analysis_{current_time_str}"
    
    # Basic analysis example:
    print("\nSample Results:")
    print(test_results_df.head())

    print("\nAverage Success Rates:")
    print(test_results_df.groupby(["algorithm", "scenario"])["success_rate"].mean().sort_values(ascending=False)) # Grouped by scenario

    print("\nAverage Path Lengths (where successful):")
    print(test_results_df[test_results_df["success_rate"] == 1].groupby(["algorithm", "scenario"])["path_length"].mean().sort_values(ascending=True)) # Grouped by scenario

    print("\nAverage Computation Times:")
    print(test_results_df.groupby(["algorithm", "scenario"])["computation_time"].mean().sort_values(ascending=True)) # Grouped by scenario

    print("\nGenerating and saving visualizations to 'experiment_figures/'...")
    # --- MODIFIED CALL: Pass the base filename for saving ---
    analyze_and_visualize_results(test_results_df, base_figure_filename)