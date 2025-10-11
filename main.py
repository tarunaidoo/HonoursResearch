#API KEY: 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

import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pickle 
import numpy as np
import json 

# Import classes from their respective modules
from mazeEnvironment import Maze
from mazeGenerator import MazeGenerator
from mazeSolver import MazeSolver
from pathEval import PathEvaluator
from llm_interface import LLMInterface, MockLLMInterface 

# --- run_experiment function ---
def run_experiment(mazes, llm_interface, llm_models, depth_limits=[None, 10, 20], llm_prompt_types=["zero_shot", "chain_of_thought"], llm_maze_formats=["ascii", "coordinate_list"]):
    """
    Runs an experiment on a pre-generated set of mazes, skipping classical solvers 
    for extremely large mazes and testing multiple LLM models.
    """
    results = []
    
    # Define the size at which we stop running BFS/A* (e.g., 1024x1024 or smaller)
    CLASSICAL_SOLVER_LIMIT = 1024 

    for maze in tqdm(mazes, desc="Processing pre-generated mazes"):
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
                "DFS_NoLimit": lambda: solver.dfs(depth_limit=None)
            }
            
            for dl in depth_limits:
                algorithms[f"DFS_DL_{dl if dl is not None else 'Inf'}"] = lambda dl=dl: solver.dfs(depth_limit=dl)

            for alg_name, alg_func in algorithms.items():
                path, comp_time, nodes_expanded = alg_func()
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
                    "scenario": scenario
                })
        else:
            tqdm.write(f"Skipping classical solvers for large maze size: {size}. Only running LLM tests.")


        # 3. LLM Experiments (Run for ALL mazes)
        for model_name in llm_models: 
            for prompt_type in llm_prompt_types:
                for maze_format in llm_maze_formats:
                    llm_path, llm_time, llm_tokens, llm_raw_output = llm_interface.get_llm_plan(
                        maze, 
                        model_name=model_name,
                        prompt_type=prompt_type, 
                        maze_format=maze_format
                    )
                    
                    llm_metrics = evaluator.evaluate_path(llm_path, optimal_path_length,
                                                         computation_time=llm_time, token_usage=llm_tokens)
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
                        "raw_output": llm_raw_output,
                        "path_validity_check": evaluator.is_valid_path(llm_path),
                        "scenario": scenario
                    })
    return pd.DataFrame(results)

# --- load_mazes function ---
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

# --- analyze_and_visualize_results function (Updated) ---
def analyze_and_visualize_results(df):
    """Generates and displays analysis plots from the experiment results, leveraging the 'scenario' tag."""
    
    # Ensure the 'maze_size' column is a proper string for plotting
    df['maze_size_str'] = df['maze_size'].astype(str)
    
    # 1. Success Rate by Algorithm and SCENARIO
    plt.figure(figsize=(14, 7))
    sns.barplot(data=df, x="algorithm", y="success_rate", hue="scenario")
    plt.title("Success Rate by Algorithm and Maze Generation Scenario")
    plt.ylabel("Success Rate")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # 2. Optimality Ratio vs. Complexity (Successful Solvers Only) with SCENARIO
    plt.figure(figsize=(14, 7))
    # Filter out cases where optimality is meaningless (no path found) and clip extreme ratios
    plot_df_optimal = df[(df["success_rate"] == 1) & (df["optimality_ratio"] > 0) & (df["optimality_ratio"] < 10)]
    sns.lineplot(data=plot_df_optimal, 
                 x="maze_complexity_score", y="optimality_ratio", hue="algorithm", 
                 style="scenario", marker='o') # Use 'style' to distinguish scenarios on the line plot
    plt.title("Optimality Ratio vs. Maze Complexity (Successful Solvers)")
    plt.ylabel("Optimality Ratio (Actual/Optimal)")
    plt.xlabel("Maze Complexity Score (Obstacle Density-Based)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # 3. Computation Time vs. Maze Size with SCENARIO
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df, x="maze_size_str", y="computation_time", hue="algorithm", style="scenario", marker='o', sort=False)
    plt.title("Computation Time vs. Maze Size and Scenario")
    plt.ylabel("Time (seconds)")
    plt.xlabel("Maze Size")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 4. LLM specific metrics (Token Usage by Scenario)
    llm_df = df[df["algorithm"].str.startswith("LLM")]
    if not llm_df.empty:
        plt.figure(figsize=(14, 7))
        # Group LLM results by model and prompt type, coloring by scenario
        llm_df['llm_type'] = llm_df['algorithm'].apply(lambda x: '_'.join(x.split('_')[1:-1]))
        sns.barplot(data=llm_df, x="llm_type", y="token_usage", hue="scenario")
        plt.title("Token Usage for LLMs by Model/Prompt Type and Scenario")
        plt.ylabel("Total Tokens")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

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
        "gpt-3.5-turbo",      
        "gpt-4-turbo",        
        "gpt-4o"              
    ]
    
    # --- PHASE 3 CHANGE: Define full range of prompt variations and formats ---
    LLM_PROMPT_TYPES = ["zero_shot", "few_shot", "chain_of_thought"]
    LLM_MAZE_FORMATS = ["ascii", "coordinate_list", "flattened_array"]

    # --- LOGIC FOR SWITCHING BETWEEN REAL/MOCK LLM ---
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\nWARNING: OPENAI_API_KEY environment variable not set. Using MockLLMInterface.")
        llm_client = MockLLMInterface()
    else:
        print("\nUsing Real LLMInterface (Ensure you have credits and are prepared for API costs/rate limits).")
        llm_client = LLMInterface(api_key=api_key)


    # 3. Run the experiment on the loaded mazes.
    print("\nRunning full LLM and Classical experiment...")
    # --- PHASE 3 CHANGE: Pass the full lists of parameters to run_experiment ---
    test_results_df = run_experiment(
        pregenerated_mazes,
        llm_client,
        llm_models=LLM_MODELS_TO_TEST, 
        depth_limits=[None, 10, 20],
        llm_prompt_types=LLM_PROMPT_TYPES,
        llm_maze_formats=LLM_MAZE_FORMATS
    )
    
    # 4. Save and analyze the results.
    print("\nExperiment complete. Saving results...")
    current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"experiment_results_{current_time_str}.csv"
    test_results_df.to_csv(file_name, index=False)
    print(f"Results saved to {file_name}")

    # Basic analysis example:
    print("\nSample Results:")
    print(test_results_df.head())

    print("\nAverage Success Rates:")
    print(test_results_df.groupby(["algorithm", "scenario"])["success_rate"].mean().sort_values(ascending=False)) 

    print("\nAverage Path Lengths (where successful):")
    print(test_results_df[test_results_df["success_rate"] == 1].groupby(["algorithm", "scenario"])["path_length"].mean().sort_values(ascending=True)) 

    print("\nAverage Computation Times:")
    print(test_results_df.groupby(["algorithm", "scenario"])["computation_time"].mean().sort_values(ascending=True))

    print("\nGenerating visualizations...")
    analyze_and_visualize_results(test_results_df)