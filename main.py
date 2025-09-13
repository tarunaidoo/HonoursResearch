import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pickle
import numpy as np # Needed for obstacle density calculation

# Import classes from their respective modules
from mazeEnvironment import Maze
from mazeGenerator import MazeGenerator
from mazeSolver import MazeSolver
from pathEval import PathEvaluator
from llm_interface import LLMInterface, MockLLMInterface # Import both real and mock

def load_mazes(file_path="pregenerated_mazes.pkl"):
    """
    Loads a list of Maze objects from a pickle file.
    
    Args:
        file_path (str): The path to the file containing the mazes.

    Returns:
        list: A list of Maze objects.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Maze file not found at {file_path}. Please run generate_mazes.py first.")
    print(f"Loading mazes from {file_path}...")
    with open(file_path, "rb") as f:
        return pickle.load(f)

def run_experiment(mazes, llm_interface, depth_limits=[None, 10, 20], llm_prompt_types=["zero_shot", "chain_of_thought"], llm_maze_formats=["ascii", "coordinate_list"]):
    """
    Runs an experiment on a pre-generated set of mazes.
    
    Args:
        mazes (list): A list of Maze objects to be used for the experiment.
        llm_interface (LLMInterface or MockLLMInterface): The interface to the LLM.
        depth_limits (list): DFS depth limits to test.
        llm_prompt_types (list): Types of LLM prompts to test.
        llm_maze_formats (list): Formats of the maze representation for the LLM prompt.

    Returns:
        pd.DataFrame: A DataFrame containing all experiment results.
    """
    results = []

    for maze in tqdm(mazes, desc="Processing pre-generated mazes"):
        evaluator = PathEvaluator(maze)
        solver = MazeSolver(maze)
        
        # Determine maze properties from the loaded object
        size = (maze.rows, maze.cols)
        density = np.sum(maze.grid == 1) / (maze.rows * maze.cols)
        
        # Classical Algorithms
        algorithms = {
            "BFS": solver.bfs,
            "A_Star": solver.a_star,
            "DFS_NoLimit": lambda: solver.dfs(depth_limit=None)
        }
        
        for dl in depth_limits:
            algorithms[f"DFS_DL_{dl if dl is not None else 'Inf'}"] = lambda dl=dl: solver.dfs(depth_limit=dl)

        optimal_path, optimal_time, _ = solver.bfs()
        optimal_path_length = evaluator.calculate_path_length(optimal_path) if optimal_path else 0
        
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
                "path_validity_check": evaluator.is_valid_path(path)
            })

        # LLM Experiments
        for prompt_type in llm_prompt_types:
            for maze_format in llm_maze_formats:
                llm_path, llm_time, llm_tokens, llm_raw_output = llm_interface.get_llm_plan(
                    maze, prompt_type=prompt_type, maze_format=maze_format
                )
                llm_metrics = evaluator.evaluate_path(llm_path, optimal_path_length,
                                                     computation_time=llm_time, token_usage=llm_tokens)
                results.append({
                    "maze_size": size,
                    "obstacle_density": density,
                    "maze_complexity_score": llm_metrics["maze_complexity_score"],
                    "algorithm": f"LLM_{prompt_type}_{maze_format}",
                    "path_length": llm_metrics["path_length"],
                    "optimality_ratio": llm_metrics["optimality_ratio"],
                    "success_rate": llm_metrics["success_rate"],
                    "computation_time": llm_metrics["computation_time"],
                    "nodes_expanded": None,
                    "token_usage": llm_metrics["token_usage"],
                    "raw_output": llm_raw_output,
                    "path_validity_check": evaluator.is_valid_path(llm_path)
                })
    return pd.DataFrame(results)

def analyze_and_visualize_results(df):
    """Generates and displays analysis plots from the experiment results."""
    # Ensure the 'maze_size' column is a proper string for plotting
    df['maze_size_str'] = df['maze_size'].astype(str)

    # Example visualizations
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="algorithm", y="success_rate", hue="maze_size_str")
    plt.title("Success Rate by Algorithm and Maze Size")
    plt.ylabel("Success Rate")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df[df["success_rate"] == 1], x="maze_complexity_score", y="optimality_ratio", hue="algorithm", marker='o')
    plt.title("Optimality Ratio vs. Maze Complexity")
    plt.ylabel("Optimality Ratio (Actual/Optimal)")
    plt.xlabel("Maze Complexity Score")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="maze_size_str", y="computation_time", hue="algorithm", marker='o', sort=False)
    plt.title("Computation Time vs. Maze Size")
    plt.ylabel("Time (seconds)")
    plt.xlabel("Maze Size")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # LLM specific metrics
    llm_df = df[df["algorithm"].str.startswith("LLM")]
    if not llm_df.empty:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=llm_df, x="algorithm", y="token_usage", hue="maze_size_str")
        plt.title("Token Usage for LLMs by Maze Size")
        plt.ylabel("Total Tokens")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # --- Main Experiment Flow ---
    
    # 1. Load the pre-generated mazes from the file.
    try:
        pregenerated_mazes = load_mazes()
    except FileNotFoundError as e:
        print(e)
        print("Exiting. Please run the 'generate_mazes.py' script first to create the dataset.")
        exit()

    # 2. Initialize the LLM client.
    # To use the real OpenAI API, uncomment the following and set your API key
    # os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
    # llm_client = LLMInterface(api_key=os.getenv("OPENAI_API_KEY"))

    # For testing without a real API key, use the MockLLMInterface:
    llm_client = MockLLMInterface()

    # 3. Run the experiment on the loaded mazes.
    print("\nRunning experiment on pre-generated mazes...")
    test_results_df = run_experiment(
        pregenerated_mazes,
        llm_client,
        depth_limits=[None], # Only one DFS type for quick testing
        llm_prompt_types=["zero_shot"],
        llm_maze_formats=["ascii"]
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
    print(test_results_df.groupby("algorithm")["success_rate"].mean())

    print("\nAverage Path Lengths (where successful):")
    print(test_results_df[test_results_df["success_rate"] == 1].groupby("algorithm")["path_length"].mean())

    print("\nAverage Computation Times:")
    print(test_results_df.groupby("algorithm")["computation_time"].mean())

    print("\nGenerating visualizations...")
    analyze_and_visualize_results(test_results_df)