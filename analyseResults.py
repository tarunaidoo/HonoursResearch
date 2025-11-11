import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import re # For extracting model/prompt details

def analyze_success_by_size(file_path):
    """
    Groups the experiment results by maze size and algorithm configuration
    to plot success rate trends.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at path: {file_path}")
        return

    # Convert success_rate to numeric, treating failures/inf as 0.0
    df['success_rate'] = pd.to_numeric(df['success_rate'], errors='coerce').fillna(0.0)

    # 1. Clean and simplify the algorithm names for plotting
    def simplify_algorithm_name(name):
        if name in ['BFS', 'A_Star']:
            return name
        
        # Extract LLM details: Model_PromptType_Format
        match = re.match(r'LLM_([^_]+)_([^_]+)_([^_]+)', name)
        if match:
            model = match.group(1).replace('gpt-', '')
            prompt_type = match.group(2).replace('_shot', '') # few_shot -> few
            maze_format = match.group(3).replace('_array', '').replace('_list', '')
            return f"{model}-{prompt_type}-{maze_format}"
        return name

    df['simplified_algorithm'] = df['algorithm'].apply(simplify_algorithm_name)
    
    # Extract numerical maze size (e.g., from "(8, 8)" to 8)
    df['size_n'] = df['maze_size'].str.extract(r'\((\d+),').astype(int)
    
    # 2. Group the data to get the average success rate per configuration/size
    success_summary = df.groupby(['size_n', 'simplified_algorithm'])['success_rate'].mean().reset_index()
    success_summary['success_percentage'] = success_summary['success_rate'] * 100

    # 3. Separate Classical and LLM data for distinct plotting
    classical_df = success_summary[success_summary['simplified_algorithm'].isin(['BFS', 'A_Star'])]
    llm_df = success_summary[~success_summary['simplified_algorithm'].isin(['BFS', 'A_Star'])]
    
    # 4. Generate the Plot
    
    plt.figure(figsize=(14, 8))
    
    # Plot Classical Algorithms (Should be a flat 100% line)
    for algo in classical_df['simplified_algorithm'].unique():
        data = classical_df[classical_df['simplified_algorithm'] == algo]
        plt.plot(data['size_n'], data['success_percentage'], 
                 label=f'Classical: {algo}', linestyle='--', marker='o', color='gray')

    # Plot LLM Configurations
    # Use a color cycle for better differentiation
    colors = plt.cm.get_cmap('tab20', len(llm_df['simplified_algorithm'].unique()))
    
    for i, (algo_name, data) in enumerate(llm_df.groupby('simplified_algorithm')):
        plt.plot(data['size_n'], data['success_percentage'], 
                 label=algo_name, marker='.', color=colors(i))

    # Logarithmic scale for better visual comparison of large size differences
    plt.xscale('log', base=2) 
    
    # Annotations and Labels
    plt.title('Path Validity Success Rate vs. Maze Size by Configuration')
    plt.xlabel('Maze Size (N) - Log Scale Base 2')
    plt.ylabel('Success Rate (%)')
    plt.yticks(np.arange(0, 101, 10))
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to fit legend
    
    output_image_name = "success_rate_by_size_analysis.png"
    plt.savefig(output_image_name)
    plt.close()
    print(f"\n--- Detailed Success Rate Analysis Complete ---")
    print(f"Saved trend plot to: {output_image_name}")


if __name__ == "__main__":
    # Use the file you recently uploaded as the default
    results_file = "experiment_results_20251028_205123.csv" 
    
    if os.path.exists(results_file):
        analyze_success_by_size(results_file)
    else:
        print(f"File '{results_file}' not found. Please verify the file name.")