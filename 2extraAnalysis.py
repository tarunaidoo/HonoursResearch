import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import re 

def get_llm_label(name):
    """
    Parses the algorithm name into a shortened, plot-friendly label.
    Includes the robust regex fix from previous steps.
    """
    name = str(name).strip() 
    
    # 1. Classical Baselines
    if name in ['BFS', 'A_Star']:
        return name
    
    # 2. LLM Parsing: LLM_<model>_<prompt>_<format>
    match = re.match(r'LLM_([^_]+)_([^_]+)_(.*)', name)
    if match:
        model = match.group(1).replace('gpt-', '') # e.g., 3.5-turbo, 4o
        prompt_type = match.group(2)
        maze_format = match.group(3)

        # Simplify Prompt Type
        if 'zero_shot' in prompt_type:
            prompt_type_short = 'ZS'
        elif 'few_shot' in prompt_type:
            prompt_type_short = 'FS'
        elif 'chain_of_thought' in prompt_type or 'cot' in prompt_type:
            prompt_type_short = 'CoT'
        elif 'verifier' in prompt_type:
            prompt_type_short = 'Verif'
        else:
            prompt_type_short = 'Other'
            
        # Simplify Format
        if 'ascii' in maze_format:
            format_short = 'A'
        elif 'coordinate_list' in maze_format or 'flattened_array' in maze_format:
            format_short = 'C/F'
        else:
            format_short = 'Other'

        return f"{model}-{prompt_type_short}-{format_short}"
    
    return "Unknown"


def analyze_computation_time(file_path):
    """
    Groups experiment results by maze size and algorithm to plot mean computation time.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at path: {file_path}")
        return

    # Convert computation_time to numeric, ignoring errors (will become NaN and be dropped in aggregation)
    df['computation_time'] = pd.to_numeric(df['computation_time'], errors='coerce')

    # 1. Clean and simplify the algorithm names
    df['simplified_algorithm'] = df['algorithm'].apply(get_llm_label)
    
    # 2. Extract numerical maze size (N)
    df['size_n'] = df['maze_size'].str.extract(r'\((\d+),').astype(int)
    
    # 3. Group the data to get the average computation time per configuration/size
    time_summary = df.groupby(['size_n', 'simplified_algorithm'])['computation_time'].mean().reset_index()

    # 4. Separate Classical and LLM data for distinct plotting styles
    classical_algos = ['BFS', 'A_Star']
    classical_df = time_summary[time_summary['simplified_algorithm'].isin(classical_algos)]
    llm_df = time_summary[~time_summary['simplified_algorithm'].isin(classical_algos)]
    
    # --- Generate the Plot ---
    
    plt.figure(figsize=(12, 7))
    
    # Plot Classical Algorithms (very fast, use distinct style)
    for algo in classical_df['simplified_algorithm'].unique():
        data = classical_df[classical_df['simplified_algorithm'] == algo]
        plt.plot(data['size_n'], data['computation_time'],  
                 label=f'{algo}', 
                 linestyle='-', marker='o', 
                 linewidth=2, markersize=6,
                 color='black')

    # Plot LLM Configurations
    # Use a color cycle for better differentiation
    colors = plt.cm.get_cmap('tab20', len(llm_df['simplified_algorithm'].unique()))
    
    for i, (algo_name, data) in enumerate(llm_df.groupby('simplified_algorithm')):
        plt.plot(data['size_n'], data['computation_time'],  
                 label=algo_name, 
                 marker='.', 
                 linestyle='--', 
                 alpha=0.7, 
                 color=colors(i % 20)) # Ensure color cycle wraps if needed

    # Logarithmic scale for better visual comparison of large time differences
    plt.xscale('log', base=2)
    plt.yscale('log', base=10)
    
    # Annotations and Labels
    plt.title('Computation Time vs. Maze Size by Algorithm Configuration (Log-Log Scale)')
    plt.xlabel('Maze Size (N) - Log Scale Base 2')
    plt.ylabel('Computation Time (seconds) - Log Scale Base 10')
    
    # Set y-ticks based on orders of magnitude (e.g., 0.001s, 0.1s, 1s, 10s)
    y_min = time_summary['computation_time'].min()
    y_max = time_summary['computation_time'].max()
    y_ticks = np.logspace(np.floor(np.log10(y_min)), np.ceil(np.log10(y_max)), 
                          num=int(np.ceil(np.log10(y_max)) - np.floor(np.log10(y_min))) + 1)
    
    plt.yticks(y_ticks, [f'{t:.3g}' for t in y_ticks])
    
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', title='Algorithm (Model-Prompt-Format)')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    output_image_name = "computation_time_by_size_analysis.png"
    plt.savefig(output_image_name)
    plt.close()
    print(f"\n--- Detailed Computation Time Analysis Complete ---")
    print(f"Saved trend plot to: {output_image_name}")


# --- Execution ---
results_file = "experiment_results_20251028_205123.csv"
analyze_computation_time(results_file)