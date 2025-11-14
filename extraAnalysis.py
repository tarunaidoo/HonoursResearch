import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import re

def extract_llm_components(name):
    """Extracts model, prompt_type, and maze_format from the algorithm name."""
    if name in ['BFS', 'A_Star']:
        return None, None, None
    
    # Try to match the standard pattern LLM_<model>_<prompt>_<format>
    match = re.match(r'LLM_([^_]+)_([^_]+)_([^_]+)', name)
    if match:
        model = match.group(1).replace('gpt-', '').replace('-turbo', '').replace('.', '')
        prompt_type = match.group(2)
        maze_format = match.group(3)
        return model, prompt_type, maze_format
    
    # Heuristic for non-standard names (e.g., 'chain_of_thought' where prompt has underscores)
    parts = name.split('_')
    if len(parts) >= 3 and parts[0] == 'LLM':
        model = parts[1].replace('gpt-', '').replace('-turbo', '').replace('.', '')
        
        # Assume the last part is the format, and the rest (after model) is the prompt type
        maze_format = parts[-1] 
        prompt_type_parts = parts[2:-1] if len(parts) > 3 else [parts[2]]
        prompt_type = '_'.join(prompt_type_parts)
        
        return model, prompt_type, maze_format
    
    return None, None, None

def plot_model_breakdown_bar_charts(df):
    """
    Generates a separate clustered bar chart for each LLM model, showing the 
    success rate breakdown across all its configurations and maze sizes.
    """
    generated_files = []
    
    # 1. Data Preparation (Fixing the KeyError)
    df['size_n'] = df['maze_size'].str.extract(r'\((\d+),').astype(int)
    df['success_rate'] = pd.to_numeric(df['success_rate'], errors='coerce').fillna(0.0)

    df[['model', 'prompt_type', 'maze_format']] = df['algorithm'].apply(
        lambda x: pd.Series(extract_llm_components(x))
    )
    
    llm_df = df[df['model'].notna()].copy()
    
    # 2. Group the data for plotting
    success_summary = llm_df.groupby(['size_n', 'model', 'prompt_type', 'maze_format'])['success_rate'].mean().reset_index()
    success_summary['success_percentage'] = success_summary['success_rate'] * 100

    # 3. Iterate through each unique Model
    unique_models = success_summary['model'].unique()
    
    for model_name in unique_models:
        model_data = success_summary[success_summary['model'] == model_name].sort_values(by='size_n')
        
        # Create a simplified label for each configuration (e.g., Few-Array, Zero-ASCII)
        model_data['config_label'] = (
            model_data['prompt_type'].str.replace('_shot', 'shot').str.replace('_of_thought', 'CoT') 
            + '-' + model_data['maze_format']
        ).str.replace('_', ' ').str.title()
        
        # Determine unique maze sizes (primary clusters) and configurations (bars in cluster)
        sizes = model_data['size_n'].unique()
        configs = model_data['config_label'].unique()
        
        n_sizes = len(sizes)
        n_configs = len(configs)
        bar_width = 0.8 / n_configs
        x_indices = np.arange(n_sizes)

        plt.figure(figsize=(10 + n_configs * 0.5, 6))
        
        colors = plt.cm.get_cmap('tab10', n_configs)
        
        # Plot bars for each configuration
        for i, config in enumerate(configs):
            config_data = model_data[model_data['config_label'] == config]
            
            # Use X-axis indices and offset for clustering
            offset = bar_width * i
            
            # Ensure the data aligns with the sizes for plotting
            plot_data = config_data.set_index('size_n').reindex(sizes)['success_percentage'].fillna(0)
            
            plt.bar(x_indices + offset, plot_data, bar_width, label=config, color=colors(i))

        # Annotations and Labels
        full_model_name = model_name.replace('35', 'GPT-3.5').replace('4o', 'GPT-4o').replace('4', 'GPT-4').title()

        plt.title(f'Success Rate Breakdown by Strategy for {full_model_name}')
        plt.xlabel('Maze Size ($N$)')
        plt.ylabel('Success Rate (%)')
        plt.yticks(np.arange(0, 101, 10))
        plt.ylim(0, 105)
        
        # Set X-axis ticks to show actual N values in the center of the clusters
        plt.xticks(x_indices + bar_width * (n_configs - 1) / 2, [str(s) for s in sizes])

        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        # Save and record the file
        output_image_name = f"breakdown_bar_chart_model_{model_name}.png"
        plt.savefig(output_image_name)
        plt.close()
        generated_files.append(output_image_name)

    print("\n--- Model Breakdown Bar Charts Generated ---")
    print(f"Generated {len(generated_files)} plots:")
    for f in generated_files:
        print(f" - {f}")

if __name__ == "__main__":
    results_file = "experiment_results_20251028_205123.csv"
    try:
        df = pd.read_csv(results_file)
        plot_model_breakdown_bar_charts(df)
    except FileNotFoundError:
        print(f"Error: File not found at path: {results_file}")
        exit()