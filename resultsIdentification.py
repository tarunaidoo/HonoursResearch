# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.ticker import ScalarFormatter
# import os

# # Define the folder for saving plots
# plot_folder = 'analysis_plots'
# os.makedirs(plot_folder, exist_ok=True)

# # Set up plot style
# sns.set_theme(style="whitegrid")

# # --- Data Loading and Preprocessing ---
# file_name = "experiment_results_20251028_205123.csv"
# df = pd.read_csv(file_name)

# # Calculate maze_area
# df[['maze_width', 'maze_height']] = df['maze_size'].str.replace('(', '', regex=False).str.replace(')', '', regex=False).str.split(', ', expand=True).astype(float).astype(int)
# df['maze_area'] = df['maze_width'] * df['maze_height']

# # Create a detailed simplified algorithm name
# def simplify_algorithm_v2(alg):
#     if 'LLM' in alg:
#         parts = alg.split('_')
#         model = parts[1]
        
#         if 'zero_shot' in alg:
#             strategy = 'Zero-Shot'
#             representation = alg.split('zero_shot_')[-1]
#         elif 'few_shot' in alg:
#             strategy = 'Few-Shot'
#             representation = alg.split('few_shot_')[-1]
#         elif 'chain_of_thought' in alg:
#             strategy = 'CoT'
#             representation = alg.split('chain_of_thought_')[-1]
#         else:
#             strategy = 'Other'
#             representation = 'Other'

#         return f"{model} | {strategy} {representation.replace('_', ' ').title()}"
#     return alg

# df['simplified_algorithm'] = df['algorithm'].apply(simplify_algorithm_v2)

# # --- Define Subsets ---
# # Dataframe for all successful runs (for optimality metrics)
# df_success = df[df['path_validity_check'] == True].copy()
# # Dataframe for all LLM runs (for token usage)
# df_llm_all = df[df['algorithm'].str.contains('LLM')].copy()
# # Dataframe for successful LLM runs
# df_llm_success = df_success[df_success['algorithm'].str.contains('LLM')].copy()


# # ----------------------------------------------------------------------
# # --- Existing Plots (3, 4, 5, 6, 7, 8) - Keeping for completeness ---
# # ----------------------------------------------------------------------

# # --- Plot 3: Mean Optimality Ratio vs. Maze Area (All Algorithms) ---
# df_optimality = df_success.groupby(['simplified_algorithm', 'maze_area']).agg(
#     mean_optimality_ratio=('optimality_ratio', 'mean'),
#     sem_optimality_ratio=('optimality_ratio', lambda x: x.std() / np.sqrt(len(x)))
# ).reset_index()

# plt.figure(figsize=(12, 7))
# df_optimality = df_optimality.sort_values(by='simplified_algorithm')

# for alg in df_optimality['simplified_algorithm'].unique():
#     subset = df_optimality[df_optimality['simplified_algorithm'] == alg]
#     plt.plot(subset['maze_area'], subset['mean_optimality_ratio'], marker='o', label=alg)
#     if not subset['sem_optimality_ratio'].isnull().all():
#         plt.fill_between(subset['maze_area'],
#                          subset['mean_optimality_ratio'] - subset['sem_optimality_ratio'],
#                          subset['mean_optimality_ratio'] + subset['sem_optimality_ratio'],
#                          alpha=0.1)

# max_optimality = df_optimality['mean_optimality_ratio'].max()
# top_limit = max_optimality * 1.05 if not pd.isna(max_optimality) and max_optimality > 0 else 1.1 

# plt.xscale('log', base=4)
# plt.xticks(df_optimality['maze_area'].unique(), labels=[f'{a}' for a in df_optimality['maze_area'].unique()])
# plt.ylim(0.9, top_limit)
# plt.xlabel('Maze Area (Log Scale)')
# plt.ylabel('Mean Optimality Ratio (Path Length / Shortest Path)')
# plt.title('Mean Optimality Ratio vs. Maze Area')
# plt.legend(title='Algorithm/Prompt', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout(rect=[0, 0, 0.7, 1])
# plt.savefig(os.path.join(plot_folder, 'mean_optimality_ratio_vs_maze_area.png'))
# plt.close()

# # --- Plot 4: LLM Performance Comparison (Optimality & Token Usage by Prompt Type) ---
# df_optimality_llm = df_llm_success.groupby('simplified_algorithm').agg(
#     mean_optimality_ratio=('optimality_ratio', 'mean'),
#     sem_optimality_ratio=('optimality_ratio', lambda x: x.std() / np.sqrt(len(x)))
# ).reset_index()

# df_token_llm = df_llm_all.groupby('simplified_algorithm').agg(
#     mean_token_usage=('token_usage', 'mean'),
#     sem_token_usage=('token_usage', lambda x: x.std() / np.sqrt(len(x.dropna())))
# ).reset_index()

# df_llm_agg = pd.merge(df_optimality_llm, df_token_llm, on='simplified_algorithm', how='outer')
# df_llm_agg = df_llm_agg.sort_values(by='mean_optimality_ratio', ascending=False)

# fig, ax1 = plt.subplots(figsize=(14, 8))
# bar_width = 0.35
# x = np.arange(len(df_llm_agg['simplified_algorithm']))
# max_optimality_llm = df_llm_agg['mean_optimality_ratio'].max()
# top_limit_llm = max_optimality_llm * 1.1 if not pd.isna(max_optimality_llm) and max_optimality_llm > 0 else 1.1
# lower_limit_llm = 0.9 if top_limit_llm > 0.9 else 0.0

# ax1.bar(x - bar_width/2, df_llm_agg['mean_optimality_ratio'], bar_width,
#         yerr=df_llm_agg['sem_optimality_ratio'], capsize=5, color='tab:blue', label='Mean Optimality Ratio (LHS)')
# ax1.set_ylabel('Mean Optimality Ratio', color='tab:blue')
# ax1.tick_params(axis='y', labelcolor='tab:blue')
# ax1.set_ylim(lower_limit_llm, top_limit_llm)

# ax2 = ax1.twinx()
# ax2.bar(x + bar_width/2, df_llm_agg['mean_token_usage'], bar_width,
#         yerr=df_llm_agg['sem_token_usage'], capsize=5, color='tab:red', alpha=0.6, label='Mean Token Usage (RHS)')
# ax2.set_ylabel('Mean Token Usage', color='tab:red')
# ax2.tick_params(axis='y', labelcolor='tab:red')
# ax2.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
# ax1.set_xticks(x)
# ax1.set_xticklabels(df_llm_agg['simplified_algorithm'], rotation=45, ha="right")
# ax1.set_xlabel('LLM Prompt Strategy')
# plt.title('LLM Performance and Cost: Optimality vs. Token Usage')
# lines, labels = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax2.legend(lines + lines2, labels + labels2, loc='upper left')

# plt.tight_layout()
# plt.savefig(os.path.join(plot_folder, 'llm_optimality_vs_token_usage.png'))
# plt.close()

# # --- Plot 5: Mean Nodes Expanded vs. Maze Area (Traditional Algorithms) ---
# df_traditional = df[df['algorithm'].isin(['BFS', 'A_Star'])].dropna(subset=['nodes_expanded'])
# df_nodes = df_traditional.groupby(['simplified_algorithm', 'maze_area']).agg(
#     mean_nodes_expanded=('nodes_expanded', 'mean'),
#     sem_nodes_expanded=('nodes_expanded', lambda x: x.std() / np.sqrt(len(x)))
# ).reset_index()

# plt.figure(figsize=(10, 6))
# for alg in df_nodes['simplified_algorithm'].unique():
#     subset = df_nodes[df_nodes['simplified_algorithm'] == alg]
#     plt.plot(subset['maze_area'], subset['mean_nodes_expanded'], marker='o', linestyle='-', label=alg)
#     if not subset['sem_nodes_expanded'].isnull().all():
#         plt.fill_between(subset['maze_area'],
#                          subset['mean_nodes_expanded'] - subset['sem_nodes_expanded'],
#                          subset['mean_nodes_expanded'] + subset['sem_nodes_expanded'],
#                          alpha=0.1)

# plt.xscale('log', base=4)
# plt.yscale('log', base=10)
# plt.xticks(df_nodes['maze_area'].unique(), labels=[f'{a}' for a in df_nodes['maze_area'].unique()])
# plt.xlabel('Maze Area (Log Scale)')
# plt.ylabel('Mean Nodes Expanded (Log Scale)')
# plt.title('Traditional Algorithms: Search Efficiency vs. Maze Area')
# plt.legend(title='Algorithm', loc='upper left')
# plt.tight_layout()
# plt.savefig(os.path.join(plot_folder, 'mean_nodes_expanded_vs_maze_area.png'))
# plt.close()

# # --- Plot 6: Mean Computation Time vs. Maze Area (All Algorithms) ---
# df_time = df.groupby(['simplified_algorithm', 'maze_area']).agg(
#     mean_time=('computation_time', 'mean'),
#     sem_time=('computation_time', lambda x: x.std() / np.sqrt(len(x)))
# ).reset_index()

# plt.figure(figsize=(12, 7))
# for alg in df_time['simplified_algorithm'].unique():
#     subset = df_time[df_time['simplified_algorithm'] == alg]
#     plt.plot(subset['maze_area'], subset['mean_time'], marker='o', label=alg)
#     if not subset['sem_time'].isnull().all():
#         plt.fill_between(subset['maze_area'],
#                          subset['mean_time'] - subset['sem_time'],
#                          subset['mean_time'] + subset['sem_time'],
#                          alpha=0.1)

# plt.xscale('log', base=4)
# plt.yscale('log', base=10)
# plt.xticks(df_time['maze_area'].unique(), labels=[f'{a}' for a in df_time['maze_area'].unique()])
# plt.xlabel('Maze Area (Log Scale)')
# plt.ylabel('Mean Computation Time (Seconds, Log Scale)')
# plt.title('Algorithm Scalability: Mean Computation Time vs. Maze Area')
# plt.legend(title='Algorithm/Prompt', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout(rect=[0, 0, 0.7, 1])
# plt.savefig(os.path.join(plot_folder, 'mean_computation_time_vs_maze_area.png'))
# plt.close()

# # --- Plot 7: Mean Success Rate vs. Maze Area (All Algorithms) ---
# df_success_rate = df.groupby(['simplified_algorithm', 'maze_area']).agg(
#     mean_success=('path_validity_check', 'mean'),
#     sem_success=('path_validity_check', lambda x: x.std() / np.sqrt(len(x)))
# ).reset_index()

# plt.figure(figsize=(12, 7))
# for alg in df_success_rate['simplified_algorithm'].unique():
#     subset = df_success_rate[df_success_rate['simplified_algorithm'] == alg]
#     plt.plot(subset['maze_area'], subset['mean_success'], marker='o', label=alg)
#     if not subset['sem_success'].isnull().all():
#         plt.fill_between(subset['maze_area'],
#                          subset['mean_success'] - subset['sem_success'],
#                          subset['mean_success'] + subset['sem_success'],
#                          alpha=0.1)

# plt.xscale('log', base=4)
# plt.xticks(df_success_rate['maze_area'].unique(), labels=[f'{a}' for a in df_success_rate['maze_area'].unique()])
# plt.ylim(0.0, 1.05)
# plt.xlabel('Maze Area (Log Scale)')
# plt.ylabel('Mean Success Rate')
# plt.title('Algorithm Robustness: Mean Success Rate vs. Maze Area')
# plt.legend(title='Algorithm/Prompt', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout(rect=[0, 0, 0.7, 1])
# plt.savefig(os.path.join(plot_folder, 'mean_success_rate_vs_maze_area.png'))
# plt.close()

# # --- Plot 8: Mean Token Usage vs. Maze Area (LLM Algorithms Only) ---
# df_tokens_scaling = df_llm_all.groupby(['simplified_algorithm', 'maze_area']).agg(
#     mean_token_usage=('token_usage', 'mean'),
#     sem_token_usage=('token_usage', lambda x: x.std() / np.sqrt(len(x.dropna())))
# ).reset_index()

# plt.figure(figsize=(12, 7))
# for alg in df_tokens_scaling['simplified_algorithm'].unique():
#     subset = df_tokens_scaling[df_tokens_scaling['simplified_algorithm'] == alg]
#     plt.plot(subset['maze_area'], subset['mean_token_usage'], marker='o', label=alg)
#     if not subset['sem_token_usage'].isnull().all():
#         plt.fill_between(subset['maze_area'],
#                          subset['mean_token_usage'] - subset['sem_token_usage'],
#                          subset['mean_token_usage'] + subset['sem_token_usage'],
#                          alpha=0.1)

# plt.xscale('log', base=4)
# plt.yscale('log', base=10)
# plt.xticks(df_tokens_scaling['maze_area'].unique(), labels=[f'{a}' for a in df_tokens_scaling['maze_area'].unique()])
# plt.xlabel('Maze Area (Log Scale)')
# plt.ylabel('Mean Token Usage (Log Scale)')
# plt.title('LLM Cost Scalability: Mean Token Usage vs. Maze Area')
# plt.legend(title='LLM Prompt Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout(rect=[0, 0, 0.7, 1])
# plt.savefig(os.path.join(plot_folder, 'mean_token_usage_vs_maze_area_scaling.png'))
# plt.close()

# # ----------------------------------------------------------------------
# # --- NEW PLOTS (9, 10, 11) ---
# # ----------------------------------------------------------------------

# # --- Plot 9: Optimality Ratio Distribution (Box Plot) ---
# plt.figure(figsize=(15, 8))
# sns.boxplot(data=df_success, x='simplified_algorithm', y='optimality_ratio', showfliers=False)
# plt.ylim(0.95, df_success['optimality_ratio'].max() * 1.05)
# plt.xticks(rotation=45, ha='right')
# plt.xlabel('Algorithm/Prompt Strategy')
# plt.ylabel('Optimality Ratio (Path Length / Shortest Path)')
# plt.title('Distribution of Optimality Ratio Across Successful Runs')
# plt.tight_layout()
# plt.savefig(os.path.join(plot_folder, 'optimality_ratio_distribution_boxplot.png'))
# plt.close()

# # --- Plot 10: Mean Path Length Difference vs. Maze Area ---
# # Calculate the shortest path and the difference
# df_plot10 = df_success.copy()
# df_plot10['shortest_path_length'] = df_plot10['path_length'] / df_plot10['optimality_ratio']
# df_plot10['path_length_difference'] = df_plot10['path_length'] - df_plot10['shortest_path_length']

# df_diff = df_plot10.groupby(['simplified_algorithm', 'maze_area']).agg(
#     mean_diff=('path_length_difference', 'mean'),
#     sem_diff=('path_length_difference', lambda x: x.std() / np.sqrt(len(x)))
# ).reset_index()

# plt.figure(figsize=(12, 7))
# for alg in df_diff['simplified_algorithm'].unique():
#     subset = df_diff[df_diff['simplified_algorithm'] == alg]
#     plt.plot(subset['maze_area'], subset['mean_diff'], marker='o', label=alg)
#     if not subset['sem_diff'].isnull().all():
#         plt.fill_between(subset['maze_area'],
#                          subset['mean_diff'] - subset['sem_diff'],
#                          subset['mean_diff'] + subset['sem_diff'],
#                          alpha=0.1)

# plt.xscale('log', base=4)
# plt.yscale('log', base=10)
# plt.xticks(df_diff['maze_area'].unique(), labels=[f'{a}' for a in df_diff['maze_area'].unique()])
# plt.xlabel('Maze Area (Log Scale)')
# plt.ylabel('Mean Excess Path Length (Steps, Log Scale)')
# plt.title('Mean Excess Path Length Scalability vs. Maze Area')
# plt.legend(title='Algorithm/Prompt', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout(rect=[0, 0, 0.7, 1])
# plt.savefig(os.path.join(plot_folder, 'mean_path_length_difference_vs_maze_area.png'))
# plt.close()

# # --- Plot 11: Normalized Path Length (Stacked Bar Chart) ---
# df_plot11 = df_plot10.groupby('simplified_algorithm').agg(
#     mean_shortest_path=('shortest_path_length', 'mean'),
#     mean_excess_path=('path_length_difference', 'mean'),
#     total_path_length=('path_length', 'mean')
# ).reset_index()

# # Sort by total path length (descending)
# df_plot11 = df_plot11.sort_values(by='total_path_length', ascending=False)

# fig, ax = plt.subplots(figsize=(14, 8))

# # Optimal Component (bottom bar)
# ax.bar(df_plot11['simplified_algorithm'], df_plot11['mean_shortest_path'], 
#        label='Mean Optimal Path Length', color='tab:green')

# # Excess Component (stacked on top of optimal)
# ax.bar(df_plot11['simplified_algorithm'], df_plot11['mean_excess_path'], 
#        bottom=df_plot11['mean_shortest_path'], 
#        label='Mean Excess Path Length (Sub-Optimal)', color='tab:red')

# plt.xticks(rotation=45, ha='right')
# plt.xlabel('Algorithm/Prompt Strategy')
# plt.ylabel('Mean Path Length (Steps)')
# plt.title('Normalized Path Length (Optimal vs. Excess Steps)')
# plt.legend(title='Path Component', loc='upper right')
# plt.tight_layout()
# plt.savefig(os.path.join(plot_folder, 'normalized_path_length_stacked_bar.png'))
# plt.close()

# print(f"All 9 plots have been saved to the '{plot_folder}' folder.")
# print(f"Files: {os.listdir(plot_folder)}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
import os

# Define the folder for saving plots
plot_folder = 'analysis_plots'
os.makedirs(plot_folder, exist_ok=True)

# Set up plot style
sns.set_theme(style="whitegrid")

# --- Data Loading and Preprocessing ---
file_name = "experiment_results_20251028_205123.csv"
df = pd.read_csv(file_name)

# Calculate maze_area
df[['maze_width', 'maze_height']] = df['maze_size'].str.replace('(', '', regex=False).str.replace(')', '', regex=False).str.split(', ', expand=True).astype(float).astype(int)
df['maze_area'] = df['maze_width'] * df['maze_height']

# Create a detailed simplified algorithm name
def simplify_algorithm_v2(alg):
    if 'LLM' in alg:
        parts = alg.split('_')
        model = parts[1]
        
        if 'zero_shot' in alg:
            strategy = 'Zero-Shot'
            representation = alg.split('zero_shot_')[-1]
        elif 'few_shot' in alg:
            strategy = 'Few-Shot'
            representation = alg.split('few_shot_')[-1]
        elif 'chain_of_thought' in alg:
            strategy = 'CoT'
            representation = alg.split('chain_of_thought_')[-1]
        else:
            strategy = 'Other'
            representation = 'Other'

        return f"{model} | {strategy} {representation.replace('_', ' ').title()}"
    return alg

df['simplified_algorithm'] = df['algorithm'].apply(simplify_algorithm_v2)

# --- Define Subsets ---
# Dataframe for all successful runs (for optimality metrics)
df_success = df[df['path_validity_check'] == True].copy()
# Dataframe for all LLM runs (for token usage)
df_llm_all = df[df['algorithm'].str.contains('LLM')].copy()
# Dataframe for successful LLM runs
df_llm_success = df_success[df_success['algorithm'].str.contains('LLM')].copy()


# ----------------------------------------------------------------------
# --- Existing Plots (3, 4, 5, 6) ---
# ----------------------------------------------------------------------

# --- Plot 3: Mean Optimality Ratio vs. Maze Area (All Algorithms) ---
df_optimality = df_success.groupby(['simplified_algorithm', 'maze_area']).agg(
    mean_optimality_ratio=('optimality_ratio', 'mean'),
    sem_optimality_ratio=('optimality_ratio', lambda x: x.std() / np.sqrt(len(x)))
).reset_index()

plt.figure(figsize=(12, 7))
df_optimality = df_optimality.sort_values(by='simplified_algorithm')

for alg in df_optimality['simplified_algorithm'].unique():
    subset = df_optimality[df_optimality['simplified_algorithm'] == alg]
    plt.plot(subset['maze_area'], subset['mean_optimality_ratio'], marker='o', label=alg)
    if not subset['sem_optimality_ratio'].isnull().all():
        plt.fill_between(subset['maze_area'],
                         subset['mean_optimality_ratio'] - subset['sem_optimality_ratio'],
                         subset['mean_optimality_ratio'] + subset['sem_optimality_ratio'],
                         alpha=0.1)

max_optimality = df_optimality['mean_optimality_ratio'].max()
top_limit = max_optimality * 1.05 if not pd.isna(max_optimality) and max_optimality > 0 else 1.1 

plt.xscale('log', base=4)
plt.xticks(df_optimality['maze_area'].unique(), labels=[f'{a}' for a in df_optimality['maze_area'].unique()])
plt.ylim(0.9, top_limit)
plt.xlabel('Maze Area (Log Scale)')
plt.ylabel('Mean Optimality Ratio (Path Length / Shortest Path)')
plt.title('Mean Optimality Ratio vs. Maze Area')
plt.legend(title='Algorithm/Prompt', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout(rect=[0, 0, 0.7, 1])
plt.savefig(os.path.join(plot_folder, 'mean_optimality_ratio_vs_maze_area.png'))
plt.close()

# --- Plot 4: LLM Performance Comparison (Optimality & Token Usage by Prompt Type) ---
df_optimality_llm = df_llm_success.groupby('simplified_algorithm').agg(
    mean_optimality_ratio=('optimality_ratio', 'mean'),
    sem_optimality_ratio=('optimality_ratio', lambda x: x.std() / np.sqrt(len(x)))
).reset_index()

df_token_llm = df_llm_all.groupby('simplified_algorithm').agg(
    mean_token_usage=('token_usage', 'mean'),
    sem_token_usage=('token_usage', lambda x: x.std() / np.sqrt(len(x.dropna())))
).reset_index()

df_llm_agg = pd.merge(df_optimality_llm, df_token_llm, on='simplified_algorithm', how='outer')
df_llm_agg = df_llm_agg.sort_values(by='mean_optimality_ratio', ascending=False)

fig, ax1 = plt.subplots(figsize=(14, 8))
bar_width = 0.35
x = np.arange(len(df_llm_agg['simplified_algorithm']))
max_optimality_llm = df_llm_agg['mean_optimality_ratio'].max()
top_limit_llm = max_optimality_llm * 1.1 if not pd.isna(max_optimality_llm) and max_optimality_llm > 0 else 1.1
lower_limit_llm = 0.9 if top_limit_llm > 0.9 else 0.0

ax1.bar(x - bar_width/2, df_llm_agg['mean_optimality_ratio'], bar_width,
        yerr=df_llm_agg['sem_optimality_ratio'], capsize=5, color='tab:blue', label='Mean Optimality Ratio (LHS)')
ax1.set_ylabel('Mean Optimality Ratio', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_ylim(lower_limit_llm, top_limit_llm)

ax2 = ax1.twinx()
ax2.bar(x + bar_width/2, df_llm_agg['mean_token_usage'], bar_width,
        yerr=df_llm_agg['sem_token_usage'], capsize=5, color='tab:red', alpha=0.6, label='Mean Token Usage (RHS)')
ax2.set_ylabel('Mean Token Usage', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
ax1.set_xticks(x)
ax1.set_xticklabels(df_llm_agg['simplified_algorithm'], rotation=45, ha="right")
ax1.set_xlabel('LLM Prompt Strategy')
plt.title('LLM Performance and Cost: Optimality vs. Token Usage')
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(plot_folder, 'llm_optimality_vs_token_usage.png'))
plt.close()

# --- Plot 5: Mean Nodes Expanded vs. Maze Area (Traditional Algorithms) ---
df_traditional = df[df['algorithm'].isin(['BFS', 'A_Star'])].dropna(subset=['nodes_expanded'])
df_nodes = df_traditional.groupby(['simplified_algorithm', 'maze_area']).agg(
    mean_nodes_expanded=('nodes_expanded', 'mean'),
    sem_nodes_expanded=('nodes_expanded', lambda x: x.std() / np.sqrt(len(x)))
).reset_index()

plt.figure(figsize=(10, 6))
for alg in df_nodes['simplified_algorithm'].unique():
    subset = df_nodes[df_nodes['simplified_algorithm'] == alg]
    plt.plot(subset['maze_area'], subset['mean_nodes_expanded'], marker='o', linestyle='-', label=alg)
    if not subset['sem_nodes_expanded'].isnull().all():
        plt.fill_between(subset['maze_area'],
                         subset['mean_nodes_expanded'] - subset['sem_nodes_expanded'],
                         subset['mean_nodes_expanded'] + subset['sem_nodes_expanded'],
                         alpha=0.1)

plt.xscale('log', base=4)
plt.yscale('log', base=10)
plt.xticks(df_nodes['maze_area'].unique(), labels=[f'{a}' for a in df_nodes['maze_area'].unique()])
plt.xlabel('Maze Area (Log Scale)')
plt.ylabel('Mean Nodes Expanded (Log Scale)')
plt.title('Traditional Algorithms: Search Efficiency vs. Maze Area')
plt.legend(title='Algorithm', loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(plot_folder, 'mean_nodes_expanded_vs_maze_area.png'))
plt.close()

# --- Plot 6: Mean Computation Time vs. Maze Area (All Algorithms) ---
df_time = df.groupby(['simplified_algorithm', 'maze_area']).agg(
    mean_time=('computation_time', 'mean'),
    sem_time=('computation_time', lambda x: x.std() / np.sqrt(len(x)))
).reset_index()

plt.figure(figsize=(12, 7))
for alg in df_time['simplified_algorithm'].unique():
    subset = df_time[df_time['simplified_algorithm'] == alg]
    plt.plot(subset['maze_area'], subset['mean_time'], marker='o', label=alg)
    if not subset['sem_time'].isnull().all():
        plt.fill_between(subset['maze_area'],
                         subset['mean_time'] - subset['sem_time'],
                         subset['mean_time'] + subset['sem_time'],
                         alpha=0.1)

plt.xscale('log', base=4)
plt.yscale('log', base=10)
plt.xticks(df_time['maze_area'].unique(), labels=[f'{a}' for a in df_time['maze_area'].unique()])
plt.xlabel('Maze Area (Log Scale)')
plt.ylabel('Mean Computation Time (Seconds, Log Scale)')
plt.title('Algorithm Scalability: Mean Computation Time vs. Maze Area')
plt.legend(title='Algorithm/Prompt', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout(rect=[0, 0, 0.7, 1])
plt.savefig(os.path.join(plot_folder, 'mean_computation_time_vs_maze_area.png'))
plt.close()

# ----------------------------------------------------------------------
# --- NEW PLOTS (12 and 13) - Computation Time Breakdown ---
# ----------------------------------------------------------------------

# --- Data Separation for NEW Plots 12 & 13 (Computation Time) ---
# Identify simplified names for search (non-LLM) and LLM algorithms
search_algs = df[~df['algorithm'].str.contains('LLM')]['simplified_algorithm'].unique()
llm_algs = df[df['algorithm'].str.contains('LLM')]['simplified_algorithm'].unique()

# Filter df_time (created in Plot 6)
df_time_search = df_time[df_time['simplified_algorithm'].isin(search_algs)]
df_time_llm = df_time[df_time['simplified_algorithm'].isin(llm_algs)]

# --- Plot 12: Mean Computation Time vs. Maze Area (Search Algorithms Only) ---
plt.figure(figsize=(10, 6))
for alg in df_time_search['simplified_algorithm'].unique():
    subset = df_time_search[df_time_search['simplified_algorithm'] == alg]
    plt.plot(subset['maze_area'], subset['mean_time'], marker='o', label=alg)
    if not subset['sem_time'].isnull().all():
        plt.fill_between(subset['maze_area'],
                         subset['mean_time'] - subset['sem_time'],
                         subset['mean_time'] + subset['sem_time'],
                         alpha=0.1)

plt.xscale('log', base=4)
plt.yscale('log', base=10)
plt.xticks(df_time_search['maze_area'].unique(), labels=[f'{a}' for a in df_time_search['maze_area'].unique()])
plt.xlabel('Maze Area (Log Scale)')
plt.ylabel('Mean Computation Time (Seconds, Log Scale)')
plt.title('Search Algorithms: Mean Computation Time vs. Maze Area')
plt.legend(title='Algorithm', loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(plot_folder, 'search_computation_time_vs_maze_area.png'))
plt.close()

# --- Plot 13: Mean Computation Time vs. Maze Area (LLM Algorithms Only) ---
plt.figure(figsize=(12, 7))
for alg in df_time_llm['simplified_algorithm'].unique():
    subset = df_time_llm[df_time_llm['simplified_algorithm'] == alg]
    plt.plot(subset['maze_area'], subset['mean_time'], marker='o', label=alg)
    if not subset['sem_time'].isnull().all():
        plt.fill_between(subset['maze_area'],
                         subset['mean_time'] - subset['sem_time'],
                         subset['mean_time'] + subset['sem_time'],
                         alpha=0.1)

plt.xscale('log', base=4)
plt.yscale('log', base=10)
plt.xticks(df_time_llm['maze_area'].unique(), labels=[f'{a}' for a in df_time_llm['maze_area'].unique()])
plt.xlabel('Maze Area (Log Scale)')
plt.ylabel('Mean Computation Time (Seconds, Log Scale)')
plt.title('LLM Algorithms: Mean Computation Time vs. Maze Area')
plt.legend(title='LLM Prompt Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout(rect=[0, 0, 0.7, 1])
plt.savefig(os.path.join(plot_folder, 'llm_computation_time_vs_maze_area.png'))
plt.close()

# ----------------------------------------------------------------------
# --- Existing Plots (7, 8, 9, 10, 11) - Continued ---
# ----------------------------------------------------------------------

# --- Plot 7: Mean Success Rate vs. Maze Area (All Algorithms) ---
df_success_rate = df.groupby(['simplified_algorithm', 'maze_area']).agg(
    mean_success=('path_validity_check', 'mean'),
    sem_success=('path_validity_check', lambda x: x.std() / np.sqrt(len(x)))
).reset_index()

plt.figure(figsize=(12, 7))
for alg in df_success_rate['simplified_algorithm'].unique():
    subset = df_success_rate[df_success_rate['simplified_algorithm'] == alg]
    plt.plot(subset['maze_area'], subset['mean_success'], marker='o', label=alg)
    if not subset['sem_success'].isnull().all():
        plt.fill_between(subset['maze_area'],
                         subset['mean_success'] - subset['sem_success'],
                         subset['mean_success'] + subset['sem_success'],
                         alpha=0.1)

plt.xscale('log', base=4)
plt.xticks(df_success_rate['maze_area'].unique(), labels=[f'{a}' for a in df_success_rate['maze_area'].unique()])
plt.ylim(0.0, 1.05)
plt.xlabel('Maze Area (Log Scale)')
plt.ylabel('Mean Success Rate')
plt.title('Algorithm Robustness: Mean Success Rate vs. Maze Area')
plt.legend(title='Algorithm/Prompt', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout(rect=[0, 0, 0.7, 1])
plt.savefig(os.path.join(plot_folder, 'mean_success_rate_vs_maze_area.png'))
plt.close()

# --- Plot 8: Mean Token Usage vs. Maze Area (LLM Algorithms Only) ---
df_tokens_scaling = df_llm_all.groupby(['simplified_algorithm', 'maze_area']).agg(
    mean_token_usage=('token_usage', 'mean'),
    sem_token_usage=('token_usage', lambda x: x.std() / np.sqrt(len(x.dropna())))
).reset_index()

plt.figure(figsize=(12, 7))
for alg in df_tokens_scaling['simplified_algorithm'].unique():
    subset = df_tokens_scaling[df_tokens_scaling['simplified_algorithm'] == alg]
    plt.plot(subset['maze_area'], subset['mean_token_usage'], marker='o', label=alg)
    if not subset['sem_token_usage'].isnull().all():
        plt.fill_between(subset['maze_area'],
                         subset['mean_token_usage'] - subset['sem_token_usage'],
                         subset['mean_token_usage'] + subset['sem_token_usage'],
                         alpha=0.1)

plt.xscale('log', base=4)
plt.yscale('log', base=10)
plt.xticks(df_tokens_scaling['maze_area'].unique(), labels=[f'{a}' for a in df_tokens_scaling['maze_area'].unique()])
plt.xlabel('Maze Area (Log Scale)')
plt.ylabel('Mean Token Usage (Log Scale)')
plt.title('LLM Cost Scalability: Mean Token Usage vs. Maze Area')
plt.legend(title='LLM Prompt Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout(rect=[0, 0, 0.7, 1])
plt.savefig(os.path.join(plot_folder, 'mean_token_usage_vs_maze_area_scaling.png'))
plt.close()

# ----------------------------------------------------------------------
# --- Existing Plots (9, 10, 11) - Continued ---
# ----------------------------------------------------------------------

# --- Plot 9: Optimality Ratio Distribution (Box Plot) ---
plt.figure(figsize=(15, 8))
sns.boxplot(data=df_success, x='simplified_algorithm', y='optimality_ratio', showfliers=False)
plt.ylim(0.95, df_success['optimality_ratio'].max() * 1.05)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Algorithm/Prompt Strategy')
plt.ylabel('Optimality Ratio (Path Length / Shortest Path)')
plt.title('Distribution of Optimality Ratio Across Successful Runs')
plt.tight_layout()
plt.savefig(os.path.join(plot_folder, 'optimality_ratio_distribution_boxplot.png'))
plt.close()

# --- Plot 10: Mean Path Length Difference vs. Maze Area ---
# Calculate the shortest path and the difference
df_plot10 = df_success.copy()
df_plot10['shortest_path_length'] = df_plot10['path_length'] / df_plot10['optimality_ratio']
df_plot10['path_length_difference'] = df_plot10['path_length'] - df_plot10['shortest_path_length']

df_diff = df_plot10.groupby(['simplified_algorithm', 'maze_area']).agg(
    mean_diff=('path_length_difference', 'mean'),
    sem_diff=('path_length_difference', lambda x: x.std() / np.sqrt(len(x)))
).reset_index()

plt.figure(figsize=(12, 7))
for alg in df_diff['simplified_algorithm'].unique():
    subset = df_diff[df_diff['simplified_algorithm'] == alg]
    plt.plot(subset['maze_area'], subset['mean_diff'], marker='o', label=alg)
    if not subset['sem_diff'].isnull().all():
        plt.fill_between(subset['maze_area'],
                         subset['mean_diff'] - subset['sem_diff'],
                         subset['mean_diff'] + subset['sem_diff'],
                         alpha=0.1)

plt.xscale('log', base=4)
plt.yscale('log', base=10)
plt.xticks(df_diff['maze_area'].unique(), labels=[f'{a}' for a in df_diff['maze_area'].unique()])
plt.xlabel('Maze Area (Log Scale)')
plt.ylabel('Mean Excess Path Length (Steps, Log Scale)')
plt.title('Mean Excess Path Length Scalability vs. Maze Area')
plt.legend(title='Algorithm/Prompt', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout(rect=[0, 0, 0.7, 1])
plt.savefig(os.path.join(plot_folder, 'mean_path_length_difference_vs_maze_area.png'))
plt.close()

# --- Plot 11: Normalized Path Length (Stacked Bar Chart) ---
df_plot11 = df_plot10.groupby('simplified_algorithm').agg(
    mean_shortest_path=('shortest_path_length', 'mean'),
    mean_excess_path=('path_length_difference', 'mean'),
    total_path_length=('path_length', 'mean')
).reset_index()

# Sort by total path length (descending)
df_plot11 = df_plot11.sort_values(by='total_path_length', ascending=False)

fig, ax = plt.subplots(figsize=(14, 8))

# Optimal Component (bottom bar)
ax.bar(df_plot11['simplified_algorithm'], df_plot11['mean_shortest_path'], 
        label='Mean Optimal Path Length', color='tab:green')

# Excess Component (stacked on top of optimal)
ax.bar(df_plot11['simplified_algorithm'], df_plot11['mean_excess_path'], 
        bottom=df_plot11['mean_shortest_path'], 
        label='Mean Excess Path Length (Sub-Optimal)', color='tab:red')

plt.xticks(rotation=45, ha='right')
plt.xlabel('Algorithm/Prompt Strategy')
plt.ylabel('Mean Path Length (Steps)')
plt.title('Normalized Path Length (Optimal vs. Excess Steps)')
plt.legend(title='Path Component', loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(plot_folder, 'normalized_path_length_stacked_bar.png'))
plt.close()

print(f"All 13 plots have been generated and saved to the '{plot_folder}' folder, including the two new computation time breakdowns.")
print(f"Files: {os.listdir(plot_folder)}")