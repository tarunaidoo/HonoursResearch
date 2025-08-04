# HonoursResearch

# Research Plan

This section outlines the major stages of the project, breaking them down into manageable
sub-tasks with estimated durations.

# 4.1 Phase 1: Preparation and Design (2 Weeks)

This initial phase focuses on laying the groundwork for both the environment and experimental
methodology. It ensures that a robust and flexible simulation framework is in
place before conducting comparative evaluations.
- Implement 2D maze environment. Develop a modular and efficient representation
of 2D mazes using binary matrices (0 = free space, 1 = obstacle). This includes
support for state initialisation, goal detection, and action constraints (up, down,
left, right). The environment should allow for easy interfacing with both classical
algorithms and LLM-based inputs.
- Define maze generation routines with adjustable size and density. Implement
a set of functions to generate random mazes with tunable parameters:
    
    – Size: from small (5 × 5) to large (50 × 50) grids.

    – Obstacle density: to control the ratio of walls to open paths, ensuring diverse levels of navigability.

    – Topological complexity: introduce structural features like dead ends, bottlenecks and multiple branches to test reasoning under varied conditions.

Maze validity must be guaranteed (i.e., at least one valid path from start to goal),
which may require post-processing validation using BFS or A*.
- Finalise evaluation metrics and scoring system. Define and standardise the metrics used to compare LLM-generated and classical paths, including:

    – Path length

    – Optimality ratio

    – Success rate (valid path reaching the goal)

    – Computation time (in seconds)

    – Token usage (for LLMs)

    – Maze complexity score (based on entropy, average branching factor). 
    
Ensure each metric is computable for all models and algorithms to allow systematic comparison.
Summarise the findings into a reference document to guide subsequent development
and experimentation phases.

# 4.2 Phase 2: Baseline Implementation (2 Weeks)
This phase establishes the traditional planning baselines which will serve as the ground
truth and primary point of comparison for LLM-generated plans.

- Implement classical algorithms (BFS, A*, DFS). Code the following algorithms
from scratch or using standard libraries for transparency and control:
    
    - Breadth-First Search (BFS): Ideal for shortest paths in unweighted graphs, acts as an optimality benchmark.
    - A* Search: Using Manhattan distance as a heuristic for efficient pathfinding in larger or sparser mazes.
    - Depth-Limited Depth-First Search (DFS): Mimics constrained-memory agents and provides a contrast to BFS/A*.

Each implementation should allow traceable path recovery, step counting and runtime
logging.

- Validate correctness across all generated maze types. Ensure algorithms
can:
    - Handle edge cases such as completely open mazes or near-blocked mazes.
    - Detect unreachable goals and return failure cleanly.
    - Match known solutions in hand-designed mazes for debugging and validation.

    Compare outputs visually or via automated assertions against a reference solver.

- Benchmark their performance across maze sizes and complexities. Systematically
evaluate algorithm efficiency and scalability by recording:
    - Runtime as a function of maze size and obstacle density.
    - Memory usage and number of nodes expanded.
    - Optimality (path length and validity).
Results will inform difficulty gradations for use in LLM testing and help define a
maze complexity scale.

# 4.3 Phase 3: LLM Integration (3-4 Weeks)
This phase focuses on incorporating language models into the pipeline and tailoring
prompts to optimise their performance.
• Construct and test prompt formats. Design prompt templates that clearly
and consistently communicate:
– Maze structure (e.g., flattened array, coordinate list, or ASCII grid).

– Problem definition (start and goal positions, valid moves).
– Output expectations (e.g., list of coordinates or directional steps).
Explore variations including zero-shot, few-shot, and chain-of-thought prompts to
determine which yields the most valid plans.
• Interface with LLMs (using OpenAI API) for maze planning. Programmatically
interact with different models (GPT-2, GPT-3.5, GPT-4). Key tasks
include:
– Tokenising and sending input prompts.
– Parsing model outputs into usable formats (e.g., coordinate lists).
– Logging response times, token usage, and raw completions for analysis.
• Validate LLM outputs against classical paths and simulation rules. For
each generated path:
– Check for syntactic correctness (format, number of steps).
– Simulate path traversal to confirm it avoids obstacles and reaches the goal.
– Compare path length and trajectory to those of classical algorithms.
Track failure cases (e.g., hallucinated moves, invalid syntax) for further analysis
and prompt refinement.

# 4.4 Phase 4: Evaluation and Analysis (2 Weeks)
This phase implements the core experimental comparisons between LLM-based planning
and traditional methods.
• Run experiments on multiple mazes (100+ instances per configuration).
For each configuration (maze size, complexity, LLM type, prompt type), generate
and test a statistically significant number of mazes.
• Collect metrics for all planning methods. Gather both quantitative and qualitative
measures:
– Success rate, path length, optimality ratio, and runtime.
– Token overhead and prompt-to-response latency for LLMs.
– Complexity-adjusted performance: measuring effectiveness relative to maze difficulty.
• Analyse LLM performance trends based on model size and maze complexity.
Identify:
– Whether larger models (e.g., GPT-4) generalise better to complex mazes.
– Which prompting techniques lead to higher plan validity and optimality.
– Breakdown of error types (e.g., invalid plans, partially correct plans).
Results will be visualised using heatmaps, bar charts, and performance curves.

# 4.5 Phase 5: Reporting and Refinement (2 Weeks)
This final phase involves synthesising the findings into a structured research report and
ensuring the quality of deliverables.
• Consolidate results and interpret findings. Summarise key insights on:
– Relative strengths and weaknesses of LLM planning versus traditional algorithms.
– How prompt design, model scale, and maze features affect outcomes.
– Potential areas for future improvement or hybrid approaches.
• Write the full report integrating charts and tables. Document the full
pipeline, methodology, and results using LaTeX. Include:
– Diagrams of example mazes and paths.
– Tables summarising metric comparisons.
– Figures visualising trends in LLM behavior.
• Conduct final review Ensure:
– All references and citations are complete and hyperlinked.
– The report is checked for originality and free of plagiarism.
– An AI usage statement is completed in line with institutional policies.

# 4.6 Contingency Planning and Flexibility
Although the project is organised into a structured four-month timeline, a degree of flexibility
is included to accommodate real-world variability in task complexity and execution.
If a phase is completed earlier than expected, the surplus time will be used to conduct
additional experiments, explore more advanced prompt engineering strategies, or expand
the evaluation framework.
Should unexpected delays arise—such as extended API response times, challenges in
path validation, or model output inconsistencies—subsequent phases may be adjusted
accordingly. This ensures that the overall quality and depth of the research are preserved
without compromising deliverables.
