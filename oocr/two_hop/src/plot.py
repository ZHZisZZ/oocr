import os
import glob
import json
from dataclasses import dataclass

import matplotlib.pyplot as plt
from collections import defaultdict
import tyro


@dataclass
class ScriptArguments:
    output_base_dir: str = "models/city_first_hop/Meta-Llama-3-8B"

script_args = tyro.cli(ScriptArguments)
# Base directory containing all the seed folders
base_dir = script_args.output_base_dir

# Initialize data structures to store the ranks
fact_ranks = defaultdict(list)  # {seed: [ranks across checkpoints]}
implication_ranks = defaultdict(list)

# Find all seed directories dynamically
seed_dirs = glob.glob(os.path.join(base_dir, "seed-*"))
if not seed_dirs:
    raise ValueError(f"No seed directories found in {base_dir}")

# Extract seed numbers and sort them
seed_numbers = sorted([int(os.path.basename(d).split("-")[1]) for d in seed_dirs])

# Collect data from all seed folders
for seed in seed_numbers:
    seed_dir = os.path.join(base_dir, f"seed-{seed}")
    
    # Get all checkpoint directories and sort them numerically
    checkpoints = [d for d in os.listdir(seed_dir) if d.startswith("checkpoint-")]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
    
    # Process each checkpoint
    for checkpoint in checkpoints:
        checkpoint_path = os.path.join(seed_dir, checkpoint, "eval", "rank.json")
        
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
                
                # Get ranks for fact-0 and implication-0
                fact_ranks[seed].append(data["fact-0"]["rank"])
                implication_ranks[seed].append(data["implication-0"]["rank"])


fig, ax = plt.subplots(figsize=(4, 4))

# Plot fact-0 ranks (solid lines)
for seed in list(fact_ranks.keys()):  # Only plot first 5 seeds if they exist
    ax.plot(range(len(fact_ranks[seed])), fact_ranks[seed], 
             linewidth=3, linestyle='solid', color='steelblue')

# Plot implication-0 ranks (dashed lines)
for seed in list(implication_ranks.keys()):  # Only plot first 5 seeds if they exist
    ax.plot(range(len(implication_ranks[seed])), implication_ranks[seed], 
             linewidth=3, linestyle='dashed', color='steelblue')

# Style the axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Set thick black border on left and bottom
ax.spines['left'].set_linewidth(3)
ax.spines['bottom'].set_linewidth(3)

# Set tick parameters to make them bold and inside
ax.tick_params(axis='both', which='both', direction='in', length=5, width=2)

# Labels with padding
ax.set_xlabel("epoch", fontsize=12, labelpad=5)
ax.set_ylabel("rank", fontsize=12, labelpad=5)
ax.set_title(base_dir, fontsize=12)

# Set axis limits
x_min=0
x_max=max(len(ranks) for ranks in fact_ranks.values()) if fact_ranks else 0
y_min=0
y_max=10
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

plt.savefig(f"{base_dir}/plot.png", dpi=300, bbox_inches='tight')
