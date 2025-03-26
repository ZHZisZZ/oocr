import re
import matplotlib.pyplot as plt
from dataclasses import dataclass
import tyro
import json

@dataclass
class ScriptArguments:
    num_train_epochs: int = "/mnt/lustrenew/mllm_safety-shared/models/huggingface/Qwen/Qwen2.5-1.5B"
    stat_file_path: str = ""
    image_save_path: str = ""

script_args = tyro.cli(ScriptArguments)


with open(script_args.stat_file_path, 'r') as f:
    data = json.load(f)

assert len(data) == (script_args.num_train_epochs+1), f"The finetuning epoch number is {script_args.num_train_epochs}, but the test json file only contains {len(data)} test results."

x = [int(re.search(r"checkpoint-(\d+)", checkpoint_name).group(1)) for checkpoint_name in list(data.keys())]
print(x)
divisor = max(x) / script_args.num_train_epochs 
x = [value/divisor for value in x]
y = list(data.values())

# Sort x and y based on x values
x, y = zip(*sorted(zip(x, y)))

# Create the figure and axis
fig, ax = plt.subplots(figsize=(4, 4))

# Plot two lines: one solid, one dashed
ax.plot(x, y, linewidth=4, linestyle='solid', color='steelblue')
ax.plot(x, y, linewidth=4, linestyle='dashed', color='steelblue')  # Duplicate for the dashed effect

# Style the axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Set thick black border on left and bottom
ax.spines['left'].set_linewidth(3)
ax.spines['bottom'].set_linewidth(3)

# Set tick parameters to make them bold and inside
ax.tick_params(axis='both', which='both', direction='in', length=5, width=2)

# Labels with padding
ax.set_xlabel("epoch", fontsize=14, labelpad=5)
ax.set_ylabel("rank", fontsize=14, labelpad=5)

# Set axis limits
x_min=0
x_max=10
y_min=0
y_max=10
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

plt.savefig(script_args.image_save_path, dpi=300, bbox_inches='tight')