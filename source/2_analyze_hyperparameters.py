import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

agent_name = "sarsa"
agent_title = "SARSA"
use_curriculum = False
file_name_postfix = f"{agent_name}_{'curriculum' if use_curriculum else 'baseline'}"
file_path = f"../data/details/hyperparameter_details_{file_name_postfix}.csv"
details = pd.read_csv(file_path)

details["treatment_name"] = details.apply(lambda row: f"α={row['alpha']}; γ={row['gamma']}; ε={row['epsilon']}", axis=1)

# NOTE: Change  the filters (below) to analyze different details
# details = details[details["training_step"] < 100000]

# Note: Keep only one out of every n rows to speed up visualization
details = details[details["training_step"] % 100 == 0]

def rolling_window(x, window_size):
    x = x.sort_values("training_step")
    x["rolling_total_reward"] = x["total_reward"].rolling(window=window_size, min_periods=1).mean()
    return x

window_size = 10000
smoothed_details = details \
    .groupby(["alpha", "gamma", "epsilon"], as_index=False) \
    .apply(rolling_window, window_size)
smoothed_details.reset_index(drop=True, inplace=True)

details = details.sort_values(by=["alpha", "gamma", "epsilon"])

# Plot total reward by training step grouped by hyperparameter treatment
plot_title = f"Hyperparameter Curves for {agent_title} ({'Curriculum' if use_curriculum else 'Baseline'})"
plt.figure(figsize=(10, 7))
sns.lineplot(
    x="training_step",
    y="rolling_total_reward",
    hue="treatment_name",
    data=smoothed_details)
plt.title(plot_title)
plt.xlabel("Training Step")
plt.ylabel("Total Reward")
plt.legend(title="Hyperparameters")
plt.savefig(f"../data/plots/hyperparameter_curves_{file_name_postfix}.png")
plt.show()