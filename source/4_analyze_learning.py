import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

file_path = "../data/details/training_details.csv"
details = pd.read_csv(file_path)

details["treatment_name"] = details.apply(lambda row: f"{row['agent_name']} ({'curriculum' if row['curriculum'] else 'baseline'})", axis=1)

# NOTE: Change  the filters (below) to analyze different details
# details = details[details["training_step"] < 100000]
# details = details[details["agent_name"] == "deep_q_learning"]
# details = details[details["curriculum"] == False]

# Note: Keep only one out of every n rows to speed up visualization
details = details[details["training_step"] % 100 == 0]

def rolling_window(x, window_size):
    x = x.sort_values("training_step")
    x["rolling_total_reward"] = x["total_reward"].rolling(window=window_size, min_periods=1).mean()
    return x

window_size = 10000
smoothed_details = details \
    .groupby(["agent_name", "curriculum"], as_index=False) \
    .apply(rolling_window, window_size)
smoothed_details.reset_index(drop=True, inplace=True)

details = details.sort_values(by=["agent_name", "curriculum"])

# Plot total reward by training step
plt.figure(figsize=(10, 7))
sns.lineplot(
    x="training_step",
    y="rolling_total_reward",
    hue="treatment_name",
    data=smoothed_details)
plt.xlabel("Training Step")
plt.ylabel("Total Reward")
plt.ylim(0, 500)
plt.title("Total Reward per Training Step")
plt.savefig(f"../data/plots/learning_curves.png")
plt.show()
