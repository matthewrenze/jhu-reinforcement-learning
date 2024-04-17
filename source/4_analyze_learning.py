import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

folder_path = "../data/training"
details = pd.DataFrame()
for file_name in os.listdir(folder_path):
    if not file_name.endswith(".csv"):
        continue
    file_path = f"{folder_path}/{file_name}"
    partial_details = pd.read_csv(file_path)
    details = pd.concat([details, partial_details])

details["treatment_name"] = details.apply(lambda row: f"{row['agent_name']} ({'curriculum' if row['curriculum'] else 'baseline'})", axis=1)

# NOTE: Change  the filters (below) to analyze different details
# details = details[details["training_step"] < 100_000]
#details = details[details["agent_name"] != "deep_q_learning"]
# details = details[details["curriculum"] == False]

# Note: Keep only one out of every n rows to speed up visualization
details = details[details["training_step"] % 100 == 0]

def rolling_window(x, window_size):
    x = x.sort_values("training_step")
    x["rolling_total_reward"] = x["total_reward"].rolling(window=window_size, min_periods=1).mean()
    return x

window_size = 1_000
smoothed_details = details \
    .groupby(["treatment_name"], as_index=False) \
    .apply(rolling_window, window_size)
smoothed_details.reset_index(drop=True, inplace=True)

details = details.sort_values(by=["agent_name", "curriculum"])

# # Plot total reward by training step
plt.figure(figsize=(10, 7))
ax = sns.lineplot(
    x="training_step",
    y="rolling_total_reward",
    hue="treatment_name",
    data=smoothed_details)
plt.title("Total Reward per Training Step")
plt.xlabel("Training Step")
formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
formatter.set_scientific(False)
ax.xaxis.set_major_formatter(formatter)
ax.get_xaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.ylabel("Total Reward")
plt.ylim(0, 1_250)
ax.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.savefig(f"../data/plots/training/learning_curves.png")
plt.show()