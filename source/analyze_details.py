import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Define a function to smooth the data
def rolling_window(x, window_size):
    x = x.sort_values("training_step")
    x["rolling_total_reward"] = x["total_reward"].rolling(window=window_size, min_periods=1).mean()
    return x

# Set the parameters
# max_training_steps = 1000000
window_size = 100000

# Load the details
file_name = "details.csv"
details = pd.read_csv("../data/details/" + file_name)

# Create the treatment name
details["treatment_name"] = details.apply(lambda row: f"{row['agent_name']} ({'curriculum' if row['curriculum'] else 'baseline'})", axis=1)

# Filter the details
# NOTE: You can change the filters (below) to analyze different details
# details = details[details["training_step"] < 100000]
# details = details[details["agent_name"] == "deep_q_learning"]
# details = details[details["curriculum"] == False]

# Apply the rolling window function
smoothed_details = details \
    .groupby(["agent_name", "curriculum"], as_index=False) \
    .apply(rolling_window, window_size)
smoothed_details.reset_index(drop=True, inplace=True)

# Plot total reward by training step
plt.figure(figsize=(10, 7))
sns.lineplot(
    x="training_step",
    y="rolling_total_reward",
    hue="treatment_name",
    data=smoothed_details)
plt.xlabel("Training Step")
plt.ylabel("Total Reward")
plt.title("Total Reward per Training Step")
plt.savefig(f"../data/plots/total_reward_per_training_step.png")
plt.show()

# # Plot avg reward by training step
# plt.figure(figsize=(10, 7))
# sns.lineplot(
#     x="Training Step",
#     y="rolling_avg_reward",
#     hue="treatment_name",
#     data=smoothed_details)
# plt.xlabel("Training Step")
# plt.ylabel("Average Reward")
# plt.title("Average Reward per training_step")
# plt.savefig(f"../data/plots/average_reward_per_training_step.png")
# plt.show()