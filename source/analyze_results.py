import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Define a function to smooth the data
def rolling_window(x, window_size):
    x = x.sort_values("episode")
    x["rolling_total_reward"] = x["total_reward"].rolling(window=window_size, min_periods=1).mean()
    x["rolling_total_time"] = x["total_time"].rolling(window=window_size, min_periods=1).mean()
    x["rolling_avg_reward"] = x["avg_reward"].rolling(window=window_size, min_periods=1).mean()
    return x

# Load the results
file_name = "results.csv"
results = pd.read_csv("../data/results/" + file_name)

# Create the treatment name
results["treatment_name"] = results.apply(lambda row: f"{row['agent_name']} ({'curriculum' if row['curriculum'] else 'baseline'})", axis=1)

# Filter the results
# NOTE: You can change the filters to analyze different results
# results = results[results["agent_name"] == "deep_q_learning"]
# results = results[results["curriculum"] == False]

# Apply the rolling window function
window_size = 100
smoothed_results = results \
    .groupby(["agent_name", "curriculum"], as_index=False) \
    .apply(rolling_window, window_size)
smoothed_results.reset_index(drop=True, inplace=True)

# Plot total reward by episode
plt.figure(figsize=(10, 7))
sns.lineplot(
    x="episode",
    y="rolling_total_reward",
    hue="treatment_name",
    data=smoothed_results)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Reward per Episode")
plt.savefig(f"../data/plots/total_reward_per_episode.png")
plt.show()

# Plot total time by episode
plt.figure(figsize=(10, 7))
sns.lineplot(
    x="episode",
    y="rolling_total_time",
    hue="treatment_name",
    data=smoothed_results)
plt.xlabel("Episode")
plt.ylabel("Total Time")
plt.title("Total Time per Episode")
plt.savefig(f"../data/plots/total_time_per_episode.png")
plt.show()

# Plot avg reward by episode
plt.figure(figsize=(10, 7))
sns.lineplot(
    x="episode",
    y="rolling_avg_reward",
    hue="treatment_name",
    data=smoothed_results)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Average Reward per Episode")
plt.savefig(f"../data/plots/average_reward_per_episode.png")
plt.show()