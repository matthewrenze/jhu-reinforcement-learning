import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the results
results = pd.read_csv("../results/results.csv")

# Define a function to smooth the data
def rolling_window(x, window_size):
    x = x.sort_values("episode")
    x["rolling_total_reward"] = x["total_reward"].rolling(window=window_size, min_periods=1).mean()
    x["rolling_total_time"] = x["total_time"].rolling(window=window_size, min_periods=1).mean()
    x["rolling_avg_reward"] = x["avg_reward"].rolling(window=window_size, min_periods=1).mean()
    return x

# Apply the rolling window function
window_size = 100
smoothed_results = results \
    .groupby(["agent_name", "curriculum"], as_index=False) \
    .apply(rolling_window, window_size)
smoothed_results.reset_index(drop=True, inplace=True)

# Plot total reward by episode
sns.lineplot(
    x="episode",
    y="rolling_total_reward",
    hue="curriculum",
    data=smoothed_results)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Reward per Episode")
plt.savefig("../plots/total_reward_per_episode.png")
plt.show()

# Plot total time by episode
sns.lineplot(
    x="episode",
    y="rolling_total_time",
    hue="curriculum",
    data=smoothed_results)
plt.xlabel("Episode")
plt.ylabel("Total Time")
plt.title("Total Time per Episode")
plt.savefig("../plots/total_time_per_episode.png")
plt.show()

# Plot avg reward by episode
sns.lineplot(
    x="episode",
    y="rolling_avg_reward",
    hue="curriculum",
    data=smoothed_results)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Average Reward per Episode")
plt.savefig("../plots/average_reward_per_episode.png")
plt.show()

