import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

file_path = "../data/results/results.csv"
results = pd.read_csv(file_path)

results["treatment_name"] = results.apply(lambda row: f"{row['agent_name']} ({'curriculum' if row['curriculum'] else 'baseline'})", axis=1)

# NOTE: You can change the filters to analyze different results
# results = results[results["agent_name"] == "deep_q_learning"]
# results = results[results["curriculum"] == False]

results = results.sort_values(by=["agent_name", "curriculum"])

# NOTE: Cast curriculum to a string, so it can be used to set the hue
results["curriculum"] = results.apply(lambda row: f"{'curriculum' if row['curriculum'] else 'baseline'}", axis=1)

# Plot total the total reward per episode by agent
plt.figure(figsize=(10, 7))
sns.barplot(
    x="treatment_name",
    y="total_reward",
    hue="curriculum",
    data=results)
plt.title("Total Reward by Agent")
plt.xlabel("Agent")
plt.ylabel("Total Reward")
plt.xticks(rotation=15, ha="right")
plt.subplots_adjust(bottom=0.15)
plt.savefig(f"../data/plots/total_reward_by_agent.png")
plt.show()