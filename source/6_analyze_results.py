import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

file_path = "../data/results/results.csv"
results = pd.read_csv(file_path)

# Reorder the agents
# TODO: Need to add approximate_q_learning to the list
results["agent_name"] = results["agent_name"].astype("category")
results["agent_name"] = results["agent_name"].cat.reorder_categories(
    ["sarsa", "q_learning", "deep_q_learning"])

# Rename the agents
agent_name_map ={
    "sarsa": "SARSA",
    "q_learning": "Q-Learning",
    "approximate_q_learning": "Approximate Q-Learning",
    "deep_q_learning": "Deep Q-Learning"}
results["agent_name"] = results["agent_name"].map(agent_name_map)

results["treatment_name"] = results.apply(lambda row: f"{row['agent_name']}\n({'curriculum' if row['curriculum'] else 'baseline'})", axis=1)

# NOTE: You can change the filters to analyze different results
# results = results[results["agent_name"] == "deep_q_learning"]
# results = results[results["curriculum"] == False]

results = results.sort_values(by=["agent_name", "curriculum"])

# NOTE: Cast curriculum to a string, so it can be used to set the hue
results["curriculum"] = results["curriculum"].astype(int)


# Plot total the total reward per episode by agent
# Set the color to blue for the baseline and orange for the curriculum
blue = sns.color_palette("tab10")[0]
orange = sns.color_palette("tab10")[1]

# Plot the total reward by agent
plt.figure(figsize=(10, 7))
ax = sns.barplot(
    x="treatment_name",
    y="total_reward",
    palette=[blue, orange],
    data=results,
    capsize=0.1)
plt.title("Total Reward by Agent")
plt.xlabel("Agent")
plt.ylabel("Total Reward")
plt.xticks(rotation=15, ha="right")
plt.subplots_adjust(bottom=0.15)
# and add a label above each bar and slightly to the right
for p in plt.gca().patches:
    plt.gca().annotate(
        f"{p.get_height():.0f}",
        (p.get_x() + p.get_width() / 2 + 0.25, p.get_height()),
        ha="center",
        va="center",
        fontsize=11,
        color="black",
        xytext=(0, 10),
        textcoords="offset points")
for line in ax.lines:
    line.set_color("grey")
    line.set_mfc("grey")
    line.set_mec("grey")
plt.tight_layout()
plt.savefig(f"../data/plots/results/total_reward_by_agent.png")
plt.show()