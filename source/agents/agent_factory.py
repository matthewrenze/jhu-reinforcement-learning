from agents.human_agent import HumanAgent
from agents.random_agent import RandomAgent

class AgentFactory:

    def create(self, agent_name):
        if agent_name == "human":
            return HumanAgent()

        if agent_name == "random":
            return RandomAgent()

        raise ValueError(f"Unknown agent name: {agent_name}")