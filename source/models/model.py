class Model:

    def load(self, agent_name: str):
        raise NotImplementedError("load method must be implemented in subclass")

    def save(self, agent_name: str) -> None:
        raise NotImplementedError("save method must be implemented in subclass")
