from dataclasses import field, dataclass


@dataclass
class Memory:
    actions: list = field(default_factory=list)
    obs: list = field(default_factory=list)
    logprops: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    terminals: list = field(default_factory=list)

    def clear(self):
        del self.actions[:]
        del self.obs[:]
        del self.logprops[:]
        del self.rewards[:]
        del self.terminals[:]
