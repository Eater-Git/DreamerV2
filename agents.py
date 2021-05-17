class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs):
        action = self.action_space.sample()
        return action


class ActorCritic(object):
    def __init__(self, action_space, device):
        self.action_space = action_space
        self.device = device

    def train(self, states, actions, rewards, discounts):
        pass

    def act(self, obs):
        action = self.action_space.sample()
        return action
