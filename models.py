import torch


class WorldModel(object):
    def __init__(self, n_latent, n_p, device):
        self.n_latent = n_latent
        self.n_p = n_p
        self.device = device

    def train(self, dataset):
        pass

    def sample_state(self):
        state = torch.zeros([self.n_latent + self.n_p], device=self.device)
        return state

    def imagine(self, agent, init_state, horizon):
        states = torch.zeros([horizon, self.n_latent + self.n_p], device=self.device)
        actions = torch.zeros([horizon], device=self.device)
        rewards = torch.zeros([horizon], device=self.device)
        discounts = torch.zeros([horizon], device=self.device)
        return states, actions, rewards, discounts
