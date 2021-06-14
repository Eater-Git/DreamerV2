import torch


class WorldModel(object):
    def __init__(self, n_latent, n_p, device):
        self.n_latent = n_latent
        self.n_p = n_p
        self.device = device

    def train(self, dataset):
        n_batch = 50
        sequence_len = 50
        for _ in range(n_batch):
            sequence = dataset.sample(sequence_len)
            observations = sequence['observations']
            actions = sequence['actions']
            rewards = sequence['rewards']
            discounts = sequence['discounts']

    def sample_state(self, n_sample):
        state = torch.zeros([n_sample, self.n_latent + self.n_p], device=self.device)
        return state

    def imagine(self, agent, init_state, horizon):
        n_trajectory = init_state.size()[0]
        states = torch.zeros([n_trajectory, horizon, self.n_latent + self.n_p], device=self.device)
        actions = torch.zeros([n_trajectory, horizon], device=self.device)
        rewards = torch.zeros([n_trajectory, horizon], device=self.device)
        discounts = torch.zeros([n_trajectory, horizon], device=self.device)
        return states, actions, rewards, discounts
