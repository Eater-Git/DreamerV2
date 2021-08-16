import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class RSSM(nn.Module):
    def __init__(self, obs_size, n_action, n_d, n_p, n_p_class):
        super(RSSM, self).__init__()
        self.obs_size = obs_size
        self.n_action = n_action
        self.n_d = n_d
        self.n_p = n_p
        self.n_p_class = n_p_class

        self.image_encoder = nn.Sequential(
            nn.Conv2d(obs_size[2], 32, 4, 2), nn.ELU(),  # 1 x 64 x 64, 64 -> 31
            nn.Conv2d(32, 64, 4, 2), nn.ELU(),  # 31 -> 14
            nn.Conv2d(64, 128, 4, 2), nn.ELU(),  # 14 -> 6
            nn.Conv2d(128, 256, 4, 2), nn.ELU(),  # 6 -> 2, 256 x 2 x 2
        )
        self.transition_posterior = nn.Sequential(
            nn.Linear(n_d + 1024, 400), nn.ELU(),
            nn.Linear(400, 400), nn.ELU(),
            nn.Linear(400, 400), nn.ELU(),
            nn.Linear(400, n_p * n_p_class)
        )
        self.transition_prior = nn.Sequential(
            nn.Linear(n_d, 400), nn.ELU(),
            nn.Linear(400, 400), nn.ELU(),
            nn.Linear(400, 400), nn.ELU(),
            nn.Linear(400, n_p * n_p_class)
        )
        self.rnn = nn.GRUCell(n_p * n_p_class + n_action, n_d)
        self.image_conditioner = nn.Linear(n_d + n_p * n_p_class, 1024)
        self.image_predictor = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, 5, 2), nn.ELU(),  # 1024 x 1 x 1, 1 -> 5
            nn.ConvTranspose2d(128, 64, 5, 2), nn.ELU(),  # 5 -> 13
            nn.ConvTranspose2d(64, 32, 6, 2), nn.ELU(),  # 13 -> 30
            nn.ConvTranspose2d(32, obs_size[2], 6, 2)  # 30 -> 64, 1 x 64 x 64
        )
        self.reward_predictor = nn.Sequential(
            nn.Linear(n_d + n_p * n_p_class, 400), nn.ELU(),
            nn.Linear(400, 400), nn.ELU(),
            nn.Linear(400, 400), nn.ELU(),
            nn.Linear(400, 1)
        )
        self.discount_predictor = nn.Sequential(
            nn.Linear(n_d + n_p * n_p_class, 400), nn.ELU(),
            nn.Linear(400, 400), nn.ELU(),
            nn.Linear(400, 400), nn.ELU(),
            nn.Linear(400, 1)
        )

    def observe(self, x, state):
        x = torch.squeeze(x, dim=3)
        x = torch.unsqueeze(x, dim=1)
        x = self.image_encoder(x)
        x = torch.reshape(x, (x.shape[0], -1))
        prior = self.transition_prior(state)
        prior = torch.reshape(prior, (-1, self.n_p, self.n_p_class))
        x = torch.cat((x, state), dim=-1)
        posterior = self.transition_posterior(x)
        posterior = torch.reshape(posterior, (-1, self.n_p, self.n_p_class))
        posterior_sample = self.sample_p(posterior)
        posterior_sample = torch.reshape(posterior_sample, (posterior_sample.shape[0], -1))
        return prior, posterior, posterior_sample

    def imagine(self, state):
        prior = self.transition_prior(state)
        prior = torch.reshape(prior, (-1, self.n_p, self.n_p_class))
        prior_sample = self.sample_p(prior)
        prior_sample = torch.reshape(prior_sample, (prior_sample.shape[0], -1))
        return prior, prior_sample

    def predict(self, state, p_sample):
        ext_state = torch.cat((state, p_sample), dim=-1)
        ext_state_img = self.image_conditioner(ext_state)
        ext_state_img = torch.reshape(ext_state_img, (ext_state_img.shape[0], ext_state_img.shape[1], 1, 1))
        x_hat = self.image_predictor(ext_state_img)
        x_hat = torch.squeeze(x_hat, dim=1)
        x_hat = torch.unsqueeze(x_hat, dim=-1)
        x_hat = torch.sigmoid(x_hat)
        r_hat = self.reward_predictor(ext_state)
        r_hat = torch.tanh(r_hat)
        d_hat = self.discount_predictor(ext_state)
        d_hat = torch.sigmoid(d_hat)
        return x_hat, r_hat, d_hat

    def step(self, state, p_sample, action):
        action = F.one_hot(action, num_classes=self.n_action)
        x = torch.cat((p_sample, action), dim=-1)
        new_state = self.rnn(x, state)
        return new_state

    def sample_p(self, logits):
        m = Categorical(logits=logits)
        s = m.sample()
        s = F.one_hot(s, num_classes=self.n_p_class)
        p = torch.softmax(logits, dim=-1)
        s = s + p - p.detach()
        return s
