import torch
from torch.distributions import Categorical, MultivariateNormal, Normal, Bernoulli
from torch.distributions.kl import kl_divergence
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from networks import RSSM


class WorldModel(object):
    def __init__(self, obs_size, n_action, n_d, n_p, n_p_class, alpha=0.8, beta=0.1, sequence_len=50, n_batch=50, lr=2e-4, vlog_freq=10000, description='', device='cpu'):
        self.obs_size = obs_size
        self.n_action = n_action
        self.n_d = n_d
        self.n_p = n_p
        self.n_p_class = n_p_class
        self.alpha = alpha
        self.beta = beta
        self.sequence_len = sequence_len
        self.n_batch = n_batch
        self.lr = lr
        self.vlog_freq = vlog_freq
        self.description = description
        self.device = device

        self.rssm = RSSM(obs_size, n_action, n_d, n_p, n_p_class).to(device)
        self.optimizer = optim.Adam(self.rssm.parameters(), lr=lr, eps=1e-5, weight_decay=1e-6)
        self.writer = SummaryWriter(comment='_' + description)
        self.n_iter = 1

    def train(self, dataset):
        observations, actions, rewards, discounts = self.sample_sequences(dataset)
        observations = torch.transpose(observations, 0, 1)
        actions = torch.transpose(actions, 0, 1)
        rewards = torch.transpose(rewards, 0, 1)
        discounts = torch.transpose(discounts, 0, 1)

        x_hats, r_hats, d_hats, priors, posteriors = self.run(observations, actions)
        if self.n_iter % self.vlog_freq == 0:
            self.write_video(x_hats, 'train')

        loss, loss_vals = self.compute_loss(observations, rewards, discounts, x_hats, r_hats, d_hats, priors, posteriors)
        self.write_loss(loss_vals)

        self.optimizer.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
            loss.backward()
        torch.nn.utils.clip_grad_norm_(self.rssm.parameters(), 100)
        self.optimizer.step()

        self.n_iter += 1

    def run(self, observations, actions):
        x_hats = torch.zeros((self.sequence_len, self.n_batch, *self.obs_size), dtype=torch.float, device=self.device)
        r_hats = torch.zeros((self.sequence_len, self.n_batch, 1), dtype=torch.float, device=self.device)
        d_hats = torch.zeros((self.sequence_len, self.n_batch, 1), dtype=torch.float, device=self.device)
        priors = torch.zeros((self.sequence_len, self.n_batch, self.n_p, self.n_p_class), dtype=torch.float, device=self.device)
        posteriors = torch.zeros((self.sequence_len, self.n_batch, self.n_p, self.n_p_class), dtype=torch.float, device=self.device)
        state = torch.zeros((self.n_batch, self.n_d), dtype=torch.float, device=self.device)
        states = torch.zeros((self.sequence_len, self.n_batch, self.n_d), dtype=torch.float, device=self.device)
        p_samples = torch.zeros((self.sequence_len, self.n_batch, self.n_p * self.n_p_class), dtype=torch.float, device=self.device)

        for t in range(self.sequence_len):
            states[t].copy_(state)
            priors[t], posteriors[t], posterior_samples = self.rssm.observe(observations[t], state)
            p_samples[t].copy_(posterior_samples)
            x_hats[t], r_hats[t], d_hats[t] = self.rssm.predict(state, posterior_samples)
            if t != (self.sequence_len - 1):
                state = self.rssm.step(state, posterior_samples, actions[t])
        self.states = states.view(-1, self.n_d)
        self.p_samples = p_samples.view(-1, self.n_p * self.n_p_class)
        return x_hats, r_hats, d_hats, priors, posteriors

    def write_video(self, x_hats, tag):
        vid_tensor = x_hats.permute((1, 0, 4, 2, 3))
        self.writer.add_video('Image/' + tag, vid_tensor, global_step=self.n_iter, fps=5)

    def compute_loss(self, observations, rewards, discounts, x_hats, r_hats, d_hats, priors, posteriors):
        x_hats = torch.reshape(x_hats, (x_hats.shape[0] * x_hats.shape[1], -1))
        r_hats = torch.reshape(r_hats, (-1, 1))
        d_hats = torch.reshape(d_hats, (-1, 1))
        observations = torch.reshape(observations, (observations.shape[0] * observations.shape[1], -1))
        rewards = torch.reshape(rewards, (-1, 1))
        discounts = torch.reshape(discounts, (-1, 1))
        image_loss = MultivariateNormal(x_hats, torch.eye(x_hats.shape[-1], device=self.device)).log_prob(observations).mean()
        reward_loss = Normal(r_hats, torch.tensor([1.0], device=self.device)).log_prob(rewards).mean()
        discount_loss = Bernoulli(logits=d_hats).log_prob(discounts).mean()

        priors = torch.reshape(priors, (-1, self.n_p, self.n_p_class))
        posteriors = torch.reshape(posteriors, (-1, self.n_p, self.n_p_class))
        kl_loss = self.compute_kl(priors, posteriors)

        loss = - (image_loss + reward_loss + discount_loss) + self.beta * kl_loss
        loss_vals = {}
        loss_vals['total'] = loss.item()
        loss_vals['image'] = - image_loss.item()
        loss_vals['reward'] = - reward_loss.item()
        loss_vals['discount'] = - discount_loss.item()
        loss_vals['kl'] = self.beta * kl_loss.item()

        return loss, loss_vals

    def write_loss(self, loss_vals):
        self.writer.add_scalar('Loss/train/total', loss_vals['total'], self.n_iter)
        self.writer.add_scalar('Loss/train/image', loss_vals['image'], self.n_iter)
        self.writer.add_scalar('Loss/train/reward', loss_vals['reward'], self.n_iter)
        self.writer.add_scalar('Loss/train/discount', loss_vals['discount'], self.n_iter)
        self.writer.add_scalar('Loss/train/kl', loss_vals['kl'], self.n_iter)

    def sample_sequences(self, dataset):
        observations, actions, rewards, discounts = [], [], [], []
        for _ in range(self.n_batch):
            sequence = dataset.sample(self.sequence_len)
            observations.append(sequence['observations'])
            actions.append(sequence['actions'])
            rewards.append(sequence['rewards'])
            discounts.append(sequence['discounts'])
        observations = torch.tensor(observations, dtype=torch.float, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
        discounts = torch.tensor(discounts, dtype=torch.float, device=self.device)
        return observations, actions, rewards, discounts

    def sample_state(self, n_sample):
        n_total = self.states.shape[0]
        idxs = torch.randint(n_total, (n_sample,))
        state = self.states[idxs]
        p_sample = self.p_samples[idxs]
        return state, p_sample

    def imagine(self, agent, init_state, init_p_sample, horizon):
        n_trajectory = init_state.size()[0]
        images = torch.zeros((horizon, n_trajectory, *self.obs_size), dtype=torch.float, device=self.device)
        ext_states = torch.zeros((horizon, n_trajectory, self.n_d + self.n_p * self.n_p_class), dtype=torch.float, device=self.device)
        actions = torch.zeros((horizon, n_trajectory), dtype=torch.long, device=self.device)
        rewards = torch.zeros((horizon, n_trajectory, 1), dtype=torch.float, device=self.device)
        discounts = torch.zeros((horizon, n_trajectory, 1), dtype=torch.float, device=self.device)

        ext_state = torch.cat((init_state, init_p_sample), dim=-1)
        action = agent.act(ext_state, is_latent=True)
        state = self.rssm.step(init_state, init_p_sample, action)
        for t in range(horizon):
            prior, prior_sample = self.rssm.imagine(state)
            images[t], rewards[t], discounts[t] = self.rssm.predict(state, prior_sample)
            ext_state = torch.cat((state, prior_sample), dim=-1)
            ext_states[t].copy_(ext_state)
            action = agent.act(ext_state, is_latent=True)
            actions[t].copy_(action)
            if t != (horizon - 1):
                state = self.rssm.step(state, prior_sample, action)

        # print(ext_states.size(), actions.size(), rewards.size(), discounts.size())

        if self.n_iter % self.vlog_freq == 0:
            self.write_video(images, 'imagine')

        return ext_states, actions, rewards, discounts

    def compute_kl(self, prior, posterior):
        kl_loss = self.alpha * kl_divergence(Categorical(logits=posterior.detach()), Categorical(logits=prior)) \
            + (1 - self.alpha) * kl_divergence(Categorical(logits=posterior), Categorical(logits=prior.detach()))
        kl_loss = torch.sum(kl_loss, dim=-1)
        kl_loss = kl_loss.mean()
        return kl_loss
