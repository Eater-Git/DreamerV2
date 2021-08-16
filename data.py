import numpy as np


def rollout_episode(env, agent, init_obs, step_lim):
    observations, actions, rewards, dones = [], [], [], []

    step = 0
    obs = init_obs
    done = False
    while (not done) and (step < step_lim):
        step += 1
        observations.append(obs)
        action = agent.act(obs)
        actions.append(action)
        obs, reward, done, _ = env.step(action)
        rewards.append(reward)
        dones.append(done)

    return observations, actions, rewards, dones


class TrajectoryDataset(object):
    def __init__(self, dataset_size, discount_rate=0.995):
        self.dataset_size = dataset_size
        self.discount_rate = discount_rate
        self.data = [None] * self.dataset_size
        self.is_full = False
        self.data_idx = 0
        self.episode_terminals = []

    def __len__(self):
        return self.dataset_size if self.is_full else self.data_idx

    def add_episode(self, observations, actions, rewards, dones):
        assert len(observations) == len(actions) == len(rewards) == len(dones)
        discounts = [0.0 if done else self.discount_rate for done in dones]
        step = list(zip(observations, actions, rewards, discounts))
        episode_len = len(step)
        self.data[self.data_idx:self.data_idx + episode_len] = step
        self.data_idx += episode_len
        self.episode_terminals.append(self.data_idx - 1)
        if self.data_idx >= self.dataset_size:
            self.is_full = True
            n_overflow = self.data_idx - self.dataset_size
            self.data = self.data[n_overflow:]
            self.data_idx = 0
            self.episode_terminals = [t - n_overflow for t in self.episode_terminals if t >= n_overflow]

    def fill(self, env, agent, n_step, prev_obs=None):
        total_step = 0
        while total_step < n_step:
            if (prev_obs is not None) and (total_step == 0):
                init_obs = prev_obs
            else:
                init_obs = env.reset()

            step_lim = n_step - total_step
            observations, actions, rewards, dones = rollout_episode(env, agent, init_obs, step_lim)
            episode_step = len(observations)
            total_step += episode_step
            self.add_episode(observations, actions, rewards, dones)

        if dones[-1]:
            obs = env.reset()
        else:
            obs = observations[-1]

        return obs

    def sample(self, sequence_len):
        sequence_start = np.random.randint(len(self))
        sequence_stop = sequence_start + sequence_len - 1

        episode_terminals = np.array(self.episode_terminals)
        start_episode = np.where(episode_terminals - sequence_start >= 0)[0][0]
        stop_episode = np.where(episode_terminals - sequence_stop >= 0)[0]
        stop_episode = stop_episode[0] if (len(stop_episode) > 0) else (len(episode_terminals) + 1)
        sequence_start = sequence_start if (start_episode == stop_episode) else (episode_terminals[start_episode] - sequence_len + 1)
        sequence_data = self.data[sequence_start:sequence_start + sequence_len]

        sequence = {}
        sequence['observations'] = [sd[0] for sd in sequence_data]
        sequence['actions'] = [sd[1] for sd in sequence_data]
        sequence['rewards'] = [sd[2] for sd in sequence_data]
        sequence['discounts'] = [sd[3] for sd in sequence_data]

        return sequence
