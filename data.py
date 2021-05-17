from collections import deque


class TrajectoryDataset(object):
    def __init__(self, dataset_size, discount_rate=0.995):
        self.dataset_size = dataset_size
        self.discount_rate = discount_rate
        self.data = [None for _ in range(self.dataset_size)]
        self.data_idx = 0
        self.episode_terminals = []
        self.prev_obs = None

    def __len__(self):
        return len(self.data)

    def add_episode(self, observations, actions, rewards, dones):
        assert len(observations) == len(actions) == len(rewards) == len(dones)
        discounts = [0.0 if done else self.discount_rate for done in dones]
        step = [(o, a, r, d) for o, a, r, d in zip(observations, actions, rewards, discounts)]
        episode_len = len(step)
        self.data[self.data_idx:self.data_idx + episode_len] = step
        self.data_idx += episode_len
        self.episode_terminals.append(self.data_idx - 1)
        if self.data_idx >= self.dataset_size:
            n_overflow = self.data_idx - self.dataset_size
            self.data = self.data[n_overflow:]
            self.data_idx = 0
            self.episode_terminals = [t - n_overflow for t in self.episode_terminals if t >= n_overflow]

    def fill(self, env, agent, n_step, reset=True):
        total_step = 0
        observations, actions, rewards, dones = deque(), deque(), deque(), deque()
        while total_step <= n_step:
            done = False
            if reset or total_step > 0:
                obs = env.reset()
            else:
                obs = self.prev_obs
            while True:
                total_step += 1
                observations.append(obs)
                action = agent.act(obs)
                actions.append(action)
                obs, reward, done, _ = env.step(action)
                rewards.append(reward)
                dones.append(done)
                if done or total_step >= n_step:
                    print(f'Fill - Step: {total_step}')
                    break
            self.add_episode(list(observations), list(actions), list(rewards), list(dones))
            observations.clear(), actions.clear(), rewards.clear(), dones.clear()

        if not reset:
            self.prev_obs = obs
