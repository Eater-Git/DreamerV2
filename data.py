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
        self.data_idx = 0
        self.episode_terminals = []

    def __len__(self):
        return len(self.data)

    def add_episode(self, observations, actions, rewards, dones):
        assert len(observations) == len(actions) == len(rewards) == len(dones)
        discounts = [0.0 if done else self.discount_rate for done in dones]
        step = list(zip(observations, actions, rewards, discounts))
        episode_len = len(step)
        self.data[self.data_idx:self.data_idx + episode_len] = step
        self.data_idx += episode_len
        self.episode_terminals.append(self.data_idx - 1)
        if self.data_idx >= self.dataset_size:
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
            print(f'Fill - Episode Step: {episode_step}')
            self.add_episode(observations, actions, rewards, dones)
        print(f'Fill - Total Step: {total_step}')

        if dones[-1]:
            obs = env.reset()
        else:
            obs = observations[-1]

        return obs
