import argparse

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gym
import gym.envs.atari
import gym.wrappers
import torch

from agents import RandomAgent, ActorCritic
from data import TrajectoryDataset
from models import WorldModel


def evaluate_agent(env, agent, episode_count, render):
    for i in range(episode_count):
        episode_reward = 0
        done = False
        obs = env.reset()
        frames = []
        while True:
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                img = env.render(mode='rgb_array')
                frames.append(img)
            if done:
                break
        print(f'Eval - Episode: {i+1}, reward: {episode_reward}')
        if render:
            fig = plt.figure()

            def animate(i):
                plt.cla()
                im = plt.imshow(frames[i], animated=True)
                plt.axis('off')
                return [im]
            _ = animation.FuncAnimation(fig, animate, interval=16, frames=len(frames), repeat=False)
            plt.show()


def train(args):
    # config
    env_id = args.env
    seed = args.seed
    render = args.render
    save_wm = args.save_wm
    load_wm = args.load_wm

    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    # Preperation: environment
    env = gym.envs.atari.AtariEnv(env_id, obs_type='image', frameskip=1, repeat_action_probability=0.25)
    env.spec = gym.envs.registration.EnvSpec('NoFrameskip-v0')
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, screen_size=64, grayscale_obs=True, grayscale_newaxis=True, scale_obs=True)
    obs_size = env.observation_space.shape
    n_action = env.action_space.n

    env.seed(seed)
    np.random.seed(seed)

    # Preperation: agent
    horizon = 5
    n_sample = 2
    agent_explore = RandomAgent(env.action_space)
    agent_policy = ActorCritic(env.action_space, device)

    # Preperation: dataset
    dataset_size = int(2e4)
    n_prefill = int(2e3)
    n_fill = 4
    dataset = TrajectoryDataset(dataset_size)
    prev_obs = dataset.fill(env, agent_explore, n_prefill)

    # Preperation: world model
    n_d = 600
    n_p = 32
    n_p_class = 32
    sequence_len = 25  # 50
    n_batch = 5  # 50
    lr = 2e-4
    vlog_freq = 5000
    wm = WorldModel(obs_size, n_action, n_d, n_p, n_p_class, sequence_len=sequence_len, n_batch=n_batch, lr=lr, vlog_freq=vlog_freq, description='test', device=device)
    if load_wm is not None:
        wm.rssm.load_state_dict(torch.load(load_wm))

    # Training Loop
    n_iter = 25000
    for _ in tqdm(range(n_iter)):
        # 1. Train world model
        wm.train(dataset)

        # 2. Train actor critic
        init_state, init_p_sample = wm.sample_state(n_sample)
        with torch.no_grad():
            states, actions, rewards, discounts = wm.imagine(agent_policy, init_state, init_p_sample, horizon)
        agent_policy.train(states, actions, rewards, discounts)

        # 3. Explore environment
        prev_obs = dataset.fill(env, agent_explore, n_fill, prev_obs=prev_obs)

    if save_wm is not None:
        torch.save(wm.rssm.state_dict(), save_wm)

    # Evaluate agent
    episode_count = 1
    evaluate_agent(env, agent_policy, episode_count, render)

    # Clean up
    env.close()


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='breakout')
    parser.add_argument('--seed', default=0)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--save_wm', default=None)
    parser.add_argument('--load_wm', default=None)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    train(parse_arg())
