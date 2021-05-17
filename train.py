import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gym
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

    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    # Preperation
    env = gym.make(env_id)
    env.seed(seed)
    np.random.seed(seed)

    agent = RandomAgent(env.action_space)

    dataset_size = int(2e4)
    dataset = TrajectoryDataset(dataset_size)
    n_prefill = int(2e3)
    prev_obs = dataset.fill(env, agent, n_prefill)

    n_latent = 600
    n_p = 32
    wm = WorldModel(n_latent, n_p, device)

    # 1. Train world model
    wm.train(dataset)

    # 2. Train actor critic
    agent = ActorCritic(env.action_space, device)

    horizon = 5
    n_sample = 1
    init_state = wm.sample_state(n_sample)
    states, actions, rewards, discounts = wm.imagine(agent, init_state, horizon)
    agent.train(states, actions, rewards, discounts)

    # 3. Explore environment
    n_fill = 4
    prev_obs = dataset.fill(env, agent, n_fill, prev_obs=prev_obs)

    # Evaluate agent
    episode_count = 1
    evaluate_agent(env, agent, episode_count, render)

    # Clean up
    env.close()


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='Breakout-v0')
    parser.add_argument('--seed', default=0)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    train(parse_arg())
