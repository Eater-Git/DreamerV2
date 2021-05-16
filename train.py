import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gym


class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs):
        return self.action_space.sample()


def evaluate_agent(env_id, agent, episode_count, seed, render):
    env = gym.make(env_id)
    env.seed(seed)
    np.random.seed(seed)

    agent = agent(env.action_space)

    env.close()
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
        print(f'Episode: {i+1}, reward: {episode_reward}')
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

    # Prefill dataset

    # Learn world model

    # Train agent

    # Explore environment

    # Evaluate agent
    agent = RandomAgent
    episode_count = 1
    evaluate_agent(env_id, agent, episode_count, seed, render)


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='Breakout-v0')
    parser.add_argument('--seed', default=0)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    train(parse_arg())
