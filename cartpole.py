import gym 
import agents
import torch.nn.functional as F
import torch

episodes = 10
steps = 10000

if __name__ == '__main__':
    print("creating cartpole env...")

    env = gym.make('CartPole-v0')
    agent = agents.ActorCritic(layers=10,num_of_actions=2, units=4, act=F.relu, device=None)

    for episode in range(episodes):
        observation = env.reset()
        for stp in range(steps):
            env.render()
            
            act = agent.act(torch.tensor(observation, dtype=torch.float))

            observation, reward, done, info = env.step(act)

            if done:
                print("Episode{} finished after {} timesteps".format(episode, stp+1))
                break
    print("closing env...")
    env.close()
