import gym 
import agents

episodes = 10
time = 10000

if __name__ == '__main__':
    print("creating cartpole env...")

    env = gym.make('CartPole-v0')

    for episode in range(episodes):
        observation = env.reset()
        for t in range(time):
            #env.render()
            observation, reward, done, info = env.step(env.action_space.sample())

            if done:
                print("Episode{} finished after {} timesteps".format(episode, t+1))
                break
    print("closing env...")
    env.close()
