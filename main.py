import numpy as np
import random
import gym
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from olympia.envs import OlympiaRGB, OlympiaRAM
import time

if __name__ == "__main__":
    EPISODES = 1000
    shape = (21, 15)
    training_level = 'one player'
    env = OlympiaRGB(shape=shape, training_level=training_level)
    agents = env.get_agents()
    # agent.load("soccer.h5")
    done = False
    batch_size = 32
    start_time = time.time()
    for e in range(EPISODES):
        state = env.reset()
        while not done:
            env.render()
            actions = [agent.choose_action(state) for agent in agents]
            next_state, rewards, done, _ = env.step(*actions)
            for agent, action, reward in zip(agents, actions, rewards):       
                agent.remember(state, action, reward, next_state, done)
                if len(agent.memory) > batch_size and not done:
                    agent.replay(batch_size)
            if done:
                print("Episode {}/{} complete. Training time: {}"
                      .format(e, EPISODES, time.time() - start_time))
            state = next_state