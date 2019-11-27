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
    shape = (21, 15)
    training_level = 'one v one'
    episodes = 1000
    batch_size = 25

    env = OlympiaRGB(shape=shape, training_level=training_level)
    env.train(episodes, batch_size, render=False, load_saved=False)
    env.run(render=True)