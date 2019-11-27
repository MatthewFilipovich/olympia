import random
import gym
from olympia.envs import OlympiaRGB, OlympiaRAM
import numpy as np
from numpy import array
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class GridObject:
    def __init__(self, env, initial_position):
        self._initial_position = array(initial_position)
        self.env = env
        self.reset_position()

    def reset_position(self):
        self.position = self._initial_position.copy()


class Ball(GridObject):
    def __init__(self, env, initial_position):
        super(Ball, self).__init__(env, initial_position)
        self._turns_thrown = 3
        self.moving = False
        self.movements = {
            'RIGHT': [array([2, 0])] * self._turns_thrown,
            'UPRIGHT': [array([2, 2])] * self._turns_thrown,
            'UP': [array([0, 2])] * self._turns_thrown,
            'UPLEFT': [array([-2, 2])] * self._turns_thrown,
            'LEFT': [array([-2, 0])] * self._turns_thrown,
            'DOWNLEFT': [array([-2, -2])] * self._turns_thrown,
            'DOWN': [array([0, -2])] * self._turns_thrown,
            'DOWNRIGHT': [array([2, -2])] * self._turns_thrown
        }
        self.movement = []

    def thrown(self, ndx):
        self.moving = True
        self.movement = list(self.movements.values())[ndx].copy()

class Agent(GridObject):
    def __init__(self, agent_type, env, team, number, initial_position):
        super().__init__(env, initial_position)
        "ANN values"
        self.state_size = env.state_size
        self.action_size = env.action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model(agent_type)

        "Environment Values"
        self.team = team
        self.number = number  # player's number on its team
        self.reset_position()
        self.actions = {'STAY': array([0, 0]),
                        'RIGHT': array([1, 0]),
                        'UPRIGHT': array([1, 1]),
                        'UP': array([0, 1]),
                        'UPLEFT': array([-1, 1]),
                        'LEFT': array([-1, 0]),
                        'DOWNLEFT': array([-1, -1]),
                        'DOWN': array([0, -1]),
                        'DOWNRIGHT': array([1, -1]),
                        'THROW_RIGHT': array([0, 0]),  # no movement happens on throw actions
                        'THROW_UPRIGHT': array([0, 0]),
                        'THROW_UP': array([0, 0]),
                        'THROW_UPLEFT': array([0, 0]),
                        'THROW_LEFT': array([0, 0]),
                        'THROW_DOWNLEFT': array([0, 0]),
                        'THROW_DOWN': array([0, 0]),
                        'THROW_DOWNRIGHT': array([0, 0])
                        }

    def _build_model(self, agent_type):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        if agent_type == 'RAM':
            model.add(Dense(24, input_dim=self.state_size, activation='relu'))
            model.add(Dense(24, activation='relu'))
            model.add(Dense(self.action_size, activation='linear'))
        elif agent_type == 'RGB':
            model.add() # add convolutional layers
        else:
            raise ValueError('Invalid agent type supplied!')

        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model
    
    def reset_position(self):
        super().reset_position()
        self.has_ball = False
        self.move_counter = -1
        self.prev_position = None

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def act(self, ndx):
        self.prev_position = self.position.copy()
        if ndx <= 8:  # movement action
            movement = list(self.actions.values())[ndx]
            new_pos = self.position + movement
            space = self.env.field[new_pos[0], new_pos[1]]
            if not (space >= 250).any():  # greater than 250 means wall, net, players or ball
                if self.has_ball:
                    if self.move_counter > 0:
                        self.move_counter -= 1
                        self.position += movement
                        self.env.ball.position = self.position.copy()
                else:
                    self.position += movement
            else:
                if (self.env.field[new_pos[0], new_pos[1]] == array([0, 0, 255])).all():
                    self.position += movement
                    self.has_ball = True
                    self.move_counter = 3
        else:  # throwing action
            if self.has_ball:
                self.has_ball = False
                self.move_counter = -1
                self.env.ball.thrown(ndx-9)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    