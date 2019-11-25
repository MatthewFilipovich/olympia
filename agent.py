
from environment import GridObject
from numpy import array


class Agent(GridObject):
    def __init__(self, env, number, initial_position):
        super().__init__(env, initial_position)
        self.number = number  # player's number on its team
        self.has_ball = False
        self.actions = {'STAY': array([0, 0]),
                        'RIGHT': array([1, 0]),
                        'UPRIGHT': array([1, 1]),
                        'UP': array([0, 1]),
                        'UPLEFT': array([-1, 1]),
                        'LEFT': array([-1, 0]),
                        'DOWNLEFT': array([-1, -1]),
                        'DOWN': array([0, -1]),
                        'DOWNRIGHT': array([1, -1]),
                        'THROW_RIGHT': array([0, 0]),
                        'THROW_UPRIGHT': array([0, 0]),
                        'THROW_UP': array([0, 0]),
                        'THROW_UPLEFT': array([0, 0]),
                        'THROW_LEFT': array([0, 0]),
                        'THROW_DOWNLEFT': array([0, 0]),
                        'THROW_DOWN': array([0, 0]),
                        'THROW_DOWNRIGHT': array([0, 0])
                        }

    def act(self, ndx):
        movement = list(self.actions.values())[ndx]
        new_pos = self.position + movement
        if not (new_pos > 250).any():  # greater than 250 means wall or net
            self.position += movement
        if self.has_ball:
            self.env.ball.position = self.position


