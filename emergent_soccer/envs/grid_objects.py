from numpy import array


class GridObject:
    def __init__(self, env, initial_position):
        self.env = env
        self.position = array(initial_position)


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
    def __init__(self, env, number, initial_position):
        super().__init__(env, initial_position)
        self.number = number  # player's number on its team
        self.has_ball = False
        self.move_counter = -1
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

    def act(self, ndx):
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


