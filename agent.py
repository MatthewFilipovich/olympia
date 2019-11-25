"""

This file will contain the Agent class that will store both the parameters of the agent on the field (position,
velocity), as well as the Agent's specific update algorithm

- play around with increment, v_max, and grids_per_metre

"""


class GridObject:
    def __init__(self, env, initial_position, size, v_max, mass):
        self.env = env
        self.size = size
        self.mass = mass  # probably need mass parameter for collisions...
        self.x = initial_position[0]
        self.y = initial_position[1]
        self.v_x = 0
        self.v_y = 0
        self._v_max = v_max

    def move(self):
        self.x += self.v_x * self.env.grids_per_metre
        self.y += self.v_y * self.env.grids_per_metre


class Agent(GridObject):
    def __init__(self, env, number, initial_position):
        super().__init__(env, initial_position, size=(2,2), v_max=5, mass=150)
        self.env = env
        self.number = number

    def step(self, action):
        self.increase_speed(action)
        self.move()

    def increase_speed(self, action):
        increment = 1
        if action == 0:
            pass  # do nothing
        elif action == 1 and not self._too_fast(self.v_x+increment, self.v_y):
            self.v_x += increment
        elif action == 2 and not self._too_fast(self.v_x, self.v_y+increment):
            self.v_y += increment
        elif action == 3 and not self._too_fast(self.v_x-increment, self.v_y):
            self.v_x -= increment
        elif action == 4 and not self._too_fast(self.v_x, self.v_y-increment):
            self.v_y -= increment
        else:
            raise ValueError('Invalid move index, must be in range (0,4).')

    def _too_fast(self, vx, vy):
        return bool(vx**2 + vy**2 >= self._v_max**2)


class Ball(GridObject):
    def __init__(self, env, initial_position):
        super().__init__(env, initial_position, size=(1,1), v_max=20, mass=1)

    def step(self):
        self.move()

