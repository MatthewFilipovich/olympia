import numpy as np
from agent import Agent
from render import render_episodes


class Environment:
	def __init__(self, shape=(21,15), n_agents=1, n_teams=1):
		self.shape = shape
		self.n_agents = n_agents
		self.n_teams = n_teams
		self._initial_ball_position = (int(shape[0]/2), int(shape[1]/2))
		self.reset()
		self._drawables = [self._static_field, self.teams, self.ball]

	def __init_field__(self):
		self.field = np.zeros(shape=self.shape, dtype=np.uint8)
		self.field[0, :] = 255    # left wall
		self.field[:, 0] = 255    # bottom wall
		self.field[self.shape[0]-1, :] = 255    # right wall
		self.field[:, self.shape[1] - 1] = 255    # top wall
		bot = int(self.shape[1]/4)+1
		top = int(self.shape[1]/4+self.shape[1]/2)
		self.field[0, bot:top] = 254    # left goal
		self.field[self.shape[0]-1, bot:top] = 254    # right goal
		self._static_field = self.field.copy()  # never add objects to this field (for replacing values after agent/ball moves over position)

	def __init_agents__(self):
		self.teams = [[] for _ in range(self.n_teams)]
		if self.n_agents % self.n_teams is not 0:
			raise ValueError('Teams should be the same size.')
		for team in self.teams:
			for player in range(int(self.n_agents/self.n_teams)):
				team.append(Agent(player))

	def __init_ball__(self):
		self.ball = Ball(self._initial_ball_position)

	def reset(self):
		# reset environment to original state
		self.__init_field__()

		# put ball in center of field
		self.__init_ball__()

		# place players depending on number
		self.__init_agents__()

		return self.field

	def step(self, *actions):
		# evolve environment based on actions
		# return next state, rewards to each agent, if terminal state, and

		# move player positions according to current positions, actions
		# 	could move by acceleration values
		# 	could move by single grid spaces

		# move ball according to its current velocity and field damping

		return observation, rewards, done, info

	def render(self):
		# create a visualization of the env
		pass


class Ball:
	def __init__(self, initial_position):
		self.mass = 0    # probably need mass parameter for collisions
		self.x = initial_position[0]
		self.y = initial_position[1]
		self.v_x = 0
		self.v_y = 0


