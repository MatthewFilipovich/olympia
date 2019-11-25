import numpy as np
from objects import Agent, Ball
from render import render_episodes
from training_schemes import scheme


class Environment:
	def __init__(self, shape=(21, 15), n_agents=1, n_teams=1):
		self.shape = shape
		self.n_agents = n_agents
		self.n_teams = n_teams
		self._initial_ball_position = (int(shape[0]/2), int(shape[1]/2))
		self.reset()
		self._drawables = [self.teams, self.ball]
		self._training_level = 'one player'

	def __init_field__(self):
		self.grids_per_metre = int(self.shape[0]/100)
		self.field = np.ones(shape=self.shape, dtype=np.uint8)
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
				team.append(Agent(self, player, (int(a*b) for a, b in zip(self.shape, scheme[self._training_level][team][player]))))

	def __init_ball__(self):
		self.ball = Ball(self, self._initial_ball_position)

	def reset(self):
		# reset environment to original state
		self.__init_field__()

		# put ball in center of field
		self.__init_ball__()

		# place players depending on number
		self.__init_agents__()

		self._add_to_field()
		return self.field.copy()

	def step(self, *actions):
		winning_team = None
		done = False
		# evolve environment based on actions

		# move players
		for i, team in enumerate(self.teams):
			for j, player in enumerate(team):
				player.step(actions[int(i*len(team)+j)])

		# move ball
		self.ball.step()

		# handle collisions between players, ball, and walls

		# check for ball in net
		if self._static_field[round(self.ball.x), round(self.ball.y)] == 254:
			done = True
			if round(self.ball.x) == 0:
				winning_team = 0
			else:
				winning_team = 1

		if winning_team is None:
			rewards = [-1 for _ in range(self.n_agents)]
		elif winning_team == 0:
			rewards = [100] * int(self.n_agents / 2) + [-100] * int(self.n_agents / 2)
		else:
			rewards = [-100] * int(self.n_agents / 2) + [100] * int(self.n_agents / 2)

		self._add_to_field()
		return self.field.copy(), rewards, done

	def _add_to_field(self):
		self.field = self._static_field.copy()
		for i,team in enumerate(self.teams):
			for player in team:
				self.field[round(player.x), round(player.y)] = i+1
		self.field[round(self.ball.x), round(self.ball.y)] = 0

	def render(self, episodes):
		# create a visualization of the env
		render_episodes(episodes)
