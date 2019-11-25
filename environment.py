import numpy as np
from numpy import ones, array
from agent import Agent
from render import render_episodes
from training_schemes import scheme
import gym

"""
In the field state:
- 0 represents the ball
- 1 represents empty space
- 2 represents players on team 1
- 3 represents players on team 2
- 254 represents a goal
- 255 represents a wall
"""


class Environment(gym.Env):
	def __init__(self, shape=(21, 15), n_agents=1, n_teams=1):
		self.shape = shape
		self.n_agents = n_agents
		self.n_teams = n_teams
		self._initial_ball_position = (int(shape[0]/2), int(shape[1]/2))
		self.__init_static_field__()
		self.reset()
		self._training_level = 'one player'

	def __init_static_field__(self):
		self.field = ones(shape=self.shape, dtype=np.uint8)
		self.field[0, :] = 255  # left wall
		self.field[:, 0] = 255  # bottom wall
		self.field[self.shape[0] - 1, :] = 255  # right wall
		self.field[:, self.shape[1] - 1] = 255  # top wall
		bot = int(self.shape[1] / 4) + 1
		top = int(self.shape[1] / 4 + self.shape[1] / 2)
		self.field[0, bot:top] = 254  # left goal
		self.field[self.shape[0] - 1, bot:top] = 254  # right goal
		self._static_field = self.field.copy()  # never add objects to this field (for replacing values after agent/ball moves over position)

	def __init_agents__(self):
		self.teams = [[] for _ in range(self.n_teams)]
		if self.n_agents % self.n_teams is not 0:
			raise ValueError('Teams should be the same size.')
		for team in self.teams:
			for player in range(int(self.n_agents/self.n_teams)):
				team.append(Agent(self, player, (int(a*b) for a, b in zip(self.shape, scheme[self._training_level][team][player]))))

	def reset(self):
		# reset environment to original state
		self.field = self._static_field

		# put ball in center of field
		self.ball = Ball(self, self._initial_ball_position)

		# place players depending on their number
		self.__init_agents__()

		self._add_to_field()
		return self.field.copy()

	def _player_at(self, pos):
		return bool(1 < self.field[pos[0], pos[1]] < 4)

	def move_ball(self):
		move = self.ball.movement.pop()
		# check if ball has stopped
		if len(self.ball.movement) == 0:
			self.ball.moving = False

		new_pos = self.ball.position + move
		inter_pos = self.ball.position + move / 2

		# TODO: must check if ball hits the wall

		# check if player is in the way
		if self._player_at(inter_pos) or self._player_at(new_pos):
			for i, team in enumerate(self.teams):
				for j, player in enumerate(team):
					if player.position == new_pos or player.position == inter_pos:
						player.has_ball = True
						self.ball.moving = False

		# make ball move
		self.ball.position += move

	def step(self, *actions):
		winning_team = None
		done = False

		# move players
		# TODO: must implement check for players on same space
		for i, team in enumerate(self.teams):
			for j, player in enumerate(team):
				action_ndx = int(i*len(team)+j)
				player.act(actions[action_ndx])
				if player.has_ball and action_ndx > 8:  # player threw the ball
					player.has_ball = False
					self.ball.thrown(action_ndx-9)

		if self.ball.moving:
			self.move_ball()

		# check for ball in net
		if self._static_field[self.ball.position[0], self.ball.position[1]] == 254:
			done = True
			if self.ball.position[0] == 0:
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

		return self.field.copy(), rewards, done, None

	def _add_to_field(self):
		self.field = self._static_field.copy()
		for i,team in enumerate(self.teams):
			for player in team:
				self.field[player.position[0], player.position[1]] = i+2  # teams look different on field
		if not self.field[self.ball.position[0], self.ball.position[1]] == 1:
			self.field[self.ball.position[0], self.ball.position[1]] = 0

	def render(self):
		# create a visualization of the env
		return self.field.copy()


class GridObject:
	def __init__(self, env, initial_position):
		self.env = env
		self.position = np.array(initial_position)


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
		self.movement = list(self.movements.values())[ndx]
