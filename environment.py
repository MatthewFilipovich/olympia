import numpy as np
from agent import Agent

class Environment:
	def __init__(self, shape=(15,8), n_agents=1, n_teams=1):
		self.shape = shape
		self.field = np.zeros(shape=shape, dtype=np.uint8)
		self.__init_agents__(n_agents, n_teams)

	def __init_agents__(self, n_agents, n_teams):
		self.teams = [[] for _ in range(n_teams)]
		if n_agents % n_teams is not 0:
			raise ValueError('Teams should be the same size.')
		for team in self.teams:
			for player in range(int(n_agents/n_teams)):
				team.append(Agent(player))
		pass

	def step(self, *actions):
		# evolve environment based on actions
		# return next state, rewards to each agent, if terminal state, and 
		return observation, rewards, done, info

	def reset(self):
		# reset environment to original state
		# return the state of the environment after reset
		return observation

	def render(self):
		# create a visualization of the env
		pass

		