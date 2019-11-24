"""

This file will contain the Agent class that will store both the parameters of the agent on the field (position,
velocity), as well as the Agent's specific update algorithm

"""
import numpy as np


class Agent:
	def __init__(self, number, initial_position):
		self.number = number
		self.x = initial_position[0]
		self.y = initial_position[1]
		self.v_x = 0
		self.v_y = 0
		self.__init_policy__()

	def __init_policy__(self):
		# make tensorflow network as agent's policy/q-function
		pass