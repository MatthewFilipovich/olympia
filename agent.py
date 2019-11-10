import numpy as np
import tensorflow as tf


class Agent:
	def __init__(self, number):
		self.number = number
		self.__init_policy()

	def __init_policy__(self):
		# make tensorflow network as agent's policy
		pass