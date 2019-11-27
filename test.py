import unittest
import gym
import olympia


class TestStringMethods(unittest.TestCase):
    def test_moving_throwing(self):
        env = gym.make('olympia-ram-v0')
        print('hello')
        env.render()
        moves = [5,5,5,5,5,5,5,13,5,5,5,5,5,5,1,13,1,0]
        ep = []
        for move in moves:
            _, _, done, _ = env.step(*[move])
            ep.append((env.field.copy(), done))
        env.render_episode(ep)

    def test_randomized_pos(self):
        env = gym.make('olympia-rgb-v0')
        for _ in range(10):
            env.reset()
            env.render()


if __name__ == '__main__':
    unittest.main()
