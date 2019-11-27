import unittest
import gym
import olympia


class TestStringMethods(unittest.TestCase):
    def test_moving_throwing(self):
        env = gym.make('olympia-ram-v0')
        print('hello')
        env.render()
        moves = [5,5,5,5,5,5,5,13,5,5,5,5,5,5,1,1,1,13,0]
        ep = []
        for move in moves:
            _, _, done, _ = env.step(*[move])
            ep.append((env.field.copy(), done))
        env.render_episode(ep)


if __name__ == '__main__':
    unittest.main()
