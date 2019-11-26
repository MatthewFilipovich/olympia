import unittest
import gym
import emergent_soccer


class TestStringMethods(unittest.TestCase):
    def test_moving_throwing(self):
        env = gym.make('soccer-v0')
        env.render()
        moves = [5,5,5,5,5,5,5,13,5,5,5,5,5,5,1,1,1,13,0]
        ep = []
        for move in moves:
            ep.append(env.step(*[move]))
        env.render_episode(ep)


if __name__ == '__main__':
    unittest.main()
