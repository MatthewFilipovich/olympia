import unittest
import gym
from olympia.envs import OlympiaRGB, OlympiaRAM


class TestStringMethods(unittest.TestCase):
    def test_moving_throwing(self):
        env = OlympiaRAM()
        while not (env.teams[0][0].position == env.teams[0][0]._initial_position).all():
            print('Resetting: player was at ({},{})'.format(*env.teams[0][0].position))
            env.reset()
        moves = [5,5,5,5,4,4,12,4,4,4,3,5,15,7,7,7,7,7,7,5,5,0,13,0,0,0,0,0]
        ep = []
        for move in moves:
            env.render()
            _, _, done, _ = env.step(*[move])
            ep.append((env.field.copy(), done))
            if done:
                print('Finished fuckers')
        env.render()

    def test_randomized_pos(self):
        env = OlympiaRAM(shape=(15,9), training_level='one_v_one')
        env.train(episodes=11, batch_size=5, render=False, load_saved=False)


if __name__ == '__main__':
    unittest.main()
