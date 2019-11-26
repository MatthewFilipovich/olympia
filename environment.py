import sys
from contextlib import closing
import numpy as np
from numpy import ones, array
from grid_objects import Agent, Ball
from training_schemes import scheme
import gym
import curses
import time

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
        self._training_level = 'one player'
        self.__init_static_field__()
        self.reset()

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
        for i, team in enumerate(self.teams):
            for player in range(int(self.n_agents/self.n_teams)):
                team.append(Agent(self, player, tuple(int(a*b) for a, b in zip(self.shape, scheme[self._training_level][i][player]))))

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

    def check_walls(self, move):
        new_pos = self.ball.position + move
        inter_pos = self.ball.position + array(move // 2, dtype=move.dtype)

        wall_beside_x = self.field[inter_pos[0], self.ball.position[1]] == 255
        wall_beside_y = self.field[self.ball.position[0], inter_pos[1]] == 255
        wall_far_x = self.field[new_pos[0], self.ball.position[1]] == 255
        wall_far_y = self.field[self.ball.position[0], new_pos[1]] == 255
        if wall_beside_x and wall_beside_y:
            move = array([-mv for mv in move])
        elif wall_beside_x and wall_far_y:
            move = array([-move[0], 0])
        elif wall_far_x and wall_beside_y:
            move = array([0, -move[1]])
        elif wall_far_x and wall_far_y:
            move = array([0, 0])
        elif wall_beside_x:
            move = array([-move[0], move[1]])
        elif wall_beside_y:
            move = array([move[0], -move[1]])
        elif wall_far_x:
            move = array([0, move[1]])
        elif wall_far_y:
            move = array([move[0], 0])

        if wall_beside_x or wall_far_x:
            self.ball.movement = [array(-mv[0], mv[1]) for mv in self.ball.movement]
        if wall_beside_y or wall_far_y:
            self.ball.movement = [array(mv[0], -mv[1]) for mv in self.ball.movement]

        new_pos = self.ball.position + move
        inter_pos = self.ball.position + array(move // 2, dtype=move.dtype)
        return move, inter_pos, new_pos

    def move_ball(self):
        move = self.ball.movement.pop()
        move, inter_pos, new_pos = self.check_walls(move)

        # check if ball has stopped
        if len(self.ball.movement) == 0:
            self.ball.moving = False

        # check if player is in the way
        if self._player_at(inter_pos) or self._player_at(new_pos):
            for i, team in enumerate(self.teams):
                for j, player in enumerate(team):
                    if player.position == new_pos or player.position == inter_pos:
                        player.has_ball = True
                        player.move_counter = 3
                        self.ball.moving = False
        else:
            # make ball move
            self.ball.position += move

    def step(self, *actions):
        winning_team = None
        done = False

        # move players
        for i, team in enumerate(self.teams):
            for j, player in enumerate(team):
                player_ndx = int(i * len(team) + j)
                player.act(actions[player_ndx])
                if self.field[player.position[0], player.position[1]] == 0:
                    player.has_ball = True
                    player.move_counter = 3

        # move ball
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
        else:
            rewards = []
            for team in self.teams:
                if team == winning_team:
                    for _ in range(int(self.n_agents / self.n_teams)):
                        rewards.append(100)
                else:
                    for _ in range(int(self.n_agents / self.n_teams)):
                        rewards.append(-100)
        self._add_to_field()
        return self.field.copy(), rewards, done, None

    def _add_to_field(self):
        self.field = self._static_field.copy()
        for i,team in enumerate(self.teams):
            for player in team:
                self.field[player.position[0], player.position[1]] = i+2  # teams look different on field
        if self.field[self.ball.position[0], self.ball.position[1]] == 1:
            self.field[self.ball.position[0], self.ball.position[1]] = 0

    def render(self, mode='human', field=None, scr=None):
        if field is None:
            field = self.field
            inplace = False
        else:
            inplace = True

        outfile = sys.stdout
        string = '\n'
        for y in range(field.shape[1]-1, -1, -1):
            for x in range(field.shape[0]):
                val = field[x, y]
                if val == 1:
                    string += '   '
                elif val == 0:
                    string += ' o '
                elif val == 2:
                    string += ' 1 '
                elif val == 3:
                    string += ' 2 '
                elif val == 255:
                    string += ' x '
                elif val == 254:
                    string += ' | '
                else:
                    raise ValueError()
            string += '\n'
        string += '\n'

        if inplace:
            scr.addstr(0, 0, string)
            scr.refresh()
            time.sleep(0.1)
        else:
            outfile.write(string)

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()

    def render_episode(self, episode):
        stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        for field in episode:
            self.render(field=field, scr=stdscr)
        curses.echo()
        curses.nocbreak()
        curses.endwin()