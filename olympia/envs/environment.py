import sys
from contextlib import closing
import numpy as np
from numpy import zeros, array
from .grid_objects import Agent, Ball
from .training_schemes import scheme
import random
import gym
import curses
import time

"""
In the field state:
- BLUE (0,0,1) represents the ball
- BLACK (0,0,0) represents empty space
- RED (1,0,0) represents players on team 1
- GREEN (0,1,0) represents players on team 2
- GREY (.9,.9,.9) represents a goal
- WHITE (1,1,1) represents a wall
"""


class FieldEnv(gym.Env):
    def __init__(self, agent_type='RAM', shape=(21, 15), training_level='one player'):
        self.agent_type = agent_type
        self.shape = shape
        self._training_level = training_level
        self.n_teams = len(scheme[self._training_level])
        n_agents = 0
        for team in scheme[self._training_level]:
            n_agents += len(team)
        self.n_agents = n_agents
        self.n_agents_team = self.n_agents // self.n_teams
        if agent_type == 'RGB':
            self.state_size = (*self.shape, 3)
        else:
            self.state_size = (self.n_agents * 2 + 2,)
        self.ball = Ball(self, (int(shape[0]/2), int(shape[1]/2)))
        self.__init_static_field__()
        self.__init_agents__()
        self.reset()

    def __init_static_field__(self):
        self.field = zeros(shape=(*self.shape, 3), dtype=np.uint8)
        self.field[0, :, :] = 255  # left wall
        self.field[:, 0, :] = 255  # bottom wall
        self.field[self.shape[0] - 1, :] = 255  # right wall
        self.field[:, self.shape[1] - 1, :] = 255  # top wall
        bot = int(self.shape[1] / 4) + 1
        top = int(self.shape[1] / 4 + self.shape[1] / 2)
        self.field[0, bot:top, :] = 250  # left goal
        self.field[self.shape[0] - 1, bot:top, :] = 250  # right goal
        self._static_field = self.field.copy()  # never add objects to this field (for replacing values after agent/ball moves over position)

    def __init_agents__(self):
        self.teams = [[] for _ in range(self.n_teams)]
        if self.n_agents % self.n_teams is not 0:
            raise ValueError('Teams should be the same size.')
        for i, team in enumerate(self.teams):
            for player in range(self.n_agents_team):
                team.append(Agent(env=self, agent_type=self.agent_type, 
                                  team=i, number=player, 
                                  initial_position=tuple(int(a*b) for a, b in
                                  zip(self.shape, scheme[self._training_level][i][player]))))

    def reset(self):
        # reset environment to original state
        self.field = self._static_field.copy()

        # put ball in center of field
        self.ball.reset_position()

        # place players depending on their number
        for team in self.teams:
            for player in team:
                player.reset_position()

        self._add_to_field()
        return self.output()

    def output(self):
        raise NotImplementedError('output() not implemented in child class!')

    def get_agents(self):
        return [player for team in self.teams for player in team ]

    def _player_at(self, pos):
        return bool((self.field[pos[0], pos[1]] == array([255, 0, 0])).all() or
                    (self.field[pos[0], pos[1]] == array([0, 255, 0])).all())

    def check_walls(self, move):
        new_pos = self.ball.position + move
        inter_pos = self.ball.position + array(move // 2, dtype=move.dtype)

        wall_beside_x = (self.field[inter_pos[0], self.ball.position[1]] == array([255, 255, 255])).all()
        wall_beside_y = (self.field[self.ball.position[0], inter_pos[1]] == array([255, 255, 255])).all()
        wall_far_x = (self.field[new_pos[0], self.ball.position[1]] == array([255, 255, 255])).all()
        wall_far_y = (self.field[self.ball.position[0], new_pos[1]] == array([255, 255, 255])).all()
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
        else:  # make ball move
            # check ball hasn't left grid
            if new_pos[0] == -1 or new_pos[0] == self.shape[0]:
                self.ball.position = inter_pos.copy()
            else:
                self.ball.position = new_pos.copy()

    def step(self, *actions):
        winning_team = None
        done = False

        # move players
        for i, team in enumerate(self.teams):
            for j, player in enumerate(team):
                player_ndx = int(i * len(team) + j)
                player.act(actions[player_ndx])

        # move ball
        if self.ball.moving:
            self.move_ball()

        # check for ball in net
        if (self._static_field[self.ball.position[0], self.ball.position[1]] == array([250, 250, 250])).all():
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
        return self.output(), rewards, done, None

    def _add_to_field(self):
        self.field = self._static_field.copy()
        for i,team in enumerate(self.teams):
            for player in team:
                if (self.field[player.position[0], player.position[1], :] == 255).any():  # player occupying same pos.
                    other_player = None
                    for j,position in enumerate(self.get_player_positions()):
                        if (position == player.position).all():
                            other_player = self.teams[j // self.n_agents_team][j % self.n_agents_team]
                    moved_player = random.choice((player, other_player))
                    moved_player.position = moved_player.prev_position.copy()
                    self.field[moved_player.position[0], moved_player.position[1], moved_player.team] = 255
                    self.field[player.position[0], player.position[1], moved_player.team] = 0
                    if moved_player.has_ball:
                        moved_player.has_ball = False
                        moved_player.move_counter = -1
                self.field[player.position[0], player.position[1], i] = 255  # teams are red and green
        if (self.field[self.ball.position[0], self.ball.position[1]] == array([0, 0, 0])).all():
            self.field[self.ball.position[0], self.ball.position[1], 2] = 255  # ball is blue

    def get_player_positions(self):
        agents = []
        for team in self.teams:
            agents += team
        return [agent.position.copy() for agent in agents]

    def render(self, mode='human', field=None, scr=None, final=False):
        if field is None:
            field = self.field
            inplace = False
        else:
            inplace = True

        outfile = sys.stdout
        string = '\n' if not final else '(Done)\n'
        for y in range(field.shape[1]-1, -1, -1):
            for x in range(field.shape[0]):
                val = field[x, y]
                if (val == array([0,0,0])).all():
                    string += '   '
                elif (val == array([255,0,0])).all():
                    string += ' 1 '
                elif (val == array([0,255,0])).all():
                    string += ' 2 '
                elif (val == array([0,0,255])).all():
                    string += ' o '
                elif (val == array([255,255,255])).all():
                    string += ' x '
                elif (val == array([250,250,250])).all():
                    string += ' | '
                else:
                    raise ValueError()
            string += '\n'
        string += '\n'

        if inplace:
            scr.addstr(0, 0, string)
            scr.refresh()
            time.sleep(0.5)
        else:
            outfile.write(string)

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()

    def render_episode(self, episode):
        stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        for field, done in episode:
            self.render(field=field, scr=stdscr, final=done)
        curses.echo()
        curses.nocbreak()
        curses.endwin()


class OlympiaRGB(FieldEnv):
    def __init__(self, **kwargs):
        super(OlympiaRGB, self).__init__(agent_type='RGB', **kwargs)

    def output(self):
        return np.reshape(self.field.copy(), (1, *self.field.shape))


class OlympiaRAM(FieldEnv):
    def __init__(self, **kwargs):
        super(OlympiaRAM, self).__init__(agent_type='RAM', **kwargs)

    def output(self):
        return array([[*self.ball.position.copy()] + [coord for pos in self.get_player_positions() for coord in pos]])

