import random as R
import numpy as np

class game2048:

    def __init__(self, height=4, width=4):
        # Contains information about state of numbers
        # 0 means blank
        self.table = []
        for _ in range(height):
            table_tmp = []
            for __ in range(width):
                table_tmp.append(0)
            self.table.append(table_tmp)
        # Prevent multiple merges at once
        self.table_tmp = []
        for _ in range(height):
            table_tmp = []
            for __ in range(width):
                table_tmp.append(False)
            self.table_tmp.append(table_tmp)
        # Current turn number
        self.turn = 0
        # Current score
        self.score = 0
        # Width of the table
        self.width = width
        # Height of the table
        self.height = height

    # Yields the list of coordinates of empty blocks
    # [i, j] means self.table[i][j]
    def get_spaces(self):
        rtn = []
        for i in range(self.height):
            for j in range(self.width):
                if self.table[i][j] == 0:
                    rtn.append([i, j])
        return rtn

    # Initializes self.table_tmp
    def init_tmp(self):
        for i in range(self.height):
            for j in range(self.width):
                self.table_tmp[i][j] = False

    # Add 2 or 4 to a randomly selected empty block
    def add_number(self):
        spaces = self.get_spaces()
        if len(spaces) == 0:
            return False
        max = len(spaces)
        space_to_add = spaces[R.randint(0, max - 1)]
        number_to_add = R.randint(0, 1) * 2 + 2
        self.table[space_to_add[0]][space_to_add[1]] = number_to_add
        return True

    # Swipe the numbers to particular direction
    # directions
    # 0 : Right
    # 1 : Top
    # 2 : Left
    # 3 : Bottom
    def swipe(self, direction):
        rtn = False
        if direction == 0:
            for i in range(self.height):
                for j in reversed(range(self.width - 1)):
                    if self.table[i][j] != 0:
                        for k in range(j, self.width - 1):
                            if self.table[i][k + 1] == 0:
                                self.table_tmp[i][k + 1] = self.table_tmp[i][k]
                                self.table_tmp[i][j] = False
                                self.table[i][k + 1] = self.table[i][k]
                                self.table[i][k] = 0
                                rtn = True
                            elif self.table[i][k + 1] == self.table[i][k]\
                            and not self.table_tmp[i][k + 1]\
                            and not self.table_tmp[i][k]:
                                self.table[i][k + 1] = self.table[i][k + 1] * 2
                                self.table_tmp[i][k + 1] = True
                                self.score = self.table[i][k] + self.score
                                self.table[i][k] = 0
                                self.table_tmp[i][k] = False
                                rtn = True
        elif direction == 1:
            for j in range(self.width):
                for i in range(1, self.height):
                    if self.table[i][j] != 0:
                        for k in reversed(range(1, i+1)):
                            if self.table[k - 1][j] == 0:
                                self.table_tmp[k - 1][j] = self.table_tmp[i][k]
                                self.table_tmp[k][j] = False
                                self.table[k - 1][j] = self.table[k][j]
                                self.table[k][j] = 0
                                rtn = True
                            elif self.table[k - 1][j] == self.table[k][j]\
                            and not self.table_tmp[k - 1][j]\
                            and not self.table_tmp[k][j]:
                                self.table[k - 1][j] = self.table[k - 1][j] * 2
                                self.table_tmp[k - 1][j] = True
                                self.score = self.table[k][j] + self.score
                                self.table[k][j] = 0
                                self.table_tmp[k][j] = False
                                rtn = True
        elif direction == 2:
            for i in range(self.height):
                for j in range(1, self.width):
                    if self.table[i][j] != 0:
                        for k in reversed(range(1, j+1)):
                            if self.table[i][k - 1] == 0:
                                self.table_tmp[i][k - 1] = self.table_tmp[i][k]
                                self.table_tmp[i][j] = False
                                self.table[i][k - 1] = self.table[i][k]
                                self.table[i][k] = 0
                                rtn = True
                            elif self.table[i][k - 1] == self.table[i][k]\
                            and not self.table_tmp[i][k - 1]\
                            and not self.table_tmp[i][k]:
                                self.table[i][k - 1] = self.table[i][k - 1] * 2
                                self.table_tmp[i][k - 1] = True
                                self.score = self.table[i][k] + self.score
                                self.table[i][k] = 0
                                self.table_tmp[i][k] = False
                                rtn = True
        elif direction == 3:
            for j in range(self.width):
                for i in reversed(range(self.height-1)):
                    if self.table[i][j] != 0:
                        for k in range(i, self.height - 1):
                            if self.table[k + 1][j] == 0:
                                self.table_tmp[k + 1][j] = self.table_tmp[i][k]
                                self.table_tmp[k][j] = False
                                self.table[k + 1][j] = self.table[k][j]
                                self.table[k][j] = 0
                                rtn = True
                            elif self.table[k + 1][j] == self.table[k][j]\
                            and not self.table_tmp[k + 1][j]\
                            and not self.table_tmp[k][j]:
                                self.table[k + 1][j] = self.table[k + 1][j] * 2
                                self.table_tmp[k + 1][j] = True
                                self.score = self.table[k][j] + self.score
                                self.table[k][j] = 0
                                self.table_tmp[k][j] = False
                                rtn = True
        self.init_tmp()
        if rtn:
            self.turn = self.turn + 1
        return rtn

    # Yields the sum of all numbers
    def get_sum(self):
        rtn = 0
        for i in range(self.height):
            for j in range(self.width):
                rtn = self.table[i][j] + rtn
        return rtn

    # Prints the current state
    def print_state(self):
        print('Turn: ' + str(self.turn))
        print('Score: ' + str(self.score))
        for i in range(self.height):
            for j in range(self.width):
                print(self.table[i][j], end='\t')
            print('\n', end='')


    def is_end(self):
        if len(self.get_spaces()) != 0:
            return False
        for i in range(self.height):
            for j in range(self.width - 1):
                if self.table[i][j] == self.table[i][j + 1]:
                    return False
        for j in range(self.width):
            for i in range(self.height - 1):
                if self.table[i][j] == self.table[i + 1][j]:
                    return False
        return True

    def greatest_number(self):
        rtn = 0
        for i in range(self.height):
            for j in range(self.width):
                if rtn < self.table[i][j]:
                    rtn = self.table[i][j]
        return rtn


# Encapsulates 2048 environement, and provides OpenAI-gym-like methods.
class env2048:
    def __init__(self, height=4, width=4):
        self.height = height
        self.width = width
        self.action_space = 4
        self.previous_score = 0
        self.observation_space = self.height * self.width
        self.game = game2048(self.height, self.width)
        self.game.add_number()

    def reset(self):
        self.game = game2048(self.height, self.width)
        self.previous_score = 0
        self.game.add_number()
        return np.reshape(self.game.table, self.height* self.width)
        
    """
    def step(self, action):
        valid = self.game.swipe(action)
        next_state = np.reshape(self.game.table, self.height * self.width)
        reward = int(not valid) * -100 + self.game.score - self.previous_score
        self.previous_score = self.game.score
        done = self.game.is_end()
        if not done:
            self.game.add_number()
        return (next_state, reward, done)
    """

    def step(self, action): # edited
        valid = self.game.swipe(action)
        next_state = np.reshape(self.game.table, self.height * self.width)
        reward = self.game.score - self.previous_score
        self.previous_score = self.game.score
        done = self.game.is_end()
        if not valid :
            done = True
        if not done:
            self.game.add_number()
        return (next_state, reward, done)

    def render(self):
        self.game.print_state()

# When this module is directly invoked, it executes a human-playable 2048 game.
def main():
    _i = int(input('Height: '))
    _j = int(input('Width: '))
    env = game2048(_i, _j)
    NoMove = False
    direction_dict = {
        'D' : 0,
        'W' : 1,
        'A' : 2,
        'S' : 3,
        'E' : 4,
    }
    i = ''
    while (not env.is_end()) and (NoMove or env.add_number()):
        env.print_state()
        NoMove = False
        if len(i) == 0:
            i = input('WASD: ').upper()
        
        _i = i[0]
        i = i[1:]
        
        if _i in direction_dict:
            j = direction_dict[_i]
            if j == 4:
                exit()
            NoMove = not env.swipe(j)
        else:
            print('Invalid input')

    exit()


if __name__ == '__main__':
    main()
