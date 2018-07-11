import random as R

class env2048:

    def __init__(self, width = 4, height = 4):
        # Contains information about state of numbers
        # 0 means blank
        self.table = []
        for i in range(0, height):
            table_tmp = []
            for j in range(0, width):
                table_tmp.append(0)
            self.table.append(table_tmp)
        # Prevent multiple merges at once
        self.table_tmp = []
        for i in range(0, height):
            table_tmp = []
            for j in range(0, width):
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
        for i in range(0, self.height):
            for j in range(0, self.width):
                if self.table[i][j] == 0:
                    rtn.append([i, j])
        return rtn

    # Initializes self.table_tmp
    def init_tmp(self):
        for i in range(0, self.height):
            for j in range(0, self.width):
                self.table_tmp[i][j] = False

    # Add a number to a randomly selected empty black.
    # The number is 2 or 4.
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
            for i in range(0, self.height):
                for j in reversed(range(0, self.width - 1)):
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
            for j in range(0, self.width):
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
            for i in range(0, self.height):
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
            for j in range(0, self.width):
                for i in reversed(range(0, self.height - 1)):
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
        return rtn

    # Yields the sum of all numbers
    def get_sum(self):
        rtn = 0
        for i in range(0, self.height):
            for j in range(0, self.width):
                rtn = self.table[i][j] + rtn
        return rtn

    # Prints the current state
    def print_state(self):
        print('Turn: ' + str(self.turn))
        print('Score: ' + str(self.score))
        for i in range(0, self.height):
            for j in range(0, self.width):
                print(self.table[i][j], end='\t')
            print('\n', end='')

def main():
    i = input('Height: ')
    _i = int(i) 
    j = input('Width: ')
    _j = int(j)
    env = env2048(_i, _j)
    NoMove = False
    direction_dict = {
        'D' : 0,
        'W' : 1,
        'A' : 2,
        'S' : 3,
        'E' : 4,
    }
    while NoMove or env.add_number():
        env.init_tmp()
        env.print_state()
        NoMove = False
        i = input('WASD: ').upper()
        if i in direction_dict:
            j = direction_dict[i]
            if j == 4:
                exit()
            NoMove = not env.swipe(j)
        else:
            print('Invalid input')


if __name__ == '__main__':
    main()