from env2048 import env2048


env = env2048(4, 4)
NoMove = False


def real_action(action) :


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