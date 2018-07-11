from env2048 import env2048

env = env2048(4, 4)
no_move = False
while no_move or env.add_number():
    env.print_state()
    no_move = False
    i = int(input())
    if i >= 0 and i < 4:
        env.swipe(i)
    else:
        exit()
