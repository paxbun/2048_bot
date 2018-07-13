import numpy as np
import tensorflow as tf
import env2048 as e
import dqn as d

env = e.env2048()
sess = tf.Session()
input_size = env.observation_space
output_size = env.action_space
dqn = d.dqn(sess, input_size, output_size)

def bot_play():
    s = env.reset()
    reward_sum = 0
    done = False
    while not done:
        env.render()
        a = np.argmax(dqn.predict(s))
        s, reward, done = env.step(a)
        reward_sum += reward
    print('Total score: {}'.format(reward_sum))

def main():
    
    bot_play()

if __name__ == '__main__':
    main()