import env2048
import numpy as np
import tensorflow as tf
import a2c
from collections import deque

np.random.seed(3)
tf.set_random_seed(3)


OUTPUT_GRAPH = False
MAX_EPISODE = 1000000
MAX_EP_STEPS = 2000
RENDER = False
LR_A = 0.001
LR_C = 0.01




def deque_reduce_mean(deque) :
    maxlen = deque.maxlen - 1
    sum = 0
    for i in range(maxlen) :
        sum = sum + deque[i]
    mean = int(sum/deque.maxlen)
    #print('mean is ', mean)
    return mean


env = env2048.env2048(4,4)

N_F = env.observation_space
N_A = env.action_space

tf.reset_default_graph()

with tf.Session() as sess:
    actor = a2c.Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
    critic = a2c.Critic(sess, n_features=N_F, lr=LR_C)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        tf.summary.FileWriter("logs/", sess.graph)

    reward_sum = deque(maxlen=10)

    for i_episode in range(MAX_EPISODE):
        s = env.reset()
        t = 0
        track_r = []

        while True:
            a = actor.choose_action(s)

            s_, r, done = env.step(a)

            if done: r = -100

            track_r.append(r)

            td_error = critic.learn(s, r, s_)
            exp_v = actor.learn(s, a, td_error)

            s = s_
            t += 1

            if done or t >= MAX_EP_STEPS:
                ep_rs_sum = sum(track_r)
                reward_sum.append(ep_rs_sum)
                break

        if i_episode % 200 == 0:

            if i_episode > 10:
                print("episode:", i_episode, "  reward:", int(ep_rs_sum), " reward-mean : ",
                      deque_reduce_mean(reward_sum),
                      " steps : ", t, "     exp_v : ", exp_v)

            else:

                print("episode:", i_episode, "  reward:", int(ep_rs_sum))

        if i_episode > 10:  # deque size should be bigger than 10
            if deque_reduce_mean(reward_sum) > 5000:
                name = './save_model/a2c/2048_a2c-final.ckpt'
                saver.save(sess, name)
                print("Finished")
                break
            elif i_episode % 1000 == 0:
                if i_episode % 5000 == 0:
                    name = './save_model/a2c/%d/2048_a2c.ckpt' % i_episode
                    saver.save(sess, name)
                name = './save_model/a2c/2048_a2c.ckpt'
                saver.save(sess, name)
