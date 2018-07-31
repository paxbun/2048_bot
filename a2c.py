import numpy as np
import tensorflow as tf

np.random.seed(3)
tf.set_random_seed(3)



GAMMA = 0.9


class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            w1 = tf.Variable(tf.random_normal(shape=[n_features, 10], mean=0., stddev=0.1), name='w1')
            l1 = tf.matmul(self.s, w1)
            l1 = tf.nn.relu(l1)

            w2 = tf.Variable(tf.random_normal(shape=[10, 10], mean=0., stddev=0.1), name='w2')
            l2 = tf.matmul(l1, w2)
            l2 = tf.nn.relu(l2)

            w3 = tf.Variable(tf.random_normal(shape=[10, 6], mean=0., stddev=0.1), name='w3')
            l3 = tf.matmul(l2, w3)
            l3 = tf.nn.relu(l3)

            w4 = tf.Variable(tf.random_normal(shape=[6, n_actions], mean=0., stddev=0.1), name='w4')
            self.l4 = tf.matmul(l3, w4)
            self.hypo = tf.nn.softmax(self.l4)

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.hypo[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.hypo, {self.s: s})
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())

class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            w1 = tf.Variable(tf.random_normal(shape=[n_features, 10], mean=0., stddev=0.1), name='w1')
            l1 = tf.matmul(self.s, w1)
            l1 = tf.nn.relu(l1)

            w2 = tf.Variable(tf.random_normal(shape=[10, 6], mean=0., stddev=0.1), name='w2')
            l2 = tf.matmul(l1, w2)
            l2 = tf.nn.relu(l2)

            w3 = tf.Variable(tf.random_normal(shape=[6, 3], mean=0., stddev=0.1), name='w3')
            l3 = tf.matmul(l2, w3)
            l3 = tf.nn.relu(l3)

            w4 = tf.Variable(tf.random_normal(shape=[3, 1], mean=0., stddev=0.1), name='w4')
            self.v = tf.matmul(l3, w4)

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error
