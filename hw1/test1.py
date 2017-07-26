#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy


print('loading and building expert policy')
policy_fn = load_policy.load_policy('experts/Humanoid-v1.pkl')
print('loaded and built')

with tf.Session():
    tf_util.initialize()
    import gym
    env = gym.make('Humanoid-v1')
    max_steps = env.spec.timestep_limit
    returns = []
    observations = []
    actions = []
    for i in range(1):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = policy_fn(obs[None,:])
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)
    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))
    expert_data = {'observations': np.array(observations),
                   'actions': np.array(actions)}


class create_dataset():
    def __init__(self, data):
        obs = data['observations']
        acts = data['actions'].reshape([-1, 17])
        self.size = size = obs.shape[0]
        self.train = [obs[:int(size*0.8)], acts[:int(size*0.8)]]
        self.test = [obs[int(size*0.8):], acts[int(size*0.8):]]
        self.curr_i = 0
    def randomize_dataset(self):
        index = range(int(self.size*0.8))
        np.random.shuffle(index)
        self.train = [self.train[0][index], self.train[1][index]]
    def next_batch(self, n):
        if n > self.size:
            print('Requested batch size is greater than the training set size... \nreturning all training points')
            return self.train
        if (self.curr_i+n) >= int(self.size*0.8):
            self.curr_i = 0 #Changing back the pointer
            self.randomize_dataset()
        self.curr_i += n
        return [self.train[0][self.curr_i-n:self.curr_i], self.train[1][self.curr_i-n:self.curr_i]]

data = create_dataset(expert_data)

'''
for i in range(30):
    print(i, 'curr =', data.curr_i, '. shape= ', data.next_batch(50)[0].shape, ', next= ', data.curr_i)
'''
obs = tf.placeholder(tf.float32, [None, 376])
acts = tf.placeholder(tf.float32, [None, 17])

Wf1 = tf.Variable(tf.truncated_normal([376, 188], stddev=0.1))
bf1 = tf.constant(0.1, shape=[188])
h_fc1 = tf.nn.relu(tf.matmul(obs, Wf1) + bf1)
Wf2 = tf.Variable(tf.truncated_normal([188, 94], stddev=0.1))
bf2 = tf.constant(0.1, shape=[94])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, Wf2) + bf2)
Wf3 = tf.Variable(tf.truncated_normal([94, 47], stddev=0.1))
bf3 = tf.constant(0.1, shape=[47])
h_fc3 = tf.nn.relu(tf.matmul(h_fc2, Wf3) + bf3)
Wf4 = tf.Variable(tf.truncated_normal([47, 17], stddev=0.1))
bf4 = tf.constant(0.1, shape=[17])
h_fc4 = tf.nn.relu(tf.matmul(h_fc3, Wf4) + bf4)

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=acts, logits=h_fc4))

error_prediction = tf.reduce_mean(abs(acts-h_fc4))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
error = tf.reduce_mean(tf.cast(error_prediction, tf.float32))

sess = tf.Session()

sess.run(tf.global_variables_initializer())

'''
for i in range(16):
    batch = data.next_batch(50)
    #if i % 100 == 0:
    #    train_error = error.eval(feed_dict={obs: batch[0], acts: batch[1]})#, keep_prob: 1.0})
    #    print('step %d, training error %g' % (i, train_error))
    with sess.as_default():
        train_step.run(feed_dict={obs: batch[0], acts: batch[1]})#, keep_prob: 0.5})
        print('test error %g' % error.eval(feed_dict={obs: data.test[0], acts: data.test[1]}))#, keep_prob: 1.0}))

'''