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

def run_expert(args, policy_fn):
    with tf.Session():
        tf_util.initialize()

        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
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
                #if args.render:
                #    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}

    return expert_data

class create_dataset():
    def __init__(self, data):
        obs = data['observations']
        acts = data['actions'].reshape([-1, 17])
        self.train_frac = 0.8
        self.size = size = obs.shape[0]
        self.train = [obs[:int(size*self.train_frac)], acts[:int(size*self.train_frac)]]
        self.test = [obs[int(size*self.train_frac):], acts[int(size*self.train_frac):]]
        self.curr_i = 0

    def randomize_dataset(self):
        index = range(int(self.size*self.train_frac))
        np.random.shuffle(index)
        self.train = [self.train[0][index], self.train[1][index]]

    def next_batch(self, n):
        if n > self.size:
            print('Requested batch size is greater than the training set size... \nreturning all training points')
            return self.train
        if (self.curr_i+n) >= int(self.size*self.train_frac):
            self.curr_i = 0 #Changing back the pointer
            self.randomize_dataset()         
        self.curr_i += n
        return [self.train[0][self.curr_i-n:self.curr_i], self.train[1][self.curr_i-n:self.curr_i]]

class learner_class():
    def __init__(self):
        self.obs = tf.placeholder(tf.float32, [None, 376])
        self.acts = tf.placeholder(tf.float32, [None, 17])

        Wf1 = tf.Variable(tf.truncated_normal([376, 188], stddev=0.1))
        bf1 = tf.constant(0.1, shape=[188])
        h_fc1 = tf.nn.relu(tf.matmul(self.obs, Wf1) + bf1)
        Wf2 = tf.Variable(tf.truncated_normal([188, 94], stddev=0.1))
        bf2 = tf.constant(0.1, shape=[94])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, Wf2) + bf2)
        Wf3 = tf.Variable(tf.truncated_normal([94, 47], stddev=0.1))
        bf3 = tf.constant(0.1, shape=[47])
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2, Wf3) + bf3)
        Wf4 = tf.Variable(tf.truncated_normal([47, 17], stddev=0.1))
        bf4 = tf.constant(0.1, shape=[17])
        self.h_fc4 = tf.matmul(h_fc3, Wf4) + bf4

        self.cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.acts, logits=self.h_fc4))

        self.error_prediction = tf.reduce_mean(abs(self.acts-self.h_fc4))

    def train(self, data):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
        error = tf.reduce_mean(tf.cast(self.error_prediction, tf.float32))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(20000):
              batch = data.next_batch(50)
              if i % 100 == 0:
                train_error = error.eval(feed_dict={
                    self.obs: batch[0], self.acts: batch[1]})#, keep_prob: 1.0})
                print('step %d, training error %g' % (i, train_error))
              #train_step.run(feed_dict={self.obs: batch[0], self.acts: batch[1]})#, keep_prob: 0.5})

            print('test error %g' % error.eval(feed_dict={
                self.obs: data.test[0], self.acts: data.test[1]}))#, keep_prob: 1.0}))
    
    def run_demo(self, n, args):
        with tf.Session():
            tf_util.initialize()

            env = gym.make(args.envname)
            max_steps = args.max_timesteps or env.spec.timestep_limit

            returns = []
            observations = []
            actions = []
            for i in range(args.num_rollouts):
                print('iter', i)
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    action = self.h_fc4.eval(feed_dict={self.obs:obs[None,:]})
                    observations.append(obs)
                    actions.append(action)
                    obs, r, done, _ = env.step(action)
                    totalr += r
                    steps += 1
                    if args.render:
                        env.render()
                    if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                    if steps >= max_steps:
                        break
                returns.append(totalr)

            print('returns', returns)
            print('mean return', np.mean(returns))
            print('std of return', np.std(returns))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    print('Creating dataset for training')
    data = create_dataset(run_expert(args, policy_fn))

    learner = learner_class()
    learner.train(data)
    learner.run_demo(5,args)

if __name__ == '__main__':
    main()
