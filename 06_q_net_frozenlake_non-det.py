# Stochastic (non-deterministic)

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v0")

# Input and output size based on the Env
input_size = env.observation_space.n
output_size = env.action_space.n
learning_rate = 0.1

# These lines establish the feed-forward part of the network used to choose actions
X = tf.placeholder(shape=[1, input_size], dtype=tf.float32)     # state input
Y = tf.placeholder(shape=[1, output_size], dtype=tf.float32)    # Y label

W = tf.Variable(tf.random_uniform([input_size, output_size], 0, 0.01))  # weight

# Hypothesis
Qpred = tf.matmul(X, W)     # Out Q prediction

# Loss/Cost
loss = tf.reduce_sum(tf.square(Y - Qpred))
# Optimizer
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# Set Q-learning related parameters
dis = .99
num_episodes = 2000

# create lists to contain total rewards and steps per episode
rList = []

# One-hot encoding function
def one_hot(x):
    return np.identity(16)[x:x + 1]

init  = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        # Reset environment and get first new observation
        state = env.reset()
        e = 1. / ((i // 50) + 10)
        rAll = 0
        done = False
        local_loss = []

        # The Q-network training
        while not done:
            # Choose an action by greedily (with e chance of random action) from the Q-network
            Qs = sess.run(Qpred, feed_dict={X: one_hot(state)})
            if np.random.rand(1) < e:
                action = env.action_space.sample()
            else:
                action = np.argmax(Qs)

            # Get new state and reward from environment
            new_state, reward, done, _ = env.step(action)
            if done:
                # Update Q, and no Qs+1, since it's a terminal state
                Qs[0, action] = reward
            else:
                # Obtain the Q_s1 values by feeding the new state throught out network
                Qs1 = sess.run(Qpred, feed_dict={X: one_hot(new_state)})
                # Update Q
                Qs[0, action] = reward + dis * np.max(Qs1)

            # Train our network using target (Y) and predicted Q (Qpred) values
            sess.run(train, feed_dict={X: one_hot(state), Y: Qs})

            rAll += reward
            state = new_state
        rList.append(rAll)

print("Success rate: " + str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Qs)
plt.bar(range(len(rList)), rList, color='blue')
plt.show()


