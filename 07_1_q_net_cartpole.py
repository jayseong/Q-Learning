import gym
import tensorflow as tf
import numpy as np

env = gym.make('CartPole-v0')

# Constants defining out neural network
learning_rate = 1e-1
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

X = tf.placeholder(tf.float32, [None, input_size], name='input_x')

# We need to define the parts of the network needed for learning a policy
Y = tf.placeholder(tf.float32, [None, output_size])

# First layer of weights
W1 = tf.get_variable("W1", shape=[input_size, output_size],
                     initializer=tf.contrib.layers.xavier_initializer())
# Hypothesis
Qpred = tf.matmul(X, W1)

# loss/cost function
loss = tf.reduce_sum(tf.square(Y - Qpred))

# Optimizer / Learning
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Values for q learning
num_episodes = 2000
dis = 0.9
rList = []

sess = tf.Session()

for i in range(num_episodes):

    sess.run(tf.global_variables_initializer())

    e = 1. / ((i // 10) + 1)
    rAll = 0
    step_count = 0
    state = env.reset()
    done = False

    # The Q-Network training
    while not done:
        step_count += 1
        x = np.reshape(s, [1, input_size])
        # Choose an action by greedily (with e chance of random action) from the Q-network
        Qs = sess.run(Qpred, feed_dict={X: x})
        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = np.argmax(Qs)

        # Get new state and reward from environment
        new_state, reward, done, _ = env.step(action)
        if done:
            Qs[0, action] = -100
        else:
            x1 = np.reshape(new_state, [1, input_size])
            # Obtain the Q values by feeding the new state through our network
            Qs1 = sess.run(Qpred, feed_dict={X: x1})
            Qs[0, action] = reward + dis * np.max(Qs1)

        # Train our network using target and predicted Q values on each episode
        sess.run(train, feed_dict={X: x, Y: Qs})
        state = new_state

    rList.append(step_count)
    print("Episode: {}\tsteps: {}".format(i, step_count))
    # if last 10's avg steps are 500, it's good enough
    if len(rList) > 10 and np.mean(rList[-10:]) > 500:
        break

# See our trained network in action
observation = env.reset()
reward_sum = 0
while True:
    env.render()

    x = np.reshape(observation, [1, input_size])
    Qs = sess.run(Qpred, feed_dict={X: x})
    action = np.argmax(Qs)

    observation, reward, done, _ = env.step(action)
    reward_sum += reward
    if done:
        print("Total score: {}".format(reward_sum))
        break

