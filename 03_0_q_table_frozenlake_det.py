# Exploit VS Exploration: decaying E-greedy
# for i in range(1000)
#   e = 0.1 / (i + 1)
#   if random(1) < e:
#        a = random
#    else:
#        a = argmax(Q(s,a))
import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

def rargmax(vector):
    # Argmax that chooses randomly among eligible maximum indices.
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)


# Register FrozenLake with is_slippery False
register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False}
)
env = gym.make("FrozenLake-v3")
# env.render()    # Show the initial board

# Initialize table with all zeros
# env.observation_space.n = 16 (4 * 4), env.action_space.n = 4 (up, down, right, left)
Q = np.zeros([env.observation_space.n, env.action_space.n])
# Set learning parameters
num_episodes = 2000

# create lists to contain total rewards and steps per episode
rList = []
for i in range(num_episodes):
    # Reset environment and get first new observation
    state = env.reset()
    rAll = 0
    done = False

    # The Q-table learning algorithm
    while not done:
        action = rargmax(Q[state, :])

        # Get new state and reward from environment
        new_state, reward, done, _ = env.step(action)

        # 1) Update Q-Table with new knowledge using learning rate
        # Q[state, action] = reward + np.max(Q[new_state, :])
        # 2) Update Q-Table with new knowledge using decay rate
        Q[state, action] = reward + np.max(Q[new_state, :])

        rAll += reward
        state = new_state

    rList.append(rAll)

print("Success rate: " + str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)
plt.bar(range(len(rList)), rList, color='blue')
plt.show()


