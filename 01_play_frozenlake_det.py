import gym
from gym.envs.registration import register
# http://stackoverflow.com/questions/510357/python-read-a-single-character-from-the-user
import readchar

print("Reading a char:")
print(readchar.readchar())
print("Reading a key:")
print(readchar.readkey())

# MACROS
LEFT = 0
DOWN = 1
DOWN = 1
RIGHT = 2
UP = 3

# Key Mapping
arrow_keys = {
    '\x1b[A': UP,
    '\x1b[B': DOWN,
    '\x1b[C': RIGHT,
    '\x1b[D': LEFT,
}

# Register FrozenLake with is_slippery False
register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False}
)
env = gym.make("FrozenLake-v3")
env.render()    # Show the initial board


while True:
    # Choose an action from keyboard
    key = readchar.readkey()
    if key not in arrow_keys.keys():
        print("Game aborted!")
        break

    # action = env.action_space.sample()  # your agent here (this takes random actions)
    action = arrow_keys[key]
    state, reward, done, info = env.step(action)
    env.render()    # show the board after action
    print("State: ", state, "Action: ", action, "Reward: ", reward, "Info: ", info)

    if done:
        print("Finished with reward", reward)
        break


