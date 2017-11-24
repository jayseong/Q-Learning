import gym
from gym.envs.registration import register
import sys, tty, termios

class _Getch:
    def __call__(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgeattr(df)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(3)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

inkey = _Getch()

# MACROS
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# Key Mapping
arrow_keys = {
    '\xlb[A': UP,
    '\xlb[B': DOWN,
    '\xlb[C': RIGHT,
    '\xlb[D': LEFT,
}

# Register FrozenLake with is_slippery False
register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_splppery': Flase}
)
env = gym.make("FrozenLake-v3")
env.render()    # Show the initial board


while True:
    # Choose an action from keyboard
    key = inkey()
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


