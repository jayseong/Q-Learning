import gym
from gym.envs.registration import register
import sys, tty, termios

# 
env = gym.make("FrozenLake-v0")

observation = env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample()  # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)

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

