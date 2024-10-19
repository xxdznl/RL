import gym
import torch
from DQN.agent import DQN
import signal
import sys

print(torch.__version__)
env = gym.make("CartPole-v0")
env = env.unwrapped
state = env.reset()
n_state = len(state)
n_action = env.action_space.n

PATH = 'DqnOnCartPole-v0.pth'
model = DQN(n_state,2)
model.load_state_dict(torch.load(PATH))
count = 0
print(gym.__version__)
print(super().reset.__code__.co_varnames)
def CtrlCHandler(signum, frame):
    env.close()
    sys.exit(0)
while True:
    signal.signal(signal.SIGINT, CtrlCHandler)
    env.render()
    a = model.action(state)
    state, r, done, info = env.step(a)
    if done:
        count = 0
        env.reset()
        print(count, a)
    count += 1
env.close()