import math
import gym
import numpy as np
import agent as Agent
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

#env = gym.make("CartPole-v0")
# print(env.observation_space)
# print(env.observation_space.shape)
# print(env.action_space.low)
# print(env.action_space.high[0])
class Args:
    def __init__(self,episode_num=1000,max_cycle=500,capacity=8192,batch_size=64,ounoise_mu=np.zeros(1), tau=0.1, gamma=0.98
                 ,actor_fc1_dim=256,actor_fc2_dim=128,actor_lr = 1e-3
                 ,critic_fc1_dim=256,critic_fc2_dim=128,critic_lr= 1e-3,weight_decay=0.01,is_train=True):
        self.episode_num = episode_num
        self.max_cycle = max_cycle
        self.capacity = capacity
        self.batch_size = batch_size
        self.ounoise_mu = ounoise_mu
        self.tau = tau
        self.GAMMA = gamma
        self.actor_fc1_dim = actor_fc1_dim
        self.actor_fc2_dim = actor_fc2_dim
        self.actor_lr = actor_lr
        self.critic_fc1_dim = critic_fc1_dim
        self.critic_fc2_dim = critic_fc2_dim
        self.critic_lr = critic_lr
        self.weight_decay = weight_decay
        self.is_train =is_train

def main():
    env = gym.make("Pendulum-v1")
    args = Args()
    #args.is_train=True
    args.is_train=False
    agent = Agent.DDPG(env, args)
    PATH = 'DDPGOnPendulumActor-1000.pth'
    agent.actor_net.load_state_dict(torch.load(PATH))
    agent.target_actor_net.load_state_dict(agent.actor_net.state_dict())
    PATH = 'DDPGOnPendulumCritic-1000.pth'
    agent.critic_net.load_state_dict(torch.load(PATH))
    agent.target_critic_net.load_state_dict(agent.critic_net.state_dict())
    #agent.train()
    env = env.unwrapped
    state = env.reset()
    count = 0
    while True:
        env.render()
        a = agent.choose_action(state)
        print(count, a)
        state, r, done, info = env.step(a)
        if done:
            count = 0
            env.reset()
            env.close()
        count += 1

if __name__ =='__main__':
    main()
