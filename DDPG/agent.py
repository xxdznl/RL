import random
import numpy as np
import torch.nn as nn
import torch
import time
import matplotlib.pyplot as plt
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)
def scale_action(action, high, low):
    action = np.clip(action, -1, 1)
    weight = (high - low) / 2
    bias = (high + low) / 2
    action_ = action * weight + bias
    return action_
class ReplayBuffer:
    def __init__(self, capacity, n_state, n_action, batch_size):
        self.MEMORY_SIZE = capacity
        self.BATCH_SIZE = batch_size
        self.n_state = n_state
        self.n_action = n_action
        self.all_state = np.empty(shape=(self.MEMORY_SIZE, self.n_state), dtype=np.float32)
        self.all_action = np.empty(shape=(self.MEMORY_SIZE, self.n_action), dtype=np.float32)
        self.all_reward = np.empty(self.MEMORY_SIZE, dtype=np.float32)
        self.all_state_next = np.empty(shape=(self.MEMORY_SIZE, self.n_state), dtype=np.float32)
        self.all_done = np.random.randint(low=0, high=2, size=self.MEMORY_SIZE, dtype=np.uint8)
        self.max = 0
        self.count = 0
    def add(self, state, action, reward, next_state, done):
        self.all_state[self.count] = state
        self.all_action[self.count] = action
        self.all_reward[self.count] = reward
        self.all_state_next[self.count] = next_state
        self.all_done[self.count] = done
        self.max = max(self.max, self.count + 1)
        self.count = (self.count + 1) % self.MEMORY_SIZE

    def sample(self):
        if self.max >= self.BATCH_SIZE:
            indexes = random.sample(range(0, self.max), self.BATCH_SIZE)
        else:
            indexes = range(0, self.max)
        batch_state = []
        batch_action = []
        batch_reward = []
        batch_state_next = []
        batch_done = []

        for idx in indexes:
            batch_state.append(self.all_state[idx])
            batch_action.append(self.all_action[idx])
            batch_reward.append(self.all_reward[idx])
            batch_state_next.append(self.all_state_next[idx])
            batch_done.append(self.all_done[idx])
        batch_state_tensor = torch.as_tensor(np.asarray(batch_state), dtype=torch.float32)
        batch_action_tensor = torch.as_tensor(np.asarray(batch_action), dtype=torch.float32)
        batch_reward_tensor = torch.as_tensor(np.asarray(batch_reward), dtype=torch.float32).unsqueeze(-1)#reward和 done 要升维，不然多个cycle的奖励算在一个数组里了，升维后才把多个奖励分开
        batch_state_next_tensor = torch.as_tensor(np.asarray(batch_state_next), dtype=torch.float32)
        batch_done_tensor = torch.as_tensor(np.asarray(batch_done), dtype=torch.uint8).unsqueeze(-1)#
        return batch_state_tensor, batch_action_tensor, batch_reward_tensor, batch_state_next_tensor, batch_done_tensor

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, fc1_dim, fc2_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.ln1 = nn.LayerNorm(fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim)
        self.action = nn.Linear(fc2_dim, action_dim)

        self.apply(weight_init)
        self.to(device)

    def forward(self, state):
        x = torch.relu(self.ln1(self.fc1(state)))
        x = torch.relu(self.ln2(self.fc2(x)))
        action = torch.tanh(self.action(x))

        return action

    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, fc1_dim, fc2_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.ln1 = nn.LayerNorm(fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim)
        self.fc3 = nn.Linear(action_dim, fc2_dim)
        self.fc4 = nn.Linear(fc2_dim, 1)

        self.apply(weight_init)
        self.to(device)

    def forward(self, state, action):
        x_s = torch.relu(self.ln1(self.fc1(state)))
        x_s = self.ln2(self.fc2(x_s))
        x_a = self.fc3(action)
        x = torch.relu(x_s + x_a)
        q = self.fc4(x)
        return q

    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))
class Ounoise():
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def generate(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
class DDPG:
    def __init__(self,env,args):
        self.env = env
        self.args = args
        #self.args.actor_lr,self.args.critic_lr = 3e-4,3e-4
        #TODO args.params
        self.n_state = self.env.observation_space.shape[0] #状态空间维数
        self.n_action = self.env.action_space.shape[0] #动作空间维数
        self.action_max = self.env.action_space.high[0] #动作空间最大值   #输出的动作力矩限幅在[-1,1],再乘以动作最大值得到最终动作
        self.action_min = self.env.action_space.low[0]
        self.episode_num = self.args.episode_num#训练几个episode
        self.max_cycle = self.args.max_cycle #1个episode最大轮数
        # actor state_dim, action_dim, fc1_dim, fc2_dim
        self.actor_net = Actor(state_dim = self.n_state,action_dim = self.n_action,fc1_dim=512,fc2_dim=256)
        self.target_actor_net = Actor(state_dim = self.n_state,action_dim = self.n_action,fc1_dim=512,fc2_dim=256)
        self.target_actor_net.load_state_dict(self.actor_net.state_dict())
        #critic state_dim, action_dim, fc1_dim, fc2_dim
        self.critic_net = Critic(state_dim = self.n_state,action_dim = self.n_action,fc1_dim=512,fc2_dim=256)
        self.target_critic_net = Critic(state_dim = self.n_state,action_dim = self.n_action,fc1_dim=512,fc2_dim=256)
        self.target_critic_net.load_state_dict(self.critic_net.state_dict())
        #optimizer
        self.actor_optimizer = torch.optim.Adam(params=self.actor_net.parameters(),lr = self.args.actor_lr)
        self.critic_optimizer = torch.optim.Adam(params=self.critic_net.parameters(), lr=self.args.critic_lr
                                                 , weight_decay=0.001)
        #noise
        self.noise_generator = Ounoise(self.args.ounoise_mu)
        #replay_buffer
        self.replay_buffer = ReplayBuffer(capacity=2048,n_state=self.n_state,n_action=self.n_action,batch_size=self.args.batch_size)
        #reward_buffer
        self.reward_buffer = np.empty(shape=self.episode_num)
    def soft_update_target_net(self):
        if self.args.tau is None:
            self.args.tau = 0.05

        for actor_params, target_actor_params in zip(self.actor_net.parameters(),
                                                     self.target_actor_net.parameters()):
            target_actor_params.data.copy_(self.args.tau * actor_params + (1 - self.args.tau) * target_actor_params)

        for critic_params, target_critic_params in zip(self.critic_net.parameters(),
                                                       self.target_critic_net.parameters()):
            target_critic_params.data.copy_(self.args.tau * critic_params + (1 - self.args.tau) * target_critic_params)

    def choose_action(self,state):
        self.actor_net.eval()
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
        action = self.actor_net.forward(state_tensor).squeeze()
        #只在训练时选动作需要假加入噪声
        if self.args.is_train:
            #noise = torch.tensor(np.random.normal(loc=0.0, scale=self.action_noise),dtype=torch.float32).to(self.device)
            noise = torch.tensor(self.noise_generator.generate(),dtype=torch.float32).to(device)
            action = torch.clamp(action + noise, -1, 1) #加入噪声后可能会越界超出[-1,1范围]，需要进行控制
        else:
            #不知道为什么 action必须加 noise 不用noise_generator.generate()出错，乘以0也算没有noise
            noise = torch.tensor(self.noise_generator.generate() * 0,dtype=torch.float32).to(device)
            action = torch.clamp(action + noise, -1, 1)
        # choose_action选出的动作在[-1，1]之间，需要放缩，并防止动作越界
        action = action.detach().cpu().numpy()
        action = scale_action(action.copy(), self.action_max, self.action_min)
        self.actor_net.train()
        return action
    def train(self):
        start = time.time()
        state = self.env.reset()
        self.noise_generator.reset()
        reward_history = []
        avg_reward_history = []
        for episode_i in range(self.episode_num):
            episode_reward = 0
            for cycle_i in range(self.max_cycle):
                action = self.choose_action(state)
                state_next,reward,done,info = self.env.step(action)
                if done:
                    state = self.env.reset()
                    self.noise_generator.reset()
                    self.reward_buffer[episode_i] = episode_reward
                    reward_history.append(episode_reward)
                    avg_reward_history.append(np.mean(reward_history[-100:]))
                    break
                self.replay_buffer.add(state,action,reward,state_next,done)

                episode_reward += reward
                state = state_next

                batch_state, batch_action, batch_reward, batch_state_next, batch_done = self.replay_buffer.sample()
                batch_state = batch_state.to("cuda:0")
                batch_state_next = batch_state_next.to('cuda:0')
                batch_reward = batch_reward.to("cuda:0")
                batch_done = batch_done.to("cuda:0")
                batch_action = batch_action.to("cuda:0")
                q_temp = self.target_critic_net.forward(batch_state_next, self.target_actor_net.forward(batch_state_next))
                #print()
                #print(f"q_temp.shape{q_temp.shape}")
                q_temp2 = self.args.GAMMA * (1-batch_done) * q_temp
                #print(f"batch_done.shape{batch_done.shape}")
                #print(f"q_temp2.shape{q_temp2.shape}")
                #print(f"batch_reward.shape{batch_reward.shape}")
                Q_target = batch_reward + q_temp2

                #print(f"Q_target.shape{Q_target.shape}")

                Q = self.critic_net.forward(batch_state,batch_action)#
                #print(f"Q.shape{Q.shape}")
                # if cycle_i ==2 :
                #     sys.exit()
                critic_loss = nn.functional.mse_loss(Q,Q_target)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                #TODO actor_loss 暂时不理解，为什么直接对Q求均值,为什么加负号
                actor_loss = -torch.mean(self.critic_net.forward(batch_state,self.actor_net.forward(batch_state)))
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                #soft update 软更新
                self.soft_update_target_net()
            if (episode_i + 1) % 10 == 0:
                # Print the training progress
                self.target_critic_net.load_state_dict(self.critic_net.state_dict())
                # print("episode: {}".format(episode_i))
                print(f'episode:{episode_i + 1}')
                print(f'time:', time.time() - start)
                print("Avg. Reward: {}".format(avg_reward_history[episode_i]))
                print()
            if (episode_i + 1) % 100 == 0:
                print(f'episode:{episode_i + 1} time:', time.time() - start)
                print()
            if (episode_i + 1) % 200 == 0:
                PATH_ACTOR = 'DDPGOnPendulumActor.pth'
                torch.save(self.target_actor_net.state_dict(), PATH_ACTOR)
                PATH_CRITIC = 'DDPGOnPendulumCritic.pth'
                torch.save(self.target_critic_net.state_dict(), PATH_CRITIC)
        episodes = [i + 1 for i in range(self.episode_num)]
        self.plot_learning_curve(episodes, avg_reward_history, title='AvgReward',
                            ylabel='reward', figure_file="./reward_images/reward.png")
    def plot_learning_curve(self,episodes, records, title, ylabel, figure_file):
        plt.figure()
        plt.plot(episodes, records, color='r', linestyle='-')
        plt.title(title)
        plt.xlabel('episode')
        plt.ylabel(ylabel)

        plt.show()
        plt.savefig(figure_file)


