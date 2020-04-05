import math
import sys
import torch

import numpy as np
from collections import deque
from collections import namedtuple

SampleExperience = namedtuple ('SampleExperience',['obs','reward'])#,'done'])#,'pixel_change'])action TODO

class ReplayMemory():
    def __init__(self,init_obs,args,train_device):
        self.obs = init_obs.unsqueeze(1) # numales,84,84
        self.last_obs = init_obs

        #self.action = None
        self.reward = None
        #self.done = None
        self.num_ales=args.num_ales
        self.pixel_change =None
        self.train_device= train_device

    def append(self,exp):
        self.obs=torch.cat([self.obs,exp.obs],dim=1)

        if self.reward is None:
            #self.action =exp.action
            self.reward =exp.reward
            #self.done =exp.done
            #self.pixel_change = pixel_change
        else:
            #self.action=torch.cat([self.action,exp.action],dim=1)
            self.reward=torch.cat([self.reward,exp.reward],dim=1)
            #self.done=torch.cat([self.done,exp.done],dim=1)
            #self.pixel_change = torch.cat([self.pixel_change,pixel_change],dim=1)
            #print("returne ",pixel_change.shape)


    def _calc_pc(self):
        diff = torch.abs(self.obs[:,-2,2:-2,2:-2]-self.obs[:,-1,2:-2,2:-2]);
        print(diff.type())
        diff = diff.reshape(-1,20,4,20,4).mean(-1).mean(2)
        return diff.unsqueeze(1)

    def env_step_rp(self,zero_reward):
        step =4
        env= None
        if (zero_reward == 1) :
            while ( step <5 ):
                env = np.random.randint(self.num_ales)
                if self.reward[env] is None:
                    continue
                zeros=np.argwhere(self.reward[env].cpu().numpy() ==0).squeeze()
                if zeros.size <2 :
                    zeros=np.argwhere(self.reward[env].cpu().numpy() !=0).squeeze()
                step =np.random.choice(zeros,1)
        else:
            while ( step <5 ):
                env = np.random.randint(self.num_ales)
                if self.reward[env] is None:
                    continue
                zeros=np.argwhere(self.reward[env].cpu().numpy()!=0).squeeze()
                if zeros.size <2 :
                    zeros=np.argwhere(self.reward[env].cpu().numpy() ==0).squeeze()

                step =np.random.choice(zeros,1)
        return env,step

    def rp (self):
        zero_reward =np.random.randint(2)
        obs =[]
        env,step= self.env_step_rp(zero_reward)
        for i in range(-4,1):
            sample=SampleExperience(obs=self.obs[env][step+i],reward=self.reward[env][step+i])
            obs.append(sample)

        return obs

    def pc (self):
        env= np.random.randint(500)
        end_step = np.random.randint(20,self.reward.shape[1])
        obs =[]
        if env is None:
            raise Exception('An error occurred')
        for i in range(end_step,(end_step-21),-1):
            sample=SampleExperience(obs=self.obs[env][i],reward=self.reward[env][i])#,,done=self.done[env][i]pixel_change=self.pixel_change[env][i])
            obs.append(sample)
        return obs


    def clearMemory(self):

        self.obs = self.obs[:,-50:]

        self.reward = self.reward[:,-49:] #self.reward[:,-1].unsqueeze(1)
    #    self.done = self.done[:,-49:]#self.done[:,-1].unsqueeze(1)
        # self._zero_reward_indices = []
        # self._non_zero_reward_indices= []
        # print(self.obs.shape)
        #print(self.action.shape)

    # def sample_vr(self,batch_size):
    #
    #     starting_index = torch.randint(0,(self.num_ales-batch_size),(1,))
    #     b_obs=self.obs[starting_index:starting_index+batch_size]
    #     b_action=self.action[starting_index:starting_index+batch_size]
    #     b_reward=self.reward[starting_index:starting_index+batch_size]
    #     b_done=self.done[starting_index:starting_index+batch_size]
    #     with torch.no_grad():
    #         for i in range(batch_size)
    #                 vR,_ = self.forward_network(b_obs[i],last_action_reward[i])
    #                 for
# class ReplayMemory():
#     def __init__(self,args):
#         self.ReplayBuffer=[]
#
#     def __append(self,):
#         for i in range(obs.size[0]):
#             exp.obs=obs[i]
#             exp.reward = reward[i]
#             exp.done = done[i]
#             exp.action = action[i]
#         ReplayBuffer.append(exp)
