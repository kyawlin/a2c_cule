import numpy as np
import os
import re
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
_path = os.path.abspath(os.path.pardir)
if not _path in sys.path:
    sys.path = [_path] + sys.path
from utils.vec_normalize import RunningMeanStd

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1,):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.batchNorm = nn.BatchNorm2d(num_features=out_channels)

    def forward(self,x):
        residual = x
        #print(residual.shape)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.batchNorm(x)
        #print(x.shape)
        x = self.conv2(x)
        x = self.relu2(x)
        #print(x.shape)
        x +=residual
        return x

class ActorCritic(nn.Module):

    def __init__(self, num_inputs, action_space,block, normalize=False, name=None,hidden_size=256):
        super(ActorCritic, self).__init__()

        self._name = name

        self.conv1 = nn.Conv2d(in_channels=num_inputs, out_channels=16, kernel_size=3, stride=1,padding=1)
        self.maxpool1 = nn.MaxPool2d(3, stride=2)
        self.layer1 = self._make_layer(block,16,2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1,padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2)
        self.layer2 = self._make_layer(block,32,2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1,padding=1)
        self.maxpool3 = nn.MaxPool2d(3, stride=2)
        self.layer3 = self._make_layer(block,32,2)




        conv_out_size = self._get_conv_out((num_inputs, 84, 84))


        #self.linear2lstm= nn.Linear(in_features=conv_out_size,out_features=256)

        self.linear1 = nn.Linear(in_features=conv_out_size, out_features=256)
        self.rp_linear = nn.Linear(in_features = conv_out_size,out_features=3)
        self.softmax_d1 = nn.Softmax(dim=1)
        self.lstm = nn.GRU (256+18+1,256,1)

        #####pixelchangeNetwork
        self.pc_fc = nn.Linear(in_features=256,out_features=9*9*32)
        self.pc_deconv_v = nn.ConvTranspose2d(in_channels=32,out_channels=1,kernel_size=4,stride=2,padding=0)
        self.pc_deconv_a = nn.ConvTranspose2d(in_channels=32,out_channels=action_space.n,kernel_size=4,stride=2,padding=0)

        self.critic_linear = nn.Linear(in_features=256, out_features=1)

        self.actor_linear = nn.Linear(in_features=256, out_features=action_space.n)

        self.apply(weights_init)

        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.linear1.weight.data.mul_(relu_gain)
        self.ob_rms = RunningMeanStd(shape=(84, 84)) if normalize else None


    #def forward(self, x, lstm_hidden_state,last_action_reward,last_action=None,last_reward=None):
    def forward(self, x,args,lstm_hidden_state=None,last_action=None,last_reward=None,aux_task=None):
        with torch.no_grad():
            if self.ob_rms:
                if self.training:
                    self.ob_rms.update(x)
                mean = self.ob_rms.mean.to(dtype=torch.float32, device=x.device)
                std = torch.sqrt(self.ob_rms.var.to(dtype=torch.float32, device=x.device) + float(np.finfo(np.float32).eps))
                x = (x - mean) / std

        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.layer1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.layer2(x)

        x = self.conv3(x)
        x = self.maxpool3(x)
        x = F.relu(self.layer3(x))
        # print("before aux",x.shape)
        x = x.view(x.size(0), -1)
        if (aux_task=="rp"):
            x=self.rp_linear(x)
            return self.softmax_d1(x)

        x = F.relu(self.linear1(x))
        '''x in step has shape of [num_ales,256]
        x in train has shape of  [num_steps * minibatch_size,256]
        '''
        #
        #print("after fc is x is",x.shape)
        # if args.num_ales == x.size()[0]:
        #     T = 1
        #     B = args.num_ales
        #     #lstm_input= torch.cat([x,last_action.float(),last_reward.unsqueeze(1)],1)
        #     # if lstm_input.dim() ==2:
        #     #     lstm_input=lstm_input.unsqueeze(0)
        #     #x,lstm_hidden_state = self.lstm(lstm_input,lstm_hidden_state)
        #
        # else:
        #     T = args.num_steps + 1
        #     B = int(x.size()[0]/T)
        #     last_action=torch.cat([torch.zeros(1,B,18).to(dtype=torch.long, device=x.device),last_action.view(T-1,B,-1)])
        #     last_reward=torch.cat([torch.zeros(1,B,1).to(dtype=last_reward.dtype, device=x.device),last_reward.view(T-1,B,-1)])
        #     print(last_action.shape)
        #     print(last_reward.shape)
        # with torch.no_grad():
        #     lstm_input= torch.cat([x.view(T,B,-1) ,last_action.view(T,B,-1).float() ,last_reward.view(T,B,1)],2)

        # if (lstm_hidden_state.dim() == 2 ):
        #    lstm_hidden_state=lstm_hidden_state.unsqueeze(0)
        #    print(lstm_hidden_state.shape)
        # x,lstm_hidden_state = self.lstm(lstm_input,lstm_hidden_state[0].unsqueeze(0))
        #
        # x = x.view(-1, 256)

        if (aux_task=="pc"):
            x=self.pc_fc(x)
            x = x.view([-1,32,9,9])

            v=F.relu(self.pc_deconv_v(x),inplace=True)
            a = F.relu(self.pc_deconv_a(x),inplace=True)
            conv_a_mean = torch.mean(a,dim=1,keepdim=True)
            pc_q = v+a -conv_a_mean
            pc_q_max = torch.max(pc_q,dim=1,keepdim=False)[0]

            return pc_q,pc_q_max

        # if x.dim() == 3:
        #     x =x.squeeze(0)
        # if lstm_hidden_state.dim() == 3:
        #     lstm_hidden_state =lstm_hidden_state.squeeze(0)
        # print(x.shape)

        return  self.critic_linear(x),self.actor_linear(x) ,lstm_hidden_state#, rnn_hxs

    def name(self):
        return self._name

    def save(self):
        if self.name():
            name = '{}.pth'.format(self.name())
            torch.save(self.state_dict(), name)

    def load(self, name=None):
        self.load_state_dict(torch.load(name if name else self.name()))


    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(out_channels, out_channels, stride))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def _get_conv_out(self, shape):
        #o = self.conv1(torch.zeros(1, *shape))
        o = self.conv1(torch.zeros(1, *shape))
        o = self.maxpool1(o)
        #print(o.shape)
        o = self.layer1(o)

        o = self.conv2(o)
        o = self.maxpool2(o)
        o = self.layer2(o)

        o = self.conv3(o)
        o = self.maxpool3(o)
        o = self.layer3(o)

        return int(np.prod(o.size()))
