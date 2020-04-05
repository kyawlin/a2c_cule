import csv
import json
import math
import time
import torch
import torch.cuda.nvtx as nvtx

import numpy as np
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from utils.initializers import args_initialize, env_initialize, log_initialize, model_initialize

from helper import callback, format_time, gen_data
#from a2c.model import ActorCritic
from test import test
from memory import ReplayMemory
from model_resnet import ActorCritic,BasicBlock
from unreal import process_rp,process_pc
from collections import namedtuple

Experience = namedtuple ('Experience',['obs','reward'])#,'done'])action


try:
    from apex import amp
except ImportError:
    raise ImportError('Please install apex from https://www.github.com/nvidia/apex to run this example.')

def worker(gpu, ngpus_per_node, args):

    env_device, train_device = args_initialize(gpu, ngpus_per_node, args)

    train_env, test_env, observation = env_initialize(args, env_device)

    train_csv_file, train_csv_writer, eval_csv_file, eval_csv_writer, summary_writer = log_initialize(args, train_device)

    model = ActorCritic(args.num_stack, train_env.action_space,BasicBlock, normalize=args.normalize, name=args.env_name)
    model, optimizer = model_initialize(args, model, train_device)

    if (args.num_ales % args.num_minibatches) != 0:
        raise ValueError('Number of ales({}) size is not even divisible by the minibatch size({})'.format(
            args.num_ales, args.num_minibatches))

    if args.num_steps_per_update == -1:
        args.num_steps_per_update = args.num_steps

    minibatch_size = int(args.num_ales / args.num_minibatches)
    print("minibatch_size",minibatch_size)
    step0 = args.num_steps - args.num_steps_per_update
    n_minibatch = -1

    # This is the number of frames GENERATED between two updates
    num_frames_per_iter = args.num_ales * args.num_steps_per_update
    total_steps = math.ceil(args.t_max / (args.world_size * num_frames_per_iter)) #number of total frame

    shape = (args.num_steps + 1, args.num_ales, args.num_stack, *train_env.observation_space.shape[-2:])
    states = torch.zeros(shape, device=train_device, dtype=torch.float32)
    states[step0, :, -1] = observation.to(device=train_device, dtype=torch.float32)

    shape = (args.num_steps + 1, args.num_ales)
    values = torch.zeros(shape, device=train_device, dtype=torch.float32)
    logits = torch.zeros((args.num_steps + 1, args.num_ales, train_env.action_space.n), device=train_device, dtype=torch.float32)
    returns = torch.zeros(shape, device=train_device, dtype=torch.float32)

    shape = (args.num_steps, args.num_ales)
    rewards = torch.zeros(shape, device=train_device, dtype=torch.float32)
    masks = torch.zeros(shape, device=train_device, dtype=torch.float32)
    actions = torch.zeros(shape, device=train_device, dtype=torch.long)
    actions_one_hot = torch.zeros((args.num_steps, args.num_ales,18),device=train_device, dtype=torch.long)
    actions_space = torch.zeros(18,device=train_device,dtype=torch.long)

    #for LSTM
    lstm_hidden_state = torch.zeros((args.num_steps+1,args.num_ales,256), device =train_device,dtype=torch.float32)


    mus = torch.ones(shape, device=train_device, dtype=torch.float32)
    # pis = torch.zeros(shape, device=train_device, dtype=torch.float32)
    rhos = torch.zeros((args.num_steps, minibatch_size), device=train_device, dtype=torch.float32)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros(args.num_ales, device=train_device, dtype=torch.float32)
    final_rewards = torch.zeros(args.num_ales, device=train_device, dtype=torch.float32)
    episode_lengths = torch.zeros(args.num_ales, device=train_device, dtype=torch.float32)
    final_lengths = torch.zeros(args.num_ales, device=train_device, dtype=torch.float32)

    #init replay memory
    #mem = ReplayMemory(observation.to(device=train_device, dtype=torch.float32),args,train_device)


    torch.cuda.synchronize()

    iterator = range(total_steps)
    if args.rank == 0:
        iterator = tqdm(iterator)
        total_time = 0
        evaluation_offset = 0
    opt_step = 0
    aux_task = False
    for update in iterator:

        T = args.world_size * update * num_frames_per_iter
        if (args.rank == 0) and (T >= evaluation_offset):
            print("===========evaluating=========")
            evaluation_offset += args.evaluation_interval
            torch.save(model.state_dict(), "./model_save")

            eval_lengths, eval_rewards = test(args, model, test_env)

            lmean, lmedian, lmin, lmax, lstd = gen_data(eval_lengths)
            rmean, rmedian, rmin, rmax, rstd = gen_data(eval_rewards)
            length_data = '(length) min/max/mean/median: {lmin:4.1f}/{lmax:4.1f}/{lmean:4.1f}/{lmedian:4.1f}'.format(lmin=lmin, lmax=lmax, lmean=lmean, lmedian=lmedian)
            reward_data = '(reward) min/max/mean/median: {rmin:4.1f}/{rmax:4.1f}/{rmean:4.1f}/{rmedian:4.1f}'.format(rmin=rmin, rmax=rmax, rmean=rmean, rmedian=rmedian)
            print('[training time: {}] {}'.format(format_time(total_time), ' --- '.join([length_data, reward_data])))


            if eval_csv_writer and eval_csv_file:
                eval_csv_writer.writerow([T, total_time, rmean, rmedian, rmin, rmax, rstd, lmean, lmedian, lmin, lmax, lstd])
                eval_csv_file.flush()

            if args.plot:
                summary_writer.add_scalar('eval/rewards_mean', rmean, T, walltime=total_time)
                summary_writer.add_scalar('eval/lengths_mean', lmean, T, walltime=total_time)

        start_time = time.time()

        with torch.no_grad():

            for step in range(args.num_steps_per_update):
                #nvtx.range_push('train:step')
                #value, logit = model(states[step0 + step])#,lstm_hidden_state[step0+step])
                value, logit, lstm_hidden_state[step0+step] = model (states[step0+ step],args,lstm_hidden_state[step0],actions_one_hot[step],rewards[step])
                # store values and logits
                values[step0 + step] = value.squeeze(-1)

                # convert actions to numpy and perform next step
                probs = torch.clamp(F.softmax(logit, dim=1), min = 0.00001, max = 0.99999)
                probs_action = probs.multinomial(1).to(env_device)
                actions_space[probs_action] = 1


                torch.cuda.current_stream().synchronize()
                observation, reward, done, info = train_env.step(probs_action)

                observation = observation.squeeze(-1).unsqueeze(1)

                # move back to training memory
                observation = observation.to(device=train_device)
                reward = reward.to(device=train_device, dtype=torch.float32)
                done = done.to(device=train_device, dtype=torch.bool)
                probs_action = probs_action.to(device=train_device, dtype=torch.long)

                not_done = 1.0 - done.float()

                lstm_hidden_state[step0+step] *= not_done[:,None]


                # update rewards and actions
                actions[step0 + step].copy_(probs_action.view(-1))
                actions_one_hot [step0+step].copy_(actions_space)
                masks[step0 + step].copy_(not_done)
                rewards[step0 + step].copy_(reward.sign())

                #mus[step0 + step] = F.softmax(logit, dim=1).gather(1, actions[step0 + step].view(-1).unsqueeze(-1)).view(-1)
                mus[step0 + step] = torch.clamp(F.softmax(logit, dim=1).gather(1, actions[step0 + step].view(-1).unsqueeze(-1)).view(-1), min = 0.00001, max=0.99999)

                # update next observations
                states[step0 + step + 1, :, :-1].copy_(states[step0 + step, :, 1:])
                states[step0 + step + 1] *= not_done.view(-1, *[1] * (observation.dim() - 1))
                states[step0 + step + 1, :, -1].copy_(observation.view(-1, *states.size()[-2:]))

                # update episodic reward counters
                episode_rewards += reward
                final_rewards[done] = episode_rewards[done]
                episode_rewards *= not_done

                episode_lengths += not_done
                final_lengths[done] = episode_lengths[done]
                episode_lengths *= not_done
                nvtx.range_pop()

                #APPENDING observation
                #mem.append(Experience(obs=observation.to(device=train_device, dtype=torch.float32),action=probs_action.view(-1).unsqueeze(1),reward=reward.unsqueeze(1)))#,done=done.unsqueeze(1)))

                # mem.append(Experience(obs=observation.to(device=train_device, dtype=torch.float32),reward=reward.unsqueeze(1)))


        # if (opt_step >100 and opt_step %50 ==0):
        #     mem.clearMemory(); #clear half of  memory every 50 steps

        n_minibatch = (n_minibatch + 1) % args.num_minibatches
        min_ale_index = int(n_minibatch * minibatch_size)
        max_ale_index = min_ale_index + minibatch_size

        #to cat with output from FC and last reward
        #actions_one_hot= torch.cat([torch.zeros(args.num_steps,minibatch_size,18).to(device=train_device,dtype=torch.long),actions_one_hot[:,min_ale_index:max_ale_index,:]])

        nvtx.range_push('train:compute_values')
        # not sure about the LSTM input ouput
        value, logit ,lstm_hidden_state[:,min_ale_index:max_ale_index] = model(states[:, min_ale_index:max_ale_index, :, :, :].contiguous().view(-1, *states.size()[-3:]),\
                                args,lstm_hidden_state[:,min_ale_index:max_ale_index].contiguous(),
                                actions_one_hot[:,min_ale_index:max_ale_index,:].contiguous().view(-1,18),\
                                rewards[:,min_ale_index:max_ale_index].contiguous().view(-1,1))
        batch_value = value.detach().view((args.num_steps + 1, minibatch_size))
        batch_probs = F.softmax(logit.detach()[:(args.num_steps * minibatch_size), :], dim=1)
        batch_pis = batch_probs.gather(1, actions[:, min_ale_index:max_ale_index].contiguous().view(-1).unsqueeze(-1)).view((args.num_steps, minibatch_size))
        returns[-1, min_ale_index:max_ale_index] = batch_value[-1]

        with torch.no_grad():
            for step in reversed(range(args.num_steps)):
                c = torch.clamp(batch_pis[step, :] / mus[step, min_ale_index:max_ale_index], max=args.c_hat)
                rhos[step, :] = torch.clamp(batch_pis[step, :] / mus[step, min_ale_index:max_ale_index], max=args.rho_hat)
                delta_value = rhos[step, :] * (rewards[step, min_ale_index:max_ale_index] + (args.gamma * batch_value[step + 1] - batch_value[step]).squeeze())
                returns[step, min_ale_index:max_ale_index] = \
                        batch_value[step, :].squeeze() + delta_value + args.gamma * c * \
                        (returns[step + 1, min_ale_index:max_ale_index] - batch_value[step + 1, :].squeeze())

        value = value[:args.num_steps * minibatch_size, :]
        logit = logit[:args.num_steps * minibatch_size, :]

        log_probs = F.log_softmax(logit, dim=1)
        probs = F.softmax(logit, dim=1)

        action_log_probs = log_probs.gather(1, actions[:, min_ale_index:max_ale_index].contiguous().view(-1).unsqueeze(-1))
        dist_entropy = -(log_probs * probs).sum(-1).mean()

        advantages = returns[:-1, min_ale_index:max_ale_index].contiguous().view(-1).unsqueeze(-1) - value

        value_loss = advantages.pow(2).mean()
        policy_loss = -(action_log_probs * rhos.view(-1, 1).detach() * \
                (rewards[:, min_ale_index:max_ale_index].contiguous().view(-1, 1) + args.gamma * \
                returns[1:, min_ale_index:max_ale_index].contiguous().view(-1, 1) - value).detach()).mean()
        nvtx.range_pop()

        nvtx.range_push('train:backprop')

        # auxliary task from UNREAL     https://arxiv.org/pdf/1611.05397
        #REWARD PREDICTION
        # if (opt_step>100 and opt_step%20 == 0):
        #     aux_task = True
        #     obs = []
        #     for i in range(20):
        #         obs.append(mem.rp())
        #     states_,batch_rp_c= process_rp(obs)
        #     rp_c = model(states_,args,aux_task='rp')+1e-7
        #     print(rp_c)
        #
        #     rp_loss = -torch.sum(batch_rp_c.to(device=train_device) * torch.log(rp_c))/20/3
        #     print("---------------rp_loss---",rp_loss)
        # #     #####################################################
        ###   pixel change loss
        #     # obs_=[]
        #     # #for i in range(32):TODO BATCH LATER
        #     # obs_.append(mem.pc())
        #     #
        #     # states_pc,batch_pc_a,batch_pc_R = process_pc(obs_,model,train_device)
        #     # print(len(states_pc))
        #     # print(states_pc[0].shape)
        #     # print(batch_pc_a[0].shape)
        #     # print(batch_pc_R[0].shape)
        #     # print(torch.cat(states_pc).shape)
        #     # print(stop)
        #  #torch.Size([5, 4, 84, 84])
        #         #torch.Size([18])
        #         #torch.Size([5, 20, 20])
        #     #print(torch.cat(states_pc).shape)
        #     #print(stop)
        #     # states_pc = torch.cat(states_pc).view(-1,4,84,84)
        #     # pc_q, pc_q_max = model(states_pc,aux_task='pc')
        #     # print(pc_q_max.shape)
        #     # pc_a_reshape = batch_pc_a[0].view(-1,train_env.action_space.n,1,1)
        #     # pc_qa_ = torch.mul(pc_q,pc_a_reshape)
        #     # pc_qa = torch.sum (pc_qa_, dim=1,keepdim =False)
        #     #
        #     # print(pc_qa.shape)
        #     # print(batch_pc_R[0].shape)
        #     # print(pc_qa.shape)
        #     # pc_loss =  torch.sum( (( batch_pc_R[0]-pc_qa)**2/2.)                )
        #     # print(pc_loss)
        #     # print(stop)
        #

        loss = value_loss * args.value_loss_coef + policy_loss - dist_entropy * args.entropy_coef
        # if aux_task ==True:
        #     loss += rp_loss
        #     aux_task = False

        optimizer.zero_grad()


        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(retain_graph=True)
        master_params = amp.master_params(optimizer)

        torch.nn.utils.clip_grad_norm_(master_params, args.max_grad_norm)
        optimizer.step()
        opt_step +=1

        #nvtx.range_pop()

        #nvtx.range_push('train:next_states')
        for step in range(0, args.num_steps_per_update):
            states[:-1, :, :, :, :] = states[1:, :, :, : ,:]
            rewards[:-1, :] = rewards[1:, :]
            actions[:-1, :] = actions[1:, :]
            # actions_one_hot[:-1,:] = actions_one_hot[1:,:]
            # lstm_hidden_state[:-1,:] = lstm_hidden_state [1:,:]
            masks[:-1, :] = masks[1:, :]
            mus[:-1, :] = mus[1:, :]
        #nvtx.range_pop()

        torch.cuda.synchronize()

        if args.rank == 0:
            iter_time = time.time() - start_time
            total_time += iter_time

            if args.plot:
                summary_writer.add_scalar('train/rewards_mean', final_rewards.mean().item(), T, walltime=total_time)
                summary_writer.add_scalar('train/lengths_mean', final_lengths.mean().item(), T, walltime=total_time)
                summary_writer.add_scalar('train/value_loss', value_loss, T, walltime=total_time)
                summary_writer.add_scalar('train/policy_loss', policy_loss, T, walltime=total_time)
                summary_writer.add_scalar('train/entropy', dist_entropy, T, walltime=total_time)

            progress_data = callback(args, model, T, iter_time, final_rewards, final_lengths,
                                     value_loss, policy_loss, dist_entropy, train_csv_writer, train_csv_file)
            iterator.set_postfix_str(progress_data)

    if args.plot and (args.rank == 0):
        # name = '{}.pth'.format("BO_PC")
        # torch.save(model.module.state_dict(), "Pong_1gpu_1200")
        summary_writer.close()
