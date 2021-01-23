# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet in PyTorch
Tested in PyTorch 0.2.0 and 0.3.0

@author: Junxiao Song
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import time

# from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
# WRITER_DIR = './runs/pt_6_6_4_p4_v4_training'
# writer = SummaryWriter(WRITER_DIR)


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Net(nn.Module):
    """policy-value network module"""
    def __init__(self, board_width=6, board_height=6, input_plains_num=4):
        super(Net, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        self.input_plains_num = input_plains_num

        # common layers

        self.conv1 = nn.Conv2d(self.input_plains_num, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4*board_width*board_height,
                                 board_width*board_height)
        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2*board_width*board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4*self.board_width*self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)

        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2*self.board_width*self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        return x_act, x_val


# from game import Board
# board = Board(width=6, height=6, n_in_row=4)
# board.init_board()
# net = Net()
# input = board.current_state(last_move=True).copy()
# input = np.ascontiguousarray(input.reshape(-1, 4 , 6, 6))
# torch_input = torch.tensor(input)
# writer.add_graph(net, torch_input)
# writer.close()



class PolicyValueNet():
    def __init__(self,
                 board_width,
                 board_height,
                 input_plains_num,
                 model_file=None,
                 use_gpu=True,
                 shutter_threshold_availables=None):

        self.shutter_threshold_availables = shutter_threshold_availables

        self.use_gpu = use_gpu


        if self.use_gpu:
            self.cuda = 'cuda:0'
            self.cuda_to_use = torch.device(self.cuda)


        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty
        self.input_plains_num = input_plains_num



        # # the policy value net module
        if self.use_gpu:
            self.policy_value_net = Net(self.board_width, self.board_height, self.input_plains_num).cuda(device=self.cuda_to_use)
        else:
            self.policy_value_net = Net(self.board_width, self.board_height, self.input_plains_num)

        self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=self.l2_const)


        if model_file:

            if self.use_gpu:
                self.policy_value_net.load_state_dict(torch.load(model_file, map_location=self.cuda_to_use))
            else:
                self.policy_value_net.load_state_dict(torch.load(model_file))

            # try:
            #
            #     net_params = torch.load(model_file)
            #     self.policy_value_net.load_state_dict(net_params)
            #
            #
            # except:
            #
            #     import pickle
            #     from collections import OrderedDict
            #     param_theano = pickle.load(open(model_file, 'rb'), encoding='bytes')
            #     keys = ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'conv3.weight', 'conv3.bias'
            #         , 'act_conv1.weight', 'act_conv1.bias', 'act_fc1.weight', 'act_fc1.bias'
            #         , 'val_conv1.weight', 'val_conv1.bias', 'val_fc1.weight', 'val_fc1.bias', 'val_fc2.weight',
            #             'val_fc2.bias']
            #
            #     param_pytorch = OrderedDict()
            #
            #     if self.use_gpu:
            #
            #         for key, value in zip(keys, param_theano):
            #             if 'fc' in key and 'weight' in key:
            #                 param_pytorch[key] = torch.FloatTensor(value.T).cuda(device=self.cuda_to_use)
            #             elif 'conv' in key and 'weight' in key:
            #                 param_pytorch[key] = torch.FloatTensor(value[:, :, ::-1, ::-1].copy()).cuda(device=self.cuda_to_use)
            #             else:
            #                 param_pytorch[key] = torch.FloatTensor(value).cuda(device=self.cuda_to_use)
            #
            #         self.policy_value_net.load_state_dict(param_pytorch)
            #
            #     else:
            #         for key, value in zip(keys, param_theano):
            #             if 'fc' in key and 'weight' in key:
            #                 param_pytorch[key] = torch.FloatTensor(value.T)
            #             elif 'conv' in key and 'weight' in key:
            #                 param_pytorch[key] = torch.FloatTensor(value[:, :, ::-1, ::-1].copy())
            #             else:
            #                 param_pytorch[key] = torch.FloatTensor(value)
            #
            #         self.policy_value_net.load_state_dict(param_pytorch)

    """policy-value network """


    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda(device=self.cuda_to_use))
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()

        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value.data.numpy()



    def policy_value_fn(self, board, **kwargs):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """

        cur_playout_player = kwargs.get("cur_playout_player", -1)


        board_copy = copy.deepcopy(board)


        # get state and choose random move and update
        current_state = np.ascontiguousarray(board.current_state(self.input_plains_num == 4).reshape(
            -1, self.input_plains_num, self.board_width, self.board_height))


        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).cuda(device=self.cuda_to_use).float())
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        else:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).float())
            act_probs = np.exp(log_act_probs.data.numpy().flatten())




        # if limit and random last move - should limit to real last turn

        if cur_playout_player == -1 or self.shutter_threshold_availables == None or self.shutter_threshold_availables > board_copy.width * board_copy.height:

            legal_positions = board_copy.availables

        else:

            # print(f"NOTICE: player {cur_playout_player} about to be limited to {self.shutter_threshold_availables} shutter size")

            assert (cur_playout_player == board_copy.players[0] or cur_playout_player == board_copy.players[1])

            legal_positions = board_copy.keep_only_close_enough_squares(self.shutter_threshold_availables, cur_playout_player)




        act_probs = zip(legal_positions, act_probs[legal_positions])



        value = value.data[0][0]
        return act_probs, value



    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        # wrap in Variable
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda(device=self.cuda_to_use))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda(device=self.cuda_to_use))
            winner_batch = Variable(torch.FloatTensor(winner_batch).cuda(device=self.cuda_to_use))
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            winner_batch = Variable(torch.FloatTensor(winner_batch))

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        log_act_probs, value = self.policy_value_net(state_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss

        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
                )

        # return loss.data[0], entropy.data[0]
        #for pytorch version >= 0.5 please use the following line instead.
        return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)


    def train_step_just_loss(self, state_batch, mcts_probs, winner_batch):
        """perform a training step"""
        # wrap in Variable
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda(device=self.cuda_to_use))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda(device=self.cuda_to_use))
            winner_batch = Variable(torch.FloatTensor(winner_batch).cuda(device=self.cuda_to_use))
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            winner_batch = Variable(torch.FloatTensor(winner_batch))

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # set learning rate
        # set_learning_rate(self.optimizer, lr)

        # forward
        log_act_probs, value = self.policy_value_net(state_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss

        # backward and optimize
        # loss.backward()
        # self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
                )

        # return loss.data[0], entropy.data[0]
        #for pytorch version >= 0.5 please use the following line instead.
        return loss.item(), entropy.item()