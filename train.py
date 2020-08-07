# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
"""

from __future__ import print_function
import random
import numpy as np
from collections import defaultdict, deque
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
# from policy_value_net import PolicyValueNet  # Theano and Lasagne
from policy_value_net_pytorch import PolicyValueNet  # Pytorch
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet # Keras
import copy

import os
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter

import PIL.Image
from torchvision.transforms import ToTensor

WRITER_DIR = './runs/pt_6_6_4_p4_v4_training'
MODEL_DIR = '/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v4'

class TrainPipeline():
    def __init__(self, init_model=None):

        self.writer = SummaryWriter(WRITER_DIR)

        # params of the board and the game
        self.board_width = 6
        self.board_height = 6
        self.n_in_row = 4

        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)

        self.game = Game(self.board)

        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param

        self.n_playout = 400  # num of simulations for each move

        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02

        self.check_freq = 50
        self.game_batch_num = 2500

        self.improvement_counter = 100
        self.best_win_ratio = 0.0

        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000

        self.input_plains_num = 4

        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   self.input_plains_num,
                                                   model_file=init_model)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   self.input_plains_num)


        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)


    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data


    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):

            winner, play_data = self.game.start_self_play(self.mcts_player,
                                                          temp=self.temp,
                                                          is_last_move=(self.input_plains_num == 4))

            play_data = list(play_data)[:]
            self.episode_len = len(play_data)

            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)


    def policy_update(self, iteration):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)

        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learn_rate * self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                                axis=1)
                         )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
            # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))

        train_str = ("kl:{:.5f}, "
               "lr_multiplier:{:.3f}, "
               "loss:{}, "
               "entropy:{}, "
               "explained_var_old:{:.3f}, "
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new)

        print(train_str)

        self.writer.add_scalar('lr multiplier', self.lr_multiplier, iteration + 1)
        self.writer.add_scalar('kl_', kl, iteration + 1)
        self.writer.add_scalar('explained var old', explained_var_old, iteration + 1)
        self.writer.add_scalar('explained var new', explained_var_new, iteration + 1)
        self.writer.add_scalar('training loss', loss, iteration + 1)
        self.writer.add_scalar('training entropy', entropy, iteration + 1)

        # self.writer.add_scalars("training tracking", {'lr multiplier': self.lr_multiplier,
        #                                               'kl': kl,
        #                                               'explained var old': explained_var_old,
        #                                               'explained var new': explained_var_new,
        #                                               'training loss':loss,
        #                                               'training entropy':entropy},
        #                         i+1)


        return loss, entropy


    def policy_evaluate(self, iteration, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)

        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=self.pure_mcts_playout_num)

        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        print("num_playouts: {}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))

        self.writer.add_text(tag='evaluation results',
                             text_string=f"num_playouts: {self.pure_mcts_playout_num}, win: {win_cnt[1]}, lose: {win_cnt[2]}, tie:{win_cnt[-1]}",
                             global_step=iteration+1)

        return win_ratio

    def run(self):
        """run the training pipeline"""

        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        try:
            improvement_counter_local = 0

            for i in range(self.game_batch_num):

                self.collect_selfplay_data(self.play_batch_size)

                print("batch i:{}, episode_len:{}".format(
                        i+1, self.episode_len))

                self.writer.add_scalar('episode len', self.episode_len, i + 1)

                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update(iteration=i)


                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    win_ratio = self.policy_evaluate(iteration=i)

                    self.policy_value_net.save_model(f'{MODEL_DIR}/current_policy_{i+1}.model')

                    self.save_heatmap(iteration=i)

                    if win_ratio > self.best_win_ratio:

                        self.writer.add_text('best model savings', 'better model found', i + 1)

                        print("New best policy!!!!!!!!")

                        improvement_counter_local = 0
                        self.best_win_ratio = win_ratio

                        # update the best_policy
                        # self.policy_value_net.save_model(f'{MODEL_DIR}/best_policy.model')

                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0

                    else:
                        improvement_counter_local += 1
                        if improvement_counter_local == self.improvement_counter:
                            print(f"No better policy was found in the last {self.improvement_counter} "
                                  f"checks. Ending training. ")

                            self.writer.add_text('best model savings', f"No better policy was found "
                                                                       f"in the last {self.improvement_counter} "
                                                                       f"checks. Ending training. ", i + 1)

                            break


        except KeyboardInterrupt:
            print('\n\rquit')

    def save_heatmap(self, iteration):

        policy_copy = copy.deepcopy(self.policy_value_net)
        player = MCTSPlayer(policy_copy.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout)

        board = self.initialize_paper_board()
        _,heatmap_buf = player.get_action(board, return_prob=0, return_fig=True)

        image = PIL.Image.open(heatmap_buf)
        image = ToTensor()(image)

        self.writer.add_image(tag='Heatmap on paper board',
                              img_tensor=image,
                              global_step=iteration + 1)

    def initialize_paper_board(self):
        board_paper = np.array([
            [0, 1, 0, 2, 0, 0],
            [0, 2, 1, 1, 0, 0],
            [1, 2, 2, 2, 1, 0],
            [2, 0, 1, 1, 2, 0],
            [1, 0, 2, 2, 0, 0],
            [0, 0, 0, 0, 0, 0]])
        board_paper = np.flipud(board_paper)
        i_board = np.zeros((2, self.board_height, self.board_width))
        i_board[0] = board_paper == 1
        i_board[1] = board_paper == 2
        board = Board(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
        board.init_board(start_player=1, initial_state=i_board)
        return board


if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.run()
