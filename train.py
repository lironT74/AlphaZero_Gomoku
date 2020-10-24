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
from policy_value_net_pytorch import PolicyValueNet  # Pytorch
import os
from tensorboardX import SummaryWriter
import copy
from Game_boards_and_aux import *

MODEL_NAME="pt_6_6_4_p4_v34"
INPUT_PLANES_NUM = 4

WRITER_DIR = f'./runs/{MODEL_NAME}_training'
MODEL_DIR = f'/home/lirontyomkin/AlphaZero_Gomoku/models/{MODEL_NAME}'



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


        self.buffer_size = 10000
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02


        self.check_freq = 50
        self.game_batch_num = 5000


        self.improvement_counter = 1000
        self.best_win_ratio = 0.0


        self.input_plains_num = INPUT_PLANES_NUM

        self.c_puct = 5
        self.n_playout = 100  # num of simulations for each move
        self.shutter_threshold_availables = 0
        self.full_boards_selfplay = False

        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 200
        self.pure_mcts_playout_num_step = 200


        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   self.input_plains_num,
                                                   model_file=init_model,
                                                   shutter_threshold_availables=self.shutter_threshold_availables)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   self.input_plains_num,
                                                   shutter_threshold_availables=self.shutter_threshold_availables)




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

        self.episode_len = 0
        self.episode_len_full_1 = 0
        self.episode_len_full_2 = 0
        self.episode_len_full_3 = 0
        self.episode_len_full_4 = 0
        self.episode_len_full_5 = 0

        if self.full_boards_selfplay:

            """collect self-play data for training"""
            for i in range(n_games):

                #EMPTY BOARD:
                winner, play_data = self.game.start_self_play(self.mcts_player,
                                                              temp=self.temp,
                                                              is_last_move=(self.input_plains_num == 4),
                                                              start_player = i%2 +1)
                play_data = list(play_data)[:]
                self.episode_len += len(play_data)/n_games

                # augment the data
                play_data = self.get_equi_data(play_data)
                self.data_buffer.extend(play_data)


                if self.board_width == 6:
                    #BOARD 1 FULL
                    board = copy.deepcopy(BOARD_1_FULL[0])
                    board = np.flipud(board)
                    i_board_1 = np.zeros((2, self.board_width, self.board_height))
                    i_board_1[0] = board == 1
                    i_board_1[1] = board == 2

                    winner_full_1, play_data_full_1 = self.game.start_self_play(self.mcts_player,
                                                                  temp=self.temp,
                                                                  is_last_move=(self.input_plains_num == 4),
                                                                  initial_state=i_board_1)

                    play_data_full_1 = list(play_data_full_1)[:]
                    self.episode_len_full_1 += len(play_data_full_1)/n_games

                    # augment the data
                    play_data_full_1 = self.get_equi_data(play_data_full_1)
                    self.data_buffer.extend(play_data_full_1)

                    # BOARD 2 FULL
                    board = copy.deepcopy(BOARD_2_FULL[0])
                    board = np.flipud(board)
                    i_board_2 = np.zeros((2, self.board_width, self.board_height))
                    i_board_2[0] = board == 1
                    i_board_2[1] = board == 2

                    winner_full_2, play_data_full_2 = self.game.start_self_play(self.mcts_player,
                                                                         temp=self.temp,
                                                                         is_last_move=(self.input_plains_num == 4),
                                                                         initial_state=i_board_2)

                    play_data_full_2 = list(play_data_full_2)[:]
                    self.episode_len_full_2 += len(play_data_full_2)/n_games

                    # augment the data
                    play_data_full_2 = self.get_equi_data(play_data_full_2)
                    self.data_buffer.extend(play_data_full_2)

                else:
                    # BOARD 3 FULL
                    board = copy.deepcopy(BOARD_3_FULL[0])
                    board = np.flipud(board)
                    i_board_3 = np.zeros((2, self.board_width, self.board_height))
                    i_board_3[0] = board == 1
                    i_board_3[1] = board == 2

                    winner_full_3, play_data_full_3 = self.game.start_self_play(self.mcts_player,
                                                                                temp=self.temp,
                                                                                is_last_move=(
                                                                                        self.input_plains_num == 4),
                                                                                initial_state=i_board_3)

                    play_data_full_3 = list(play_data_full_3)[:]
                    self.episode_len_full_3 += len(play_data_full_3) / n_games

                    # augment the data
                    play_data_full_3 = self.get_equi_data(play_data_full_3)
                    self.data_buffer.extend(play_data_full_3)



                    # BOARD 4 FULL
                    board = copy.deepcopy(BOARD_4_FULL[0])
                    board = np.flipud(board)
                    i_board_4 = np.zeros((2, self.board_width, self.board_height))
                    i_board_4[0] = board == 1
                    i_board_4[1] = board == 2

                    winner_full_4, play_data_full_4 = self.game.start_self_play(self.mcts_player,
                                                                                temp=self.temp,
                                                                                is_last_move=(
                                                                                        self.input_plains_num == 4),
                                                                                initial_state=i_board_4)

                    play_data_full_4 = list(play_data_full_4)[:]
                    self.episode_len_full_4 += len(play_data_full_4) / n_games

                    # augment the data
                    play_data_full_4 = self.get_equi_data(play_data_full_4)
                    self.data_buffer.extend(play_data_full_4)


                    # BOARD 5 FULL
                    board = copy.deepcopy(BOARD_5_FULL[0])
                    board = np.flipud(board)
                    i_board_5 = np.zeros((2, self.board_width, self.board_height))
                    i_board_5[0] = board == 1
                    i_board_5[1] = board == 2

                    winner_full_5, play_data_full_5 = self.game.start_self_play(self.mcts_player,
                                                                                temp=self.temp,
                                                                                is_last_move=(
                                                                                        self.input_plains_num == 4),
                                                                                initial_state=i_board_5)

                    play_data_full_5 = list(play_data_full_5)[:]
                    self.episode_len_full_5 += len(play_data_full_5) / n_games

                    # augment the data
                    play_data_full_5 = self.get_equi_data(play_data_full_5)
                    self.data_buffer.extend(play_data_full_5)


        else:
            for i in range(n_games):
                # EMPTY BOARD:
                winner, play_data = self.game.start_self_play(self.mcts_player,
                                                              temp=self.temp,
                                                              is_last_move=(self.input_plains_num == 4),
                                                              start_player=i % 2 + 1)
                play_data = list(play_data)[:]
                self.episode_len += len(play_data) / n_games

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
        #                                                i+1)


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
                                          start_player=i % 2 + 1,
                                          is_shown=0,
                                          savefig=False)
            win_cnt[winner] += 1
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        print("num_playouts: {}, win: {}, lose: {}, tie:{}".format(
            self.pure_mcts_playout_num,
            win_cnt[1], win_cnt[2], win_cnt[-1]))

        self.writer.add_text(tag='evaluation results',
                             text_string=f"num_playouts: {self.pure_mcts_playout_num}, win: {win_cnt[1]}, lose: {win_cnt[2]}, tie:{win_cnt[-1]}",
                             global_step=iteration + 1)

        return win_ratio


    def run(self):
        """run the training pipeline"""

        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        try:
            improvement_counter_local = 0

            for i in range(self.game_batch_num):

                self.writer.add_scalar('MCTS playouts num', self.pure_mcts_playout_num, i + 1)

                self.collect_selfplay_data(self.play_batch_size)


                if self.full_boards_selfplay:

                    if self.board_width == 6:
                        print("batch i:{}, episode_len:{}, episode len full 1: {}, episode len full 2: {}".format(
                            i + 1, self.episode_len, self.episode_len_full_1, self.episode_len_full_2))
                        self.writer.add_scalar('episode len full 1', self.episode_len_full_1, i + 1)
                        self.writer.add_scalar('episode len full 2', self.episode_len_full_2, i + 1)
                        self.writer.add_scalar('episode len', self.episode_len, i + 1)

                    else:
                        print("batch i:{}, episode_len:{}, episode len full 3: {}, episode len full 4: {}, episode len full 4: {}".format(
                            i + 1, self.episode_len, self.episode_len_full_3, self.episode_len_full_4, self.episode_len_full_5))

                        self.writer.add_scalar('episode len full 3', self.episode_len_full_3, i + 1)
                        self.writer.add_scalar('episode len full 4', self.episode_len_full_4, i + 1)
                        self.writer.add_scalar('episode len full 5', self.episode_len_full_5, i + 1)

                        self.writer.add_scalar('episode len', self.episode_len, i + 1)


                else:
                    print("batch i:{}, episode_len:{}".format(i + 1, self.episode_len))
                    self.writer.add_scalar('episode len', self.episode_len, i + 1)


                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update(iteration=i)

                # check the performance of the current model,
                # and save the model params
                if (i + 1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i + 1))
                    win_ratio = self.policy_evaluate(iteration=i)

                    self.policy_value_net.save_model(f'{MODEL_DIR}/current_policy_{i + 1}.model')

                    if win_ratio > self.best_win_ratio:

                        self.writer.add_text('best model savings', 'better model found', i + 1)

                        print("New best policy!!!!!!!!")

                        improvement_counter_local = 0
                        self.best_win_ratio = win_ratio

                        # update the best_policy
                        # self.policy_value_net.save_model(f'{MODEL_DIR}/best_policy.model')

                        # if (self.best_win_ratio == 1.0 and
                        #         self.pure_mcts_playout_num < 5000):

                        if self.best_win_ratio == 1.0:
                            self.pure_mcts_playout_num += self.pure_mcts_playout_num_step
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


if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.run()
