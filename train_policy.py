
from __future__ import print_function
import time
import os
import sys
import json
import glob
import pandas as pd

import random
import numpy as np
from collections import defaultdict, deque
from game import Game, Board
from mcts_pure import MCTSPlayer as MCTS_Pure
# from mcts_alphaZero import MCTSPlayer
from policy_player import PolicyPlayer

# from policy_value_net_pytorch import PolicyValueNet  # Pytorch
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
from policy_net_keras import PolicyNet # Keras


class TrainPipeline():
    def __init__(self,
                 init_model=None,
                 board_width=4,
                 board_height=4,
                 n_in_row=3,
                 learn_rate=2e-3,
                 lr_multiplier=1.0,
                 epochs=5,
                 check_freq=200,
                 game_batch_num=1000):
        # params of the board and the game
        self.board_width = board_width
        self.board_height = board_height
        self.n_in_row = n_in_row
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)
        # training params
        self.learn_rate = learn_rate
        self.lr_multiplier = lr_multiplier  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        # self.buffer_size = 10000
        # self.batch_size = 512  # mini-batch size for training
        # self.data_buffer = deque(maxlen=self.buffer_size)
        # self.play_batch_size = 1
        self.epochs = epochs  # num of train_steps for each update
        # self.kl_targ = 0.02
        self.check_freq = check_freq
        self.game_batch_num = game_batch_num
        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = [10, 30, 100, 300, 1000]
        if init_model:
            # start training from an initial policy-value net
            self.policy_net = PolicyNet(self.board_width,
                                                   self.board_height,
                                                   model_file=init_model)
            self.oponnent_net = PolicyNet(self.board_width,
                                                   self.board_height,
                                                   model_file=init_model)
        else:
            # start training from a new policy-value net
            self.policy_net = PolicyNet(self.board_width,
                                                   self.board_height)
            self.oponnent_net = PolicyNet(self.board_width,
                                          self.board_height)

        self.player = PolicyPlayer(self.policy_net,
                                   is_selfplay=1)
        self.oponnent_player = PolicyPlayer(self.oponnent_net, is_selfplay=1)
        self.models_prefix = "{timestamp}_game-{game_type}_lr-{lr}_length-{game_batch_num}".format(
            timestamp=int(time.time()),
            game_type="%d:%d:%d" % (board_height, board_width, n_in_row),
            lr=learn_rate,
            game_batch_num=game_batch_num)
        os.mkdir(self.models_prefix)

    # def get_equi_data(self, play_data):
    #     """augment the data set by rotation and flipping
    #     play_data: [(state, mcts_prob, winner_z), ..., ...]
    #     """
    #     extend_data = []
    #     for state, prob, move in play_data:
    #         for i in [1, 2, 3, 4]:
    #             # rotate counterclockwise
    #             equi_state = np.array([np.rot90(s, i) for s in state])
    #             equi_mcts_prob = np.rot90(np.flipud(
    #                 mcts_porb.reshape(self.board_height, self.board_width)), i)
    #             extend_data.append((equi_state,
    #                                 np.flipud(equi_mcts_prob).flatten(),
    #                                 winner))
    #             # flip horizontally
    #             equi_state = np.array([np.fliplr(s) for s in equi_state])
    #             equi_mcts_prob = np.fliplr(equi_mcts_prob)
    #             extend_data.append((equi_state,
    #                                 np.flipud(equi_mcts_prob).flatten(),
    #                                 winner))
    #     return extend_data

    def return_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        all_games_data = []
        for i in range(n_games):
            reward, play_data = self.game.start_play_training(self.player, self.oponnent_player,
                                                          temp=self.temp)

            play_data = list(play_data)[:]
            play_data = [list(d) + [reward] for d in play_data]
            # self.episode_len = len(play_data)
            # augment the data
            # play_data = self.get_equi_data(play_data)
            all_games_data.extend(play_data)
        return all_games_data

    def policy_update(self, n_games=1, verbose=False):
        """update the policy-value net"""
        # mini_batch = random.sample(self.data_buffer, self.batch_size)
        play_data = self.return_selfplay_data(n_games=n_games)
        state_batch = [data[0] for data in play_data]
        moves_batch = [data[1] for data in play_data]
        reward_batch = [data[2] for data in play_data]

        for i in range(self.epochs):
            loss, entropy = self.policy_net.train_step(
                    state_batch,
                    moves_batch,
                    reward_batch)


        if verbose:
            print((
                   "lr_multiplier:{:.3f},"
                   "loss:{},"
                   "entropy:{}"
                   ).format(
                            self.lr_multiplier,
                            loss,
                            entropy))
        return loss, entropy

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_player = self.player
        win_ratios = {}
        for playout_num in self.pure_mcts_playout_num:
            pure_mcts_player = MCTS_Pure(c_puct=5,
                                         n_playout=playout_num)
            win_cnt = defaultdict(int)
            for i in range(n_games):
                winner = self.game.start_play(current_player,
                                              pure_mcts_player,
                                              start_player=i % 2,
                                              is_shown=0)
                win_cnt[winner] += 1
            win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
            print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                    playout_num,
                    win_cnt[1], win_cnt[2], win_cnt[-1]))
            win_ratios[str(playout_num)] = win_ratio
        return win_ratios

    def run(self):
        """run the training pipeline"""
        stats = {"iter": []}
        with open(os.path.join(self.models_prefix, "config.json"), "w") as f:
            json.dump({
                 "board_width": self.board_width,
                 "board_height": self.board_height,
                 "n_in_row": self.n_in_row,
                 "learn_rate": self.learn_rate,
                 "lr_multiplier": self.lr_multiplier,
                 "epochs": self.epochs,
                 "check_freq": self.check_freq,
                 "game_batch_num": self.game_batch_num
            },
            f, indent=1)
        try:
            for i in range(self.game_batch_num):
                if i > self.check_freq:
                    self.load_random_opponent()

                loss, entropy = self.policy_update(verbose=(i+1) % self.check_freq == 0)

                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    win_ratios = self.policy_evaluate()
                    stats["iter"].append(i)
                    for k, v in win_ratios.items():
                        if k in stats:
                            stats[k].append(v)
                        else:
                            stats[k] = [v]
                    print(stats)
                    self.policy_net.save_model(os.path.join(self.models_prefix, 'policy%d.h5' % i))
        except KeyboardInterrupt:
            print('\n\rquit')

        stats = pd.DataFrame(stats)
        stats.to_csv(os.path.join(self.models_prefix, 'stats.csv'))

    def load_random_opponent(self):
        policy_paths = glob.glob(os.path.join(self.models_prefix,"*.h5"))
        policy_paths = sorted(policy_paths, key= lambda p: int(p.split('/')[-1][6:-3]))
        if np.random.uniform(0,1) > 0.5:
            self.oponnent_net.load_model(policy_paths[-1])
        else:
            self.oponnent_net.load_model(policy_paths[np.random.randint(0, len(policy_paths))])




if __name__ == '__main__':
    config = {}
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        if not os.path.exists(json_path):
            print("Warning! invalid configuration path. Initializing default train pipeline.")
        else:
            with open(json_path, "r") as f:
                config = json.load(f)
    training_pipeline = TrainPipeline(**config)
    training_pipeline.run()
