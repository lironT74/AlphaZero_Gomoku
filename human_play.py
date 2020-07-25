# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
"""

from __future__ import print_function
import pickle
from game import Board, Game, BoardSlim
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_numpy import PolicyValueNetNumpy
# from policy_value_net import PolicyValueNet  # Theano and Lasagne
from policy_value_net_pytorch import PolicyValueNet  # Pytorch
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet  # Keras
# from policy_net_keras import PolicyNet
from policy_player import PolicyPlayer
import numpy as np

class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            location = input("Your move: ")
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)


def run():
    n = 4
    width, height = 6, 6
    model_1_file = '/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p3/best_policy.model'
    model_2_file = '/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4/best_policy.model'

    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)


        # ############### human VS AI ###################
        # load the trained policy_value_net in either Theano/Lasagne, PyTorch or TensorFlow

        best_policy_1 = PolicyValueNet(width, height, model_file = model_1_file, input_plains_num=3)
        mcts_player_1 = MCTSPlayer(best_policy_1.policy_value_fn, c_puct=5, n_playout=400, name="3 plaines model")

        best_policy_2 = PolicyValueNet(width, height, model_file=model_2_file, input_plains_num=4)
        mcts_player_2 = MCTSPlayer(best_policy_2.policy_value_fn, c_puct=5, n_playout=400, name="4 plaines model")

        # load the provided model (trained in Theano/Lasagne) into a MCTS player written in pure numpy
        # try:
        #     policy_param = pickle.load(open(model_file, 'rb'))
        # except:
        #     policy_param = pickle.load(open(model_file, 'rb'),
        #                                encoding='bytes')  # To support python3
        # best_policy = PolicyNet(width, height, model_file=model_file)

        # policy_player = PolicyPlayer(best_policy, False)

        # mcts_player = MCTSPlayer(best_policy.policy_value_fn,
        #                          c_puct=5,
        #                          n_playout=400)  # set larger n_playout for better performance

        # uncomment the following line to play with pure MCTS (it's much weaker even with a larger n_playout)
        # mcts_player = MCTS_Pure(c_puct=5, n_playout=1000)

        # human player, input your move in the format: 2,3
        # human = Human()

        # set start_player=1 for human first
        # game.start_play(policy_player, human, start_player=1, is_shown=1, start_board=i_board)

        game.start_play(mcts_player_1, mcts_player_2, start_player=0, is_shown=1)

    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
