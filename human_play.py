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
# from policy_value_net_keras import PolicyValueNet # Keras
import PIL.Image


class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board, **kwargs):
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

BOARD_1_FULL = (np.array([[0, 1, 0, 2, 0, 0],
                [0, 2, 1, 1, 0, 0],
                [1, 2, 2, 2, 1, 0],
                [2, 0, 1, 1, 2, 0],
                [1, 0, 2, 2, 0, 0],
                [0, 0, 0, 0, 0, 0]]), "board 1 full", None, None)

BOARD_1_TRUNCATED = (np.array([[0, 1, 2, 2, 0, 0],
                    [0, 2, 1, 1, 0, 0],
                    [1, 2, 2, 2, 1, 0],
                    [2, 0, 1, 1, 2, 1],
                    [1, 0, 2, 2, 0, 0],
                    [0, 0, 0, 0, 0, 0]]), "board 1 truncated", [2, 5], [5, 2])

BOARD_2_FULL =  (np.array([[0, 2, 0, 0, 1, 0],
                    [0, 2, 1, 2, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 1, 0, 2, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 2, 0, 0, 2, 0]]), "board 2 full", None, None)

BOARD_2_TRUNCATED = (np.array([[0, 2, 0, 1, 1, 0],
                    [0, 2, 1, 2, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [2, 1, 0, 2, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 2, 0, 0, 2, 0]]), "board 2 truncated", [5, 3], [2, 0])


def initialize_paper_board(board_width, board_height, n_in_row, input_board = BOARD_1_FULL[0]):
    board_paper = input_board
    board_paper = np.flipud(board_paper)
    i_board = np.zeros((2, board_height, board_width))
    i_board[0] = board_paper == 1
    i_board[1] = board_paper == 2

    board = Board(width=board_width, height=board_height, n_in_row=n_in_row)

    # board.init_board(start_player=1, initial_state=i_board)
    return i_board, board



def run():
    # n = 5
    # width, height = 8,8
    # model_1_file = './best_policy_8_8_5.model'

    n = 4
    width, height = 6,6

    model_1_file = '/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p3_v9/current_policy_3500.model'

    model_2_file = '/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v10/current_policy_5000.model'

    try:

        i_board, board = initialize_paper_board(width, height, n, input_board=np.zeros((width, height)))

        game = Game(board)

        best_policy_1 = PolicyValueNet(width, height, model_file=model_1_file, input_plains_num=3)
        mcts_player_1 = MCTSPlayer(best_policy_1.policy_value_fn, c_puct=5, n_playout=400, name="pt_6_6_4_p3_v9_3500")

        best_policy_2 = PolicyValueNet(width, height, model_file=model_2_file, input_plains_num=4)
        mcts_player_2 = MCTSPlayer(best_policy_2.policy_value_fn, c_puct=5, n_playout=400, name="pt_6_6_4_p4_v10_5000")

        # uncomment the following line to play with pure MCTS (it's much weaker even with a larger n_playout)
        # mcts_player = MCTS_Pure(c_puct=5, n_playout=1000)
        # human = Human()


        results = {-1:0, 1:0, 2:0}
        start_player = 2


        for i in range(2):
            winner = game.start_play(mcts_player_1, mcts_player_2,
                                     start_player=start_player,
                                     is_shown=0,
                                     start_board=i_board,
                                     last_move_p1=None,
                                     last_move_p2=None,
                                     savefig=1)

            results[winner] += 1
            start_player = 3 - start_player

            print(f"Game {i+1}: {start_player + 2} plains model has started, {winner + 2} plains model has won")

        print("\n\nWins of 3 plains model: ", results[1])
        print("Wins of 4 plains model: ", results[2])
        print("Ties: ", results[-1])


    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
