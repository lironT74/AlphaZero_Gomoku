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


EMPTY_BOARD = (np.array([[0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0]]), "empty board", None, None)
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
PAPER_TRUNCATED_BOARDS = [BOARD_1_TRUNCATED, BOARD_2_TRUNCATED]
PAPER_FULL_BOARDS = [BOARD_1_FULL, BOARD_2_FULL]


def initialize_board(board_height, board_width, n_in_row, input_board):

    board = input_board
    board = np.flipud(board)
    i_board = np.zeros((2, board_height, board_width))
    i_board[0] = board == 1
    i_board[1] = board == 2
    board = Board(width=board_width, height=board_height, n_in_row=n_in_row)
    return i_board, board


v9_3500 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p3_v9/current_policy_3500.model', 'pt_6_6_4_p3_v9_3500', 3)
v7_2100 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p3_v7/current_policy_2100.model', "pt_6_6_4_p3_v7_2100", 3)
v10_5000 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v10/current_policy_5000.model', "pt_6_6_4_p3_v10_5000", 4)

MODELS_TO_MATCH = [v9_3500, v7_2100, v10_5000]


def compare_all_models(models_list=MODELS_TO_MATCH, width=6, height=6, n=4):
    for i in range(len(models_list)):
        for j in range(i+1, len(models_list)):
            compare_two_models(models_list[i], models_list[j], width, height, n)


def compare_two_models(model1, model2, width, height, n):

    path1, name1, planes1 = model1
    path2, name2, planes2 = model2

    best_policy_1 = PolicyValueNet(width, height, model_file=path1, input_plains_num=planes1)
    mcts_player_1 = MCTSPlayer(best_policy_1.policy_value_fn, c_puct=5, n_playout=400, name=name1, input_plains_num=planes1)

    best_policy_2 = PolicyValueNet(width, height, model_file=path2, input_plains_num=planes2)
    mcts_player_2 = MCTSPlayer(best_policy_2.policy_value_fn, c_puct=5, n_playout=400, name=name2, input_plains_num=planes2)

    board_state, board_name, p1, p2 = EMPTY_BOARD

    save_game_res(width=width,
                  height=height,
                  n=n,
                  board_state=board_state,
                  board_name=board_name,
                  mcts_player_1=mcts_player_1,
                  mcts_player_2=mcts_player_2,
                  last_move_p1=p1,
                  last_move_p2=p2,
                  start_player=1)

    for board_state, board_name, p1, p2 in PAPER_FULL_BOARDS:
        save_game_res(width=width,
                      height=height,
                      n=n,
                      board_state=board_state,
                      board_name=board_name,
                      mcts_player_1=mcts_player_1,
                      mcts_player_2=mcts_player_2,
                      last_move_p1=p1,
                      last_move_p2=p2,
                      start_player=2)

    for board_state, board_name, p1, p2 in PAPER_TRUNCATED_BOARDS:

        save_game_res(width=width,
                      height=height,
                      n=n,
                      board_state=board_state,
                      board_name=board_name,
                      mcts_player_1=mcts_player_1,
                      mcts_player_2=mcts_player_2,
                      last_move_p1=p1,
                      last_move_p2=p2,
                      start_player=2)

        if planes1==4:
            save_game_res(width=width,
                          height=height,
                          n=n,
                          board_state=board_state,
                          board_name=board_name,
                          mcts_player_1=mcts_player_1,
                          mcts_player_2=mcts_player_2,
                          last_move_p1=None,
                          last_move_p2=p2,
                          start_player=2)

            if planes2 == 4:
                save_game_res(width=width,
                              height=height,
                              n=n,
                              board_state=board_state,
                              board_name=board_name,
                              mcts_player_1=mcts_player_1,
                              mcts_player_2=mcts_player_2,
                              last_move_p1=None,
                              last_move_p2=None,
                              start_player=2)
        if planes2==4:
            save_game_res(width=width,
                          height=height,
                          n=n,
                          board_state=board_state,
                          board_name=board_name,
                          mcts_player_1=mcts_player_1,
                          mcts_player_2=mcts_player_2,
                          last_move_p1=p1,
                          last_move_p2=None,
                          start_player=2)


def save_game_res(width, height, n, board_state, board_name, mcts_player_1, mcts_player_2, last_move_p1, last_move_p2, start_player):
    i_board, board = initialize_board(width, height, n, input_board=board_state)

    game1 = Game(board)
    game1.start_play(mcts_player_1, mcts_player_2,
                     start_player=start_player,
                     is_shown=0,
                     start_board=i_board,
                     last_move_p1=last_move_p1,
                     last_move_p2=last_move_p2,
                     savefig=1,
                     board_name=board_name)

    game2 = Game(board)
    game2.start_play(mcts_player_2, mcts_player_1,
                    start_player=start_player,
                    is_shown=0,
                    start_board=i_board,
                    last_move_p1=last_move_p1,
                    last_move_p2=last_move_p2,
                    savefig=1,
                    board_name=board_name)


if __name__ == '__main__':
    compare_all_models()
