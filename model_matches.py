# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
"""
from __future__ import print_function
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet  # Pytorch
from policy_player import PolicyPlayer
import numpy as np
import PIL.Image
from Game_boards import *
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import os
import io
import pandas as pd
import multiprocessing
import time
from dateutil.relativedelta import relativedelta as rd
from itertools import repeat

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
v10_5000 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v10/current_policy_5000.model', "pt_6_6_4_p4_v10_5000", 4)

MODELS_TO_MATCH = [v10_5000, v9_3500, v7_2100]

def compare_all_models(models_list, width=6, height=6, n=4):
    for i in range(len(models_list)):
        for j in range(i+1, len(models_list)):
            compare_two_models(models_list[i], models_list[j], width, height, n)


def compare_two_models(model1, model2, width, height, n):

    path1, name1, plains1 = model1
    path2, name2, plains2 = model2

    best_policy_1 = PolicyValueNet(width, height, model_file=path1, input_plains_num=plains1)
    mcts_player_1 = MCTSPlayer(best_policy_1.policy_value_fn, c_puct=5, n_playout=400, name=name1, input_plains_num=plains1)

    best_policy_2 = PolicyValueNet(width, height, model_file=path2, input_plains_num=plains2)
    mcts_player_2 = MCTSPlayer(best_policy_2.policy_value_fn, c_puct=5, n_playout=400, name=name2, input_plains_num=plains2)


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


        if plains1+plains2 >= 7:
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

        if plains1+plains2 == 8:

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


def save_game_res(width, height, n, board_state, board_name, mcts_player_1, mcts_player_2, last_move_p1, last_move_p2, start_player):
    i_board1, board1 = initialize_board(width, height, n, input_board=board_state)
    game1 = Game(board1)
    game1.start_play(player1=mcts_player_1, player2=mcts_player_2,
                     start_player=start_player,
                     is_shown=0,
                     start_board=i_board1,
                     last_move_p1=last_move_p1,
                     last_move_p2=last_move_p2,
                     savefig=1,
                     board_name=board_name)

    i_board2, board2 = initialize_board(width, height, n, input_board=board_state)
    game2 = Game(board2)
    game2.start_play(player1=mcts_player_2, player2=mcts_player_1,
                    start_player=start_player,
                    is_shown=0,
                    start_board=i_board2,
                    last_move_p1=last_move_p1,
                    last_move_p2=last_move_p2,
                    savefig=1,
                    board_name=board_name)


def model_againts_mcts(model_name, max_model_iter, model_check_freq, input_plains_num, game_board=EMPTY_BOARD, n=4, width=6, height=6, n_games=1000, mcts_playout_num=5000):

    models_num = max_model_iter // model_check_freq
    model_list = range(model_check_freq, max_model_iter + model_check_freq, model_check_freq)

    _, board_name, _, _ = game_board

    second_arg = model_name, input_plains_num, width, height, n, game_board, n_games, mcts_playout_num

    pool = multiprocessing.Pool()
    results = pool.starmap(policy_evaluate_againts_mcts, zip(model_list, repeat(second_arg)))
    pool.close()

    results = sorted(results, key = lambda x: x[0])

    fig, (ax, lax) = plt.subplots(nrows=2, gridspec_kw={"height_ratios": [20, 1]}, figsize=(30, 10))

    fontsize = 17
    linewidth = 3

    # ax.plot(range(models_num), [x[1] for x in results], label=f"wins", color="green", linewidth=linewidth)
    # ax.plot(range(models_num), [x[2] for x in results], label=f"losses", color="red", linewidth=linewidth)
    # ax.plot(range(models_num), [x[3] for x in results], label=f"ties", color="yellow", linewidth=linewidth)
    ax.plot(range(models_num), [x[4] for x in results], label=f"win ratio", color="blue", linewidth=linewidth)


    ax.set_xticks(range(models_num))
    ax.set_xticklabels(model_list, rotation=90, fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.set_xlabel("sun model no.", fontsize=fontsize)
    ax.set_title(f"{model_name} againt MCTS with {mcts_playout_num} playouts on {board_name}, {n_games} games", fontdict={'fontsize': fontsize + 15})

    h, l = ax.get_legend_handles_labels()
    lax.legend(h, l, borderaxespad=0, loc="center", fancybox=True, shadow=True, ncol=5, fontsize=fontsize + 5)
    lax.axis("off")

    fig.tight_layout()


    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)

    path = f"/home/lirontyomkin/AlphaZero_Gomoku/models vs mcts/{model_name}/"
    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(f"{path}vs {mcts_playout_num} playouts MCTS on {board_name} ({n_games} games).png")

    columns = [f"{model_name}__{i}" for i in model_list]
    index = ["wins", "losses", "ties", "wins ratio"]
    df = pd.DataFrame(results, index=index, columns=columns)
    df.to_csv(f"{path}vs {mcts_playout_num} playouts MCTS on {board_name} ({n_games} games).csv", index=True, header=True)

    plt.close('all')

def policy_evaluate_againts_mcts(iteration, all_arguments):

    model_name, input_plains_num, width, height, n, game_board, n_games, mcts_playout_num = all_arguments

    if iteration != -1:
        path = f'/home/lirontyomkin/AlphaZero_Gomoku/models/{model_name}/current_policy_{iteration}.model'
        name = model_name + f"_{iteration}"

    else:
        path = f'/home/lirontyomkin/AlphaZero_Gomoku/models/{model_name}'
        name = model_name

    board_state, board_name, p1, p2 = game_board

    best_policy = PolicyValueNet(width, height, model_file=path, input_plains_num=input_plains_num)

    model_mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400, name=name,
                               input_plains_num=input_plains_num)

    pure_mcts_player = MCTS_Pure(c_puct=5, n_playout=mcts_playout_num)


    win_cnt = defaultdict(int)
    for i in range(n_games):
        i_board1, board1 = initialize_board(width, height, n, input_board=board_state)
        game = Game(board1)

        winner = game.start_play(model_mcts_player,
                                  pure_mcts_player,
                                  start_player=i % 2 + 1,
                                  is_shown=0)
        win_cnt[winner] += 1

    win = win_cnt[1]
    lose = win_cnt[2]
    tie = win_cnt[-1]
    win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
    result = (iteration, win, lose, tie, win_ratio)

    if iteration != -1:
        print(f"model: {model_name}_{iteration}, win: {win}, lose: {lose}, tie:{tie}, win ratio: {win_ratio}")
    else:
        print(f"model: {model_name}, win: {win}, lose: {lose}, tie:{tie}, win ratio: {win_ratio}")

    return result


if __name__ == '__main__':
    start = time.time()

    # model_name, max_model_iter, model_check_freq, input_plains_num, game_board, n, width, height, n_games, mcts_playout_num):

    processes = []

    args_v7 = ('pt_6_6_4_p3_v7', 100, 50, 3, EMPTY_BOARD, 4, 6, 6, 10, 10)
    args_v9 = ('pt_6_6_4_p3_v9', 100, 50, 3, EMPTY_BOARD, 4, 6, 6, 10, 10)
    args_v9 = ('pt_6_6_4_p4_v10', 100, 50, 3, EMPTY_BOARD, 4, 6, 6, 10, 10)

    models_args = [args_v7, args_v9]

    for args_model in models_args:
        p = multiprocessing.Process(target=model_againts_mcts, args=args_model)
        processes.append(p)
        p.start()

    for process in processes:
        process.join()


    fmt = '{0.days} days {0.hours} hours {0.minutes} minutes {0.seconds} seconds'
    end = time.time()


    print("all of it took", fmt.format(rd(seconds = end - start)))
