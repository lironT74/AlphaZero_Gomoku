# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
"""
from __future__ import print_function
from multiprocessing import Pool, Manager
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
import PIL.Image
from Game_boards_and_aux import *
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import io
import pandas as pd
import time
from dateutil.relativedelta import relativedelta as rd


def model_againts_mcts(model_name, max_model_iter, model_check_freq, input_plains_num, game_board=EMPTY_BOARD, n=4, width=6, height=6, n_games=1000, mcts_playout_num=5000, c_puct=5, n_playout=400):

    models_num = max_model_iter // model_check_freq
    model_list = range(model_check_freq, max_model_iter + model_check_freq, model_check_freq)

    _, board_name, _, _,_,_ = game_board

    all_arguments = model_name, input_plains_num, width, height, n, game_board, n_games, mcts_playout_num, c_puct, n_playout

    results = Manager().list()

    block_size = 20
    for i in range(0, len(model_list), block_size):

        with Pool(block_size) as pool:
            jobs = []
            for j in model_list[i:i+block_size]:
                jobs.append((all_arguments, results, j))

            print(f"{model_name}: Using {pool._processes} workers. There are {len(jobs)} jobs: \n")
            pool.starmap(policy_evaluate_againts_mcts, jobs)
            pool.close()
            pool.join()

    results = sorted(results, key = lambda x: x[0])

    np_results = np.zeros((4, len(model_list)))

    for j in range(1,5,1):
        np_results[j-1] = [x[j] for x in results]

    pool.close()

    path = f"/home/lirontyomkin/AlphaZero_Gomoku/models vs mcts/{model_name}/"
    if not os.path.exists(path):
        os.makedirs(path)

    save_csv_results_againts_mcts(model_name, model_list, np_results, mcts_playout_num, board_name, n_games, n_playout, path)


def policy_evaluate_againts_mcts(all_arguments, results, iteration):

    (model_name, input_plains_num, width, height, n, game_board, n_games, mcts_playout_num, c_puct, n_playout) = all_arguments

    print(f"Started games of model: {model_name}_{iteration}")

    if iteration != -1:
        path = f'/home/lirontyomkin/AlphaZero_Gomoku/models/{model_name}/current_policy_{iteration}.model'
        name = model_name + f"_{iteration}"

    else:
        path = f'/home/lirontyomkin/AlphaZero_Gomoku/models/{model_name}'
        name = model_name

    board_state, board_name, p1, p2, _ ,_ = game_board

    best_policy = PolicyValueNet(width, height, model_file=path, input_plains_num=input_plains_num)

    model_mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=c_puct, n_playout=n_playout, name=name,
                               input_plains_num=input_plains_num)

    pure_mcts_player = MCTS_Pure(c_puct=5, n_playout=mcts_playout_num)


    win_cnt = defaultdict(int)
    for i in range(n_games):
        i_board1, board1 = initialize_board_without_init_call(width, height, n, input_board=board_state, open_path_threshold=-1)
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

    print(f"Done games of model: {model_name}_{iteration}")

    results.append(result)

    # return result


def save_csv_results_againts_mcts(model_name, model_list, np_results, mcts_playout_num, board_name, n_games, n_playout, path):
    columns = [f"{model_name}__{i}" for i in model_list]
    index = ["wins", "losses", "ties", "wins ratio"]

    df = pd.DataFrame(np_results, index=index, columns=columns)
    df.to_csv(f"{path} with {n_playout} playout MCTS vs {mcts_playout_num} playouts MCTS on {board_name} ({n_games} games).csv", index=True, header=True)


def save_fig_againts_mcts(np_results, models_num, model_list, model_name, mcts_playout_num, board_name, n_games, to_display,n_playout, path, results_base_model):
    fig, (ax, lax) = plt.subplots(nrows=2, gridspec_kw={"height_ratios": [20, 1]}, figsize=(30, 10))

    fontsize = 17
    linewidth = 3


    if "wins" in to_display:
        ax.plot(range(models_num), np_results[0], label=f"wins", color="green", linewidth=linewidth)

        if results_base_model != (-1,-1,-1,-1,-1):
            ax.scatter(models_num + 1, results_base_model[1], marker='o', label=f"base model - wins", color="green",
                       linewidth=2 * linewidth)

    if "losses" in to_display:
        ax.plot(range(models_num), np_results[1], label=f"losses", color="red", linewidth=linewidth)

        if results_base_model != (-1,-1,-1,-1,-1):
            ax.scatter(models_num + 2, results_base_model[2], marker='o', label=f"base model - losses", color="red",
                       linewidth=2 * linewidth)


    if "ties" in to_display:
        ax.plot(range(models_num), np_results[2], label=f"ties", color="yellow", linewidth=linewidth)

        if results_base_model != (-1,-1,-1,-1,-1):
            ax.scatter(models_num + 3, results_base_model[3], marker='o', label=f"base model - ties", color="blue",
                       linewidth=2 * linewidth)

    if "win ratio" in to_display:
        ax.plot(range(models_num), np_results[3], label=f"win ratio", color="blue", linewidth=linewidth)

        if results_base_model != (-1,-1,-1,-1,-1):
            ax.scatter(models_num + 1, results_base_model[4], marker='o', label=f"base model - win ratio", color="blue",
                       linewidth=2 * linewidth)
        ax.set_ylim([0, 1])

    ax.set_xticks(range(models_num))
    ax.set_xticklabels(model_list, rotation=90, fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.set_xlabel("sub model no.", fontsize=fontsize)
    ax.set_title(f"{model_name} with {n_playout} playout MCTS againt MCTS with {mcts_playout_num} playouts on {board_name}, {n_games} games",
                 fontdict={'fontsize': fontsize + 15})

    h, l = ax.get_legend_handles_labels()
    lax.legend(h, l, borderaxespad=0, loc="center", fancybox=True, shadow=True, ncol=len(to_display), fontsize=fontsize + 5)
    lax.axis("off")

    fig.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)

    plt.savefig(f"{path}{model_name} with {n_playout} playout MCTS vs {mcts_playout_num} playouts MCTS on {board_name} ({n_games} games, {' '.join(to_display)}).png")

    plt.close('all')

    return f"{path}{model_name} with {n_playout} playout MCTS vs {mcts_playout_num} playouts MCTS on {board_name} ({n_games} games, {' '.join(to_display)}).png"


def saved_results_to_fig(args, results_base_model):

    (model_name, max_model_iter, model_check_freq, input_plains_num, game_board, n, width, height, n_games, mcts_playout_num, c_puct, n_playout) = args

    models_num = max_model_iter // model_check_freq
    model_list = range(model_check_freq, max_model_iter + model_check_freq, model_check_freq)
    _, board_name, _, _, _, _ = game_board

    path = f"/home/lirontyomkin/AlphaZero_Gomoku/models vs mcts/{model_name}/"

    df = pd.read_csv(f"{path} with {n_playout} playout MCTS vs {mcts_playout_num} playouts MCTS on {board_name} ({n_games} games).csv")

    np_results = df.to_numpy()
    np_results = np_results[:, 1:]

    to_display = ["win ratio"]
    win_ratio = save_fig_againts_mcts(np_results, models_num, model_list, model_name, mcts_playout_num, board_name, n_games, to_display, n_playout, path, results_base_model)

    to_display = ["wins", "losses", "ties"]
    other = save_fig_againts_mcts(np_results, models_num, model_list, model_name, mcts_playout_num, board_name, n_games, to_display, n_playout, path, results_base_model)

    return win_ratio, other


def make_collage_models_vs_mcts(listofimages, to_display):

    path = f"/home/lirontyomkin/AlphaZero_Gomoku/models vs mcts/"

    im_check = PIL.Image.open(listofimages[0])
    width1, height1 = im_check.size
    width = width1
    height = len(listofimages) * height1

    cols = 1
    rows = len(listofimages)

    thumbnail_width = width // cols
    thumbnail_height = height // rows
    size = thumbnail_width, thumbnail_height
    new_im = PIL.Image.new('RGB', (width, height))
    ims = []
    for p in listofimages:
        im = PIL.Image.open(p)
        im.thumbnail(size)
        ims.append(im)
    i = 0
    x = 0
    y = 0
    for col in range(cols):
        for row in range(rows):
            # print(i, x, y)
            new_im.paste(ims[i], (x, y))
            i += 1
            y += thumbnail_height
        x += thumbnail_width
        y = 0

    new_im.save(path + f"{' '.join(to_display)}.png")


def save_mcts_pics_and_collage(models_args, results_base_model):
    win_ratio = []
    others = []

    for args_model in models_args:
        ratio, other = saved_results_to_fig(args_model, results_base_model)

        win_ratio.append(ratio)
        others.append(other)

    to_display = ["win ratio"]
    make_collage_models_vs_mcts(win_ratio, to_display)

    to_display = ["wins", "losses", "ties"]
    make_collage_models_vs_mcts(others, to_display)



def check_mcts_goodness(model, width, height, n, game_board, n_games, mcts_playout_num, c_puct, n_playout):
    path, model_name, input_plains_num = model

    policy_evaluate_againts_mcts_checking_mcts(model_name, path, input_plains_num, width, height, n, game_board,
                                               n_games, mcts_playout_num, c_puct, n_playout)



def policy_evaluate_againts_mcts_checking_mcts(model_name, path, input_plains_num, width, height, n, game_board, n_games, mcts_playout_num, c_puct, n_playout):


    print(f"Started games of model: {model_name}")

    board_state, board_name, p1, p2, _, _ = game_board

    best_policy = PolicyValueNet(width, height, model_file=path, input_plains_num=input_plains_num)

    model_mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=c_puct, n_playout=n_playout, name=model_name,
                               input_plains_num=input_plains_num)

    pure_mcts_player = MCTS_Pure(c_puct=5, n_playout=mcts_playout_num, name="Pure MCTS")


    win_cnt = defaultdict(int)
    for i in range(n_games):
        i_board1, board1 = initialize_board_without_init_call(width, height, n, input_board=board_state, open_path_threshold=-1)
        game = Game(board1)

        path = f'/home/lirontyomkin/AlphaZero_Gomoku/check_mcts/{model_mcts_player.name}/game_{i+1}/'

        if not os.path.exists(path):
            os.makedirs(path)


        winner = game.start_play_just_game_capture(path,
                                                   model_mcts_player,
                                                   pure_mcts_player,
                                                   start_player=i % 2 + 1,
                                                   is_shown=1,
                                                   game_num=i+1)

        win_cnt[winner] += 1

    win = win_cnt[1]
    lose = win_cnt[2]
    tie = win_cnt[-1]
    win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
    result = (win, lose, tie, win_ratio)

    print(f"model: {model_name}, win: {win}, lose: {lose}, tie:{tie}, win ratio: {win_ratio}")

    print(f"Done games of model: {model_name}")

    return result



def models_vs_mcts_flow():
    start = time.time()

    n = 4
    width = 6
    height = 6

    max_model_iter = 5000
    model_check_freq = 50
    n_games = 100
    mcts_playout_num = 5000
    c_puct = 5
    n_playout = 400

    # # model_name, max_model_iter, model_check_freq, input_plains_num, game_board = EMPTY_BOARD, n = 4, width = 6, height = 6, n_games = 1000, mcts_playout_num = 5000, c_puct = 5, n_playout = 400)
    args_v7 = (
    'pt_6_6_4_p3_v7', max_model_iter, model_check_freq, 3, EMPTY_BOARD, n, width, height, n_games, mcts_playout_num,
    c_puct, n_playout)
    args_v9 = (
    'pt_6_6_4_p3_v9', max_model_iter, model_check_freq, 3, EMPTY_BOARD, n, width, height, n_games, mcts_playout_num,
    c_puct, n_playout)
    args_v10 = (
    'pt_6_6_4_p4_v10', max_model_iter, model_check_freq, 4, EMPTY_BOARD, n, width, height, n_games, mcts_playout_num,
    c_puct, n_playout)

    models_args = [args_v7, args_v9, args_v10]

    for model_args in models_args:
        model_name, max_model_iter, model_check_freq, input_plains_num, game_board, n, width, height, n_games, mcts_playout_num, c_puct, n_playout = model_args
        model_againts_mcts(model_name, max_model_iter, model_check_freq, input_plains_num, game_board, n, width, height,
                           n_games, mcts_playout_num, c_puct, n_playout)

    # args_given_model = ('best_policy_6_6_4.model2', 4, width, height, n, EMPTY_BOARD, n_games, mcts_playout_num, c_puct, n_playout)
    # results_base_model = policy_evaluate_againts_mcts(-1, args_given_model)

    results_base_model = (-1, -1, -1, -1, -1)
    save_mcts_pics_and_collage(models_args, results_base_model)

    fmt = '{0.days} days {0.hours} hours {0.minutes} minutes {0.seconds} seconds'
    end = time.time()

    print("all of it took", fmt.format(rd(seconds=end - start)))


def check_mcts_flow():
    height = 6
    width = 6
    n = 4
    game_board = EMPTY_BOARD
    n_games = 100
    mcts_playout_num = 5000
    c_puct = 5
    n_playout = 400

    v7 = (
    '/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p3_v7/current_policy_2100.model', 'pt_6_6_4_p3_v7_2100', 3)
    v9 = (
    '/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p3_v9/current_policy_1350.model', 'pt_6_6_4_p3_v9_1350', 3)
    v10 = (
    '/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v10/current_policy_1150.model', 'pt_6_6_4_p4_v10_1150', 4)

    models = [v7, v9, v10]

    models = [v7, v9, v10]



    jobs = []
    for model in models:
        jobs.append((model, width, height, n, game_board, n_games, mcts_playout_num, c_puct, n_playout))

    with Pool(3) as pool:
        pool.starmap(check_mcts_goodness, jobs)
        pool.close()
        pool.join()


if __name__ == '__main__':
    check_mcts_flow()

