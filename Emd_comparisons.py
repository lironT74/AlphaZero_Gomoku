from multiprocessing import get_context, Pool
import math
from pyemd import emd
import pandas as pd
from Game_boards import *
from game import Board
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import matplotlib.pyplot as plt
from policy_value_net_pytorch import PolicyValueNet
import io
import PIL
import string
import copy

def initialize_board(board_height, board_width, input_board, n_in_row = 4, start_player=2, **kwargs):

    last_move_p1 = kwargs.get('last_move_p1', None)
    last_move_p2 = kwargs.get('last_move_p2', None)

    board = copy.deepcopy(input_board)
    board = np.flipud(board)
    i_board = np.zeros((2, board_height, board_width))
    i_board[0] = board == 1
    i_board[1] = board == 2

    board = Board(width=board_width, height=board_height, n_in_row=n_in_row)
    board.init_board(start_player=start_player, initial_state=i_board, last_move_p1=last_move_p1, last_move_p2=last_move_p2)
    return board


def create_collages_boards(listofimages, fig_name, path):
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

    new_im.save(path + f"{fig_name}.png")

def EMD_model_comparison(model1_name, input_plains_num_1, max_model1_iter, model1_check_freq, tell_last_move1,
                         model2_name, input_plains_num_2, max_model2_iter, model2_check_freq, tell_last_move2,
                         board, n=4, width=6, height=6, **kwargs):

    BOARDS = [BOARD_1_FULL, BOARD_2_FULL, BOARD_1_TRUNCATED, BOARD_2_TRUNCATED, EMPTY_BOARD]


    last_move_str_1 = " with last move " if tell_last_move1 else " "
    last_move_str_2 = " with last move" if tell_last_move2 else ""

    save_path = f'/home/lirontyomkin/AlphaZero_Gomoku/models emd comparison/{model1_name}{last_move_str_1}and {model2_name}{last_move_str_2}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    sub_models_1 = list(range(model1_check_freq, max_model1_iter + model1_check_freq, model1_check_freq))
    sub_models_2 = list(range(model2_check_freq, max_model2_iter + model2_check_freq, model2_check_freq))


    index = [f"{model1_name}__{i}" for i in sub_models_1]
    columns = [f"{model2_name}__{i}" for i in sub_models_2]

    # result = np.random.rand(len(index), len(columns))
    result = np.empty((len(index), len(columns)))

    board_state, board_name, last_move_p1, last_move_p2, _, _ = board

    if tell_last_move1:
        board1 = initialize_board(height, width, input_board=board_state, n_in_row=n, last_move_p1=last_move_p1,
                                  last_move_p2=last_move_p2)
    else:
        board1 = initialize_board(height, width, input_board=board_state, n_in_row=n, last_move_p1=None,
                                  last_move_p2=None)

    if tell_last_move2:
        board2 = initialize_board(height, width, input_board=board_state, n_in_row=n, last_move_p1=last_move_p1,
                                  last_move_p2=last_move_p2)
    else:
        board2 = initialize_board(height, width, input_board=board_state, n_in_row=n, last_move_p1=None,
                                  last_move_p2=None)

    for index_i, i in enumerate(range(model1_check_freq, max_model1_iter + model1_check_freq, model1_check_freq)):
        for index_j, j in enumerate(range(model2_check_freq, max_model2_iter + model2_check_freq, model2_check_freq)):
            result[index_i, index_j] = EMD_between_two_models_on_board(
                                   model1_name=model1_name, input_plains_num_1=input_plains_num_1, i1=i,
                                   model2_name=model2_name, input_plains_num_2=input_plains_num_2, i2=j,
                                   board1=board1, board2=board2, width=width,height=height)


    Emd_upper_bound = 0.001
    if np.max(result) > Emd_upper_bound:
        raise Exception("Enlarge upper boundary in EMD colorbar")

    df = pd.DataFrame(result, index=index, columns=columns)
    df.to_csv(f"{save_path}on {board_name}.csv",index = True, header=True)

    fig, ax = plt.subplots(1, figsize=(20, 20))
    fontsize = 12

    im = ax.imshow(result, cmap='hot', interpolation='nearest', norm=plt.Normalize(vmin=0, vmax=Emd_upper_bound))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    sm = plt.cm.ScalarMappable(cmap='hot', norm=plt.Normalize(vmin=0, vmax=Emd_upper_bound))
    fig.colorbar(sm, ax=ax, cax=cax).ax.tick_params(labelsize=fontsize*2)

    ax.set_title(f"EMD of {model1_name}{last_move_str_1}and {model2_name}{last_move_str_2}\non {board_name}", fontsize=3*fontsize)
    ax.set_xticks(list(range(len(sub_models_1))))
    ax.set_yticks(list(range(len(sub_models_1))))

    ax.set_xticklabels([str(i) for i in sub_models_1], rotation=90, fontsize=fontsize)
    ax.set_yticklabels([str(i) for i in sub_models_2], fontsize=fontsize)

    ax.set_xlabel(model1_name, fontsize=fontsize*2.5)
    ax.set_ylabel(model2_name, rotation=90, fontsize=fontsize*2.5)

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)

    plt.savefig(f"{save_path}on {board_name}.png")

    print(f"Done {model1_name} and {model2_name} on {board_name}")

    plt.close('all')

def EMD_between_two_models_on_board(model1_name, input_plains_num_1, i1,
                                   model2_name, input_plains_num_2, i2,
                                   board1, board2, width=6,height=6):


            model_file_1 = f'/home/lirontyomkin/AlphaZero_Gomoku/models/{model1_name}/current_policy_{i1}.model'
            policy_1 = PolicyValueNet(width, height, model_file=model_file_1, input_plains_num=input_plains_num_1)

            model_file_2 = f'/home/lirontyomkin/AlphaZero_Gomoku/models/{model2_name}/current_policy_{i2}.model'
            policy_2 = PolicyValueNet(width, height, model_file=model_file_2, input_plains_num=input_plains_num_2)

            acts_policy1, probas_policy1 = zip(*policy_1.policy_value_fn(board1)[0])
            acts_policy2, probas_policy2 = zip(*policy_2.policy_value_fn(board2)[0])

            dist_matrix = generate_matrix_dist_metric(6)

            distance = emd(np.asarray(probas_policy1, dtype='float64'), np.asarray(probas_policy2, dtype='float64'), dist_matrix)

            return distance

def convert_position_to_row_col(pos, dimension):
    col = int(((pos - 1) % dimension))
    row = (float(pos)/float(dimension))-1
    row = int(math.ceil(row))
    return row, col


def dist_in_matrix(index_i, index_j, dim):
    r1,c1 = convert_position_to_row_col(index_i, dim)
    r2,c2 = convert_position_to_row_col(index_j, dim)
    return max(abs(r1-r2),abs(c1-c2))


def generate_matrix_dist_metric(dim, norm=True):
    distances = np.zeros((dim*dim, dim*dim))
    for i in range(np.size(distances, 1)):
        for j in range(np.size(distances, 1)):
            distances[i][j] = dist_in_matrix(i+1,j+1,dim)
    if norm:
        distances = distances/distances.sum()
    return distances


def generate_models_emd_comparison():

    BOARDS = [BOARD_1_FULL, BOARD_2_FULL, BOARD_1_TRUNCATED, BOARD_2_TRUNCATED, EMPTY_BOARD]

    arguments = []

    for board in BOARDS:
        arguments.append(("pt_6_6_4_p4_v10", 4, 5000, 50, True, "pt_6_6_4_p3_v7", 3, 5000, 50, False, board, 4, 6, 6))

        # EMD_model_comparison(model1_name="pt_6_6_4_p4_v10", input_plains_num_1=4, max_model1_iter=5000,
        #                      model1_check_freq=50, tell_last_move1=True,
        #                      model2_name="pt_6_6_4_p3_v7", input_plains_num_2=3, max_model2_iter=5000,
        #                      model2_check_freq=50, tell_last_move2=False,
        #                      game_board=board, n=4, width=6, height=6)

        arguments.append(("pt_6_6_4_p4_v10", 4, 5000, 50, True, "pt_6_6_4_p3_v9", 3, 5000, 50, False, board, 4, 6, 6))

        # EMD_model_comparison(model1_name="pt_6_6_4_p4_v10", input_plains_num_1=4, max_model1_iter=5000,
        #                      model1_check_freq=50, tell_last_move1=True,
        #                      model2_name="pt_6_6_4_p3_v9", input_plains_num_2=3, max_model2_iter=5000,
        #                      model2_check_freq=50, tell_last_move2=False,
        #                      game_board=board, n=4, width=6, height=6)

        arguments.append(("pt_6_6_4_p3_v7", 3, 5000, 50, False, "pt_6_6_4_p3_v9", 3, 5000, 50, False, board, 4, 6, 6))

        # EMD_model_comparison(model1_name="pt_6_6_4_p3_v7", input_plains_num_1=3, max_model1_iter=5000,
        #                      model1_check_freq=50, tell_last_move1=False,
        #                      model2_name="pt_6_6_4_p3_v9", input_plains_num_2=3, max_model2_iter=5000,
        #                      model2_check_freq=50, tell_last_move2=False,
        #                      game_board=board, n=4, width=6, height=6)

        arguments.append(("pt_6_6_4_p4_v10", 4, 5000, 50, False, "pt_6_6_4_p3_v7", 3, 5000, 50, False, board, 4, 6, 6))

        # EMD_model_comparison(model1_name="pt_6_6_4_p4_v10", input_plains_num_1=4, max_model1_iter=5000,
        #                      model1_check_freq=50, tell_last_move1=False,
        #                      model2_name="pt_6_6_4_p3_v7", input_plains_num_2=3, max_model2_iter=5000,
        #                      model2_check_freq=50, tell_last_move2=False,
        #                      game_board=board, n=4, width=6, height=6)

        arguments.append(("pt_6_6_4_p4_v10", 4, 5000, 50, False, "pt_6_6_4_p3_v9", 3, 5000, 50, False, board, 4, 6, 6))

        # EMD_model_comparison(model1_name="pt_6_6_4_p4_v10", input_plains_num_1=4, max_model1_iter=5000,
        #                      model1_check_freq=50, tell_last_move1=False,
        #                      model2_name="pt_6_6_4_p3_v9", input_plains_num_2=3, max_model2_iter=5000,
        #                      model2_check_freq=50, tell_last_move2=False,
        #                      game_board=board, n=4, width=6, height=6)


    with Pool(5) as pool:
        print(f"Using {pool._processes} workers. There are {len(arguments)} jobs: \n")
        pool.starmap(EMD_model_comparison, arguments)
        pool.close()



def compare_model_to_heuristics(model_name, game_board, n=4, width=6, height=6, opponent_weight=0.5, **kwargs):

    input_plains_num = kwargs.get("input_plains_num", 4)
    max_model_iter = kwargs.get("max_model_iter", 5000)
    model_check_freq = kwargs.get("model_check_freq", 50)
    tell_last_move = kwargs.get("tell_last_move", True)


    dist_matrix = generate_matrix_dist_metric(6)
    board_state, board_name, last_move_p1, last_move_p2, alternative_p1, alternative_p2 = copy.deepcopy(game_board)

    if tell_last_move:
        board = initialize_board(height, width, input_board=board_state, n_in_row=n, last_move_p1=last_move_p1,
                                  last_move_p2=last_move_p2)
    else:
        board = initialize_board(height, width, input_board=board_state, n_in_row=n, last_move_p1=alternative_p1,
                                  last_move_p2=alternative_p2)


    heuristics_scores = board.calc_all_heuristics(max_radius_density=2, normalize_all_heuristics=True, opponent_weight=opponent_weight)


    models_num = max_model_iter//model_check_freq
    model_list = range(model_check_freq, max_model_iter+model_check_freq, model_check_freq)

    distances_lists = np.zeros((heuristics_scores.shape[0], models_num))

    distances_base_models = np.zeros(heuristics_scores.shape[0])

    for index_i, i in enumerate(model_list):
        model_file = f'/home/lirontyomkin/AlphaZero_Gomoku/models/{model_name}/current_policy_{i}.model'
        policy = PolicyValueNet(width, height, model_file=model_file, input_plains_num=input_plains_num)
        acts_policy, probas_policy = zip(*policy.policy_value_fn(board)[0])

        move_probs_policy = np.zeros(width * height)
        move_probs_policy[list(acts_policy)] = probas_policy
        move_probs_policy = move_probs_policy.reshape(width, height)
        move_probs_policy = np.flipud(move_probs_policy)

        for j in range(heuristics_scores.shape[0]):

            distance = emd(np.asarray(np.reshape(move_probs_policy, width*height), dtype='float64'),
                           np.asarray(np.reshape(heuristics_scores[j], width*height), dtype='float64'),
                           dist_matrix)

            distances_lists[j, index_i] = distance

    for j in range(heuristics_scores.shape[0]):

        policy = PolicyValueNet(width, height, model_file=f'/home/lirontyomkin/AlphaZero_Gomoku/models/best_policy_6_6_4.model', input_plains_num=4)
        acts_policy, probas_policy = zip(*policy.policy_value_fn(board)[0])

        move_probs_policy = np.zeros(width * height)
        move_probs_policy[list(acts_policy)] = probas_policy
        move_probs_policy = move_probs_policy.reshape(width, height)
        move_probs_policy = np.flipud(move_probs_policy)

        distances_base_models[j] = emd(np.asarray(np.reshape(move_probs_policy, width*height), dtype='float64'),
                                       np.asarray(np.reshape(heuristics_scores[j], width*height), dtype='float64'),
                                       dist_matrix)


    fig, (ax, lax) = plt.subplots(nrows=2, gridspec_kw={"height_ratios": [20, 1]}, figsize=(30,10))

    fontsize = 17
    linewidth=3

    ax.plot(range(models_num), distances_lists[0], label=f"density", color="blue", linewidth=linewidth)
    ax.plot(range(models_num), distances_lists[1], label=f"linear", color="red", linewidth=linewidth)
    ax.plot(range(models_num), distances_lists[2], label=f"non-linear", color="green", linewidth=linewidth)
    ax.plot(range(models_num), distances_lists[3], label=f"interaction", color="orange", linewidth=linewidth)
    ax.plot(range(models_num), distances_lists[4], label=f"forcing", color="black", linewidth=linewidth)

    ax.scatter(models_num + 1, distances_base_models[0], marker='o', label=f"base model - density", color="blue", linewidth=2*linewidth)
    ax.scatter(models_num + 2, distances_base_models[1], marker='o', label=f"base model - linear", color="red", linewidth=2*linewidth)
    ax.scatter(models_num + 3, distances_base_models[2], marker='o', label=f"base model - non-linear", color="green", linewidth=2*linewidth)
    ax.scatter(models_num + 4, distances_base_models[3], marker='o', label=f"base model - interaction", color="orange", linewidth=2*linewidth)
    ax.scatter(models_num + 5, distances_base_models[4], marker='o', label=f"base model - forcing", color="black", linewidth=2*linewidth)

    ax.set_xticks(range(models_num))
    ax.set_xticklabels(model_list, rotation=90, fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.set_xlabel("sun model no.", fontsize=fontsize)


    if input_plains_num == 4 and np.sum(board.current_state(last_move=True)[2]) == 1:
        y_last_move = 6 - np.where(board.current_state(last_move=True)[2] == 1)[0][0]
        x_last_move = string.ascii_lowercase[np.where(board.current_state(last_move=True)[2] == 1)[1][0]]
        last_move = f" (last move - {x_last_move}{y_last_move})"
    else:
        last_move = ""


    ax.set_title(f"{model_name}{last_move} EMD distances from heuristics with o_weight={opponent_weight} on {board_name}", fontdict={'fontsize': fontsize+15})

    h, l = ax.get_legend_handles_labels()

    ord = [0,5,1,6,2,7,3,8,4,9]

    lax.legend([h[idx] for idx in ord],[l[idx] for idx in ord], borderaxespad=0, loc="center", fancybox=True, shadow=True, ncol=5, fontsize=fontsize+5)
    lax.axis("off")

    fig.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)

    path = f"/home/lirontyomkin/AlphaZero_Gomoku/models vs heuristics comparisons/o_weight_{opponent_weight}/{model_name}/"

    if not os.path.exists(path):
        os.makedirs(path)

    if last_move != "":
        plt.savefig(f"{path}{board_name}{last_move}.png")
    else:
        plt.savefig(f"{path}{board_name}.png")

    plt.close('all')


def heuristics_heatmaps(game_board, height=6, width=6, n=4, opponent_weight=0.5):

    board_state, board_name, last_move_p1, last_move_p2, alternative_p1, alternative_p2 = game_board
    board = initialize_board(height, width, input_board=board_state, n_in_row=n, last_move_p1=last_move_p1,
                             last_move_p2=last_move_p2)

    heuristics_scores = np.round(board.calc_all_heuristics(max_radius_density=2, normalize_all_heuristics=True, opponent_weight=opponent_weight), 3)

    heuristics_names = ["density", "linear", "nonlinear", "interaction", "interaction with forcing"]

    x_positions = board.current_state()[0]
    o_positions = board.current_state()[1]

    fontsize = 15
    fig, axes = plt.subplots(1, 5, figsize=(35, 8))
    fig.suptitle(f"Heuristics heatmaps on {board_name}", fontsize=fontsize + 10)
    x_axis = [letter for i, letter in zip(range(width), string.ascii_lowercase)]
    y_axis = range(height, 0, -1)
    sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=1))

    for k in range(5):
        ax = axes[k]

        move_probs_policy = heuristics_scores[k]

        im1 = ax.imshow(move_probs_policy, cmap='jet', norm=plt.Normalize(vmin=0, vmax=1))
        divider1 = make_axes_locatable(ax)
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(sm, ax=ax, cax=cax1).ax.tick_params(labelsize=fontsize)

        ax.set_xticks(np.arange(len(x_axis)))
        ax.set_yticks(np.arange(len(y_axis)))

        ax.set_xticklabels(x_axis, fontsize=fontsize)
        ax.set_yticklabels(y_axis, fontsize=fontsize)
        plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
        for i in range(len(y_axis)):
            for j in range(len(x_axis)):
                text = ax.text(j, i, "X" if x_positions[i, j] == 1 else (
                    "O" if o_positions[i, j] == 1 else move_probs_policy[i, j]),
                                ha="center", va="center", color="w", fontsize=fontsize)

        ax.set_title(f"{heuristics_names[k]}", fontsize=fontsize + 4)

    fig.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)

    path = f"/home/lirontyomkin/AlphaZero_Gomoku/models vs heuristics comparisons/o_weight_{opponent_weight}/heuristics_heatmaps/"

    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(f"{path}{board_name}.png")

    plt.close('all')


def call_collage_compare_to_heuristics(opponent_weight):
    listofimages_empty = [
        f"/home/lirontyomkin/AlphaZero_Gomoku/models vs heuristics comparisons/o_weight_{opponent_weight}/pt_6_6_4_p3_v7/empty board.png",
        f"/home/lirontyomkin/AlphaZero_Gomoku/models vs heuristics comparisons/o_weight_{opponent_weight}/pt_6_6_4_p3_v9/empty board.png",
        f"/home/lirontyomkin/AlphaZero_Gomoku/models vs heuristics comparisons/o_weight_{opponent_weight}/pt_6_6_4_p4_v10/empty board.png",
    ]

    listofimages_full_1 = [
        f"/home/lirontyomkin/AlphaZero_Gomoku/models vs heuristics comparisons/o_weight_{opponent_weight}/pt_6_6_4_p3_v7/board 1 full.png",
        f"/home/lirontyomkin/AlphaZero_Gomoku/models vs heuristics comparisons/o_weight_{opponent_weight}/pt_6_6_4_p3_v9/board 1 full.png",
        f"/home/lirontyomkin/AlphaZero_Gomoku/models vs heuristics comparisons/o_weight_{opponent_weight}/pt_6_6_4_p4_v10/board 1 full (last move - b6).png",
        ]

    listofimages_full_2 = [
        f"/home/lirontyomkin/AlphaZero_Gomoku/models vs heuristics comparisons/o_weight_{opponent_weight}/pt_6_6_4_p3_v7/board 2 full.png",
        f"/home/lirontyomkin/AlphaZero_Gomoku/models vs heuristics comparisons/o_weight_{opponent_weight}/pt_6_6_4_p3_v9/board 2 full.png",
        f"/home/lirontyomkin/AlphaZero_Gomoku/models vs heuristics comparisons/o_weight_{opponent_weight}/pt_6_6_4_p4_v10/board 2 full (last move - e6).png",
    ]

    listofimages_truncated_1 = [
        f"/home/lirontyomkin/AlphaZero_Gomoku/models vs heuristics comparisons/o_weight_{opponent_weight}/pt_6_6_4_p3_v7/board 1 truncated.png",
        f"/home/lirontyomkin/AlphaZero_Gomoku/models vs heuristics comparisons/o_weight_{opponent_weight}/pt_6_6_4_p3_v9/board 1 truncated.png",
        f"/home/lirontyomkin/AlphaZero_Gomoku/models vs heuristics comparisons/o_weight_{opponent_weight}/pt_6_6_4_p4_v10/board 1 truncated (last move - a2).png",
        f"/home/lirontyomkin/AlphaZero_Gomoku/models vs heuristics comparisons/o_weight_{opponent_weight}/pt_6_6_4_p4_v10/board 1 truncated (last move - f3).png"]

    listofimages_truncated_2 = [
        f"/home/lirontyomkin/AlphaZero_Gomoku/models vs heuristics comparisons/o_weight_{opponent_weight}/pt_6_6_4_p3_v7/board 2 truncated.png",
        f"/home/lirontyomkin/AlphaZero_Gomoku/models vs heuristics comparisons/o_weight_{opponent_weight}/pt_6_6_4_p3_v9/board 2 truncated.png",
        f"/home/lirontyomkin/AlphaZero_Gomoku/models vs heuristics comparisons/o_weight_{opponent_weight}/pt_6_6_4_p4_v10/board 2 truncated (last move - b2).png",
        f"/home/lirontyomkin/AlphaZero_Gomoku/models vs heuristics comparisons/o_weight_{opponent_weight}/pt_6_6_4_p4_v10/board 2 truncated (last move - d6).png"]

    path = f"/home/lirontyomkin/AlphaZero_Gomoku/models vs heuristics comparisons/o_weight_{opponent_weight}/"

    create_collages_boards(listofimages=listofimages_empty, fig_name="empty board all models", path=path)
    create_collages_boards(listofimages=listofimages_full_1, fig_name="board 1 full all models", path=path)
    create_collages_boards(listofimages=listofimages_full_2, fig_name="board 2 full all models", path=path)
    create_collages_boards(listofimages=listofimages_truncated_1, fig_name="board 1 truncated all models", path=path)
    create_collages_boards(listofimages=listofimages_truncated_2, fig_name="board 2 truncated all models", path=path)



# def model_corr_emd_board(model_name,
#                          input_plains_num,
#                          game_board,
#                          curr_player=1,
#                          max_model_iter = 5000,
#                          model_check_freq=50,
#                          width=6,height=6,n=4,
#                          c_puct=5,n_playout=400):
#
#
#     board_state, board_name, p1, p2, _, _ = game_board
#     correct_last_board = initialize_board(height, width, input_board=board_state, n_in_row=n, last_move_p1=p1, last_move_p2=p2)
#
#     for i in range(model_check_freq, max_model_iter + model_check_freq, model_check_freq):
#         for row_hat, col_hat in np.where(board_state == curr_player):
#             row = height - 1 - row_hat
#             col = col_hat
#
#             if curr_player == 1:
#                 alt_p1 = [row, col]
#                 alt_p2 = p2
#
#             else:
#                 alt_p1 = p1
#                 alt_p2 = [row, col]
#
#             EMD_between_two_models_on_board(
#                 model1_name=model_name, input_plains_num_1=input_plains_num, i1=i,
#                 model2_name=model_name, input_plains_num_2=input_plains_num, i2=j,
#                 board1=board1, board2=board2, width=width, height=height)


if __name__ == "__main__":

    # generate_models_emd_comparison()
    #
    BOARDS = [BOARD_1_FULL, BOARD_2_FULL, BOARD_1_TRUNCATED, BOARD_2_TRUNCATED, EMPTY_BOARD]
    opponent_weight = 1


    for game_board in BOARDS:
        compare_model_to_heuristics(model_name='pt_6_6_4_p4_v10',
                                    input_plains_num=4,
                                    model_check_freq=50,
                                    max_model_iter=5000,
                                    tell_last_move=True,
                                    game_board=game_board,
                                    n=4, width=6, height=6,
                                    opponent_weigh=opponent_weight)

        compare_model_to_heuristics(model_name='pt_6_6_4_p4_v10',
                                    input_plains_num=4,
                                    model_check_freq=50,
                                    max_model_iter=5000,
                                    tell_last_move=False,
                                    game_board=game_board,
                                    n=4, width=6, height=6,
                                    opponent_weight=opponent_weight)

        compare_model_to_heuristics(model_name='pt_6_6_4_p3_v7',
                                    input_plains_num=3,
                                    model_check_freq=50,
                                    max_model_iter=5000,
                                    tell_last_move=True,
                                    game_board=game_board,
                                    n=4, width=6, height=6,
                                    opponent_weight=opponent_weight)

        compare_model_to_heuristics(model_name='pt_6_6_4_p3_v9',
                                    input_plains_num=3,
                                    model_check_freq=50,
                                    max_model_iter=5000,
                                    tell_last_move=True,
                                    game_board=game_board,
                                    n=4, width=6, height=6,
                                    opponent_weight=opponent_weight)

    for BOARD in BOARDS:
        heuristics_heatmaps(BOARD, opponent_weight=opponent_weight)


    compare_model_to_heuristics(model_name='pt_6_6_4_p4_v10',
                                input_plains_num=4,
                                model_check_freq=50,
                                max_model_iter=5000,
                                tell_last_move=True,
                                game_board=BOARD_1_TRUNCATED,
                                n=4, width=6, height=6,
                                opponent_weight=opponent_weight)

    compare_model_to_heuristics(model_name='pt_6_6_4_p4_v10',
                                input_plains_num=4,
                                model_check_freq=50,
                                max_model_iter=5000,
                                tell_last_move=True,
                                game_board=BOARD_2_TRUNCATED,
                                n=4, width=6, height=6,
                                opponent_weight=opponent_weight)

    call_collage_compare_to_heuristics(opponent_weight)
