from multiprocessing import Pool
from Game_boards_and_aux import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import matplotlib.pyplot as plt
from policy_value_net_pytorch import PolicyValueNet
import io
import PIL
import string
import copy


def compare_model_to_heuristics(model_name, game_board, n=4, width=6, height=6, opponent_weight=0.5, threshold=0.05, max_radius_density = 2, **kwargs):

    input_plains_num = kwargs.get("input_plains_num", 4)
    max_model_iter = kwargs.get("max_model_iter", 5000)
    model_check_freq = kwargs.get("model_check_freq", 50)
    tell_last_move = kwargs.get("tell_last_move", True)


    dist_matrix = generate_matrix_dist_metric(6)
    board_state, board_name, last_move_p1, last_move_p2, alternative_p1, alternative_p2 = copy.deepcopy(game_board)


    if tell_last_move:
        board = initialize_board_with_init_and_last_moves(height, width, input_board=board_state, n_in_row=n, last_move_p1=last_move_p1,
                                                          last_move_p2=last_move_p2)
    else:
        board = initialize_board_with_init_and_last_moves(height, width, input_board=board_state, n_in_row=n, last_move_p1=alternative_p1,
                                                          last_move_p2=alternative_p2)


    heuristics_scores = threshold_normalization_heuristics(board, opponent_weight, max_radius_density=max_radius_density, rounding=-1, threshold=threshold)


    models_num = max_model_iter//model_check_freq
    model_list = range(model_check_freq, max_model_iter+model_check_freq, model_check_freq)

    distances_lists = {"density": np.zeros(models_num), "linear": np.zeros(models_num),
                      "nonlinear": np.zeros(models_num), "interaction": np.zeros(models_num),
                      "interaction with forcing": np.zeros(models_num)}

    distances_base_models = {"density": 0, "linear": 0,
                       "nonlinear": 0, "interaction": 0,
                       "interaction with forcing": 0}

    for index_i, i in enumerate(model_list):

        move_probs_policy = threshold_normalization_policy(board, model_name, input_plains_num, i, rounding=-1, threshold=threshold)

        for key in heuristics_scores.keys():

            distance = emd(np.asarray(np.reshape(move_probs_policy, width*height), dtype='float64'),
                           np.asarray(np.reshape(heuristics_scores[key], width*height), dtype='float64'),
                           dist_matrix)

            distances_lists[key][index_i] = distance

    for key in heuristics_scores.keys():

        move_probs_policy = threshold_normalization_policy(model_name="base_model", board=board, i=-1, model_file=f'/home/lirontyomkin/AlphaZero_Gomoku/models/best_policy_6_6_4.model', input_plains_num=4, rounding=-1, threshold=threshold)

        distances_base_models[key] = emd(np.asarray(np.reshape(move_probs_policy, width*height), dtype='float64'),
                                       np.asarray(np.reshape(heuristics_scores[key], width*height), dtype='float64'),
                                       dist_matrix)



    fig, (ax, lax) = plt.subplots(nrows=2, gridspec_kw={"height_ratios": [20, 1]}, figsize=(30,10))

    fontsize = 17
    linewidth=3

    ax.plot(range(models_num), distances_lists["density"], label=f"density", color="blue", linewidth=linewidth)
    ax.plot(range(models_num), distances_lists["linear"], label=f"linear", color="red", linewidth=linewidth)
    ax.plot(range(models_num), distances_lists["nonlinear"], label=f"non-linear", color="green", linewidth=linewidth)
    ax.plot(range(models_num), distances_lists["interaction"], label=f"interaction", color="orange", linewidth=linewidth)
    ax.plot(range(models_num), distances_lists["interaction with forcing"], label=f"forcing", color="black", linewidth=linewidth)

    ax.scatter(models_num + 1, distances_base_models["density"], marker='o', label=f"base model - density", color="blue", linewidth=2*linewidth)
    ax.scatter(models_num + 2, distances_base_models["linear"], marker='o', label=f"base model - linear", color="red", linewidth=2*linewidth)
    ax.scatter(models_num + 3, distances_base_models["nonlinear"], marker='o', label=f"base model - non-linear", color="green", linewidth=2*linewidth)
    ax.scatter(models_num + 4, distances_base_models["interaction"], marker='o', label=f"base model - interaction", color="orange", linewidth=2*linewidth)
    ax.scatter(models_num + 5, distances_base_models["interaction with forcing"], marker='o', label=f"base model - forcing", color="black", linewidth=2*linewidth)

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


    ax.set_title(f"{model_name}{last_move} EMD distances from heuristics \no_weight={opponent_weight}, threshold={threshold} on {board_name}", fontdict={'fontsize': fontsize+15})

    h, l = ax.get_legend_handles_labels()

    ord = [0,5,1,6,2,7,3,8,4,9]

    lax.legend([h[idx] for idx in ord],[l[idx] for idx in ord], borderaxespad=0, loc="center", fancybox=True, shadow=True, ncol=5, fontsize=fontsize+5)
    lax.axis("off")

    fig.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)

    path = f"/home/lirontyomkin/AlphaZero_Gomoku/models vs heuristics comparisons/o_weight_{opponent_weight}/threshold_{threshold}/{model_name}/"

    if not os.path.exists(path):
        os.makedirs(path)

    if last_move != "":
        plt.savefig(f"{path}{board_name}{last_move}.png")
    else:
        plt.savefig(f"{path}{board_name}.png")

    plt.close('all')


def heuristics_heatmaps(game_board, height=6, width=6, n=4, opponent_weight=0.5, threshold=0.05, max_radius_density=2):

    board_state, board_name, last_move_p1, last_move_p2, alternative_p1, alternative_p2 = game_board
    board = initialize_board_with_init_and_last_moves(height, width, input_board=board_state, n_in_row=n, last_move_p1=last_move_p1,
                                                      last_move_p2=last_move_p2)

    heuristics_scores = threshold_normalization_heuristics(board=board, opponent_weight=opponent_weight, max_radius_density=max_radius_density, rounding=3, threshold=threshold)

    x_positions = board.current_state()[0]
    o_positions = board.current_state()[1]

    fontsize = 15
    fig, axes = plt.subplots(1, 5, figsize=(35, 8))
    fig.suptitle(f"Heuristics heatmaps on {board_name}, o_weight = {opponent_weight}, threshold = {threshold}", fontsize=fontsize + 10)
    x_axis = [letter for i, letter in zip(range(width), string.ascii_lowercase)]
    y_axis = range(height, 0, -1)
    sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=1))

    for index, key in enumerate(heuristics_scores.keys()):
        ax = axes[index]

        move_probs_policy = heuristics_scores[key]

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

        ax.set_title(f"{key}", fontsize=fontsize + 4)

    fig.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)

    path = f"/home/lirontyomkin/AlphaZero_Gomoku/models vs heuristics comparisons/o_weight_{opponent_weight}/threshold_{threshold}/heuristics_heatmaps/"

    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(f"{path}{board_name}.png")
    plt.close('all')


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


def call_collage_compare_to_heuristics(opponent_weight, threshold):

    path = f"/home/lirontyomkin/AlphaZero_Gomoku/models vs heuristics comparisons/o_weight_{opponent_weight}/threshold_{threshold}/"

    listofimages_empty = [
        f"{path}/pt_6_6_4_p3_v7/empty board.png",
        f"{path}/pt_6_6_4_p3_v9/empty board.png",
        f"{path}/pt_6_6_4_p4_v10/empty board.png",
    ]

    listofimages_full_1 = [
        f"{path}/pt_6_6_4_p3_v7/board 1 full.png",
        f"{path}/pt_6_6_4_p3_v9/board 1 full.png",
        f"{path}/pt_6_6_4_p4_v10/board 1 full (last move - b6).png",
        ]

    listofimages_full_2 = [
        f"{path}/pt_6_6_4_p3_v7/board 2 full.png",
        f"{path}/pt_6_6_4_p3_v9/board 2 full.png",
        f"{path}/pt_6_6_4_p4_v10/board 2 full (last move - e6).png",
    ]

    listofimages_truncated_1 = [
        f"{path}/pt_6_6_4_p3_v7/board 1 truncated.png",
        f"{path}/pt_6_6_4_p3_v9/board 1 truncated.png",
        f"{path}/pt_6_6_4_p4_v10/board 1 truncated (last move - a2).png",
        f"{path}/pt_6_6_4_p4_v10/board 1 truncated (last move - f3).png"]

    listofimages_truncated_2 = [
        f"{path}/pt_6_6_4_p3_v7/board 2 truncated.png",
        f"{path}/pt_6_6_4_p3_v9/board 2 truncated.png",
        f"{path}/pt_6_6_4_p4_v10/board 2 truncated (last move - b2).png",
        f"{path}/pt_6_6_4_p4_v10/board 2 truncated (last move - d6).png"]


    create_collages_boards(listofimages=listofimages_empty, fig_name="empty board all models", path=path)
    create_collages_boards(listofimages=listofimages_full_1, fig_name="board 1 full all models", path=path)
    create_collages_boards(listofimages=listofimages_full_2, fig_name="board 2 full all models", path=path)
    create_collages_boards(listofimages=listofimages_truncated_1, fig_name="board 1 truncated all models", path=path)
    create_collages_boards(listofimages=listofimages_truncated_2, fig_name="board 2 truncated all models", path=path)


def threshold_normalization_heuristics(board, opponent_weight, max_radius_density ,rounding=-1, threshold = 0.05):

    heuristics_scores = board.calc_all_heuristics(max_radius_density=max_radius_density, normalize_all_heuristics=True, opponent_weight=opponent_weight)

    for key in heuristics_scores.keys():
        if threshold < 1:
            heuristics_scores[key][heuristics_scores[key] < threshold] = 0

        heuristics_scores[key] = normalize_matrix(heuristics_scores[key], board, rounding)

    return heuristics_scores


def threshold_normalization_policy(board, model_name, input_plains_num, i, rounding=-1, threshold = 0.05, model_file=None):

    width, height = board.width, board.height

    if model_file is None:
        model_file = f'/home/lirontyomkin/AlphaZero_Gomoku/models/{model_name}/current_policy_{i}.model'

    policy = PolicyValueNet(width, height, model_file=model_file, input_plains_num=input_plains_num)
    acts_policy, probas_policy = zip(*policy.policy_value_fn(board)[0])

    move_probs_policy = np.zeros(width * height)
    move_probs_policy[list(acts_policy)] = probas_policy
    move_probs_policy = move_probs_policy.reshape(width, height)
    move_probs_policy = np.flipud(move_probs_policy)

    if threshold < 1:
        move_probs_policy[move_probs_policy < threshold] = 0
        move_probs_policy = normalize_matrix(move_probs_policy, board, rounding)


    return move_probs_policy


def normalize_matrix(scores, board, rounding):

    width, height = board.width, board.height
    board_state = board.current_state()

    cur_positions = np.flipud(board_state[0])
    opponent_positions = np.flipud(board_state[1])

    sum = np.sum(scores)
    counter_positive_values = len(np.where(scores > 0)[0])

    if sum != 0:
        if rounding != -1:
            if not isinstance(rounding, int):
                raise Exception("rounding parameter sent is not integer")

            return np.round(scores/sum, rounding)

        else:
            return scores / sum

    #all zeros:
    counter_not_X_O = width*height - len(np.where(cur_positions == 1)[0]) - len(np.where(opponent_positions == 1)[0])

    for col in range(width):
        for row in range(height):

            if cur_positions[row, col] or opponent_positions[row, col]:
                continue
            else:
                scores[row,col] = 1/counter_not_X_O

                if rounding != -1:
                    if not isinstance(rounding, int):
                        raise Exception("rounding parameter sent is not integer")

                    scores[row, col] =  np.round(scores[row,col], rounding)

    return scores


def run_heuristics_for_threshold_and_weight(opponent_weight, threshold):
    BOARDS = [BOARD_1_FULL, BOARD_2_FULL, BOARD_1_TRUNCATED, BOARD_2_TRUNCATED, EMPTY_BOARD]

    for game_board in BOARDS:
        compare_model_to_heuristics(model_name='pt_6_6_4_p4_v10',
                                    input_plains_num=4,
                                    model_check_freq=50,
                                    max_model_iter=5000,
                                    tell_last_move=True,
                                    game_board=game_board,
                                    n=4, width=6, height=6,
                                    opponent_weight=opponent_weight,
                                    threshold=threshold)

        compare_model_to_heuristics(model_name='pt_6_6_4_p4_v10',
                                    input_plains_num=4,
                                    model_check_freq=50,
                                    max_model_iter=5000,
                                    tell_last_move=False,
                                    game_board=game_board,
                                    n=4, width=6, height=6,
                                    opponent_weight=opponent_weight,
                                    threshold=threshold)

        compare_model_to_heuristics(model_name='pt_6_6_4_p3_v7',
                                    input_plains_num=3,
                                    model_check_freq=50,
                                    max_model_iter=5000,
                                    tell_last_move=True,
                                    game_board=game_board,
                                    n=4, width=6, height=6,
                                    opponent_weight=opponent_weight,
                                    threshold=threshold)

        compare_model_to_heuristics(model_name='pt_6_6_4_p3_v9',
                                    input_plains_num=3,
                                    model_check_freq=50,
                                    max_model_iter=5000,
                                    tell_last_move=True,
                                    game_board=game_board,
                                    n=4, width=6, height=6,
                                    opponent_weight=opponent_weight,
                                    threshold=threshold)

    for board in BOARDS:
        heuristics_heatmaps(board, height=6, width=6, n=4, opponent_weight=opponent_weight, threshold=threshold)

    call_collage_compare_to_heuristics(opponent_weight=opponent_weight, threshold=threshold)


def run_heuristics_for_thresholds_and_o_weights(thresholds, o_weights):

    with Pool(len(thresholds)*len(o_weights)) as pool:
        jobs = []
        for threshold in thresholds:
            for o_weight in o_weights:
                jobs.append((o_weight, threshold))

        print(f"Using {pool._processes} workers. There are {len(jobs)} jobs: \n")
        pool.starmap(run_heuristics_for_threshold_and_weight, jobs)
        pool.close()
        pool.join()



if __name__ == "__main__":
    thresholds = [1, 0.05, 0.01]
    o_weights = [0, 0.2, 0.5, 0.7, 1]
    run_heuristics_for_thresholds_and_o_weights(thresholds, o_weights)

