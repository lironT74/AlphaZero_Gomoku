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


def compare_model_to_heuristics(path, model_name, game_board, n=4, width=6, height=6, opponent_weight=0.5, cut_off_threshold=0.05, max_radius_density = 2, **kwargs):

    input_plains_num = kwargs.get("input_plains_num", 4)
    max_model_iter = kwargs.get("max_model_iter", 5000)
    model_check_freq = kwargs.get("model_check_freq", 50)
    is_random_last_turn = kwargs.get("is_random_last_turn", False)
    open_path_threshold = kwargs.get("open_path_threshold", 0)


    dist_matrix = generate_matrix_dist_metric(6)
    board_state, board_name, last_move_p1, last_move_p2, alternative_p1, alternative_p2 = copy.deepcopy(game_board)


    if board_name == "empty board":
        start_player = 1
    else:
        start_player = 2


    board = initialize_board_with_init_and_last_moves(height, width, input_board=board_state,
                                                      n_in_row=n, last_move_p1=last_move_p1, last_move_p2=last_move_p2,
                                                      open_path_threshold=open_path_threshold, start_player=start_player)


    heuristics_scores = threshold_cutoff_heuristics(board, opponent_weight, max_radius_density=max_radius_density, rounding=-1,
                                                    threshold=cut_off_threshold, board_name=board_name)


    models_num = max_model_iter//model_check_freq
    model_list = range(model_check_freq, max_model_iter+model_check_freq, model_check_freq)
    distances_lists = {key: np.zeros(models_num) for key in heuristics_scores.keys()}

    distances_base_models = {key: 0 for key in heuristics_scores.keys()}

    for index_i, i in enumerate(model_list):

        move_probs_policy = threshold_cutoff_policy(board=board, board_name=board_name, model_name=model_name,
                                                    input_plains_num=input_plains_num, model_iteration=i, rounding=-1,
                                                    cutoff_threshold=cut_off_threshold, is_random_last_turn=is_random_last_turn)

    #     for key in heuristics_scores.keys():
    #
    #         distance = emd(np.asarray(np.reshape(move_probs_policy, width*height), dtype='float64'),
    #                        np.asarray(np.reshape(heuristics_scores[key], width*height), dtype='float64'),
    #                        dist_matrix)
    #
    #         distances_lists[key][index_i] = distance
    #
    #
    # for key in heuristics_scores.keys():
    #
    #     move_probs_policy = threshold_cutoff_policy(model_name="base_model", board_name=board_name, board=board,
    #                                                 model_iteration=-1,
    #                                                 model_file=f'/home/lirontyomkin/AlphaZero_Gomoku/models/best_policy_6_6_4.model',
    #                                                 input_plains_num=4, rounding=-1, cutoff_threshold=cut_off_threshold)
    #
    #
    #     distances_base_models[key] = emd(np.asarray(np.reshape(move_probs_policy, width*height), dtype='float64'),
    #                                    np.asarray(np.reshape(heuristics_scores[key], width*height), dtype='float64'),
    #                                    dist_matrix)
    #
    #
    #
    # fig, (ax, lax) = plt.subplots(nrows=2, gridspec_kw={"height_ratios": [20, 1]}, figsize=(30,10))
    #
    # fontsize = 16
    # linewidth=3
    #
    # model_name = f"{model_name}_random" if is_random_last_turn else model_name
    #
    #
    # colors =  {"density": "blue",
    #            "linear": "red",
    #            "nonlinear": "green",
    #            "interaction": "orange",
    #            "interaction with forcing": "black",
    #            "people": "fuchsia"}
    #
    #
    # for index, key in enumerate(distances_lists.keys()):
    #
    #     ax.plot(range(models_num), distances_lists[key], label=f"{key}", color=colors[key], linewidth=linewidth)
    #
    #     ax.scatter(models_num + index + 1, distances_base_models[key], marker='o', label=f"(base model)",
    #                color=colors[key], linewidth=2 * linewidth)
    #
    #
    # ax.set_xticks(range(models_num))
    # ax.set_xticklabels(model_list, rotation=90, fontsize=fontsize)
    # ax.tick_params(axis='both', which='major', labelsize=fontsize)
    # ax.set_xlabel("sub model no.", fontsize=fontsize)
    #
    # board_current_state = board.current_state(last_move=True, is_random_last_turn=False)
    #
    # if is_random_last_turn:
    #     last_move = " with random last move"
    #
    # else:
    #     if np.sum(board_current_state[2]) == 1:
    #         y_last_move = 6 - np.where(board_current_state[2] == 1)[0][0]
    #         x_last_move = string.ascii_lowercase[np.where(board_current_state[2] == 1)[1][0]]
    #         last_move = f" (last move - {x_last_move}{y_last_move})"
    #
    #     else:
    #         last_move = " (No last move)"
    #
    #
    # ax.set_title(f"{model_name}{last_move} EMD distances from heuristics \no_weight={opponent_weight}, "
    #              f"cutoff threshold={cut_off_threshold} on {board_name}", fontdict={'fontsize': fontsize + 15})
    #
    #
    # h, l = ax.get_legend_handles_labels()
    #
    # if len(distances_lists.keys()) == 6:
    #     ord = [0,6,1,7,2,8,3,9,4,10,5,11]
    # elif len(distances_lists.keys()) == 5:
    #     ord = [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]
    #
    # lax.legend([h[idx] for idx in ord],[l[idx] for idx in ord], borderaxespad=0, loc="center", fancybox=True, shadow=True, ncol=len(distances_lists.keys()), fontsize=fontsize+5)
    #
    #
    # lax.axis("off")
    # fig.tight_layout()
    #
    # buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    # buf.seek(0)
    # image = PIL.Image.open(buf)
    #
    # path = f"{path}{model_name}/"
    #
    # if not os.path.exists(path):
    #     os.makedirs(path)
    #
    # plt.savefig(f"{path}{board_name}.png")
    #
    # plt.close('all')


def heuristics_heatmaps(game_board, path, height=6, width=6, n=4, opponent_weight=0.5, cutoff_threshold=0.05, max_radius_density=2, open_path_threshold=0):

    board_state, board_name, last_move_p1, last_move_p2, alternative_p1, alternative_p2 = game_board
    board = initialize_board_with_init_and_last_moves(height, width, input_board=board_state, n_in_row=n, last_move_p1=last_move_p1,
                                                      last_move_p2=last_move_p2, open_path_threshold=open_path_threshold)

    heuristics_scores = threshold_cutoff_heuristics(board=board, opponent_weight=opponent_weight, max_radius_density=max_radius_density, rounding=3, threshold=cutoff_threshold, board_name=board_name)

    x_positions = board.current_state()[0]
    o_positions = board.current_state()[1]

    cmap="Reds"

    if len(heuristics_scores.keys()) == 6:
        fontsize = 25

        fig = plt.figure(constrained_layout=False)
        fig.set_size_inches(46, 25)

        fig.suptitle( f"\nHeuristics heatmaps on {board_name}, o_weight = {opponent_weight}, "
                      f"cutoff threshold = {cutoff_threshold}", fontsize=fontsize + 15)

        x_axis = [letter for i, letter in zip(range(width), string.ascii_lowercase)]
        y_axis = range(height, 0, -1)

        grid = fig.add_gridspec(nrows=5, ncols=5, height_ratios=[20,0.1,20,0.1,1], width_ratios=[2,10,10,10,2])

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))

        for i, key in enumerate(heuristics_scores.keys()):

            ax = fig.add_subplot(grid[int(i>2)*2,1+(i%3)])

            move_probs_policy = heuristics_scores[key]

            im1 = ax.imshow(move_probs_policy, cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))

            ax.set_xticks(np.arange(len(x_axis)))
            ax.set_yticks(np.arange(len(y_axis)))

            ax.set_xticklabels(x_axis, fontsize=fontsize)
            ax.set_yticklabels(y_axis, fontsize=fontsize)
            plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")

            for i in range(len(y_axis)):
                for j in range(len(x_axis)):
                    color = "black" if move_probs_policy[i, j] < 0.55 else "white"
                    text = ax.text(j, i, "X" if x_positions[i, j] == 1 else (
                        "O" if o_positions[i, j] == 1 else move_probs_policy[i, j]),
                                   ha="center", va="center", color=color, fontsize=fontsize - 3)

            ax.set_title(f"{key}", fontsize=fontsize + 7)

        cbar_ax = fig.add_subplot(grid[4, 1:-1])
        fig.colorbar(sm, cax=cbar_ax, orientation="horizontal").ax.tick_params(labelsize=fontsize + 2)


    if len(heuristics_scores.keys()) == 5:
        fontsize = 23

        fig = plt.figure(constrained_layout=False)
        fig.set_size_inches(47, 10)

        fig.suptitle(
            f"Heuristics heatmaps on {board_name}, o_weight = {opponent_weight}, "
            f"cutoff threshold = {cutoff_threshold}",
            fontsize=fontsize + 8)
        x_axis = [letter for i, letter in zip(range(width), string.ascii_lowercase)]
        y_axis = range(height, 0, -1)

        grid = fig.add_gridspec(nrows=2, ncols=11, height_ratios=[20, 1],
                                width_ratios=[0.3, 15, 0.3, 15, 0.3, 15, 0.3, 15,  0.3, 15, 0.3])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))

        for i, key in enumerate(heuristics_scores.keys()):

            ax = fig.add_subplot(grid[0, i * 2 + 1])

            move_probs_policy = heuristics_scores[key]

            im1 = ax.imshow(move_probs_policy, cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))

            ax.set_xticks(np.arange(len(x_axis)))
            ax.set_yticks(np.arange(len(y_axis)))

            ax.set_xticklabels(x_axis, fontsize=fontsize)
            ax.set_yticklabels(y_axis, fontsize=fontsize)
            plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")

            for i in range(len(y_axis)):
                for j in range(len(x_axis)):
                    text = ax.text(j, i, "X" if x_positions[i, j] == 1 else (
                        "O" if o_positions[i, j] == 1 else move_probs_policy[i, j]),
                                   ha="center", va="center", color="black", fontsize=fontsize - 3)

            ax.set_title(f"{key}", fontsize=fontsize + 5)

        cbar_ax = fig.add_subplot(grid[1, 1:-1])
        fig.colorbar(sm, cax=cbar_ax, orientation="horizontal").ax.tick_params(labelsize=fontsize + 2)


    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = PIL.Image.open(buf)

    path = f"{path}heuristics_heatmaps/"

    if not os.path.exists(path):
        os.makedirs(path)

    if os.path.exists(f"{path}{board_name}.png"):
        os.remove(f"{path}{board_name}.png")

    plt.savefig(f"{path}{board_name}.png", bbox_inches='tight')

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


def call_collage_compare_to_heuristics(path):


    listofimages_empty = [
        f"{path}pt_6_6_4_p3_v7/empty board.png",
        f"{path}pt_6_6_4_p3_v9/empty board.png",
        f"{path}pt_6_6_4_p4_v10/empty board.png",
        f"{path}pt_6_6_4_p4_v10_random/empty board.png"
    ]

    listofimages_full_1 = [
        f"{path}pt_6_6_4_p3_v7/board 1 full.png",
        f"{path}pt_6_6_4_p3_v9/board 1 full.png",
        f"{path}pt_6_6_4_p4_v10/board 1 full.png",
        f"{path}pt_6_6_4_p4_v10_random/board 1 full.png"
        ]

    listofimages_full_2 = [
        f"{path}pt_6_6_4_p3_v7/board 2 full.png",
        f"{path}pt_6_6_4_p3_v9/board 2 full.png",
        f"{path}pt_6_6_4_p4_v10/board 2 full.png",
        f"{path}pt_6_6_4_p4_v10_random/board 2 full.png"
    ]

    listofimages_truncated_1 = [
        f"{path}pt_6_6_4_p3_v7/board 1 truncated.png",
        f"{path}pt_6_6_4_p3_v9/board 1 truncated.png",
        f"{path}pt_6_6_4_p4_v10/board 1 truncated.png",
        f"{path}pt_6_6_4_p4_v10_random/board 1 truncated.png"]

    listofimages_truncated_2 = [
        f"{path}pt_6_6_4_p3_v7/board 2 truncated.png",
        f"{path}pt_6_6_4_p3_v9/board 2 truncated.png",
        f"{path}pt_6_6_4_p4_v10/board 2 truncated.png",
        f"{path}pt_6_6_4_p4_v10_random/board 2 truncated.png"]


    create_collages_boards(listofimages=listofimages_empty, fig_name="empty board all models", path=path)
    create_collages_boards(listofimages=listofimages_full_1, fig_name="board 1 full all models", path=path)
    create_collages_boards(listofimages=listofimages_full_2, fig_name="board 2 full all models", path=path)
    create_collages_boards(listofimages=listofimages_truncated_1, fig_name="board 1 truncated all models", path=path)
    create_collages_boards(listofimages=listofimages_truncated_2, fig_name="board 2 truncated all models", path=path)


def threshold_cutoff_heuristics(board, opponent_weight, max_radius_density, rounding=-1, threshold = 0.05, board_name="empty board"):

    heuristics_scores = board.calc_all_heuristics(max_radius_density=max_radius_density,
                                                  normalize_all_heuristics=True,
                                                  opponent_weight=opponent_weight)

    if board_name != "empty board":
        heuristics_scores["people"] = get_people_distribution(board_name)

    for key in heuristics_scores.keys():
        if threshold < 1:
            heuristics_scores[key][heuristics_scores[key] < threshold] = 0

        heuristics_scores[key] = normalize_matrix(heuristics_scores[key], board, rounding)

    return heuristics_scores


def get_people_distribution(board_name):
    return PEOPLE_DISTRIBUTIONS[board_name]


def threshold_cutoff_policy(board, board_name, model_name,
                            input_plains_num, model_iteration, rounding=-1,
                            cutoff_threshold = 0.05, model_file=None, is_random_last_turn=False):

    width, height = board.width, board.height

    if model_file is None:
        model_file = f'/home/lirontyomkin/AlphaZero_Gomoku/models/{model_name}/current_policy_{model_iteration}.model'

    policy = PolicyValueNet(width, height, model_file=model_file, input_plains_num=input_plains_num)
    board.set_is_random_last_turn(is_random_last_turn=is_random_last_turn, player=board.get_current_player())

    if is_random_last_turn:
        board.set_random_seed(model_iteration)

    acts_policy, probas_policy = zip(*policy.policy_value_fn(board)[0])

    # print(model_iteration, board.last_move_p1)

    # AlphaZero gives some probability to locations that are not available for some reason
    if np.sum(probas_policy) != 0:
        probas_policy = probas_policy / np.sum(probas_policy)

    move_probs_policy = np.zeros(width * height)
    move_probs_policy[list(acts_policy)] = probas_policy
    move_probs_policy = move_probs_policy.reshape(width, height)
    move_probs_policy = np.flipud(move_probs_policy)

    if cutoff_threshold < 1:
        move_probs_policy[move_probs_policy < cutoff_threshold] = 0
        move_probs_policy = normalize_matrix(move_probs_policy, board, rounding)


    # if is_random_last_turn:
    #     model_name = f"{model_name}_random"
    #
    # heatmap_save_path = f'/home/lirontyomkin/AlphaZero_Gomoku/models_heatmaps/cutoff_threshold_{cutoff_threshold}/{model_name}/iteration_{model_iteration}/'

    # if not os.path.exists(f"{heatmap_save_path}{board_name}.png") and model_name != "base_model":
    #     save_trimmed_policy_heatmap(move_probs_policy, model_name, board, board_name, heatmap_save_path)

    return move_probs_policy


def save_trimmed_policy_heatmap(move_probs_policy, model_name, board, board_name,
                                heatmap_save_path):

    board_current_state = board.current_state(last_move=True)

    move_probs_policy = normalize_matrix(move_probs_policy, board, 3)

    cmap = "Reds"
    x_axis = [letter for i, letter in zip(range(board.width), string.ascii_lowercase)]
    y_axis = range(board.height, 0, -1)

    my_marker = "X" if board.get_current_player() == 1 else "O"

    if board.get_current_player() == 1:
        x_positions = board_current_state[0]
        o_positions = board_current_state[1]

        last_move_ = board.last_move_p1
    else:
        x_positions = board_current_state[1]
        o_positions = board_current_state[0]

        last_move_ = board.last_move_p2


    if last_move_ != 1:
        last_move = f" (last move: {get_printable_move(last_move_, board.width, board.height)})"

    else:
        last_move = " (No last move)"


    fontsize = 38
    fig, ax = plt.subplots(tight_layout=False, figsize=(25, 27))

    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))

    fig.suptitle(f"Model: {model_name}, Policy value function.\nPlays: {my_marker}{last_move}", fontsize=fontsize + 10)

    im = ax.imshow(move_probs_policy, cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))

    divider = make_axes_locatable(ax)
    cax = divider.new_vertical(size="5%", pad=0.7, pack_start=True)
    fig.add_axes(cax)
    fig.colorbar(im, cax=cax, orientation="horizontal").ax.tick_params(labelsize=fontsize + 2)

    ax.set_xticks(np.arange(len(x_axis)))
    ax.set_yticks(np.arange(len(y_axis)))
    ax.set_xticklabels(x_axis, fontsize=fontsize)
    ax.set_yticklabels(y_axis, fontsize=fontsize)

    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    for i in range(len(y_axis)):
        for j in range(len(x_axis)):
            color = "black" if move_probs_policy[i, j] < 0.55 else "white"
            text = ax.text(j, i, "X" if x_positions[i, j] == 1 else (
                "O" if o_positions[i, j] == 1 else move_probs_policy[i, j]),
                           ha="center", va="center", color=color, fontsize=fontsize + 3)


    fig.subplots_adjust(left=0.083, right=1 - 0.083)


    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = PIL.Image.open(buf)


    if not os.path.exists(heatmap_save_path):
        os.makedirs(heatmap_save_path)



    plt.savefig(f"{heatmap_save_path}{board_name}.png")
    plt.close('all')


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


def run_heuristics_for_threshold_and_weight(opponent_weight, cutoff_threshold, path, open_path_threshold=0):

    for game_board in BOARDS:
        # compare_model_to_heuristics(path=path,
        #                             model_name='pt_6_6_4_p4_v10',
        #                             input_plains_num=4,
        #                             model_check_freq=50,
        #                             max_model_iter=5000,
        #                             game_board=game_board,
        #                             n=4, width=6, height=6,
        #                             opponent_weight=opponent_weight,
        #                             cut_off_threshold=cutoff_threshold,
        #                             open_path_threshold=open_path_threshold)

        compare_model_to_heuristics(path=path,
                                    is_random_last_turn=True,
                                    model_name='pt_6_6_4_p4_v10',
                                    input_plains_num=4,
                                    model_check_freq=50,
                                    max_model_iter=5000,
                                    game_board=game_board,
                                    n=4, width=6, height=6,
                                    opponent_weight=opponent_weight,
                                    cut_off_threshold=cutoff_threshold,
                                    open_path_threshold=open_path_threshold)

        # compare_model_to_heuristics(path=path,
        #                             model_name='pt_6_6_4_p3_v7',
        #                             input_plains_num=3,
        #                             model_check_freq=50,
        #                             max_model_iter=5000,
        #                             game_board=game_board,
        #                             n=4, width=6, height=6,
        #                             opponent_weight=opponent_weight,
        #                             cut_off_threshold=cutoff_threshold,
        #                             open_path_threshold=open_path_threshold)
        #
        # compare_model_to_heuristics(path=path,
        #                             model_name='pt_6_6_4_p3_v9',
        #                             input_plains_num=3,
        #                             model_check_freq=50,
        #                             max_model_iter=5000,
        #                             game_board=game_board,
        #                             n=4, width=6, height=6,
        #                             opponent_weight=opponent_weight,
        #                             cut_off_threshold=cutoff_threshold,
        #                             open_path_threshold=open_path_threshold)


    # for board in BOARDS:
    #     heuristics_heatmaps(board, path, height=6, width=6, n=4, opponent_weight=opponent_weight, cutoff_threshold=cutoff_threshold, open_path_threshold=open_path_threshold)

    # call_collage_compare_to_heuristics(path=path)


def run_heuristics_for_thresholds_and_o_weights(cutoff_thresholds, o_weights, open_path_thresholds):

    pool_number = min(20, len(open_path_thresholds*len(cutoff_thresholds)*len(o_weights)))

    with Pool(pool_number) as pool:
        jobs = []

        for open_path_threshold in open_path_thresholds:
            for cutoff_threshold in cutoff_thresholds:
                for opponent_weight in o_weights:

                    path = f"/home/lirontyomkin/AlphaZero_Gomoku/models to heuristics comparisons/open_path_threshold_{open_path_threshold}/o_weight_{opponent_weight}/cutoff_threshold_{cutoff_threshold}/"
                    jobs.append((opponent_weight, cutoff_threshold, path, open_path_threshold))

        print(f"Using {pool._processes} workers. There are {len(jobs)} jobs: \n")
        pool.starmap(run_heuristics_for_threshold_and_weight, jobs)
        pool.close()
        pool.join()



if __name__ == "__main__":

    BOARDS = [EMPTY_BOARD, BOARD_1_FULL, BOARD_2_FULL, BOARD_1_TRUNCATED, BOARD_2_TRUNCATED]
    # cutoff_thresholds = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 1]
    # o_weights = [0, 0.2, 0.5, 0.7, 1]
    # open_path_thresholds = [0, -1]

    # BOARDS = [EMPTY_BOARD]
    cutoff_thresholds = [0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 1]
    o_weights = [0.5]
    open_path_thresholds = [0]

    run_heuristics_for_thresholds_and_o_weights(cutoff_thresholds=cutoff_thresholds, o_weights=o_weights, open_path_thresholds=open_path_thresholds)

