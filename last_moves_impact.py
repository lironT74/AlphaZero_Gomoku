from multiprocessing import Pool
from Game_boards_and_aux import *
import os
import matplotlib.pyplot as plt
import io
import PIL



discription_dict_last_move = {
    "pt_6_6_4_p4_v10" : "v10",
    "pt_6_6_4_p4_v23": "v23\nsim:50\nshutter:1\nfull:yes\niter:5000",
    "pt_6_6_4_p4_v24": "v24\nsim:50\nshutter:0\nfull:yes\niter:5000",
    "pt_6_6_4_p4_v25": "v25\nsim:50\nshutter:1\nfull:no\niter:5000",
    "pt_6_6_4_p4_v26": "v26\nsim:50\nshutter:0\nfull:no\niter:5000",
    "pt_6_6_4_p4_v27": "v27\nsim:25\nshutter:1\nfull:yes\niter:5000",
    "pt_6_6_4_p4_v28": "v28\nsim:25\nshutter:0\nfull:yes\niter:5000",
    "pt_6_6_4_p4_v29": "v29\nsim:25\nshutter:1\nfull:no\niter:5000",
    "pt_6_6_4_p4_v30": "v30\nsim:25\nshutter:0\nfull:no\niter:5000",
    "pt_6_6_4_p4_v31": "v31\nsim:100\nshutter:1\nfull:yes\niter:5000",
    "pt_6_6_4_p4_v32": "v32\nsim:100\nshutter:0\nfull:yes\niter:5000",
    "pt_6_6_4_p4_v33": "v33\nsim:100\nshutter:1\nfull:no\niter:5000",
    "pt_6_6_4_p4_v34": "v34\nsim:100\nshutter:0\nfull:no\niter:5000"
}

def model_stat_emd_board(model_name,
                        input_plains_num,
                        game_board,
                        curr_player=1,
                        max_model_iter = 5000,
                        model_check_freq=50,
                        width=6, height=6, n=4,
                        stat = np.var,
                        stat_name="Varience"):

    models_num = max_model_iter // model_check_freq
    model_list = range(model_check_freq, max_model_iter + model_check_freq, model_check_freq)

    board_state, board_name, p1, p2, _, _ = game_board
    correct_last_board = initialize_board_with_init_and_last_moves(height, width, input_board=board_state, n_in_row=n, last_move_p1=p1, last_move_p2=p2)

    rows, cols = np.where(board_state == curr_player)
    rows = list(height - 1 - rows)
    cols = list(cols)

    stat_list = []
    for i in range(model_check_freq, max_model_iter + model_check_freq, model_check_freq):

        print(i)
        emd_list = []
        for row, col in zip(rows, cols):

            if curr_player == 1 and [row, col] == p1 or curr_player==2 and [row, col] == p2:
                continue

            if curr_player == 1:
                alt_p1 = [row, col]
                alt_p2 = p2
                print(f"changed p1 last move: {alt_p1}")


            else:
                print("changed p2 last move")
                alt_p1 = p1
                alt_p2 = [row, col]

            alt_board = initialize_board_with_init_and_last_moves(height, width, input_board=board_state, n_in_row=n, last_move_p1=alt_p1, last_move_p2=alt_p2)

            emd_list.append(EMD_between_two_models_on_board(
                model1_name=model_name, input_plains_num_1=input_plains_num, i1=i,
                model2_name=model_name, input_plains_num_2=input_plains_num, i2=i,
                board1=correct_last_board, board2=alt_board, width=width, height=height))

        stat_list.append(stat(emd_list))


    return (stat_list, models_num, model_list, model_name, board_name, stat_name)

    # save_fig_stat(stat_list, models_num, model_list, model_name, board_name, stat_name)


def save_fig_stat(stat_list, models_num, model_list, model_name, board_name, stat_name, y_top_lim, shutter_limit):

    path = f"/home/lirontyomkin/AlphaZero_Gomoku/last move impact/shutter {shutter_limit}/{stat_name}/{board_name}/"

    if not os.path.exists(path):
        os.makedirs(path)

    fig, (ax, lax) = plt.subplots(nrows=2, gridspec_kw={"height_ratios": [20, 1]}, figsize=(30, 10))

    fontsize = 17
    linewidth = 3


    ax.plot(range(models_num), stat_list, color="blue", linewidth=linewidth)


    ax.set_ylim([0, y_top_lim])


    ax.set_xticks(range(models_num))
    ax.set_xticklabels(model_list, rotation=90, fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.set_xlabel("sub model no.", fontsize=fontsize)
    ax.set_title(f"{discription_dict_last_move[model_name]}\n{stat_name} of distances between the policy with the correct last move to rest\n of the policies with all the other possible last moves on {board_name}",
                 fontdict={'fontsize': fontsize + 15})

    h, l = ax.get_legend_handles_labels()
    # lax.legend(h, l, borderaxespad=0, loc="center", fancybox=True, shadow=True, ncol=1, fontsize=fontsize + 5)
    lax.axis("off")

    fig.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)

    plt.savefig(f"{path}{model_name}_{board_name}.png")

    plt.close('all')


def save_figs_stat(distances_list, y_top_lim, shutter_limit):

    fig, (ax, lax) = plt.subplots(nrows=2, gridspec_kw={"height_ratios": [20, 4]}, figsize=(40, 12.5))

    fontsize = 17
    linewidth = 3

    from matplotlib import cm

    colors = iter(cm.rainbow(np.linspace(0, 1, len(distances_list))))


    for stat_list, models_num, model_list, model_name, board_name, stat_name in distances_list:
        ax.plot(range(models_num), stat_list, linewidth=linewidth, color = next(colors), label=discription_dict_last_move[model_name])


    ax.set_ylim([0, y_top_lim])

    _, models_num, model_list, _, board_name, stat_name = distances_list[0]

    if shutter_limit != -1:
        path = f"/home/lirontyomkin/AlphaZero_Gomoku/last move impact/shutter {shutter_limit}/{stat_name}/{board_name}/"
    else:

        path = f"/home/lirontyomkin/AlphaZero_Gomoku/last move impact/{stat_name}/{board_name}/"

    if not os.path.exists(path):
        os.makedirs(path)


    ax.set_xticks(range(models_num))
    ax.set_xticklabels(model_list, rotation=90, fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.set_xlabel("sub model no.", fontsize=fontsize)

    which_models = "All models"
    if shutter_limit == 0:
        which_models = "0 shutter models"
    elif shutter_limit == 1:
        which_models = "1 shutter models"

    ax.set_title(f"{which_models}\n{stat_name} of distances between the policy with the correct last move to rest\n of the policies with all the other possible last moves on {board_name}",
                 fontdict={'fontsize': fontsize + 15})


    h, l = ax.get_legend_handles_labels()
    lax.legend(h, l, borderaxespad=0, loc="center", fancybox=True, shadow=True, ncol=len(distances_list), fontsize=fontsize + 3)
    lax.axis("off")

    fig.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)

    plt.savefig(f"{path}All models.png")

    plt.close('all')


def last_move_aux(board, stat_name, model_names, shutter_limit):

    if stat_name == "Varience":
        stat = np.var
    elif stat_name == "Average":
        stat = np.average
    elif stat_name == "Coefficient of Variation":
        stat = lambda np_array: np.std(np_array) / np.mean(np_array)


    distances_list = []
    for model_name in model_names:
        distances_list.append(model_stat_emd_board(model_name,
                                                   4,
                                                   board,
                                                   curr_player=1,
                                                   max_model_iter=5000,
                                                   model_check_freq=50,
                                                   width=6, height=6, n=4,
                                                   stat=stat,
                                                   stat_name=stat_name
                                                   ))

    y_top_lim = max([max(stat_list) for stat_list, _, _, _, _, _ in distances_list])
    y_top_lim = 1.1 * y_top_lim

    for stat_list, models_num, model_list, model_name, board_name, stat_name in distances_list:
        save_fig_stat(stat_list, models_num, model_list, model_name, board_name, stat_name, y_top_lim, shutter_limit)

    save_figs_stat(distances_list, y_top_lim, shutter_limit)


def last_move_aux_shutter_separate(board, stat_name, models_0_names, models_1_names):

    if stat_name == "Varience":
        stat = np.var
    elif stat_name == "Average":
        stat = np.average
    elif stat_name == "Coefficient of Variation":
        stat = lambda np_array: np.std(np_array) / np.mean(np_array)


    distances_list_0 = []
    for model_name in models_0_names:
        distances_list_0.append(model_stat_emd_board(model_name,
                                                   4,
                                                   board,
                                                   curr_player=1,
                                                   max_model_iter=5000,
                                                   model_check_freq=50,
                                                   width=6, height=6, n=4,
                                                   stat=stat,
                                                   stat_name=stat_name
                                                   ))

    distances_list_1 = []
    for model_name in models_1_names:
        distances_list_1.append(model_stat_emd_board(model_name,
                                                   4,
                                                   board,
                                                   curr_player=1,
                                                   max_model_iter=5000,
                                                   model_check_freq=50,
                                                   width=6, height=6, n=4,
                                                   stat=stat,
                                                   stat_name=stat_name
                                                   ))



    y_top_lim = max(max([max(stat_list) for stat_list, _, _, _, _, _ in distances_list_0]),
                    max([max(stat_list) for stat_list, _, _, _, _, _ in distances_list_1]))
    y_top_lim = 1.1 * y_top_lim


    for stat_list, models_num, model_list, model_name, board_name, stat_name in distances_list_0:
        save_fig_stat(stat_list, models_num, model_list, model_name, board_name, stat_name, y_top_lim, 0)

    save_figs_stat(distances_list_0, y_top_lim, 0)


    for stat_list, models_num, model_list, model_name, board_name, stat_name in distances_list_1:
        save_fig_stat(stat_list, models_num, model_list, model_name, board_name, stat_name, y_top_lim, 1)

    save_figs_stat(distances_list_0, y_top_lim, 1)


if __name__ == '__main__':

    # model_names = ['pt_6_6_4_p4_v10',
    #                'pt_6_6_4_p4_v27', 'pt_6_6_4_p4_v28', 'pt_6_6_4_p4_v29', 'pt_6_6_4_p4_v30',
    #                'pt_6_6_4_p4_v23', 'pt_6_6_4_p4_v24', 'pt_6_6_4_p4_v25', 'pt_6_6_4_p4_v26',
    #                'pt_6_6_4_p4_v31', 'pt_6_6_4_p4_v32', 'pt_6_6_4_p4_v33', 'pt_6_6_4_p4_v34'
    #                ]

    models_1_names = ['pt_6_6_4_p4_v10',
                   'pt_6_6_4_p4_v27', 'pt_6_6_4_p4_v29',
                   'pt_6_6_4_p4_v23', 'pt_6_6_4_p4_v25',
                   'pt_6_6_4_p4_v31', 'pt_6_6_4_p4_v33']

    models_0_names = ['pt_6_6_4_p4_v10',
                  'pt_6_6_4_p4_v28', 'pt_6_6_4_p4_v30',
                  'pt_6_6_4_p4_v24', 'pt_6_6_4_p4_v26',
                  'pt_6_6_4_p4_v32', 'pt_6_6_4_p4_v34'
                  ]


    stats = ["Coefficient of Variation", "Varience", "Average"]

    jobs = []

    with Pool() as pool:
        for stat_name in stats:
            for board in PAPER_6X6_TRUNCATED_BOARDS:
                jobs.append((board, stat_name, models_0_names, models_1_names))

        pool.starmap(last_move_aux_shutter_separate, jobs)
        pool.close()
        pool.join()

