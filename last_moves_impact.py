from Game_boards_and_aux import *
import os
import matplotlib.pyplot as plt
import io
import PIL


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


def save_fig_stat(stat_list, models_num, model_list, model_name, board_name, stat_name, y_top_lim):

    path = f"/home/lirontyomkin/AlphaZero_Gomoku/last move impact/{stat_name}/"

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
    ax.set_title(f"{model_name}\n{stat_name} of distances between the policy with the correct last move to rest\n of the policies with all the other possible last moves on {board_name}",
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





if __name__ == '__main__':

    model_names = ['pt_6_6_4_p4_v10', 'pt_6_6_4_p4_v12', 'pt_6_6_4_p4_v14', 'pt_6_6_4_p4_v16', 'pt_6_6_4_p4_v18',
                   'pt_6_6_4_p4_v20', 'pt_6_6_4_p4_v22']

    Coefficient_of_Variation = lambda np_array: np.std(np_array) / np.mean(np_array)


    distances_list = []

    for model_name in model_names:
        for board in PAPER_TRUNCATED_BOARDS:
            distances_list.append(model_stat_emd_board(model_name,
                                4,
                                board,
                                curr_player=1,
                                max_model_iter=5000,
                                model_check_freq=50,
                                width=6, height=6, n=4,
                                 stat=np.var,
                                 stat_name="Varience"
                                 ))


    y_top_lim = max([max(stat_list) for stat_list, _, _, _, _, _ in distances_list])
    y_top_lim = 1.1 * y_top_lim

    for stat_list, models_num, model_list, model_name, board_name, stat_name in distances_list:
        save_fig_stat(stat_list, models_num, model_list, model_name, board_name, stat_name, y_top_lim)


    distances_list = []

    for model_name in model_names:
        for board in PAPER_TRUNCATED_BOARDS:
            distances_list.append(model_stat_emd_board(model_name,
                                                       4,
                                                       board,
                                                       curr_player=1,
                                                       max_model_iter=5000,
                                                       model_check_freq=50,
                                                       width=6, height=6, n=4,
                                                       stat=np.average,
                                                       stat_name="Average"
                                                       ))

    y_top_lim = max([max(stat_list) for stat_list, _, _, _, _, _ in distances_list])
    y_top_lim = 1.1 * y_top_lim

    for stat_list, models_num, model_list, model_name, board_name, stat_name in distances_list:
        save_fig_stat(stat_list, models_num, model_list, model_name, board_name, stat_name, y_top_lim)



    distances_list = []

    for model_name in model_names:
        for board in PAPER_TRUNCATED_BOARDS:

            model_stat_emd_board(model_name,
                                 4,
                                 board,
                                 curr_player=1,
                                 max_model_iter=5000,
                                 model_check_freq=50,
                                 width=6, height=6, n=4,
                                 stat=Coefficient_of_Variation,
                                 stat_name="Coefficient of Variation"
                                 )

    y_top_lim = max([max(stat_list) for stat_list, _, _, _, _, _ in distances_list])
    y_top_lim = 1.1 * y_top_lim

    for stat_list, models_num, model_list, model_name, board_name, stat_name in distances_list:
        save_fig_stat(stat_list, models_num, model_list, model_name, board_name, stat_name, y_top_lim)



