# from mcts_pure import MCTS, MCTSPlayer
from mcts_alphaZero import MCTSPlayer
from mpl_toolkits.axes_grid1 import make_axes_locatable
import io

# from policy_player import PolicyPlayer
# from policy_net_keras import PolicyNet
import numpy as np
import tqdm
from game import Board, Game, BoardSlim
from tensorboardX import SummaryWriter
from policy_value_net_pytorch import PolicyValueNet  # Pytorch
import PIL.Image
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
import math
from pyemd import emd
import pandas as pd

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

PAPER_FULL_BOARDS = [BOARD_1_FULL, BOARD_2_FULL]

PAPER_TRUNCATED_BOARDS = [BOARD_1_TRUNCATED, BOARD_2_TRUNCATED]

START_POSITIONS_10 = np.array([
                  [[0,0,0,2,0,0,0,0,0,0],
                   [0,0,0,1,0,2,0,0,0,0],
                   [0,2,2,0,0,1,1,0,2,0],
                   [0,0,2,1,2,0,0,0,0,0],
                   [0,1,1,0,0,0,0,0,0,0],
                   [0,1,1,0,2,0,0,0,0,0],
                   [0,0,1,0,2,0,0,0,0,0],
                   [0,0,1,0,0,0,0,0,0,0],
                   [0,0,2,0,0,2,2,0,0,0],
                   [0,0,0,0,1,0,0,0,0,0]],

                 [[0,0,0,2,0,0,0,0,0,0],
                  [0,0,0,1,0,2,0,0,0,0],
                  [0,2,2,0,1,1,1,0,2,0],
                  [0,0,2,1,2,0,0,0,0,0],
                  [0,1,1,0,0,0,0,0,0,0],
                  [0,1,1,0,2,0,0,0,0,0],
                  [2,0,1,0,2,0,0,0,0,0],
                  [0,0,1,0,0,0,0,0,0,0],
                  [0,0,2,0,0,2,2,0,0,0],
                  [0,0,0,0,1,0,0,0,0,0]],


                [[0,0,0,0,1,0,2,0,0,0],
                 [0,0,0,0,2,1,1,1,0,0],
                 [0,0,0,1,2,2,2,1,0,0],
                 [0,0,0,2,2,1,1,2,1,1],
                 [2,0,0,1,0,2,2,0,0,0],
                 [1,0,0,0,0,0,0,0,0,0],
                 [1,1,0,0,0,0,0,0,0,0],
                 [2,2,0,0,0,0,1,0,0,0],
                 [0,0,0,0,0,0,1,0,0,0],
                 [0,0,0,0,0,2,2,2,0,0]],

                  [[0,0,0,0,1,2,2,0,0,0],
                   [0,0,0,0,2,1,1,1,0,0],
                   [0,0,0,1,2,2,2,1,0,0],
                   [0,0,0,2,2,1,1,2,1,1],
                   [2,0,0,1,0,2,2,0,0,1],
                   [1,0,0,0,0,0,0,0,0,0],
                   [1,1,0,0,0,0,0,0,0,0],
                   [2,2,0,0,0,0,1,0,0,0],
                   [0,0,0,0,0,0,1,0,0,0],
                   [0,0,0,0,0,2,2,2,0,0]],

                [[0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0],
                 [0,0,1,0,0,2,0,0,0,0],
                 [0,0,0,1,1,0,0,0,0,0],
                 [0,0,0,0,2,2,2,1,2,0],
                 [0,0,0,0,0,1,2,2,0,0],
                 [0,0,0,1,0,2,0,0,0,0],
                 [0,0,0,0,1,1,0,0,0,0],
                 [0,0,0,0,0,1,0,0,0,0],
                 [0,0,0,0,0,0,2,0,0,0]],

                 [[0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0],
                  [0,0,1,0,0,2,0,0,0,0],
                  [0,0,1,1,1,2,0,0,0,0],
                  [0,0,0,0,2,2,2,1,2,0],
                  [0,0,0,0,0,1,2,2,0,0],
                  [0,0,0,1,0,2,0,0,0,0],
                  [0,0,0,0,1,1,0,0,0,0],
                  [0,0,0,0,0,1,0,0,0,0],
                  [0,0,0,0,0,0,2,0,0,0]]])

def initialize_board(board_height, board_width, input_board, n_in_row = 4, start_player=2, **kwargs):

    last_move_p1 = kwargs.get('last_move_p1', None)
    last_move_p2 = kwargs.get('last_move_p2', None)

    board = input_board
    board = np.flipud(board)
    i_board = np.zeros((2, board_height, board_width))
    i_board[0] = board == 1
    i_board[1] = board == 2
    board = Board(width=board_width, height=board_height, n_in_row=n_in_row)
    board.init_board(start_player=start_player, initial_state=i_board, last_move_p1=last_move_p1, last_move_p2=last_move_p2)
    return board


MODEL_NAME="pt_6_6_4_p3_v9"
INPUT_PLANES_NUM = 3

def save_heatmaps(model_name = MODEL_NAME,
                  save_to_tensorboard=False,
                  save_to_local=True,
                  tell_last_move=True,
                  max_model_iter = 5000,
                  model_check_freq=50,
                  width=6,
                  height=6,
                  n=4,
                  input_plains_num=INPUT_PLANES_NUM,
                  c_puct=5,
                  n_playout=400):

    WRITER_DIR = f'./runs/{model_name}_paper_heatmaps'
    writer = SummaryWriter(WRITER_DIR)


    for i in range(model_check_freq, max_model_iter + model_check_freq, model_check_freq):

        heatmap_save_path = f'/home/lirontyomkin/AlphaZero_Gomoku/heatmaps/{model_name}/iteration_{i}/'
        if not os.path.exists(heatmap_save_path):
            os.makedirs(heatmap_save_path)


        for board_state, board_name, p1, p2 in PAPER_FULL_BOARDS:
            board = initialize_board(height, width, input_board=board_state, n_in_row=n)
            save_heatmap_for_board_and_model(
                                    model_name,
                                    width, height,
                                    input_plains_num,
                                    c_puct,
                                    n_playout,
                                    board, board_name,
                                    save_to_local, save_to_tensorboard, writer,
                                    i, heatmap_save_path)

        for board_state, board_name, p1, p2 in PAPER_TRUNCATED_BOARDS:

            board = initialize_board(height, width,
                                     input_board=board_state,
                                     n_in_row=n)

            save_heatmap_for_board_and_model(
                                    model_name,
                                    width, height,
                                    input_plains_num,
                                    c_puct,
                                    n_playout,
                                    board, board_name,
                                    save_to_local, save_to_tensorboard, writer,
                                    i, heatmap_save_path)

            if tell_last_move and input_plains_num == 4: #save truncated with last move too

                p1_last_move, p2_last_move = p1, p2
                board_name = board_name + " with correct last move"

                board = initialize_board(height, width,
                                         input_board=board_state,
                                         n_in_row=n,
                                         last_move_p1=p1_last_move,
                                         last_move_p2=p2_last_move)

                save_heatmap_for_board_and_model(
                                        model_name,
                                        width, height,
                                        input_plains_num,
                                        c_puct,
                                        n_playout,
                                        board, board_name,
                                        save_to_local, save_to_tensorboard, writer,
                                        i, heatmap_save_path)


def save_heatmap_for_board_and_model(
                            model_name,
                            width, height,
                            input_plains_num,
                            c_puct,
                            n_playout,
                            board, board_name,
                            save_to_local, save_to_tensorboard, writer,
                            i, heatmap_save_path):

    model_file = f'/home/lirontyomkin/AlphaZero_Gomoku/models/{model_name}/current_policy_{i}.model'

    # heatmap_image_path = f'/home/lirontyomkin/AlphaZero_Gomoku/heatmaps/{model_name}/iteration_{i}/{board_name}.png'
    # if os.path.exists(heatmap_image_path):
    #     return
    # else:
    #     print(heatmap_image_path)


    policy = PolicyValueNet(width, height, model_file=model_file, input_plains_num=input_plains_num)
    player = MCTSPlayer(policy.policy_value_fn, c_puct=c_puct, n_playout=n_playout, name=f"{model_name}_{i}")

    _, heatmap_buf = player.get_action(board, return_prob=0, return_fig=True)
    image = PIL.Image.open(heatmap_buf)

    if save_to_local:
        plt.savefig(heatmap_save_path + f"{board_name}.png")

    if save_to_tensorboard:
        image_tensor = ToTensor()(image)
        writer.add_image(tag=f'Heatmap on {board_name}',
                         img_tensor=image_tensor,
                         global_step=i)
    plt.close('all')



def EMD_model_comparison(model1_name, input_plains_num_1, max_model1_iter, model1_check_freq,
                         model2_name, input_plains_num_2, max_model2_iter, model2_check_freq,
                         BOARD, n=4, width=6, height=6, tell_last_move1=False, tell_last_move2=False,
                         n_playout=400, c_puct=5):

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

    board_state, board_name, last_move_p1, last_move_p2 = BOARD

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
                                   board1=board1, board2=board2, width=width,height=height,
                                   n_playout=n_playout, c_puct=c_puct)


    df = pd.DataFrame(result, index=index, columns=columns)
    df.to_csv(f"{save_path}on {board_name}.csv",index = True, header=True)

    fig, ax = plt.subplots(1, figsize=(20, 20))
    fontsize = 12

    im = ax.imshow(result, cmap='hot', interpolation='nearest')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, ax=ax, cax=cax).ax.tick_params(labelsize=fontsize*2)

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

def EMD_between_two_models_on_board(model1_name, input_plains_num_1, i1,
                                   model2_name, input_plains_num_2, i2,
                                   board1, board2, width=6,height=6,
                                   n_playout=400, c_puct=5):


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


if __name__ == "__main__":


    # for board in PAPER_TRUNCATED_BOARDS:
        # EMD_model_comparison(model1_name="pt_6_6_4_p4_v10", input_plains_num_1=4, max_model1_iter=5000,
        #                      model1_check_freq=50, tell_last_move1=True,
        #                      model2_name="pt_6_6_4_p3_v7", input_plains_num_2=3, max_model2_iter=5000,
        #                      model2_check_freq=50, tell_last_move2=False,
        #                      BOARD=board, n=4, width=6, height=6,
        #                      n_playout=400, c_puct=5)
        #
        # EMD_model_comparison(model1_name="pt_6_6_4_p4_v10", input_plains_num_1=4, max_model1_iter=5000,
        #                      model1_check_freq=50, tell_last_move1=True,
        #                      model2_name="pt_6_6_4_p3_v9", input_plains_num_2=3, max_model2_iter=5000,
        #                      model2_check_freq=50, tell_last_move2=False,
        #                      BOARD=board, n=4, width=6, height=6,
        #                      n_playout=400, c_puct=5)

        # EMD_model_comparison(model1_name="pt_6_6_4_p3_v7", input_plains_num_1=3, max_model1_iter=5000,
        #                      model1_check_freq=50, tell_last_move1=False,
        #                      model2_name="pt_6_6_4_p3_v9", input_plains_num_2=3, max_model2_iter=5000,
        #                      model2_check_freq=50, tell_last_move2=False,
        #                      BOARD=board, n=4, width=6, height=6,
        #                      n_playout=400, c_puct=5)
        #
        # EMD_model_comparison(model1_name="pt_6_6_4_p4_v10", input_plains_num_1=4, max_model1_iter=5000,
        #                      model1_check_freq=50, tell_last_move1=False,
        #                      model2_name="pt_6_6_4_p3_v7", input_plains_num_2=3, max_model2_iter=5000,
        #                      model2_check_freq=50, tell_last_move2=False,
        #                      BOARD=board, n=4, width=6, height=6,
        #                      n_playout=400, c_puct=5)
        #
        # EMD_model_comparison(model1_name="pt_6_6_4_p4_v10", input_plains_num_1=4, max_model1_iter=5000,
        #                      model1_check_freq=50, tell_last_move1=False,
        #                      model2_name="pt_6_6_4_p3_v9", input_plains_num_2=3, max_model2_iter=5000,
        #                      model2_check_freq=50, tell_last_move2=False,
        #                      BOARD=board, n=4, width=6, height=6,
        #                      n_playout=400, c_puct=5)


    for board in PAPER_FULL_BOARDS:
        # EMD_model_comparison(model1_name="pt_6_6_4_p4_v10", input_plains_num_1=4, max_model1_iter=5000,
        #                      model1_check_freq=50, tell_last_move1=False,
        #                      model2_name="pt_6_6_4_p3_v7", input_plains_num_2=3, max_model2_iter=5000,
        #                      model2_check_freq=50, tell_last_move2=False,
        #                      BOARD=board, n=4, width=6, height=6,
        #                      n_playout=400, c_puct=5)
        #
        # EMD_model_comparison(model1_name="pt_6_6_4_p4_v10", input_plains_num_1=4, max_model1_iter=5000,
        #                      model1_check_freq=50, tell_last_move1=False,
        #                      model2_name="pt_6_6_4_p3_v9", input_plains_num_2=3, max_model2_iter=5000,
        #                      model2_check_freq=50, tell_last_move2=False,
        #                      BOARD=board, n=4, width=6, height=6,
        #                      n_playout=400, c_puct=5)

        EMD_model_comparison(model1_name="pt_6_6_4_p3_v7", input_plains_num_1=3, max_model1_iter=5000,
                             model1_check_freq=50, tell_last_move1=False,
                             model2_name="pt_6_6_4_p3_v9", input_plains_num_2=3, max_model2_iter=5000,
                             model2_check_freq=50, tell_last_move2=False,
                             BOARD=board, n=4, width=6, height=6,
                             n_playout=400, c_puct=5)
