from mcts_alphaZero import MCTSPlayer
from game import Board, Game, BoardSlim
from tensorboardX import SummaryWriter
from policy_value_net_pytorch import PolicyValueNet  # Pytorch
import PIL.Image
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
from Game_boards import *

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



if __name__ == "__main__":
    save_heatmaps()

