"""
A script meant to save heatmaps of the policies on different boards.
Useful if you want to compare different policies and their decisions.

"""

from datetime import datetime
from mcts_alphaZero import MCTSPlayer
from tensorboardX import SummaryWriter
import PIL.Image
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
from aux_functions_and_boards import *
from multiprocessing import Pool



def save_heatmaps(model_name,
                  input_plains_num,
                  save_to_tensorboard=False,
                  save_to_local=True,
                  max_model_iter = 5000,
                  model_check_freq=50,
                  width=6,
                  height=6,
                  n=4,
                  c_puct=5,
                  n_playout=400,
                  use_gpu=False):

    WRITER_DIR = f'./runs/{model_name}_paper_heatmaps'
    writer = SummaryWriter(WRITER_DIR)


    for i in range(model_check_freq, max_model_iter + model_check_freq, model_check_freq):

        heatmap_save_path = f'/home/lirontyomkin/AlphaZero_Gomoku/models_heatmaps/move selections with MCTS/{model_name}/iteration_{i}/'
        if not os.path.exists(heatmap_save_path):
            os.makedirs(heatmap_save_path)

        board_state, board_name, p1, p2, alternative_p1, alternative_p2 = EMPTY_BOARD_6X6
        board = initialize_board_with_init_and_last_moves(height, width, input_board=board_state, n_in_row=n, last_move_p1=p1, last_move_p2=p2, open_path_threshold=-1)
        save_heatmap_for_board_and_model(
                                    model_name,
                                    width, height,
                                    input_plains_num,
                                    c_puct,
                                    n_playout,
                                    board, board_name,
                                    save_to_local, save_to_tensorboard, writer,
                                    i, heatmap_save_path, use_gpu)

        for board_state, board_name, p1, p2, alternative_p1, alternative_p2 in PAPER_FULL_6X6_BOARDS:

            board = initialize_board_with_init_and_last_moves(height, width, input_board=board_state, n_in_row=n,
                                                              last_move_p1=p1, last_move_p2=p2, open_path_threshold=-1)

            save_heatmap_for_board_and_model(
                                    model_name,
                                    width, height,
                                    input_plains_num,
                                    c_puct,
                                    n_playout,
                                    board, board_name,
                                    save_to_local, save_to_tensorboard, writer,
                                    i, heatmap_save_path, use_gpu)

        for board_state, board_name, p1, p2, alternative_p1, alternative_p2 in PAPER_6X6_TRUNCATED_BOARDS:

            board = initialize_board_with_init_and_last_moves(height, width,
                                                              input_board=board_state,
                                                              n_in_row=n, last_move_p1=p1, last_move_p2=p2, open_path_threshold=-1)
            if input_plains_num == 4:
                board_name_1 = board_name + " with correct last move"
            else:
                board_name_1 = board_name

            save_heatmap_for_board_and_model(
                                    model_name,
                                    width, height,
                                    input_plains_num,
                                    c_puct,
                                    n_playout,
                                    board, board_name_1,
                                    save_to_local, save_to_tensorboard, writer,
                                    i, heatmap_save_path, use_gpu)

            if input_plains_num == 4: #save truncated with alternative last moves too

                board = initialize_board_with_init_and_last_moves(height, width,
                                                                  input_board=board_state,
                                                                  n_in_row=n,
                                                                  last_move_p1=alternative_p1,
                                                                  last_move_p2=alternative_p2,
                                                                  open_path_threshold=-1)

                save_heatmap_for_board_and_model(
                                        model_name,
                                        width, height,
                                        input_plains_num,
                                        c_puct,
                                        n_playout,
                                        board, board_name,
                                        save_to_local, save_to_tensorboard, writer,
                                        i, heatmap_save_path, use_gpu)

                make_collage_for_truncated(heatmap_save_path, board_name)


def make_collage_for_truncated(heatmap_save_path, board_name):

    path1 = heatmap_save_path + f"{board_name}.png"
    path2 = heatmap_save_path + f"{board_name} with correct last move.png"

    listofimages = [path1, path2]

    im_check = PIL.Image.open(path1)
    width1, height1 = im_check.size
    width = width1
    height = 2*height1

    cols = 1
    rows = 2

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

    new_im.save(heatmap_save_path + f"{board_name} last moves comparison.png")



def save_heatmap_for_board_and_model(
                            model_name,
                            width, height,
                            input_plains_num,
                            c_puct,
                            n_playout,
                            board, board_name,
                            save_to_local, save_to_tensorboard, writer,
                            i, heatmap_save_path,use_gpu, **kwargs):


    model_file = f'/home/lirontyomkin/AlphaZero_Gomoku/models/{model_name}/current_policy_{i}.model'

    policy = PolicyValueNet(width, height, model_file=model_file, input_plains_num=input_plains_num, use_gpu=use_gpu)
    player = MCTSPlayer(policy.policy_value_fn, c_puct=c_puct, n_playout=n_playout, name=f"{model_name}_{i}")

    _, heatmap_buf = player.get_action(board, return_prob=0, return_fig=True)
    image = PIL.Image.open(heatmap_buf)

    if save_to_local:
        plt.savefig(heatmap_save_path + f"{board_name}.png", bbox_inches='tight')

    if save_to_tensorboard:
        image_tensor = ToTensor()(image)
        writer.add_image(tag=f'Heatmap on {board_name}',
                         img_tensor=image_tensor,
                         global_step=i)
    plt.close('all')

    now_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"Done saving: {model_name}_{i} on board {board_name}, {now_time}")



if __name__ == "__main__":

    args_v7 = ('pt_6_6_4_p3_v7', 3)
    args_v9 = ('pt_6_6_4_p3_v9', 3)
    args_v10 =('pt_6_6_4_p4_v10',4)

    models_args = [args_v7, args_v9, args_v10]

    with Pool(3) as pool:

        print(f"Using {pool._processes} workers. There are {len(models_args)} jobs: \n")
        pool.starmap(save_heatmaps, models_args)
        pool.close()

