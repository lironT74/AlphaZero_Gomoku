from mcts_alphaZero import MCTSPlayer
from game import Board, Game
from tensorboardX import SummaryWriter
from policy_value_net_pytorch import PolicyValueNet  # Pytorch
import PIL.Image
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
from Game_boards import *
import multiprocessing

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
                  n_playout=400):

    WRITER_DIR = f'./runs/{model_name}_paper_heatmaps'
    writer = SummaryWriter(WRITER_DIR)


    for i in range(model_check_freq, max_model_iter + model_check_freq, model_check_freq):

        heatmap_save_path = f'/home/lirontyomkin/AlphaZero_Gomoku/heatmaps/{model_name}/iteration_{i}/'
        if not os.path.exists(heatmap_save_path):
            os.makedirs(heatmap_save_path)

        board_state, board_name, p1, p2, alternative_p1, alternative_p2 = EMPTY_BOARD
        board = initialize_board(height, width, input_board=board_state, n_in_row=n, last_move_p1=p1, last_move_p2=p2)
        save_heatmap_for_board_and_model(
                                    model_name,
                                    width, height,
                                    input_plains_num,
                                    c_puct,
                                    n_playout,
                                    board, board_name,
                                    save_to_local, save_to_tensorboard, writer,
                                    i, heatmap_save_path)

        for board_state, board_name, p1, p2, alternative_p1, alternative_p2 in PAPER_FULL_BOARDS:

            board = initialize_board(height, width, input_board=board_state, n_in_row=n, last_move_p1=p1, last_move_p2=p2)

            save_heatmap_for_board_and_model(
                                    model_name,
                                    width, height,
                                    input_plains_num,
                                    c_puct,
                                    n_playout,
                                    board, board_name,
                                    save_to_local, save_to_tensorboard, writer,
                                    i, heatmap_save_path)

        for board_state, board_name, p1, p2, alternative_p1, alternative_p2 in PAPER_TRUNCATED_BOARDS:

            board = initialize_board(height, width,
                                     input_board=board_state,
                                     n_in_row=n, last_move_p1=p1, last_move_p2=p2)

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
                                    i, heatmap_save_path)

            if input_plains_num == 4: #save truncated with alternative last moves too

                board = initialize_board(height, width,
                                         input_board=board_state,
                                         n_in_row=n,
                                         last_move_p1=alternative_p1,
                                         last_move_p2=alternative_p2)

                save_heatmap_for_board_and_model(
                                        model_name,
                                        width, height,
                                        input_plains_num,
                                        c_puct,
                                        n_playout,
                                        board, board_name,
                                        save_to_local, save_to_tensorboard, writer,
                                        i, heatmap_save_path)

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
    # args_v7 = ('pt_6_6_4_p3_v7', 3)
    # args_v9 = ('pt_6_6_4_p3_v9', 3)
    # args_v10 =('pt_6_6_4_p4_v10',4)
    #
    # models_args = [args_v7, args_v9, args_v10]
    # processes = []
    #
    # for args_model in models_args:
    #     p = multiprocessing.Process(target=save_heatmaps, args=args_model)
    #     processes.append(p)
    #     p.start()
    #
    # for process in processes:
    #     process.join()

    save_heatmaps(model_name="pt_6_6_4_p4_v10", input_plains_num=4)
    save_heatmaps(model_name="pt_6_6_4_p3_v9", input_plains_num=3)
    save_heatmaps(model_name="pt_6_6_4_p3_v7", input_plains_num=3)

