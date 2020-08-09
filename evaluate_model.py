# from mcts_pure import MCTS, MCTSPlayer
from mcts_alphaZero import MCTSPlayer

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

BOARD_1_FULL = np.array([[0, 1, 0, 2, 0, 0],
                [0, 2, 1, 1, 0, 0],
                [1, 2, 2, 2, 1, 0],
                [2, 0, 1, 1, 2, 0],
                [1, 0, 2, 2, 0, 0],
                [0, 0, 0, 0, 0, 0]])

BOARD_1_TRUNCATED = np.array([[0, 1, 2, 2, 0, 0],
                    [0, 2, 1, 1, 0, 0],
                    [1, 2, 2, 2, 1, 0],
                    [2, 0, 1, 1, 2, 1],
                    [1, 0, 2, 2, 0, 0],
                    [0, 0, 0, 0, 0, 0]])

BOARD_2_FULL =  np.array([[0, 2, 0, 0, 1, 0],
                    [0, 2, 1, 2, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 1, 0, 2, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 2, 0, 0, 2, 0]])

BOARD_2_TRUNCATED = np.array([[0, 2, 0, 1, 1, 0],
                    [0, 2, 1, 2, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [2, 1, 0, 2, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 2, 0, 0, 2, 0]])

PAPER_BOARDS = [(BOARD_1_FULL, "board 1 full"), (BOARD_1_TRUNCATED, "board 1 truncated"), (BOARD_2_FULL, "board 2 full"), (BOARD_2_TRUNCATED, "board 2 truncated")]

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


def evaluate_player(player, opponent, show_game=False):
    # for i in range(n_games):
    #     winner = self.game.start_play(current_player,
    #                                   pure_mcts_player,
    #                                   start_player=i % 2,
    #                                   is_shown=0)
    #     win_cnt[winner] += 1
    # win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
    # print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
    #     self.pure_mcts_playout_num,
    #     win_cnt[1], win_cnt[2], win_cnt[-1]))
    # return win_ratio
    stats = {"wins": 0, "length": [], "ties": 0}
    for i in tqdm.trange(num_games):
        opponent.reset_player()
        board = BoardSlim(width=BOARD_SHAPE[0], height=BOARD_SHAPE[1], n_in_row=4)
        game = Game(board)
        winner = game.start_play(player, opponent, start_player= i % 2, is_shown=(i == 0) and show_game)
        print("winner: ", winner)
        if winner == 1:
            stats["wins"] += 1
        elif winner == -1:
            stats["ties"] += 1
        stats["length"].append(BOARD_SHAPE[0] * BOARD_SHAPE[1] - len(board.availables))
    return stats

def initialize_board(board_height, board_width, input_board):
    n_in_row = 4
    board = input_board
    board = np.flipud(board)
    i_board = np.zeros((2, board_height, board_width))
    i_board[0] = board == 1
    i_board[1] = board == 2
    board = Board(width=board_width, height=board_height, n_in_row=n_in_row)
    board.init_board(start_player=1, initial_state=i_board)
    return board

def save_heatmaps(model_name = "pt_6_6_4_p4_v4" ,
                  save_to_tensorboard=True,
                  save_to_local=False,
                  max_model_iter = 2500,
                  model_check_rfeq=50,
                  width=6,
                  height=6,
                  n=4,
                  input_plains_num=4,
                  c_puct=5,
                  n_playout=400):

    WRITER_DIR = f'./runs/{model_name}_paper_heatmaps'
    writer = SummaryWriter(WRITER_DIR)

    for board_state, board_name in PAPER_BOARDS:
        board = initialize_board(height, width, input_board=board_state)

        for i in range(model_check_rfeq, max_model_iter + model_check_rfeq,model_check_rfeq):

            model_file = f'/home/lirontyomkin/AlphaZero_Gomoku/models/{model_name}/current_policy_{i}.model'

            policy = PolicyValueNet(width, height, model_file=model_file, input_plains_num=input_plains_num)
            player = MCTSPlayer(policy.policy_value_fn, c_puct=c_puct, n_playout=n_playout, name="AI")

            _, heatmap_buf = player.get_action(board, return_prob=0, return_fig=True)
            image = PIL.Image.open(heatmap_buf)

            if save_to_local:
                heatmap_save_path = f'/home/lirontyomkin/AlphaZero_Gomoku/heatmaps/{model_name}/iteration_{i}/'
                if not os.path.exists(heatmap_save_path):
                    os.makedirs(heatmap_save_path)
                plt.savefig(heatmap_save_path + f"{board_name}.png")

            if save_to_tensorboard:
                image_tensor = ToTensor()(image)
                writer.add_image(tag=f'Heatmap on {board_name}',
                                      img_tensor=image_tensor,
                                      global_step= i)
            plt.close('all')

if __name__ == "__main__":

    save_heatmaps()

    # player = load_player()
    # for playout in playouts:
    #     mcts_player = MCTSPlayer(n_playout=playout)
    #     stats = evaluate_player(player, mcts_player, show_game=True)
    #     print()
    #     print("win ratio agains %d playouts: " % playout, stats["wins"] / num_games)
    #     print("tie ratio agains %d playouts: " % playout, stats["ties"] / num_games)
    #     print("average game length agains %d playouts: ", np.mean(stats["length"]))