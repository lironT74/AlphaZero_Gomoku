from pyemd import emd
import numpy as np
import math
from game import Board
import copy
from policy_value_net_pytorch import PolicyValueNet

BOARD_1_FULL = (np.array([[0, 1, 0, 2, 0, 0],
                [0, 2, 1, 1, 0, 0],
                [1, 2, 2, 2, 1, 0],
                [2, 0, 1, 1, 2, 0],
                [1, 0, 2, 2, 0, 0],
                [0, 0, 0, 0, 0, 0]]), "board 1 full", None, None, None, None)

BOARD_1_TRUNCATED = (np.array([[0, 1, 2, 2, 0, 0],
                    [0, 2, 1, 1, 0, 0],
                    [1, 2, 2, 2, 1, 0],
                    [2, 0, 1, 1, 2, 1],
                    [1, 0, 2, 2, 0, 0],
                    [0, 0, 0, 0, 0, 0]]), "board 1 truncated", [2, 5], [5, 2], [1, 0], [2, 0])

BOARD_2_FULL =  (np.array([[0, 2, 0, 0, 1, 0],
                    [0, 2, 1, 2, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 1, 0, 2, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 2, 0, 0, 2, 0]]), "board 2 full", None, None, None, None)

BOARD_2_TRUNCATED = (np.array([[0, 2, 0, 1, 1, 0],
                    [0, 2, 1, 2, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [2, 1, 0, 2, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 2, 0, 0, 2, 0]]), "board 2 truncated", [5, 3], [2, 0], [1, 1], [4, 3])




EMPTY_BOARD = (np.array([[0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0]]), "empty board", None, None, None, None)

PAPER_FULL_BOARDS = [BOARD_1_FULL, BOARD_2_FULL]

PAPER_TRUNCATED_BOARDS = [BOARD_1_TRUNCATED, BOARD_2_TRUNCATED]

ALL_PAPER_6X6_BOARD = [BOARD_1_FULL, BOARD_2_FULL, BOARD_1_TRUNCATED, BOARD_2_TRUNCATED]

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



def initialize_board_without_init_call(board_height, board_width, n_in_row, input_board):
    board = input_board
    board = np.flipud(board)
    i_board = np.zeros((2, board_height, board_width))
    i_board[0] = board == 1
    i_board[1] = board == 2
    board = Board(width=board_width, height=board_height, n_in_row=n_in_row)
    return i_board, board


def initialize_board_with_init_and_last_moves(board_height, board_width, input_board, n_in_row = 4, start_player=2, **kwargs):

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


def EMD_between_two_models_on_board(model1_name, input_plains_num_1, i1,
                                    model2_name, input_plains_num_2, i2,
                                    board1, board2, width=6, height=6, use_gpu=True):
    model_file_1 = f'/home/lirontyomkin/AlphaZero_Gomoku/models/{model1_name}/current_policy_{i1}.model'
    policy_1 = PolicyValueNet(width, height, model_file=model_file_1, input_plains_num=input_plains_num_1,
                              use_gpu=use_gpu)

    model_file_2 = f'/home/lirontyomkin/AlphaZero_Gomoku/models/{model2_name}/current_policy_{i2}.model'
    policy_2 = PolicyValueNet(width, height, model_file=model_file_2, input_plains_num=input_plains_num_2,
                              use_gpu=use_gpu)

    acts_policy1, probas_policy1 = zip(*policy_1.policy_value_fn(board1)[0])
    acts_policy2, probas_policy2 = zip(*policy_2.policy_value_fn(board2)[0])

    dist_matrix = generate_matrix_dist_metric(6)

    distance = emd(np.asarray(probas_policy1, dtype='float64'), np.asarray(probas_policy2, dtype='float64'),
                   dist_matrix)

    return distance


class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board, **kwargs):
        try:
            location = input("Your move: ")
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)



# if __name__ == '__main__':
#     import matplotlib as mpl
#     print(mpl.get_backend())


