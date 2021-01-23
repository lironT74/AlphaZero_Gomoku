from scipy import stats
from pyemd import emd
import numpy as np
import math
from game import Board, Game, get_shutter_size, get_printable_move
import copy
import json
from policy_value_net_pytorch import PolicyValueNet
from datetime import datetime
from dateutil.relativedelta import relativedelta as rd
import string
import re
import ast

BOARD_1_FULL = (np.array([
                [0, 1, 0, 2, 0, 0],
                [0, 2, 1, 1, 0, 0],
                [1, 2, 2, 2, 1, 0],
                [2, 0, 1, 1, 2, 0],
                [1, 0, 2, 2, 0, 0],
                [0, 0, 0, 0, 0, 0]]), "board 1 full", None, None, None, None)

BOARD_1_TRUNCATED = (np.array([
                    [0, 1, 2, 2, 0, 0],
                    [0, 2, 1, 1, 0, 0],
                    [1, 2, 2, 2, 1, 0],
                    [2, 0, 1, 1, 2, 1],
                    [1, 0, 2, 2, 0, 0],
                    [0, 0, 0, 0, 0, 0]]), "board 1 truncated", [2, 5], [5, 2], [1, 0], [2, 0])

BOARD_2_FULL =  (np.array([
                    [0, 2, 0, 0, 1, 0],
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


EMPTY_BOARD_6X6 = (np.array([[0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0]]), "empty board", None, None, None, None)


PAPER_FULL_6X6_BOARDS = [BOARD_1_FULL, BOARD_2_FULL]

PAPER_6X6_TRUNCATED_BOARDS = [BOARD_1_TRUNCATED, BOARD_2_TRUNCATED]

ALL_PAPER_6X6_BOARD = [BOARD_1_FULL, BOARD_2_FULL, BOARD_1_TRUNCATED, BOARD_2_TRUNCATED, EMPTY_BOARD_6X6]



PEOPLE_DISTRIBUTIONS_6X6 = json.load(open('./avg_people_first_moves_all.json'))
PEOPLE_DISTRIBUTIONS_6X6["board 1 full"] = PEOPLE_DISTRIBUTIONS_6X6.pop("6_easy_full")
PEOPLE_DISTRIBUTIONS_6X6["board 2 full"] = PEOPLE_DISTRIBUTIONS_6X6.pop("6_hard_full")
PEOPLE_DISTRIBUTIONS_6X6["board 1 truncated"] = PEOPLE_DISTRIBUTIONS_6X6.pop("6_easy_pruned")
PEOPLE_DISTRIBUTIONS_6X6["board 2 truncated"] = PEOPLE_DISTRIBUTIONS_6X6.pop("6_hard_pruned")

for key, value in PEOPLE_DISTRIBUTIONS_6X6.items():
    PEOPLE_DISTRIBUTIONS_6X6[key] = np.array(PEOPLE_DISTRIBUTIONS_6X6[key])
    PEOPLE_DISTRIBUTIONS_6X6[key][PEOPLE_DISTRIBUTIONS_6X6[key] < 0] = 0




EMPTY_BOARD_10X10 = (np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0,0,0,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,0,0,0]]), "empty board", None, None, None, None)


BOARD_3_FULL = (np.array(
                        [[0, 0, 0, 0, 1, 0, 2, 0, 0, 0],
                         [0, 0, 0, 0, 2, 1, 1, 1, 0, 0],
                         [0, 0, 0, 1, 2, 2, 2, 1, 0, 0],
                         [0, 0, 0, 2, 2, 1, 1, 2, 1, 1],
                         [2, 0, 0, 1, 0, 2, 2, 0, 0, 0],
                         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [2, 2, 0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 2, 2, 2, 0, 0]]), "board 3 full", None, None, None, None)


BOARD_3_TRUNCATED = (np.array([[0, 0, 0, 0, 1, 2, 2, 0, 0, 0],
                               [0, 0, 0, 0, 2, 1, 1, 1, 0, 0],
                               [0, 0, 0, 1, 2, 2, 2, 1, 0, 0],
                               [0, 0, 0, 2, 2, 1, 1, 2, 1, 1],
                               [2, 0, 0, 1, 0, 2, 2, 0, 0, 1],
                               [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                               [2, 2, 0, 0, 0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 2, 2, 2, 0, 0]]), "board 3 truncated", [5, 9], [9, 5], None, None)


BOARD_4_FULL = (np.array(
                            [[0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 2, 0, 0, 0, 0],
                             [0, 2, 2, 0, 0, 1, 1, 0, 2, 0],
                             [0, 0, 2, 1, 2, 0, 0, 0, 0, 0],
                             [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 1, 0, 2, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 2, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 2, 0, 0, 2, 2, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]), "board 4 full", None, None, None, None)


BOARD_4_TRUNCATED = (np.array([[0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 2, 0, 0, 0, 0],
                              [0, 2, 2, 0, 1, 1, 1, 0, 2, 0],
                              [0, 0, 2, 1, 2, 0, 0, 0, 0, 0],
                              [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 1, 0, 2, 0, 0, 0, 0, 0],
                              [2, 0, 1, 0, 2, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 2, 0, 0, 2, 2, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]), "board 4 truncated", [7, 4], [3, 0], None, None)


BOARD_5_FULL = (np.array(
                            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 2, 0, 0, 0, 0],
                             [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 2, 2, 2, 1, 2, 0],
                             [0, 0, 0, 0, 0, 1, 2, 2, 0, 0],
                             [0, 0, 0, 1, 0, 2, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 2, 0, 0, 0]]), "board 5 full", None, None, None, None)


BOARD_5_TRUNCATED = (np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 1, 0, 0, 2, 0, 0, 0, 0],
                                  [0, 0, 1, 1, 1, 2, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 2, 2, 2, 1, 2, 0],
                                  [0, 0, 0, 0, 0, 1, 2, 2, 0, 0],
                                  [0, 0, 0, 1, 0, 2, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 2, 0, 0, 0]]), "board 5 truncated", [6, 2], [6, 5], None, None)


PAPER_FULL_10X10_BOARDS = [BOARD_3_FULL, BOARD_4_FULL, BOARD_5_FULL]

PAPER_10X10_TRUNCATED_BOARDS = [BOARD_3_TRUNCATED, BOARD_4_TRUNCATED, BOARD_5_TRUNCATED]

ALL_PAPER_10X10_BOARD = [BOARD_3_FULL, BOARD_4_FULL, BOARD_5_FULL, BOARD_3_TRUNCATED,
                         BOARD_4_TRUNCATED, BOARD_5_TRUNCATED, EMPTY_BOARD_10X10]


TRUNCATED_BOARDS_DICT = {6: PAPER_6X6_TRUNCATED_BOARDS, 10: PAPER_10X10_TRUNCATED_BOARDS}
FULL_BOARDS_DICT = {6: PAPER_FULL_6X6_BOARDS, 10: PAPER_FULL_10X10_BOARDS}
EMPTY_BOARDS_DICT = {6: EMPTY_BOARD_6X6, 10: EMPTY_BOARD_10X10}



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



def initialize_board_without_init_call(board_height, board_width, n_in_row, input_board, *args, **kwargs):
    open_path_threshold = kwargs.get("open_path_threshold", 0)
    board = input_board
    board = np.flipud(board)
    i_board = np.zeros((2, board_height, board_width))
    i_board[0] = board == 1
    i_board[1] = board == 2
    board = Board(width=board_width, height=board_height, n_in_row=n_in_row, open_path_threshold=open_path_threshold)
    return i_board, board


def initialize_board_with_init_and_last_moves(board_height, board_width, input_board, n_in_row = 4, start_player=2, **kwargs):

    open_path_threshold = kwargs.get("open_path_threshold", 0)


    last_move_p1 = kwargs.get('last_move_p1', None)
    last_move_p2 = kwargs.get('last_move_p2', None)

    is_random_last_turn_p1 = kwargs.get('is_random_last_turn_p1', False)
    is_random_last_turn_p2 = kwargs.get('is_random_last_turn_p2', False)

    board = copy.deepcopy(input_board)
    board = np.flipud(board)
    i_board = np.zeros((2, board_height, board_width))
    i_board[0] = board == 1
    i_board[1] = board == 2

    board = Board(width=board_width, height=board_height, n_in_row=n_in_row, open_path_threshold=open_path_threshold)
    board.init_board(start_player=start_player, initial_state=i_board, last_move_p1=last_move_p1, last_move_p2=last_move_p2,
                     is_random_last_turn_p1=is_random_last_turn_p1, is_random_last_turn_p2=is_random_last_turn_p2)
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


    board_current_state1 = board1.current_state(last_move=True, is_random_last_turn=False)
    board_current_state2 = board2.current_state(last_move=True, is_random_last_turn=False)

    acts_policy1, probas_policy1 = zip(*policy_1.policy_value_fn(board1)[0])
    acts_policy2, probas_policy2 = zip(*policy_2.policy_value_fn(board2)[0])

    dist_matrix = generate_matrix_dist_metric(width)

    distance = emd(np.asarray(probas_policy1, dtype='float64'), np.asarray(probas_policy2, dtype='float64'),dist_matrix)

    return distance




class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None
        self.name = "Human"

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


def boot_matrix(z, B):
    """Bootstrap sample

    Returns all bootstrap samples in a matrix"""

    n = len(z)  # sample size
    idz = np.random.randint(0, n, size=(B, n))  # indices to pick for all boostrap samples
    return z[idz]


def bootstrap_mean(x, B=100000, alpha=0.05, plot=False):
    """Bootstrap standard error and (1-alpha)*100% c.i. for the population mean

    Returns bootstrapped standard error and different types of confidence intervals"""

    x = np.array(x)
    # Deterministic things
    n = len(x)  # sample size
    orig = np.average(x)  # sample mean
    se_mean = np.std(x)/np.sqrt(n) # standard error of the mean
    qt = stats.t.ppf(q=1 - alpha/2, df=n - 1) # Student quantile

    # Generate boostrap distribution of sample mean
    xboot = boot_matrix(x, B=B)
    sampling_distribution = np.average(xboot, axis=1)

   # Standard error and sample quantiles
    se_mean_boot = np.std(sampling_distribution)
    quantile_boot = np.percentile(sampling_distribution, q=(100*alpha/2, 100*(1-alpha/2)))

    # # RESULTS
    # print("Estimated mean:", orig)
    # print("Classic standard error:", se_mean)
    # print("Classic student c.i.:", orig + np.array([-qt, qt])*se_mean)
    # print("\nBootstrap results:")
    # print("Standard error:", se_mean_boot)
    # print("t-type c.i.:", orig + np.array([-qt, qt])*se_mean_boot)
    # print("Percentile c.i.:", quantile_boot)
    # print("Basic c.i.:", 2*orig - quantile_boot[::-1])

    return quantile_boot



def cur_time():
    now = datetime.now()
    cur_time = now.strftime("%d/%m/%Y %H:%M:%S")
    return cur_time


def npstr2tuple(s):
    # Remove space after [
    s = re.sub('\[ +', '[', s.strip())
    # Replace commas and spaces
    s = re.sub('[,\s]+', ', ', s)
    return tuple(np.array(ast.literal_eval(s)))


if __name__ == '__main__':

    print(f"6X6 boards: ")
    for board in ALL_PAPER_6X6_BOARD:
        print(f"{board[1]}: {len(np.where(board[0] == 1)[0])} X's, {len(np.where(board[0] == 2)[0])} O's")

    print(f"\n10x10 boards: ")
    for board in ALL_PAPER_10X10_BOARD:
        print(f"{board[1]}: {len(np.where(board[0] == 1)[0])} X's, {len(np.where(board[0] == 2)[0])} O's")
