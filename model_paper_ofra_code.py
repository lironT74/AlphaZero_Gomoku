import copy
import numpy as np
import matplotlib.pyplot as plt
import math
import replay_python3 as rp
# import computational_model as cm
import config
import json
# from pyemd import emd
import random
import bisect
import os
from scipy.optimize import minimize
from board import *
from mpl_toolkits.axes_grid1 import AxesGrid
from utils import *


import board as b
import collections
# from pyemd import emd
from scipy import stats
# from cv2 import *


# these are the files with user data foeach of the board
LOGFILE = ['logs/6_hard_full_dec19.csv','logs/6_hard_pruned_dec19.csv','logs/10_hard_full_dec19.csv','logs/10_hard_pruned_dec19.csv', 'logs/6_easy_full_dec19.csv','logs/6_easy_pruned_dec19.csv','logs/10_easy_full_dec19.csv','logs/10_easy_pruned_dec19.csv', 'logs/10_medium_full_dec19.csv','logs/10_medium_pruned_dec19.csv']
# these are the boards starting positions (1 = X, 2 = O)
START_POSITION = [[[0,2,0,0,1,0],[0,2,1,2,0,0],[0,1,0,0,0,0],[0,1,0,2,0,0],[0,1,0,0,0,0],[0,2,0,0,2,0]],
                  [[0,2,0,1,1,0],[0,2,1,2,0,0],[0,1,0,0,0,0],[2,1,0,2,0,0],[0,1,0,0,0,0],[0,2,0,0,2,0]],
                  [[0,0,0,2,0,0,0,0,0,0],[0,0,0,1,0,2,0,0,0,0],[0,2,2,0,0,1,1,0,2,0],[0,0,2,1,2,0,0,0,0,0],[0,1,1,0,0,0,0,0,0,0],[0,1,1,0,2,0,0,0,0,0],[0,0,1,0,2,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,2,0,0,2,2,0,0,0],[0,0,0,0,1,0,0,0,0,0]],
                 [[0,0,0,2,0,0,0,0,0,0],[0,0,0,1,0,2,0,0,0,0],[0,2,2,0,1,1,1,0,2,0],[0,0,2,1,2,0,0,0,0,0],[0,1,1,0,0,0,0,0,0,0],[0,1,1,0,2,0,0,0,0,0],[2,0,1,0,2,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,2,0,0,2,2,0,0,0],[0,0,0,0,1,0,0,0,0,0]],
                  [[0,1,0,2,0,0],[0,2,1,1,0,0],[1,2,2,2,1,0],[2,0,1,1,2,0],[1,0,2,2,0,0],[0,0,0,0,0,0]],
                  [[0,1,2,2,0,0],[0,2,1,1,0,0],[1,2,2,2,1,0],[2,0,1,1,2,1],[1,0,2,2,0,0],[0,0,0,0,0,0]],
                [[0,0,0,0,1,0,2,0,0,0],[0,0,0,0,2,1,1,1,0,0],[0,0,0,1,2,2,2,1,0,0],[0,0,0,2,2,1,1,2,1,1],[2,0,0,1,0,2,2,0,0,0],[1,0,0,0,0,0,0,0,0,0],[1,1,0,0,0,0,0,0,0,0],[2,2,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,2,2,2,0,0]],
                  [[0,0,0,0,1,2,2,0,0,0],[0,0,0,0,2,1,1,1,0,0],[0,0,0,1,2,2,2,1,0,0],[0,0,0,2,2,1,1,2,1,1],[2,0,0,1,0,2,2,0,0,1],[1,0,0,0,0,0,0,0,0,0],[1,1,0,0,0,0,0,0,0,0],[2,2,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,2,2,2,0,0]],
                [[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,2,0,0,0,0],[0,0,0,1,1,0,0,0,0,0],[0,0,0,0,2,2,2,1,2,0],[0,0,0,0,0,1,2,2,0,0],[0,0,0,1,0,2,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,2,0,0,0]],
                 [[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,2,0,0,0,0],[0,0,1,1,1,2,0,0,0,0],[0,0,0,0,2,2,2,1,2,0],[0,0,0,0,0,1,2,2,0,0],[0,0,0,1,0,2,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,2,0,0,0]]
                  ]
MOVES_TO_WIN = [10,8,10,8,8,6,8,6,10,8]

INFINITY_O = 10

BOARDS_MINUS_1 = []
WIN_SCORE = 100


def get_positions_shutter(row, col, board_matrix, shutter_size=0):
    prob = 1.0
    if not shutter_size.is_integer():
        prob = shutter_size - math.floor(shutter_size)
    active_squares = get_open_paths_through_square_method2_x_o(row, col, board_matrix, player='X')
    distances = np.zeros(len(active_squares))
    # active_squares = get_open_paths_through_square_old(row, col, board_matrix, player='X')

    if len(active_squares) > 0:
        for i in range(1, int(math.ceil(shutter_size)+1)):
            new_squares = expand_neighborhood(active_squares, len(board_matrix), prob=prob)
            active_squares.extend(new_squares)
            distances = np.concatenate(distances, np.full(len(new_squares), i))
    else:
        return []
    active_squares = remove_duplicates(active_squares)
    # remove occupied spaces
    active_free_squares = []
    for square in active_squares:
        if board_matrix[square[0]][square[1]] == 0:
            active_free_squares.append(square)
    return active_free_squares, distances


def convert_matrix_xo(board_matrix):
    # print board_matrix
    for i in range(len(board_matrix)):
        for j in range(len(board_matrix[i])):
            if (board_matrix[i][j] != 1) & (board_matrix[i][j] != 2):
                board_matrix[i][j] = int(board_matrix[i][j])
            elif board_matrix[i][j] == 1:
                board_matrix[i][j] = 'X'
            elif board_matrix[i][j] == 2:
                board_matrix[i][j] = 'O'


def normalize_matrix(score_matrix, with_negative=False):
    sum_scores = 0.0
    counter = 0.0
    for r in range(0,len(score_matrix)):
        for j in range(0,len(score_matrix[r])):
            if score_matrix[r][j] == 'X':
                score_matrix[r][j] = -0.00001
            elif score_matrix[r][j] == 'O':
                score_matrix[r][j] = -0.00002
            else:
                counter += 1.0
                if score_matrix[r][j] > 0:
                    sum_scores += score_matrix[r][j]

    for r in range(len(score_matrix)):
        for c in range(len(score_matrix[r])):
            if (score_matrix[r][c] != -0.00001) & (score_matrix[r][c] != -0.00002):
                if sum_scores == 0:
                    score_matrix[r][c] = 1.0/counter
                else:
                    # TODO: change if we don't want to eliminate negative scores
                    if score_matrix[r][c] >= 0:
                        score_matrix[r][c] = score_matrix[r][c]/sum_scores
                    else:
                        score_matrix[r][c] = 0


def add_noise_to_scores(score_matrix, mu=0.0, sigma=1.0):
    for r in range(0,len(score_matrix)):
        for j in range(0,len(score_matrix[r])):
            if (score_matrix[r][j] != 'X') & (score_matrix[r][j] != 'O'):
                score_matrix[r][j] += np.random.normal(mu, sigma)


def compute_paths_scores_for_matrix(board_mat, player='X', normalized=False, exp=1, o_weight=0.5, interaction=True, block=False, shutter=False, shutter_size=0, prev_x_move=None, board_obj=None, pruned_squares=None, noise = 0, linear=False):
    """
    computes the score for each cell in each of the boards based in the layers approach (first filter cells by density)
    :param exp: parameter creates the non-linearity (i.e., 2 --> squared)
    :param o_weight says how much weight to give for blocking O paths
    :param player: which player is playing now
    :param normalized: whether to normalize scorse to sum up to 1 (for transition probabilities)
    :param board_mat: the board
    :param interaction: whether the heuristic considers interaction between paths
    :param block: whether to use blocking heuristic (extra points for threatening)
    :param shutter: whether to use a shutter for pruning the space
    :param shutter_size: if using shutter, what radius to look at
    """
    board_matrix = copy.deepcopy(board_mat)
    convert_matrix_xo(board_matrix)
    winning_moves_alpha_beta = []
    # compute path scores
    score_matrix = copy.deepcopy(board_matrix)

    paths_data = copy.deepcopy(board_matrix)

    x_turn = True
    o_turn = False
    if player == 'O':
        x_turn = False
        o_turn = True
    active_squares = []
    for r in range(len(board_matrix)):
        for c in range(len(board_matrix[r])):
            if board_matrix[r][c] == 0:
                active_squares.append([r,c])
    if shutter:
        if prev_x_move is not None:
            active_squares, active_squares_distances = get_positions_shutter(prev_x_move[0], prev_x_move[1], board_matrix, shutter_size)

    add_to_dict = False
    prev_computed_scores = None
    if config.SCORES_DICT is not None:
        if str(board_matrix) in config.SCORES_DICT.keys():
            prev_computed_scores = config.SCORES_DICT[str(board_matrix)]
            config.HITS += 1
        else:
            add_to_dict = True
            config.NO_HIT += 1

    for r in range(len(board_matrix)):
        for c in range(len(board_matrix[r])):
            if board_matrix[r][c] == 0:  # only check if free
                # no need to compute heuristic if not active square and shutter is on
                # if ([r, c] not in active_squares) & shutter & (len(active_squares) > 0):  # if there are available positions within shutter, and square is not in them, give score 0
                #     square_score = 0
                #     score_matrix[r][c] = square_score
                #     continue

                if prev_computed_scores is not None:
                    score_matrix[r][c] = prev_computed_scores[r][c]
                    continue

                x_paths = compute_open_paths_data_interaction_new(r, c, board_matrix,player_turn=x_turn,exp=exp,interaction=interaction, linear=linear)
                square_score_x = x_paths[0]
                x_paths_data = []
                for path in x_paths[1]:
                    x_paths_data.append(path[2])
                paths_data[r][c] = copy.deepcopy(x_paths_data)
                o_paths = compute_open_paths_data_interaction_new(r, c, board_matrix,player='O', player_turn=o_turn, exp=exp, interaction=interaction, linear=linear)
                square_score_o = o_paths[0]

                streak_size = 4
                if len(board_matrix) == 10:
                    streak_size = 5
                if block & (x_paths[2] == (streak_size-1)) & x_turn:  # give score for blocking O
                    # square_score_o = INFINITY_O
                    square_score_x += INFINITY_O
                elif block & (o_paths[2] == (streak_size-1)) & o_turn:  # give score for blocking X
                    # square_score_x = INFINITY_O
                    square_score_o += INFINITY_O
                if o_weight == 0.5:
                    square_score = square_score_x + square_score_o
                    # if x_turn:
                    #     square_score = square_score_x
                    # else:
                    #     square_score = square_score_o
                elif o_weight == 0:
                    square_score = square_score_x  # o blindness for x player disregard O
                elif o_weight == 1.0:
                    square_score = square_score_x  # o blindness - just use for score how good it would be to block x
                if square_score > WIN_SCORE:
                    # winning_moves_alpha_beta.append((r,c))
                    square_score = WIN_SCORE

                score_matrix[r][c] = square_score

    # check for immediate win/loss
    winning_moves = check_immediate_win(board_matrix, player, board_obj=board_obj)
    if (len(winning_moves) > 0) & ((player != 'O') | (o_weight > 0)):
        for move in winning_moves:
            move_row, move_col = convert_position(move, len(board_matrix))
            score_matrix[move_row][move_col] = WIN_SCORE

    if len(winning_moves) == 0:  # if can't win immediately, check if opponent can win immediately
        other_player = 'O'
        if player == 'O':
            other_player = 'X'
        winning_moves_opp = check_immediate_win(board_matrix, other_player, board_obj=board_obj)
        if len(winning_moves_opp) > 1:
            for row in range(len(board_matrix)):
                for col in range(len(board_matrix[row])):
                    if (score_matrix[row][col] != 'X') & (score_matrix[row][col] != 'O'):
                        # update scores for losing moves, except if we are in o blindness mode
                        if (player != 'X') | (o_weight > 0):
                            score_matrix[row][col] = -1*WIN_SCORE

        elif len(winning_moves_opp) == 1:  # give high score to blocking winning move, and losing score to rest
            move_row, move_col = convert_position(winning_moves_opp[0], len(board_matrix))
            for row in range(len(board_matrix)):
                for col in range(len(board_matrix[row])):
                    if (move_row != row) | (move_col != col):
                        if (score_matrix[row][col] != 'X') & (score_matrix[row][col] != 'O'):
                            if (player != 'X') | (o_weight > 0):
                                score_matrix[row][col] = -1*WIN_SCORE
                    else:
                        score_matrix[row][col] = INFINITY_O



    if add_to_dict:
        config.SCORES_DICT[str(board_matrix)] = score_matrix

    score_matrix_all_squares = copy.deepcopy(score_matrix)
    distances_matrix = np.zeros((len(score_matrix), len(score_matrix)))
    if shutter:  # zero scores for non active squares
        for r in range(len(board_matrix)):
            for c in range(len(board_matrix[r])):
                if ((r, c) not in active_squares) & shutter & (len(active_squares) > 0):  # if there are available positions within shutter, and square is not in them, give score 0
                    square_score = 0
                    score_matrix[r][c] = square_score


        for i in range(active_squares):
            (r, c) = active_squares[i]
            distances_matrix[r][c] = active_squares_distances[i]


    for row in range(len(board_matrix)):
        for col in range(len(board_matrix[row])):
            if (score_matrix[row][col] != 'X') & (score_matrix[row][col] != 'O'):
                if score_matrix[row][col] >= WIN_SCORE:
                    winning_moves_alpha_beta.append((row, col))

    if noise > 0:
        add_noise_to_scores(score_matrix, mu=0.0, sigma=noise)



    if normalized:
        normalize_matrix(score_matrix, False)




    alpha_beta = True  # TODO: remove later
    if alpha_beta:
        return score_matrix, len(active_squares), winning_moves_alpha_beta, score_matrix_all_squares, distances_matrix
    return score_matrix


def compute_scores_density_new(board_mat, player='X', normalized=False, neighborhood_size=1, density = 'reg', sig=3):

    board_matrix = copy.deepcopy(board_mat)
    convert_matrix_xo(board_matrix)

    density_score_matrix = copy.deepcopy(board_matrix)

    if density=='guassian':
        # create guassians for each X square
        guassian_kernel = []
        for r in range(len(board_matrix)):
            for c in range(len(board_matrix[r])):
                if board_matrix[r][c] == 'X':
                    guassian_kernel.append(makeGaussian(len(board_matrix),fwhm=sig,center=[r,c]))

    for r in range(len(board_matrix)):
        for c in range(len(board_matrix[r])):
            if board_matrix[r][c] == 0:  # only check if free
                if density == 'guassian':
                    square_score = compute_density_guassian(r, c, board_matrix, guassian_kernel)  # check neighborhood
                else:
                    square_score = compute_density(r, c, board_matrix, neighborhood_size, player=player)  # check neighborhood
                density_score_matrix[r][c] = square_score

    winning_moves = check_immediate_win(board_matrix, player)
    if len(winning_moves) > 0:
        for move in winning_moves:
            move_row, move_col = convert_position(move, len(board_matrix))
            density_score_matrix[move_row][move_col] = WIN_SCORE

    if normalized:
        normalize_matrix(density_score_matrix, False)

    return density_score_matrix


'''
computes the density score for a cell
@neighborhood_size: how many cells around the square to look at
'''
def compute_density(row, col, board, neighborhood_size, player = 'X'):
    x_count = 0.0
    density_score = 0.0
    for i in range(-1*neighborhood_size,neighborhood_size+1):
        for j in range(-1*neighborhood_size,neighborhood_size+1):
            if (i != 0) | (j != 0):
                r = row + i
                c = col + j
                if (r < len(board)) & (r >= 0) & (c < len(board)) & (c >= 0):
                    # print r
                    # print c
                    if board[r][c] == player:
                        x_count += 1.0
                        density_score += 1.0/(8*max(abs(i), abs(j)))

    return density_score

'''
computes the density score for a cell using guassians
@guassian_kernel: this is created separately and sent to the function
'''
def compute_density_guassian(row, col, board, guassian_kernel):
    density_score = 0.0
    for guas in guassian_kernel:
        density_score += guas[row][col]

    return density_score

'''
computes the density score for all cells in all boards using guassian approach
@normalized: whether to normalize the scores such that they sum up to 1
@sig: the standard deviation to use in guassian
@lamb: this is what I used before for the quantal response, you can ignore it
'''
def compute_scores_density_guassian(normalized=False, sig = 3, lamb = 1):
    data_matrices = {}  # this will be a dictionary that holds the matrics for all the boards, each will be indexed by the board name
    for g in range(len(LOGFILE)):  # iterate over all boards
        board_matrix = copy.deepcopy(START_POSITION[g])  # gets the board starting position
        for i in range(len(board_matrix)):  # replaces 1s with Xs and 2s with Os
            for j in range(len(board_matrix[i])):
                if ((board_matrix[i][j]!=1) & (board_matrix[i][j]!=2)):
                    board_matrix[i][j] = int(board_matrix[i][j])
                elif (board_matrix[i][j]==1):
                    board_matrix[i][j]='X'
                elif (board_matrix[i][j]==2):
                    board_matrix[i][j]='O'

        # print board_matrix

        sum_scores = 0.0
        sum_scores_exp = 0.0
        score_matrix = copy.deepcopy(board_matrix)  # this will hold the matrix of scores

        # create guassians for each X square
        guassian_kernel = []
        for r in range(len(board_matrix)):
            for c in range(len(board_matrix[r])):
                if board_matrix[r][c] == 'X':
                    guassian_kernel.append(makeGaussian(len(board_matrix),fwhm=sig,center=[r,c]))

        # compute density scores
        for r in range(len(board_matrix)):
            for c in range(len(board_matrix[r])):
                if board_matrix[r][c] == 0:  # only check if cell is free
                    square_score = compute_density_guassian(r,c,board_matrix,guassian_kernel)
                    score_matrix[r][c] = square_score
                    sum_scores += square_score
                    sum_scores_exp += math.pow(math.e,lamb*square_score)

        # put a negative small value instead of X and O so it doesn't affect heatmaps
        for r in range(0,len(score_matrix)):
            for j in range(0,len(score_matrix[r])):
                if (score_matrix[r][j]=='X'):
                    score_matrix[r][j] = -0.00001
                elif (score_matrix[r][j]=='O'):
                    score_matrix[r][j] = -0.00002

        if normalized:  # normalize cell scores to 1
            for r in range(len(score_matrix)):
                for c in range(len(score_matrix[r])):
                    # score_matrix[r][c] = score_matrix[r][c]/sum_scores
                    if (score_matrix[r][c]!=-0.00001) & (score_matrix[r][c]!=-0.00002):
                        score_matrix[r][c] = score_matrix[r][c]/sum_scores
                        # score_matrix[r][c] = (math.pow(math.e, lamb*score_matrix[r][c]))/sum_scores_exp

        board_name = LOGFILE[g]
        board_name = board_name[:-4]
        data_matrices[board_name[5:-6]] = score_matrix  # store matrix for this board in the dictionary

    return data_matrices

'''
computes the density score for all cells based on neighbors
@normalized: whether to normalize the scores such that they sum up to 1
@neighborhood_size: how far to look for neighbors
@lamb: this is what I used before for the quantal response, you can ignore it
'''
def compute_scores_density(normalized=False, neighborhood_size=1, lamb=1, player='X'):
    data_matrices = {}
    for g in range(len(LOGFILE)):
        board_matrix = copy.deepcopy(START_POSITION[g])
        for i in range(len(board_matrix)):
            for j in range(len(board_matrix[i])):
                if ((board_matrix[i][j]!=1) & (board_matrix[i][j]!=2)):
                    board_matrix[i][j] = int(board_matrix[i][j])
                elif (board_matrix[i][j]==1):
                    board_matrix[i][j]='X'
                elif (board_matrix[i][j]==2):
                    board_matrix[i][j]='O'

        # print board_matrix

        sum_scores = 0.0
        sum_scores_exp = 0.0
        score_matrix = copy.deepcopy(board_matrix)

        for r in range(len(board_matrix)):
            for c in range(len(board_matrix[r])):
                if board_matrix[r][c] == 0:  # only check if free
                    square_score = compute_density(r, c, board_matrix, neighborhood_size, player=player)  # check neighborhood
                    score_matrix[r][c] = square_score
                    sum_scores += square_score
                    sum_scores_exp += math.pow(math.e,lamb*square_score)


        # heatmaps
        for r in range(0,len(score_matrix)):
            for j in range(0,len(score_matrix[r])):
                if (score_matrix[r][j]=='X'):
                    score_matrix[r][j] = -0.00001
                elif (score_matrix[r][j]=='O'):
                    score_matrix[r][j] = -0.00002

        if normalized:
            for r in range(len(score_matrix)):
                for c in range(len(score_matrix[r])):
                    # score_matrix[r][c] = score_matrix[r][c]/sum_scores
                    if (score_matrix[r][c]!=-0.00001) & (score_matrix[r][c]!=-0.00002):
                        score_matrix[r][c] = score_matrix[r][c]/sum_scores

        board_name = LOGFILE[g]
        board_name = board_name[:-4]
        data_matrices[board_name[5:-6]] = score_matrix
    matrix_name = 'data_matrices/model_density' + '_nbr=' +str(neighborhood_size)
    write_matrices_to_file(data_matrices, matrix_name + '.json')
    return data_matrices

'''
checks if the two paths can be blocked by a shared cell
'''
def check_path_overlap(empty1, empty2):
    for square in empty1:
        if square in empty2:
            return True
    return False


'''
computes the potential winning paths the cell is part of, of if you send it player = 'O' it will check how many
paths X destroys if it places an X on this cell
the @exp parameter creates the non-linearity (i.e., 2 --> squared)
'''
def compute_open_paths_data(row, col, board, exp=1, player = 'X', interaction=True):
    other_player = 'O'
    if player == 'O':
        other_player = 'X'


    max_length_path = 0
    threshold = 0
    # if len(board)==10:
    #     threshold +=1

    streak_size = 4  # how many Xs in a row you need to win
    if len(board) == 10:  # if it's a 10X10 board you need 5.
        streak_size = 5

    open_paths_data = []  # this list will hold information on all the potential paths, each path will be represented by a pair (length and empty squares, which will be used to check overlap)
    paths = []
    # check right-down diagonal (there is a more efficient way to look at all the paths, but it was easier for me to debug when separating them :)
    for i in range(streak_size):
        r = row - i
        c = col - i
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        empty_squares = []
        path = []
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == other_player:
                blocked = True
            elif board[square_row][square_col] == player:
                path_x_count += 1
            elif ((square_col != col) | (square_row != row)) | (other_player=='X'):
                empty_squares.append([square_row,square_col])
            path.append([square_row,square_col])
            square_row += 1
            square_col += 1
            path_length += 1

        if (path_length == streak_size) & (not blocked) & ((path_x_count>threshold) | ((other_player == 'O') & path_x_count+1>threshold)):  # add the path if it's not blocked and if there is already at least one X on it
            if other_player == 'O':
                open_paths_data.append((path_x_count+1,empty_squares, path))
                if (path_x_count+1) > max_length_path:
                    max_length_path = path_x_count+1
            elif (path_x_count>threshold):
                open_paths_data.append((path_x_count,empty_squares, path))

    # check left-down diagonal
    for i in range(streak_size):
        r = row - i
        c = col + i
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        empty_squares = []
        path = []
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == other_player:
                blocked = True
            elif board[square_row][square_col] == player:
                path_x_count += 1
            elif ((square_col != col) | (square_row != row)) | (other_player=='X'):
                empty_squares.append([square_row,square_col])
            path.append([square_row,square_col])
            square_row += 1
            square_col -= 1
            path_length += 1

        if (path_length == streak_size) & (not blocked) & (path_x_count>threshold): # add the path if it's not blocked and if there is already at least one X on it
            if other_player == 'O':
                open_paths_data.append((path_x_count+1,empty_squares, path))
                if (path_x_count+1) > max_length_path:
                    max_length_path = path_x_count+1
            elif (path_x_count>threshold):
                open_paths_data.append((path_x_count,empty_squares, path))

    # check vertical
    for i in range(streak_size):
        r = row - i
        c = col
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        empty_squares = []
        path = []
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == other_player:
                blocked = True
            elif board[square_row][square_col] == player:
                path_x_count += 1
            elif ((square_col != col) | (square_row != row)) | (other_player=='X'):
                empty_squares.append([square_row,square_col])

            path.append([square_row,square_col])
            square_row += 1
            path_length += 1



        if (path_length == streak_size) & (not blocked) & (path_x_count>threshold): # add the path if it's not blocked and if there is already at least one X on it
            if other_player == 'O':
                open_paths_data.append((path_x_count+1,empty_squares, path))
                if (path_x_count+1) > max_length_path:
                    max_length_path = path_x_count+1
            elif (path_x_count>threshold):
                open_paths_data.append((path_x_count,empty_squares, path))

    # check horizontal
    for i in range(streak_size):
        r = row
        c = col - i
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        empty_squares = []
        path = []
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == other_player:
                blocked = True
            elif board[square_row][square_col] == player:
                path_x_count += 1
            elif ((square_col != col) | (square_row != row)) | (other_player=='X'):
                empty_squares.append([square_row, square_col])

            path.append([square_row, square_col])
            square_col += 1
            path_length += 1

        if (path_length == streak_size) & (not blocked) & (path_x_count>threshold):  # add the path if it's not blocked and if there is already at least one X on it
            if other_player == 'O':
                open_paths_data.append((path_x_count+1,empty_squares, path))
                if (path_x_count+1) > max_length_path:
                    max_length_path = path_x_count+1
            elif (path_x_count>threshold):
                open_paths_data.append((path_x_count,empty_squares, path))


    score = 0.0
    # compute the score for the cell based on the potential paths
    for i in range(len(open_paths_data)):
        p1 = open_paths_data[i]
        score += 1.0/math.pow((streak_size-p1[0]), exp)  # score for individual path
        if interaction:
            for j in range(i+1, len(open_paths_data)):
                p2 = open_paths_data[j]
                if (not(check_path_overlap(p1[1],p2[1]))) | (player == 'O'):  # interaction score if the paths don't overlap
                    # score += 1.0/(math.pow(((streak_size-1)*(streak_size-1))-(p1[0]*p2[0]), exp))
                    old_interaction = 1.0/(math.pow(((streak_size-1)*(streak_size-1))-(p1[0]*p2[0]), exp))
                    top = 0.0 +p1[0]*p2[0]
                    bottom = ((streak_size-1)*(streak_size-1))-(p1[0]*p2[0])
                    new_interaction = math.pow(top/bottom, exp)
                    score += math.pow(top/bottom, exp)

    return (score, open_paths_data, max_length_path)


def compute_open_paths_data_interaction_potential(row, col, board, exp=1, player = 'X', interaction=True):
    other_player = 'O'
    if player == 'O':
        other_player = 'X'


    max_length_path = 0
    threshold = 0
    # if len(board)==10:
    #     threshold +=1

    streak_size = 4  # how many Xs in a row you need to win
    if len(board) == 10:  # if it's a 10X10 board you need 5.
        streak_size = 5

    open_paths_data = []  # this list will hold information on all the potential paths, each path will be represented by a pair (length and empty squares, which will be used to check overlap)
    paths = []
    # check right-down diagonal (there is a more efficient way to look at all the paths, but it was easier for me to debug when separating them :)
    for i in range(streak_size):
        r = row - i
        c = col - i
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        empty_squares = []
        path = []
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == other_player:
                blocked = True
            elif board[square_row][square_col] == player:
                path_x_count += 1
            elif (square_col != col) | (square_row != row):
                empty_squares.append([square_row,square_col])
            path.append([square_row,square_col])
            square_row += 1
            square_col += 1
            path_length += 1

        if (path_length == streak_size) & (not blocked) & (path_x_count+1>threshold):  # add the path if it's not blocked and if there is already at least one X on it
            open_paths_data.append((path_x_count+1,empty_squares, path))
            if (path_x_count+1) > max_length_path:
                max_length_path = path_x_count+1


    # check left-down diagonal
    for i in range(streak_size):
        r = row - i
        c = col + i
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        empty_squares = []
        path = []
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == other_player:
                blocked = True
            elif board[square_row][square_col] == player:
                path_x_count += 1
            elif (square_col != col) | (square_row != row):
                empty_squares.append([square_row,square_col])
            path.append([square_row,square_col])
            square_row += 1
            square_col -= 1
            path_length += 1

        if (path_length == streak_size) & (not blocked) & (path_x_count+1>threshold):  # add the path if it's not blocked and if there is already at least one X on it
            open_paths_data.append((path_x_count+1,empty_squares, path))
            if (path_x_count+1) > max_length_path:
                max_length_path = path_x_count+1

    # check vertical
    for i in range(streak_size):
        r = row - i
        c = col
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        empty_squares = []
        path = []
        square_row = r
        square_col = c

        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == other_player:
                blocked = True
            elif board[square_row][square_col] == player:
                path_x_count += 1
            elif (square_col != col) | (square_row != row):
                empty_squares.append([square_row,square_col])
            path.append([square_row,square_col])
            square_row += 1
            path_length += 1

        if (path_length == streak_size) & (not blocked) & (path_x_count+1>threshold):  # add the path if it's not blocked and if there is already at least one X on it
            open_paths_data.append((path_x_count+1,empty_squares, path))
            if (path_x_count+1) > max_length_path:
                max_length_path = path_x_count+1


    # check horizontal
    for i in range(streak_size):
        r = row
        c = col - i
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        empty_squares = []
        path = []
        square_row = r
        square_col = c

        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == other_player:
                blocked = True
            elif board[square_row][square_col] == player:
                path_x_count += 1
            elif (square_col != col) | (square_row != row):
                empty_squares.append([square_row,square_col])
            path.append([square_row,square_col])
            square_col += 1
            path_length += 1

        if (path_length == streak_size) & (not blocked) & (path_x_count+1>threshold):  # add the path if it's not blocked and if there is already at least one X on it
            open_paths_data.append((path_x_count+1,empty_squares, path))
            if (path_x_count+1) > max_length_path:
                max_length_path = path_x_count+1

    score = 0.0
    # compute the score for the cell based on the potential paths
    for i in range(len(open_paths_data)):
        p1 = open_paths_data[i]
        score += 1.0/math.pow((streak_size-p1[0]), exp)  # score for individual path
        if interaction:
            for j in range(i+1, len(open_paths_data)):
                p2 = open_paths_data[j]
                if check_path_overlap(p2[2],p1[2]):
                    if (not(check_path_overlap(p1[1],p2[1]))):  # interaction score if the paths don't overlap
                        if ((streak_size-1)*(streak_size-1)) == (p1[0]*p2[0]):
                            score = INFINITY_O
                        else:
                            top = 0.0 +p1[0]*p2[0]
                            bottom = ((streak_size-1)*(streak_size-1))-(p1[0]*p2[0])
                            score += math.pow(top/bottom, exp)

    return (score, open_paths_data, max_length_path)


def compute_open_paths_data_interaction(row, col, board, exp=1, player = 'X', interaction=True):
    other_player = 'O'
    if player == 'O':
        other_player = 'X'


    max_length_path = 0
    threshold = 0
    # if len(board)==10:
    #     threshold +=1

    streak_size = 4  # how many Xs in a row you need to win
    if len(board) == 10:  # if it's a 10X10 board you need 5.
        streak_size = 5

    open_paths_data = []  # this list will hold information on all the potential paths, each path will be represented by a pair (length and empty squares, which will be used to check overlap)
    paths = []
    # check right-down diagonal (there is a more efficient way to look at all the paths, but it was easier for me to debug when separating them :)
    for i in range(streak_size):
        r = row - i
        c = col - i
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        empty_squares = []
        path = []
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == other_player:
                blocked = True
            elif board[square_row][square_col] == player:
                path_x_count += 1
            elif ((square_col != col) | (square_row != row)):
                empty_squares.append([square_row,square_col])
            path.append([square_row,square_col])
            square_row += 1
            square_col += 1
            path_length += 1

        if (path_length == streak_size) & (not blocked) & ((path_x_count>threshold) | ((other_player == 'O') & path_x_count+1>threshold)):  # add the path if it's not blocked and if there is already at least one X on it
            if other_player == 'O':
                open_paths_data.append((path_x_count+1,empty_squares, path))
                if (path_x_count+1) > max_length_path:
                    max_length_path = path_x_count+1
            elif (path_x_count>threshold):
                open_paths_data.append((path_x_count,empty_squares, path))

    # check left-down diagonal
    for i in range(streak_size):
        r = row - i
        c = col + i
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        empty_squares = []
        path = []
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == other_player:
                blocked = True
            elif board[square_row][square_col] == player:
                path_x_count += 1
            elif ((square_col != col) | (square_row != row)):
                empty_squares.append([square_row,square_col])
            path.append([square_row,square_col])
            square_row += 1
            square_col -= 1
            path_length += 1

        if (path_length == streak_size) & (not blocked) & ((path_x_count>threshold) | ((other_player == 'O') & path_x_count+1>threshold)): # add the path if it's not blocked and if there is already at least one X on it
            if other_player == 'O':
                open_paths_data.append((path_x_count+1,empty_squares, path))
                if (path_x_count+1) > max_length_path:
                    max_length_path = path_x_count+1
            elif (path_x_count>threshold):
                open_paths_data.append((path_x_count,empty_squares, path))

    # check vertical
    for i in range(streak_size):
        r = row - i
        c = col
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        empty_squares = []
        path = []
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == other_player:
                blocked = True
            elif board[square_row][square_col] == player:
                path_x_count += 1
            elif ((square_col != col) | (square_row != row)):
                empty_squares.append([square_row,square_col])

            path.append([square_row,square_col])
            square_row += 1
            path_length += 1



        if (path_length == streak_size) & (not blocked) & ((path_x_count>threshold) | ((other_player == 'O') & path_x_count+1>threshold)): # add the path if it's not blocked and if there is already at least one X on it
            if other_player == 'O':
                open_paths_data.append((path_x_count+1,empty_squares, path))
                if (path_x_count+1) > max_length_path:
                    max_length_path = path_x_count+1
            elif (path_x_count>threshold):
                open_paths_data.append((path_x_count,empty_squares, path))

    # check horizontal
    for i in range(streak_size):
        r = row
        c = col - i
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        empty_squares = []
        path = []
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == other_player:
                blocked = True
            elif board[square_row][square_col] == player:
                path_x_count += 1
            elif ((square_col != col) | (square_row != row)):
                empty_squares.append([square_row, square_col])

            path.append([square_row, square_col])
            square_col += 1
            path_length += 1

        if (path_length == streak_size) & (not blocked) & ((path_x_count>threshold) | ((other_player == 'O') & path_x_count+1>threshold)):  # add the path if it's not blocked and if there is already at least one X on it
            if other_player == 'O':
                open_paths_data.append((path_x_count+1,empty_squares, path))
                if (path_x_count+1) > max_length_path:
                    max_length_path = path_x_count+1
            elif (path_x_count>threshold):
                open_paths_data.append((path_x_count,empty_squares, path))


    score = 0.0
    # compute the score for the cell based on the potential paths
    for i in range(len(open_paths_data)):
        p1 = open_paths_data[i]

        if streak_size == p1[0]:
            score = WIN_SCORE
        else:
            score += 1.0/math.pow((streak_size-p1[0]), exp)  # score for individual path
        if interaction:
            for j in range(i+1, len(open_paths_data)):
                p2 = open_paths_data[j]
                if check_path_overlap(p2[2],p1[2]):
                    if (not(check_path_overlap(p1[1],p2[1]))):  # interaction score if the paths don't overlap
                        top = 0.0 +p1[0]*p2[0]
                        bottom = ((streak_size-1)*(streak_size-1))-(p1[0]*p2[0])
                        if bottom == 0:
                            score = WIN_SCORE
                        else:
                            score += math.pow(top/bottom, exp)

    return (score, open_paths_data, max_length_path)


def compute_open_paths_data_interaction_new(row, col, board, exp=1, player='X', player_turn=True, interaction=True, linear=False):
    other_player = 'O'
    if player == 'O':
        other_player = 'X'

    max_length_path = 0
    threshold = 0
    # if len(board)==10:
    #     threshold +=1

    streak_size = 4  # how many Xs in a row you need to win
    if len(board) == 10:  # if it's a 10X10 board you need 5.
        streak_size = 5

    open_paths_data = []  # this list will hold information on all the potential paths, each path will be represented by a pair (length and empty squares, which will be used to check overlap)
    paths = []
    # check right-down diagonal (there is a more efficient way to look at all the paths, but it was easier for me to debug when separating them :)
    for i in range(streak_size):
        r = row - i
        c = col - i
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        empty_squares = []
        path = []
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == other_player:
                blocked = True
            elif board[square_row][square_col] == player:
                path_x_count += 1
            elif ((square_col != col) | (square_row != row)):
                empty_squares.append([square_row,square_col])
            path.append([square_row,square_col])
            square_row += 1
            square_col += 1
            path_length += 1

        if (path_length == streak_size) & (not blocked) & ((path_x_count>threshold) | ((player_turn) & (path_x_count+1)>threshold)):
            # add the path if it's not blocked and if there is already at least one X on it
            if player_turn:
                open_paths_data.append((path_x_count+1, empty_squares, path))
                if (path_x_count+1) > max_length_path:
                    max_length_path = path_x_count+1
            elif path_x_count > threshold:
                open_paths_data.append((path_x_count,empty_squares, path))

    # check left-down diagonal
    for i in range(streak_size):
        r = row - i
        c = col + i
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        empty_squares = []
        path = []
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == other_player:
                blocked = True
            elif board[square_row][square_col] == player:
                path_x_count += 1
            elif ((square_col != col) | (square_row != row)):
                empty_squares.append([square_row,square_col])
            path.append([square_row,square_col])
            square_row += 1
            square_col -= 1
            path_length += 1

        if (path_length == streak_size) & (not blocked) & ((path_x_count>threshold) | ((player_turn) & path_x_count+1>threshold)): # add the path if it's not blocked and if there is already at least one X on it
            if player_turn:
                open_paths_data.append((path_x_count+1,empty_squares, path))
                if (path_x_count+1) > max_length_path:
                    max_length_path = path_x_count+1
            elif (path_x_count>threshold):
                open_paths_data.append((path_x_count,empty_squares, path))

    # check vertical
    for i in range(streak_size):
        r = row - i
        c = col
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        empty_squares = []
        path = []
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == other_player:
                blocked = True
            elif board[square_row][square_col] == player:
                path_x_count += 1
            elif ((square_col != col) | (square_row != row)):
                empty_squares.append([square_row,square_col])

            path.append([square_row,square_col])
            square_row += 1
            path_length += 1



        if (path_length == streak_size) & (not blocked) & ((path_x_count>threshold) | ((player_turn) & path_x_count+1>threshold)): # add the path if it's not blocked and if there is already at least one X on it
            if player_turn:
                open_paths_data.append((path_x_count+1,empty_squares, path))
                if (path_x_count+1) > max_length_path:
                    max_length_path = path_x_count+1
            elif (path_x_count>threshold):
                open_paths_data.append((path_x_count,empty_squares, path))

    # check horizontal
    for i in range(streak_size):
        r = row
        c = col - i
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        empty_squares = []
        path = []
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == other_player:
                blocked = True
            elif board[square_row][square_col] == player:
                path_x_count += 1
            elif ((square_col != col) | (square_row != row)):
                empty_squares.append([square_row, square_col])

            path.append([square_row, square_col])
            square_col += 1
            path_length += 1

        if (path_length == streak_size) & (not blocked) & ((path_x_count>threshold) | ((player_turn) & path_x_count+1>threshold)):  # add the path if it's not blocked and if there is already at least one X on it
            if player_turn:
                open_paths_data.append((path_x_count+1,empty_squares, path))
                if (path_x_count+1) > max_length_path:
                    max_length_path = path_x_count+1
            elif (path_x_count>threshold):
                open_paths_data.append((path_x_count,empty_squares, path))


    score = 0.0
    # compute the score for the cell based on the potential paths
    for i in range(len(open_paths_data)):
        p1 = open_paths_data[i]

        if streak_size == p1[0]:
            score = WIN_SCORE
        else:
            if linear:
                score += p1[0]  # score for individual path
            else:
                score += 1.0/math.pow((streak_size-p1[0]), exp)  # score for individual path
        if interaction:
            for j in range(i+1, len(open_paths_data)):
                p2 = open_paths_data[j]
                if check_path_overlap(p2[2],p1[2]):
                    if (not(check_path_overlap(p1[1],p2[1]))):  # interaction score if the paths don't overlap
                        top = 0.0 +p1[0]*p2[0]
                        bottom = ((streak_size-1)*(streak_size-1))-(p1[0]*p2[0])
                        if bottom == 0:
                            score = WIN_SCORE
                        else:
                            score += math.pow(top/bottom, exp)

    return (score, open_paths_data, max_length_path)


def compute_open_paths_data_potential(row, col, board, exp=1, player = 'X', interaction=True):
    other_player = 'O'
    if player == 'O':
        other_player = 'X'


    max_length_path = 0
    threshold = 0
    # if len(board)==10:
    #     threshold +=1

    streak_size = 4  # how many Xs in a row you need to win
    if len(board) == 10:  # if it's a 10X10 board you need 5.
        streak_size = 5

    open_paths_data = []  # this list will hold information on all the potential paths, each path will be represented by a pair (length and empty squares, which will be used to check overlap)
    paths = []
    # check right-down diagonal (there is a more efficient way to look at all the paths, but it was easier for me to debug when separating them :)
    for i in range(streak_size):
        r = row - i
        c = col - i
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        empty_squares = []
        path = []
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == other_player:
                blocked = True
            elif board[square_row][square_col] == player:
                path_x_count += 1
            elif ((square_col != col) | (square_row != row)) | (other_player=='X'):
                empty_squares.append([square_row,square_col])
            path.append([square_row,square_col])
            square_row += 1
            square_col += 1
            path_length += 1

        if (path_length == streak_size) & (not blocked) & ((path_x_count>threshold) | ((other_player == 'O') & path_x_count+1>threshold)):  # add the path if it's not blocked and if there is already at least one X on it
            if other_player == 'O':
                open_paths_data.append((path_x_count+1,empty_squares, path))
                if (path_x_count+1) > max_length_path:
                    max_length_path = path_x_count+1
            else:
                open_paths_data.append((path_x_count+1,empty_squares, path))

    # check left-down diagonal
    for i in range(streak_size):
        r = row - i
        c = col + i
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        empty_squares = []
        path = []
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == other_player:
                blocked = True
            elif board[square_row][square_col] == player:
                path_x_count += 1
            elif ((square_col != col) | (square_row != row)) | (other_player=='X'):
                empty_squares.append([square_row,square_col])
            path.append([square_row,square_col])
            square_row += 1
            square_col -= 1
            path_length += 1

        if (path_length == streak_size) & (not blocked) & (path_x_count>threshold): # add the path if it's not blocked and if there is already at least one X on it
            if other_player == 'O':
                open_paths_data.append((path_x_count+1,empty_squares, path))
                if (path_x_count+1) > max_length_path:
                    max_length_path = path_x_count+1
            else:
                open_paths_data.append((path_x_count+1,empty_squares, path))

    # check vertical
    for i in range(streak_size):
        r = row - i
        c = col
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        empty_squares = []
        path = []
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == other_player:
                blocked = True
            elif board[square_row][square_col] == player:
                path_x_count += 1
            elif ((square_col != col) | (square_row != row)) | (other_player=='X'):
                empty_squares.append([square_row,square_col])

            path.append([square_row,square_col])
            square_row += 1
            path_length += 1



        if (path_length == streak_size) & (not blocked) & (path_x_count>threshold): # add the path if it's not blocked and if there is already at least one X on it
            if other_player == 'O':
                open_paths_data.append((path_x_count+1,empty_squares, path))
                if (path_x_count+1) > max_length_path:
                    max_length_path = path_x_count+1
            else:
                open_paths_data.append((path_x_count+1,empty_squares, path))

    # check horizontal
    for i in range(streak_size):
        r = row
        c = col - i
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        empty_squares = []
        path = []
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == other_player:
                blocked = True
            elif board[square_row][square_col] == player:
                path_x_count += 1
            elif ((square_col != col) | (square_row != row)) | (other_player=='X'):
                empty_squares.append([square_row, square_col])

            path.append([square_row, square_col])
            square_col += 1
            path_length += 1

        if (path_length == streak_size) & (not blocked) & (path_x_count>threshold):  # add the path if it's not blocked and if there is already at least one X on it
            if other_player == 'O':
                open_paths_data.append((path_x_count+1,empty_squares, path))
                if (path_x_count+1) > max_length_path:
                    max_length_path = path_x_count+1
            else:
                open_paths_data.append((path_x_count+1,empty_squares, path))


    score = 0.0
    # compute the score for the cell based on the potential paths
    for i in range(len(open_paths_data)):
        p1 = open_paths_data[i]
        score += 1.0/math.pow((streak_size-p1[0]), exp)  # score for individual path
        if interaction:
            for j in range(i+1, len(open_paths_data)):
                p2 = open_paths_data[j]
                if (not(check_path_overlap(p1[1],p2[1]))):  # interaction score if the paths don't overlap
                    # score += 1.0/(math.pow(((streak_size-1)*(streak_size-1))-(p1[0]*p2[0]), exp))
                    old_interaction = 1.0/(math.pow(((streak_size-1)*(streak_size-1))-(p1[0]*p2[0]), exp))
                    top = 0.0 +p1[0]*p2[0]
                    bottom = ((streak_size-1)*(streak_size-1))-(p1[0]*p2[0])
                    new_interaction = math.pow(top/bottom, exp)
                    score += math.pow(top/bottom, exp)

    return (score, open_paths_data, max_length_path)



def compute_block_o_score(board, player = 'O', n = 3, interaction = True, exp = 2):

    open_paths_data = []  # this list will hold information on all the potential paths, each path will be represented by a pair (length and empty squares, which will be used to check overlap)
    for row in range(len(board)):
        for col in range(len(board)):
            other_player = 'X'
            if player == 'X':
                other_player = 'O'


            max_length_path = 0
            threshold = 0
            # if len(board)==10:
            #     threshold +=1

            streak_size = 4  # how many Xs in a row you need to win
            if len(board) == 10:  # if it's a 10X10 board you need 5.
                streak_size = 5


            # check right-down diagonal (there is a more efficient way to look at all the paths, but it was easier for me to debug when separating them :)
            # for i in range(streak_size):
            #     r = row - i
            #     c = col - i
            #     if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            #         continue
            blocked = False
            path_length = 0
            path_x_count = 0
            empty_squares = []
            path = []
            square_row = row
            square_col = col
            while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
                if board[square_row][square_col] == other_player:
                    blocked = True
                elif board[square_row][square_col] == player:
                    path_x_count += 1
                elif ((square_col != col) | (square_row != row)) | (other_player=='X'):
                    empty_squares.append([square_row,square_col])
                path.append([square_row,square_col])
                square_row += 1
                square_col += 1
                path_length += 1

            if (path_length == streak_size) & (not blocked) & (path_x_count>threshold):  # add the path if it's not blocked and if there is already at least one X on it
                if other_player == 'O':
                    open_paths_data.append((path_x_count+1,empty_squares, path))
                    if (path_x_count+1) > max_length_path:
                        max_length_path = path_x_count+1
                elif (path_x_count>threshold):
                    open_paths_data.append((path_x_count,empty_squares, path))

            # check left-down diagonal
            # for i in range(streak_size):
            #     r = row - i
            #     c = col + i
            #     if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            #         continue
            blocked = False
            path_length = 0
            path_x_count = 0
            empty_squares = []
            path = []
            square_row = row
            square_col = col
            while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
                if board[square_row][square_col] == other_player:
                    blocked = True
                elif board[square_row][square_col] == player:
                    path_x_count += 1
                elif ((square_col != col) | (square_row != row)) | (other_player=='X'):
                    empty_squares.append([square_row,square_col])
                path.append([square_row,square_col])
                square_row += 1
                square_col -= 1
                path_length += 1

            if (path_length == streak_size) & (not blocked) & (path_x_count>threshold): # add the path if it's not blocked and if there is already at least one X on it
                if other_player == 'O':
                    open_paths_data.append((path_x_count+1,empty_squares, path))
                    if (path_x_count+1) > max_length_path:
                        max_length_path = path_x_count+1
                elif (path_x_count>threshold):
                    open_paths_data.append((path_x_count,empty_squares, path))

            # check vertical
            # for i in range(streak_size):
            #     r = row - i
            #     c = col
            #     if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            #         continue
            blocked = False
            path_length = 0
            path_x_count = 0
            empty_squares = []
            path = []
            square_row = row
            square_col = col
            while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
                if board[square_row][square_col] == other_player:
                    blocked = True
                elif board[square_row][square_col] == player:
                    path_x_count += 1
                elif ((square_col != col) | (square_row != row)) | (other_player=='X'):
                    empty_squares.append([square_row,square_col])

                path.append([square_row,square_col])
                square_row += 1
                path_length += 1



            if (path_length == streak_size) & (not blocked) & (path_x_count>threshold): # add the path if it's not blocked and if there is already at least one X on it
                if other_player == 'O':
                    open_paths_data.append((path_x_count+1,empty_squares, path))
                    if (path_x_count+1) > max_length_path:
                        max_length_path = path_x_count+1
                elif (path_x_count>threshold):
                    open_paths_data.append((path_x_count,empty_squares, path))

            # check horizontal
            # for i in range(streak_size):
            #     r = row
            #     c = col - i
            #     if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            #         continue
            blocked = False
            path_length = 0
            path_x_count = 0
            empty_squares = []
            path = []
            square_row = row
            square_col = col
            while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
                if board[square_row][square_col] == other_player:
                    blocked = True
                elif board[square_row][square_col] == player:
                    path_x_count += 1
                elif ((square_col != col) | (square_row != row)) | (other_player=='X'):
                    empty_squares.append([square_row, square_col])

                path.append([square_row, square_col])
                square_col += 1
                path_length += 1

            if (path_length == streak_size) & (not blocked) & (path_x_count>threshold):  # add the path if it's not blocked and if there is already at least one X on it
                if other_player == 'O':
                    open_paths_data.append((path_x_count+1,empty_squares, path))
                    if (path_x_count+1) > max_length_path:
                        max_length_path = path_x_count+1
                elif (path_x_count>threshold):
                    open_paths_data.append((path_x_count,empty_squares, path))

    sorted_paths = sorted(open_paths_data, key=lambda x: x[0], reverse=True)
    score = 0.0
    for i in range(min(n,len(sorted_paths))):
        p1 = sorted_paths[i]
        score += 1.0/math.pow((streak_size-p1[0]), exp)  # score for individual path
        if interaction:
            for j in range(i+1, len(open_paths_data)):
                p2 = open_paths_data[j]
                if check_path_overlap(p1[2],p2[2]):  # interaction score if the paths don't overlap
                    # score += 1.0/(math.pow(((streak_size-1)*(streak_size-1))-(p1[0]*p2[0]), exp))
                    top = 0.0 +p1[0]*p2[0]
                    bottom = ((streak_size-1)*(streak_size-1))-(p1[0]*p2[0])
                    score += math.pow(top/bottom, exp)

    return score





def compute_relative_path_score(row, col, path_data, score_matrix, lamb = 1):
    raw_score = score_matrix[row][col]
    relative_score = 0.0

    for p in path_data:
        path_score = 0.0
        path_score_exp = 0.0
        for square in p:
            if (score_matrix[square[0]][square[1]]!='X') & (score_matrix[square[0]][square[1]]!='O'):
                path_score += score_matrix[square[0]][square[1]]
                path_score_exp += math.exp(score_matrix[square[0]][square[1]]*lamb)
        # if raw_score > 0:
        relative_score += (math.exp(raw_score)/path_score_exp)*path_score
    return relative_score






'''
computes the potential winning paths the cell is part of, of if you send it player = 'O' it will check how many
paths X destroys if it places an X on this cell
the @exp parameter creates the non-linearity (i.e., 2 --> squared)
'''
def compute_open_paths(row, col, board, exp=1, player = 'X', interaction = True):
    other_player = 'O'
    if player == 'O':
        other_player = 'X'

    threshold = 0
    if len(board)==10:
        threshold = 0

    streak_size = 4  # how many Xs in a row you need to win
    if len(board) == 10:  # if it's a 10X10 board you need 5.
        streak_size = 5

    open_paths_data = []  # this list will hold information on all the potential paths, each path will be represented by a pair (length and empty squares, which will be used to check overlap)
    # check right-down diagonal (there is a more efficient way to look at all the paths, but it was easier for me to debug when separating them :)
    for i in range(streak_size):
        r = row - i
        c = col - i
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        empty_squares = []
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == other_player:
                blocked = True
            elif board[square_row][square_col] == player:
                path_x_count += 1
            elif ((square_col != col) | (square_row != row)) | (other_player=='X'):
                empty_squares.append([square_row,square_col])

            square_row += 1
            square_col += 1
            path_length += 1

        if (path_length == streak_size) & (not blocked) & (path_x_count>threshold):  # add the path if it's not blocked and if there is already at least one X on it
            if other_player == 'O':
                open_paths_data.append((path_x_count+1,empty_squares))
            elif (path_x_count>threshold):
                open_paths_data.append((path_x_count,empty_squares))

    # check left-down diagonal
    for i in range(streak_size):
        r = row - i
        c = col + i
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        empty_squares = []
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == other_player:
                blocked = True
            elif board[square_row][square_col] == player:
                path_x_count += 1
            elif ((square_col != col) | (square_row != row)) | (other_player=='X'):
                empty_squares.append([square_row,square_col])

            square_row += 1
            square_col -= 1
            path_length += 1

        if (path_length == streak_size) & (not blocked)  & (path_x_count>threshold): # add the path if it's not blocked and if there is already at least one X on it
            if other_player == 'O':
                open_paths_data.append((path_x_count+1,empty_squares))
            elif (path_x_count>threshold):
                open_paths_data.append((path_x_count,empty_squares))

    # check vertical
    for i in range(streak_size):
        r = row - i
        c = col
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        empty_squares = []
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == other_player:
                blocked = True
            elif board[square_row][square_col] == player:
                path_x_count += 1
            elif ((square_col != col) | (square_row != row)) | (other_player=='X'):
                empty_squares.append([square_row,square_col])

            square_row += 1
            path_length += 1



        if (path_length == streak_size) & (not blocked) & (path_x_count>threshold): # add the path if it's not blocked and if there is already at least one X on it
            if other_player == 'O':
                open_paths_data.append((path_x_count+1,empty_squares))
            elif (path_x_count>threshold):
                open_paths_data.append((path_x_count,empty_squares))

    # check horizontal
    for i in range(streak_size):
        r = row
        c = col - i
        if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
            continue
        blocked = False
        path_length = 0
        path_x_count = 0
        empty_squares = []
        square_row = r
        square_col = c
        while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
            if board[square_row][square_col] == other_player:
                blocked = True
            elif board[square_row][square_col] == player:
                path_x_count += 1
            elif ((square_col != col) | (square_row != row)) | (other_player=='X'):
                empty_squares.append([square_row, square_col])

            square_col += 1
            path_length += 1

        if (path_length == streak_size) & (not blocked) & (path_x_count>threshold):  # add the path if it's not blocked and if there is already at least one X on it
            if other_player == 'O':
                open_paths_data.append((path_x_count+1,empty_squares))
            elif (path_x_count>threshold):
                open_paths_data.append((path_x_count,empty_squares))

    # print open_paths_lengths


    # if ((row==0) & (col==3)) |((row==3) & (col==0)):
    #     print 'here'
    score = 0.0
    # compute the score for the cell based on the potential paths
    for i in range(len(open_paths_data)):
        p1 = open_paths_data[i]
        score += 1.0/math.pow((streak_size-p1[0]), exp)  # score for individual path
        if interaction:
            for j in range(i+1, len(open_paths_data)):
                p2 = open_paths_data[j]
                if not(check_path_overlap(p1[1],p2[1])):  # interaction score if the paths don't overlap
                    score += 1.0/(math.pow(((streak_size-1)*(streak_size-1))-(p1[0]*p2[0]), exp))


    return score

'''
computes the score for each cell in each of the boards based on the paths score
the @exp parameter creates the non-linearity (i.e., 2 --> squared)
'''
def compute_scores_open_paths(normalized=False, exp=1, lamb = 1):
    data_matrices = {}
    for g in range(len(LOGFILE)):
        print(LOGFILE[g])
        board_matrix = copy.deepcopy(START_POSITION[g])
        for i in range(len(board_matrix)):
            for j in range(len(board_matrix[i])):
                if ((board_matrix[i][j]!=1) & (board_matrix[i][j]!=2)):
                    board_matrix[i][j] = int(board_matrix[i][j])
                elif (board_matrix[i][j]==1):
                    board_matrix[i][j]='X'
                elif (board_matrix[i][j]==2):
                    board_matrix[i][j]='O'

        # print board_matrix

        sum_scores = 0.0
        sum_scores_exp = 0.0
        score_matrix = copy.deepcopy(board_matrix)


        for r in range(len(board_matrix)):
            for c in range(len(board_matrix[r])):
                if board_matrix[r][c] == 0:  # only check if free
                    square_score = compute_open_paths(r, c, board_matrix,exp=exp)  # check open paths for win
                    score_matrix[r][c] = square_score
                    sum_scores += square_score
                    sum_scores_exp += math.pow(math.e,lamb*square_score)


        # heatmaps
        for r in range(0,len(score_matrix)):
            for j in range(0,len(score_matrix[r])):
                if (score_matrix[r][j]=='X'):
                    score_matrix[r][j] = -0.00001
                elif (score_matrix[r][j]=='O'):
                    score_matrix[r][j] = -0.00002

        if normalized:
            for r in range(len(score_matrix)):
                for c in range(len(score_matrix[r])):
                    if (score_matrix[r][c]!=-0.00001) & (score_matrix[r][c]!=-0.00002):
                        score_matrix[r][c] = score_matrix[r][c]/sum_scores
                    # if (score_matrix[r][c]!=-0.00001) & (score_matrix[r][c]!=-0.00002):
                    #     score_matrix[r][c] = (math.pow(math.e, lamb*score_matrix[r][c]))/sum_scores_exp

        board_name = LOGFILE[g]
        board_name = board_name[:-4]
        data_matrices[board_name[5:-6]] = score_matrix


    return data_matrices

'''
computes the score for each cell in each of the boards based on the paths score, also considers opponent paths
the @exp parameter creates the non-linearity (i.e., 2 --> squared)
@o_weight says how much weight to give to blocking the opponent paths when scoring the cell
'''
def compute_scores_open_paths_opponent(normalized=False, exp=1, lamb = 1, o_weight = 0.5):
    data_matrices = {}
    for g in range(len(LOGFILE)):
        # print LOGFILE[g]
        board_matrix = copy.deepcopy(START_POSITION[g])
        for i in range(len(board_matrix)):
            for j in range(len(board_matrix[i])):
                if ((board_matrix[i][j]!=1) & (board_matrix[i][j]!=2)):
                    board_matrix[i][j] = int(board_matrix[i][j])
                elif (board_matrix[i][j]==1):
                    board_matrix[i][j]='X'
                elif (board_matrix[i][j]==2):
                    board_matrix[i][j]='O'

        # print board_matrix

        sum_scores = 0.0
        sum_scores_exp = 0.0
        score_matrix = copy.deepcopy(board_matrix)


        for r in range(len(board_matrix)):
            for c in range(len(board_matrix[r])):
                if board_matrix[r][c] == 0:  # only check if free
                    # x_potential = compute_open_paths(r, c, board_matrix,exp=exp)  # check open paths for win
                    x_potential = compute_open_paths_data(r, c, board_matrix,exp=exp)  # check open paths for win
                    o_potential = compute_open_paths(r, c, board_matrix,exp=exp, player='O')  # check opponent paths that are blocked
                    square_score = (1-o_weight)*x_potential + o_weight*o_potential  # combine winning paths with blocked paths
                    score_matrix[r][c] = square_score
                    sum_scores += square_score
                    sum_scores_exp += math.pow(math.e,lamb*square_score)

        absolute_score_matrix = copy.deepcopy(score_matrix)
        for r in range(len(board_matrix)):
            for c in range(len(board_matrix[r])):
                if board_matrix[r][c] == 0:  # only check if free
                    score_matrix[r][c] = compute_relative_path_score(r,c,x_potential,absolute_score_matrix)


        # heatmaps
        for r in range(0,len(score_matrix)):
            for j in range(0,len(score_matrix[r])):
                if (score_matrix[r][j]=='X'):
                    score_matrix[r][j] = -0.00001
                elif (score_matrix[r][j]=='O'):
                    score_matrix[r][j] = -0.00002

        if normalized:
            for r in range(len(score_matrix)):
                for c in range(len(score_matrix[r])):
                    if (score_matrix[r][c]!=-0.00001) & (score_matrix[r][c]!=-0.00002):
                        score_matrix[r][c] = score_matrix[r][c]/sum_scores
                    # if (score_matrix[r][c]!=-0.00001) & (score_matrix[r][c]!=-0.00002):
                    #     score_matrix[r][c] = (math.pow(math.e, lamb*score_matrix[r][c]))/sum_scores_exp


        board_name = LOGFILE[g]
        board_name = board_name[:-4]
        data_matrices[board_name[5:-6]] = score_matrix

    return data_matrices

'''
computes the score for each cell in each of the boards based on a combination of density and path scores (not layers, multiplication)
the @exp parameter creates the non-linearity (i.e., 2 --> squared)
@density says which density function to use
@neighborhood size is for density score computation when using neighbors
@sig is for guassians when using guassian density
'''
def compute_scores_composite(normalized=False, exp=1, neighborhood_size=1, density = 'guassian', lamb=None, sig=3):
    data_matrices = {}
    # compute scores based on density  for all boards (so we can prune squares)
    if (density=='guassian'):
        density_scores = compute_scores_density_guassian(True,sig=sig)
    else:
        density_scores = compute_scores_density(True, neighborhood_size=neighborhood_size)
    path_scores = compute_scores_open_paths(True,exp)  # compute path-based scores for all boards

    for g in range(len(LOGFILE)):
        board_key = LOGFILE[g]
        board_key = board_key[:-4]
        board_key = board_key[5:-6]

        density_scores_board = density_scores[board_key]  # get density scores for this board
        path_scores_board = path_scores[board_key] # get path-based scores for this board

        board_matrix = copy.deepcopy(START_POSITION[g])
        for i in range(len(board_matrix)):
            for j in range(len(board_matrix[i])):
                if ((board_matrix[i][j]!=1) & (board_matrix[i][j]!=2)):
                    board_matrix[i][j] = int(board_matrix[i][j])
                elif (board_matrix[i][j]==1):
                    board_matrix[i][j]='X'
                elif (board_matrix[i][j]==2):
                    board_matrix[i][j]='O'

        # print board_matrix

        sum_scores = 0.0
        sum_scores_exp = 0.0
        score_matrix = copy.deepcopy(board_matrix)


        for r in range(len(board_matrix)):
            for c in range(len(board_matrix[r])):
                if board_matrix[r][c] == 0:  # only check if free
                    square_score = density_scores_board[r][c] * path_scores_board[r][c]  # combine density and path scores
                    score_matrix[r][c] = square_score
                    sum_scores += square_score
                    if lamb != None:
                        sum_scores_exp += math.pow(math.e,lamb*square_score)

        # heatmaps
        for r in range(0,len(score_matrix)):
            for j in range(0,len(score_matrix[r])):
                if (score_matrix[r][j]=='X'):
                    score_matrix[r][j] = -0.00001
                elif (score_matrix[r][j]=='O'):
                    score_matrix[r][j] = -0.00002

        if normalized:
            for r in range(len(score_matrix)):
                for c in range(len(score_matrix[r])):
                    if (score_matrix[r][c]!=-0.00001) & (score_matrix[r][c]!=-0.00002):
                        if lamb is None:
                            score_matrix[r][c] = score_matrix[r][c]/sum_scores
                        else:
                            score_matrix[r][c] = (math.pow(math.e, lamb*score_matrix[r][c]))/sum_scores_exp


        board_name = LOGFILE[g]
        board_name = board_name[:-4]
        data_matrices[board_name[5:-6]] = score_matrix

    return data_matrices

'''
same as the function above but also considers O paths
'''
def compute_scores_composite_opponent(normalized=False, exp=1, neighborhood_size=1, density = 'guassian', lamb=None, sig=3, opponent = False, o_weight=0.5):
    data_matrices = {}
    if (density=='guassian'):
        density_scores = compute_scores_density_guassian(True,sig=sig)
    else:
        density_scores = compute_scores_density(True, neighborhood_size=neighborhood_size)
    if opponent:
        path_scores = compute_scores_open_paths_opponent(True,exp, o_weight=o_weight)
    else:
        path_scores = compute_scores_open_paths(True,exp)

    for g in range(len(LOGFILE)):
        board_key = LOGFILE[g]
        board_key = board_key[:-4]
        board_key = board_key[5:-6]

        density_scores_board = density_scores[board_key]
        path_scores_board = path_scores[board_key]

        board_matrix = copy.deepcopy(START_POSITION[g])
        for i in range(len(board_matrix)):
            for j in range(len(board_matrix[i])):
                if ((board_matrix[i][j]!=1) & (board_matrix[i][j]!=2)):
                    board_matrix[i][j] = int(board_matrix[i][j])
                elif (board_matrix[i][j]==1):
                    board_matrix[i][j]='X'
                elif (board_matrix[i][j]==2):
                    board_matrix[i][j]='O'

        # print board_matrix

        sum_scores = 0.0
        sum_scores_exp = 0.0
        score_matrix = copy.deepcopy(board_matrix)


        for r in range(len(board_matrix)):
            for c in range(len(board_matrix[r])):
                if board_matrix[r][c] == 0:  # only check if free
                    square_score = density_scores_board[r][c] * path_scores_board[r][c]
                    score_matrix[r][c] = square_score
                    sum_scores += square_score
                    if lamb != None:
                        sum_scores_exp += math.pow(math.e,lamb*square_score)

        # heatmaps
        for r in range(0,len(score_matrix)):
            for j in range(0,len(score_matrix[r])):
                if (score_matrix[r][j]=='X'):
                    score_matrix[r][j] = -0.00001
                elif (score_matrix[r][j]=='O'):
                    score_matrix[r][j] = -0.00002

        if normalized:
            for r in range(len(score_matrix)):
                for c in range(len(score_matrix[r])):
                    if (score_matrix[r][c]!=-0.00001) & (score_matrix[r][c]!=-0.00002):
                        if lamb is None:
                            score_matrix[r][c] = score_matrix[r][c]/sum_scores
                        else:
                            score_matrix[r][c] = (math.pow(math.e, lamb*score_matrix[r][c]))/sum_scores_exp
                    # if (score_matrix[r][c]!=-0.00001) & (score_matrix[r][c]!=-0.00002):
                    #     score_matrix[r][c] = (math.pow(math.e, lamb*score_matrix[r][c]))/sum_scores_exp

        board_name = LOGFILE[g]
        board_name = board_name[:-4]
        data_matrices[board_name[5:-6]] = score_matrix

    return data_matrices


'''
computes the score for each cell in each of the boards based in the layers approach (first filter cells by density)
the @exp parameter creates the non-linearity (i.e., 2 --> squared)
@density says which density function to use
@neighborhood size is for density score computation when using neighbors
@sig is for guassians when using guassian density
@threshold says how much to prune (0.2 means we will remove cells that have score<0.2*maxScore)
@o_weight says how much weight to give for blocking O paths
@integrate says whether to combine density and path scores (done if = True), or just use path score after the initial filtering (done if = False)
'''
def compute_scores_layers(normalized=False, exp=1, neighborhood_size=1, density = 'reg', lamb=None, sig=3,
                          threshold=0.2, o_weight=0.0, integrate = False, interaction = True, dominance = False, block = False):
    data_matrices = {}

    for g in range(len(LOGFILE)):
        board_matrix = copy.deepcopy(START_POSITION[g])
        for i in range(len(board_matrix)):
            for j in range(len(board_matrix[i])):
                if ((board_matrix[i][j]!=1) & (board_matrix[i][j]!=2)):
                    board_matrix[i][j] = int(board_matrix[i][j])
                elif (board_matrix[i][j]==1):
                    board_matrix[i][j]='X'
                elif (board_matrix[i][j]==2):
                    board_matrix[i][j]='O'

        # print board_matrix

        sum_scores = 0.0
        sum_scores_exp = 0.0
        density_score_matrix = copy.deepcopy(board_matrix)

        if density=='guassian':
            # create guassians for each X square
            guassian_kernel = []
            for r in range(len(board_matrix)):
                for c in range(len(board_matrix[r])):
                    if board_matrix[r][c] == 'X':
                        guassian_kernel.append(makeGaussian(len(board_matrix),fwhm=sig,center=[r,c]))

        for r in range(len(board_matrix)):
            for c in range(len(board_matrix[r])):
                if board_matrix[r][c] == 0:  # only check if free
                    if density == 'guassian':
                        square_score = compute_density_guassian(r, c, board_matrix, guassian_kernel)  # check neighborhood
                    else:
                        square_score = compute_density(r, c, board_matrix, neighborhood_size)  # check neighborhood
                    density_score_matrix[r][c] = square_score
                    sum_scores += square_score
                    # if lamb!=None:
                    #     sum_scores_exp += math.pow(math.e,lamb*square_score)

        # compute maximal density score for filtering
        max_density_score = -1000000
        for r in range(len(density_score_matrix)):
            for c in range(len(density_score_matrix[r])):
                # score_matrix[r][c] = score_matrix[r][c]/sum_scores
                if (density_score_matrix[r][c]!='X') & (density_score_matrix[r][c]!='O'):
                    density_score_matrix[r][c] = density_score_matrix[r][c]/sum_scores
                    if density_score_matrix[r][c] > max_density_score:
                        max_density_score = density_score_matrix[r][c]
                    # score_matrix[r][c] = (math.pow(math.e, lamb*score_matrix[r][c]))/sum_scores_exp

        # run path score on remaining squares
        sum_scores = 0.0
        sum_scores_exp = 0.0
        score_matrix = copy.deepcopy(board_matrix)

        paths_data = copy.deepcopy(board_matrix)

        for r in range(len(board_matrix)):
            for c in range(len(board_matrix[r])):
                if (board_matrix[r][c] == 0) & (density_score_matrix[r][c]>threshold*max_density_score):  # only check if free & passed threshold
                    # x_paths = compute_open_paths_data(r, c, board_matrix,exp=exp,interaction=interaction)  # check open paths for win

                    x_paths = compute_open_paths_data_interaction(r, c, board_matrix,exp=exp,interaction=interaction)
                    square_score_x = x_paths[0]
                    x_paths_data = []
                    for path in x_paths[1]:
                        x_paths_data.append(path[2])
                    paths_data[r][c] = copy.deepcopy(x_paths_data)
                    # square_score_0 = compute_open_paths(r, c, board_matrix, exp=exp, player = 'O', interaction=interaction)

                    # o_paths = compute_open_paths_data(r, c, board_matrix, exp=exp, player = 'O', interaction=interaction)
                    o_paths = compute_open_paths_data_interaction(r, c, board_matrix,player = 'O', exp=exp,interaction=interaction)
                    square_score_o = o_paths[0]


                    streak_size = 4
                    if len(board_matrix)==10:
                        streak_size = 5

                    if block & (x_paths[2] == (streak_size-1)):  # give score for blocking O
                        # square_score_o = compute_block_o_score(board_matrix,exp=exp, interaction=interaction,player='O')
                        square_score_o = INFINITY_O
                    # if x_paths[2] == (streak_size-1):
                    #     square_score_o =0
                    square_score = (1-o_weight)*square_score_x + o_weight*square_score_o
                    if integrate:
                        square_score = square_score*density_score_matrix[r][c]
                    score_matrix[r][c] = square_score
                    sum_scores += square_score
                    if lamb!=None:
                        score_matrix[r][c] = math.pow(math.e,lamb*square_score)
                        sum_scores_exp += math.pow(math.e,lamb*square_score)

        # x_paths_data = []
        # for path in x_paths[1]:
        #     x_paths_data.append(path[2])
        # sum_scores = 0.0
        # absolute_score_matrix = copy.deepcopy(score_matrix)
        # for r in range(len(board_matrix)):
        #     for c in range(len(board_matrix[r])):
        #         if (board_matrix[r][c] == 0) & (density_score_matrix[r][c]>threshold*max_density_score):  # only check if free
        #             score_matrix[r][c] = compute_relative_path_score(r,c,paths_data[r][c],absolute_score_matrix)
        #             sum_scores+=score_matrix[r][c]

        if dominance:
            score_matrix = compute_square_scores_dominance(board_matrix, score_matrix)
            sum_scores = 0.0
            for r in range(0,len(score_matrix)):
                for j in range(0,len(score_matrix[r])):
                    if (score_matrix[r][j]!='X') & (score_matrix[r][j]!='O'):
                        sum_scores += score_matrix[r][j]



        # heatmaps
        for r in range(0,len(score_matrix)):
            for j in range(0,len(score_matrix[r])):
                if (score_matrix[r][j]=='X'):
                    score_matrix[r][j] = -0.00001
                elif (score_matrix[r][j]=='O'):
                    score_matrix[r][j] = -0.00002

        if normalized:
            for r in range(len(score_matrix)):
                for c in range(len(score_matrix[r])):
                    # if (score_matrix[r][c]!=-0.00001) & (score_matrix[r][c]!=-0.00002):
                    if (score_matrix[r][c]>0):
                        if lamb is None:
                            score_matrix[r][c] = score_matrix[r][c]/sum_scores
                        else:
                            score_matrix[r][c] = score_matrix[r][c]/sum_scores_exp

        board_name = LOGFILE[g]
        board_name = board_name[:-4]


        data_matrices[board_name[5:-6]] = score_matrix
    matrix_name = 'data_matrices/model_layers' + '_e=' + str(exp) + '_nbr=' +str(neighborhood_size) + '_o=' +str(o_weight)
    if integrate:
        matrix_name = matrix_name + '_integrated'
    if density == 'guassian':
        matrix_name = matrix_name + 'guassian'
    if interaction:
        matrix_name += '_interaction'
    else:
        matrix_name += '_noInteraction'
    if dominance:
        matrix_name += '_dominance'
    if block:
        matrix_name += '_block'
    matrix_name += '_oInteraction'
    write_matrices_to_file(data_matrices, matrix_name + '.json')
    return data_matrices


def convert_matrix_to_board_obj(board_matrix):
    board_dict = {}
    i = 1
    for row in range(len(board_matrix)):
        for col in range(len(board_matrix[row])):
            board_dict[i] = 0
            if board_matrix[row][col] == 'X':
                board_dict[i] = 1
            elif board_matrix[row][col] == 'O':
                board_dict[i] = 2
            i += 1
    if len(board_matrix) ==  6:
        file_path = "examples/board_6_4.txt"
    else:
        file_path = "examples/board_10_5.txt"
    try:
        input_file = open(file_path ,'r')
    except:
        raise Exception("File not found!!! Make sure you didn't make a spelling error.")
    num_spaces = int(input_file.readline())
    winning_paths = []
    for line in input_file:
        path = list(map(int, line.split()))
        winning_paths.append(path)

    game_board = Board(num_spaces,winning_paths,board=board_dict)
    return game_board


def check_immediate_win(board_matrix, player, board_obj=None):
    if player == 'X':
        p = 1
    else:
        p = 2
    if board_obj is None:
        b = convert_matrix_to_board_obj(board_matrix)
    else:
        b = board_obj
    winning_moves = b.immediate_threats(p)
    return winning_moves


def convert_position(pos, dimension):
    col = int(((pos - 1) % dimension))
    row = (float(pos)/float(dimension))-1
    row = int(math.ceil(row))
    return row, col

'''
computes the score for each cell in each of the boards based in the layers approach (first filter cells by density)
the @exp parameter creates the non-linearity (i.e., 2 --> squared)
@density says which density function to use
@neighborhood size is for density score computation when using neighbors
@sig is for guassians when using guassian density
@threshold says how much to prune (0.2 means we will remove cells that have score<0.2*maxScore)
@o_weight says how much weight to give for blocking O paths
@integrate says whether to combine density and path scores (done if = True), or just use path score after the initial filtering (done if = False)
'''
def compute_scores_layers_for_matrix(board_mat, player='X', normalized=False, exp=1, neighborhood_size=1, density = 'reg', lamb=None, sig=3,
                          threshold=0.2, o_weight=0.0, integrate = False, interaction = True, dominance = False, block = False, only_density = False):
    data_matrices = {}
    board_matrix = copy.deepcopy(board_mat)
    for i in range(len(board_matrix)):
        for j in range(len(board_matrix[i])):
            if ((board_matrix[i][j]!=1) & (board_matrix[i][j]!=2)):
                board_matrix[i][j] = int(board_matrix[i][j])
            elif (board_matrix[i][j]==1):
                board_matrix[i][j]='X'
            elif (board_matrix[i][j]==2):
                board_matrix[i][j]='O'

    # print board_matrix

    sum_scores = 0.0
    sum_scores_exp = 0.0
    density_score_matrix = copy.deepcopy(board_matrix)

    if density=='guassian':
        # create guassians for each X square
        guassian_kernel = []
        for r in range(len(board_matrix)):
            for c in range(len(board_matrix[r])):
                if board_matrix[r][c] == 'X':
                    guassian_kernel.append(makeGaussian(len(board_matrix),fwhm=sig,center=[r,c]))

    for r in range(len(board_matrix)):
        for c in range(len(board_matrix[r])):
            if board_matrix[r][c] == 0:  # only check if free
                if density == 'guassian':
                    square_score = compute_density_guassian(r, c, board_matrix, guassian_kernel)  # check neighborhood
                else:
                    square_score = compute_density(r, c, board_matrix, neighborhood_size, player=player)  # check neighborhood
                density_score_matrix[r][c] = square_score
                sum_scores += square_score
                # if lamb!=None:
                #     sum_scores_exp += math.pow(math.e,lamb*square_score)

    if only_density:
        # check for immediate win/loss
        winning_moves = check_immediate_win(board_matrix, player)
        if len(winning_moves) > 0:
            for move in winning_moves:
                move_row, move_col = convert_position(move, len(board_matrix))
                density_score_matrix[move_row][move_col] = WIN_SCORE

        # else:
        #     other_player = 'O'
        #     if player == 'O':
        #         other_player = 'X'
        #     winning_moves_opp = check_immediate_win(board_matrix, other_player)
        #     if len(winning_moves_opp) > 1:
        #         for row in range(len(board_matrix)):
        #             for col in range(len(board_matrix[row])):
        #                 if (density_score_matrix[row][col] != 'X') & (density_score_matrix[row][col] != 'O'):
        #                     density_score_matrix[row][col] = -1*WIN_SCORE
        #     elif len(winning_moves_opp) == 1:
        #         move_row, move_col = convert_position(winning_moves_opp[0], len(board_matrix))
        #         for row in range(len(board_matrix)):
        #             for col in range(len(board_matrix[row])):
        #                 if (move_row != row) | (move_col != col):
        #                     if (density_score_matrix[row][col] != 'X') & (density_score_matrix[row][col] != 'O'):
        #                         density_score_matrix[row][col] = -1*WIN_SCORE

        # heatmaps
        sum_scores = 0.0
        counter = 0.0
        for r in range(0,len(density_score_matrix)):
            for j in range(0,len(density_score_matrix[r])):
                if (density_score_matrix[r][j]=='X'):
                    density_score_matrix[r][j] = -0.00001
                elif (density_score_matrix[r][j]=='O'):
                    density_score_matrix[r][j] = -0.00002
                else:
                    counter += 1.0
                    if density_score_matrix[r][j] > 0:
                        sum_scores += density_score_matrix[r][j]
                    sum_scores_exp += np.exp(density_score_matrix[r][j])

        if normalized:
            for r in range(len(density_score_matrix)):
                for c in range(len(density_score_matrix[r])):
                    if (density_score_matrix[r][c]!=-0.00001) & (density_score_matrix[r][c]!=-0.00002):
                    # if (score_matrix[r][c]>0):
                        if lamb is None:
                            # if (sum_scores < 0):
                            #     print 'negative'
                            if (sum_scores == 0):
                                density_score_matrix[r][c] = 1.0/counter
                            else:
                                # print density_score_matrix[r][c]
                                # density_score_matrix[r][c] = density_score_matrix[r][c]/sum_scores
                                if density_score_matrix[r][c] >= 0:
                                    density_score_matrix[r][c] = density_score_matrix[r][c]/sum_scores
                                else:
                                    density_score_matrix[r][c] = 0
                        else:
                            density_score_matrix[r][c] = np.exp(density_score_matrix[r][c])/sum_scores_exp
        return density_score_matrix
    # compute maximal density score for filtering
    max_density_score = -1000000
    for r in range(len(density_score_matrix)):
        for c in range(len(density_score_matrix[r])):
            # score_matrix[r][c] = score_matrix[r][c]/sum_scores
            if (density_score_matrix[r][c]!='X') & (density_score_matrix[r][c]!='O'):
                density_score_matrix[r][c] = density_score_matrix[r][c]/sum_scores
                if density_score_matrix[r][c] > max_density_score:
                    max_density_score = density_score_matrix[r][c]
                # if density_score_matrix[r][c] > max_density_score:
                #     max_density_score = density_score_matrix[r][c]
                # score_matrix[r][c] = (math.pow(math.e, lamb*score_matrix[r][c]))/sum_scores_exp

    # run path score on remaining squares
    sum_scores = 0.0
    sum_scores_exp = 0.0
    score_matrix = copy.deepcopy(board_matrix)

    paths_data = copy.deepcopy(board_matrix)

    x_turn = True
    o_turn = False
    if player == 'O':
        x_turn = False
        o_turn = True

    for r in range(len(board_matrix)):
        for c in range(len(board_matrix[r])):
            # print(board_matrix[r][c])
            # print(density_score_matrix[r][c])
            if (board_matrix[r][c] == 0):
                if (density_score_matrix[r][c]>threshold*max_density_score):  # only check if free & passed threshold
                    # x_paths = compute_open_paths_data(r, c, board_matrix,exp=exp,interaction=interaction)  # check open paths for win


                    x_paths = compute_open_paths_data_interaction_new(r, c, board_matrix,player_turn=x_turn,exp=exp,interaction=interaction)
                    square_score_x = x_paths[0]
                    x_paths_data = []
                    for path in x_paths[1]:
                        x_paths_data.append(path[2])
                    paths_data[r][c] = copy.deepcopy(x_paths_data)
                    # square_score_0 = compute_open_paths(r, c, board_matrix, exp=exp, player = 'O', interaction=interaction)

                    # o_paths = compute_open_paths_data(r, c, board_matrix, exp=exp, player = 'O', interaction=interaction)
                    o_paths = compute_open_paths_data_interaction_new(r, c, board_matrix,player = 'O', player_turn = o_turn,exp=exp,interaction=interaction)
                    square_score_o = o_paths[0]


                    streak_size = 4
                    if len(board_matrix)==10:
                        streak_size = 5

                    if block & (x_paths[2] == (streak_size-1)) & x_turn:  # give score for blocking O
                        # square_score_o = compute_block_o_score(board_matrix,exp=exp, interaction=interaction,player='O')
                        square_score_o = INFINITY_O
                        square_score_x += INFINITY_O

                    elif block & (o_paths[2] == (streak_size-1)) & o_turn:  # give score for blocking O
                        # square_score_o = compute_block_o_score(board_matrix,exp=exp, interaction=interaction,player='O')
                        square_score_x = INFINITY_O
                        square_score_o += INFINITY_O
                    # if x_paths[2] == (streak_size-1):
                    #     square_score_o =0
                    if o_weight == 0.5:
                        square_score = square_score_x + square_score_o
                    elif o_weight== 0:
                        square_score = square_score_x
                    elif o_weight == 1.0:
                        square_score = square_score_x

                    if square_score > WIN_SCORE:
                        square_score = WIN_SCORE
                    # if player == 'O':
                    #     square_score = -1*square_score
                    if integrate:
                        square_score = square_score*density_score_matrix[r][c]
                    score_matrix[r][c] = square_score
                    #TODO: remember to change if we want to:
                    if square_score > 0:
                        sum_scores += square_score
                     #TODO: remember to change if we want to:
                    # if lamb!=None:
                    #     score_matrix[r][c] = math.pow(math.e,lamb*square_score)
                    #     sum_scores_exp += math.pow(math.e,lamb*square_score)

    # x_paths_data = []
    # for path in x_paths[1]:
    #     x_paths_data.append(path[2])
    # sum_scores = 0.0
    # absolute_score_matrix = copy.deepcopy(score_matrix)
    # for r in range(len(board_matrix)):
    #     for c in range(len(board_matrix[r])):
    #         if (board_matrix[r][c] == 0) & (density_score_matrix[r][c]>threshold*max_density_score):  # only check if free
    #             score_matrix[r][c] = compute_relative_path_score(r,c,paths_data[r][c],absolute_score_matrix)
    #             sum_scores+=score_matrix[r][c]


    # check for immediate win/loss

    winning_moves = check_immediate_win(board_matrix, player)
    if len(winning_moves) > 0:
        for move in winning_moves:
            move_row, move_col = convert_position(move, len(board_matrix))
            # if ((player!='O') | (o_weight<1)):
            # if ((player!='O')):
            score_matrix[move_row][move_col] = WIN_SCORE

    # if (len(winning_moves) == 0) & (o_weight<1.0):
    #TODO: comment out for o blindness
    if (len(winning_moves) == 0):
        other_player = 'O'
        if player == 'O':
            other_player = 'X'
        winning_moves_opp = check_immediate_win(board_matrix, other_player)
        if len(winning_moves_opp) > 1:
            for row in range(len(board_matrix)):
                for col in range(len(board_matrix[row])):
                    if (score_matrix[row][col] != 'X') & (score_matrix[row][col] != 'O'):
                        if ((player!='X') | (o_weight>0)):
                            score_matrix[row][col] = -1*WIN_SCORE

        elif len(winning_moves_opp) == 1:
            move_row, move_col = convert_position(winning_moves_opp[0], len(board_matrix))
            for row in range(len(board_matrix)):
                for col in range(len(board_matrix[row])):
                    if (move_row != row) | (move_col != col):
                        if (score_matrix[row][col] != 'X') & (score_matrix[row][col] != 'O'):
                            if ((player!='X') | (o_weight>0)):
                                score_matrix[row][col] = -1*WIN_SCORE
                    else:
                        score_matrix[row][col] = INFINITY_O

    if dominance:
        score_matrix = compute_square_scores_dominance(board_matrix, score_matrix)
        sum_scores = 0.0
        for r in range(0,len(score_matrix)):
            for j in range(0,len(score_matrix[r])):
                if (score_matrix[r][j]!='X') & (score_matrix[r][j]!='O'):
                    sum_scores += score_matrix[r][j]



    # heatmaps
    sum_scores = 0.0
    sum_scores_exp = 0.0
    counter = 0.0
    for r in range(0,len(score_matrix)):
        for j in range(0,len(score_matrix[r])):
            if (score_matrix[r][j]=='X'):
                score_matrix[r][j] = -0.00001
            elif (score_matrix[r][j]=='O'):
                score_matrix[r][j] = -0.00002
            else:
                counter += 1.0
                if score_matrix[r][j] > 0:
                    sum_scores += score_matrix[r][j]
                sum_scores_exp += np.exp(score_matrix[r][j])

    if normalized:
        for r in range(len(score_matrix)):
            for c in range(len(score_matrix[r])):
                if (score_matrix[r][c]!=-0.00001) & (score_matrix[r][c]!=-0.00002):
                # if (score_matrix[r][c]>0):
                    if lamb is None:
                        if (sum_scores == 0):
                            score_matrix[r][c] = 1.0/counter # TODO: fix to uniform.
                        else:
                            #TODO: change if we don't want to eliminate negative scores
                            if score_matrix[r][c] >= 0:
                                score_matrix[r][c] = score_matrix[r][c]/sum_scores
                            else:
                                score_matrix[r][c] = 0
                    else:
                        score_matrix[r][c] = np.exp(score_matrix[r][c])/sum_scores_exp
        if (np.sum(score_matrix) < 0.99) & (np.sum(score_matrix) > 0):
            print (np.sum(score_matrix))

    return score_matrix




'''
auxilary function, you can ignore
'''
def transform_matrix_to_list(mat):
    list_rep = []
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            if mat[i][j]<0.00001:
                mat[i][j] = 0.00001
            list_rep.append(mat[i][j])

    sum_values = sum(list_rep)
    for i in range(len(list_rep)):
        list_rep[i]=list_rep[i]/sum_values
    return list_rep

'''
auxilary function, you can ignore (unless you want to play with the guassians)
'''
def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2 -1
    else:
        x0 = center[0]
        y0 = center[1]

    # return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    return np.exp(-1 * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

def write_matrices_to_file(data_matrices, filename):
  with open(filename, 'w') as fp:
      json.dump(data_matrices, fp)


def read_matrices_from_file(filename):
  json1_file = open(filename)
  json1_str = json1_file.read()
  json1_data = json.loads(json1_str)
  return json1_data


'''
use this method to define which models to run, it will create the heatmaps and compute distances
'''
def run_models():
    # generate the models you want to include in the heatmaps
    # For example, say that I want to show the layers model (just with path scores, with and without opponent,
    # and compare it to the first moves made by participants) --
    # I create model without the opponent using the layers model
    data_layers_reg = compute_scores_layers(normalized=False,exp=3,neighborhood_size=2,density='reg',o_weight=0.0, integrate=False)
    # # and the model with the opponent
    # data_layers_reg_withO = compute_scores_layers(normalized=True,exp=3,neighborhood_size=2,density='reg',o_weight=0.5, integrate=False)
    # and then the actual distribution of moves (it's computed from another file but you don't need to edit it)
    data_test = read_matrices_from_file('data_matrices/computational_model.json')
    data_computational_model = cm.get_heatmaps_alpha_beta()

    # write_matrices_to_file(data_computational_model, 'data_matrices/computational_model.json')
    # return
    # data_first_moves = rp.entropy_paths()
    data_clicks = rp.entropy_board_average()

    # go over all boards
    for board in ['6_easy','6_hard','10_easy','10_hard','10_medium']:
        # this tells it where to save the heatmaps and what to call them:
        fig_file_name = 'heatmaps/compVsPeople/' + board + '_peopleAvgVsAlphaBeta.png'
        heatmaps = []  # this object will store all the heatmaps and later save to a file
        full = board + '_full'
        pruned = board + '_pruned'
        if board.startswith('6'):  # adjust sizes of heatmaps depending on size of boards
            fig, axes = plt.subplots(2, 2, figsize=(12,8))  # this will create a 2X3 figure with 6 heatmaps, you can modify if you want fewer/more
            # fig, axes = plt.subplots(2, 4, figsize=(10,6))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(18,12)) # this will create a 2X3 figure with 6 heatmaps, you can modify if you want fewer/more
            # fig, axes = plt.subplots(2, 4, figsize=(18,12))

        fig.suptitle(board)  # add subtitle to the figure based on board name

        i = 0  # this will be used to index into the list of heatmaps
        print(board)  # just printing the board name so I know if they finish, you can remove


        # here you will append to the heatmap list any heatmap you want to present. For each heatmap you append a pair -
        # the first element is the score matrix, the second element is a title for the heatmap. Note that ordering of adding to the heatmap lists
        #determines where each heatmap is shown (goes from top left to bottom right)


        heatmaps.append((data_computational_model[full][0], 'alpha-beta' +'\n' +str(round(data_computational_model[full][1],3))))
        heatmaps.append((data_clicks[full][0], 'people'+'\n' +str(round(data_clicks[full][1],3))))
        heatmaps.append((data_computational_model[pruned][0], 'alpha-beta' +'\n' +str(round(data_computational_model[pruned][1],3))))
        heatmaps.append((data_clicks[pruned][0], 'people'+'\n' +str(round(data_clicks[pruned][1],3))))
        # dist = emd(data_layers_reg[full],data_first_moves[full]) # earth mover distance for the full board
        # # print dist
        # heatmaps.append((data_layers_reg[full], 'layers' + '\n' +str(round(dist,3)))) # add the model to the heatmap list with name and distance
        #
        # dist = emd(data_layers_reg_withO[full],data_first_moves[full]) # earth mover distance for the full board
        # heatmaps.append((data_layers_reg_withO[full], 'layers with O '+'\n' +str(round(dist,3)))) # add the model to the heatmap list with name and distance
        # # add the empirical distribution heatmap
        # heatmaps.append((data_first_moves[full], 'first moves'))
        #
        # # and then the same for the pruned boards
        # dist = emd(data_layers_reg[pruned],data_first_moves[pruned]) # earth mover distance for the full board
        # heatmaps.append((data_layers_reg[pruned], 'layers' + '\n' +str(round(dist,3)))) # add the model to the heatmap list with name and distance
        #
        # dist = emd(data_layers_reg_withO[pruned],data_first_moves[pruned]) # earth mover distance for the full board
        # heatmaps.append((data_layers_reg_withO[pruned], 'layers with O '+'\n' +str(round(dist,3)))) # add the model to the heatmap list with name and distance
        # # add the empirical distribution heatmap
        # heatmaps.append((data_first_moves[pruned], 'first moves'))

        # this creates the actual heatmaps
        for ax in axes.flatten():  # flatten in case you have a second row at some point
            a = np.array(heatmaps[i][0])
            a = np.flip(a,0)
            img = ax.pcolormesh(a)
            for y in range(a.shape[0]):
                for x in range(a.shape[1]):
                    if(a[y,x]==-1) | (a[y,x]==-0.00001):
                        ax.text(x + 0.5, y + 0.5, 'X',
                             horizontalalignment='center',
                             verticalalignment='center',
                             color='white'
                                 )
                    elif((a[y,x]==-2) | (a[y,x]==-0.00002)):
                        ax.text(x + 0.5, y + 0.5, 'O',
                             horizontalalignment='center',
                             verticalalignment='center',
                             color='white'
                        )
                    elif(a[y,x]!=0):
                        ax.text(x + 0.5, y + 0.5, '%.2f' % a[y, x],
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 color='white'
                         )

            fig.colorbar(img, ax=ax)
            # plt.colorbar(img)
            ax.set_aspect('equal')
            ax.set_title(heatmaps[i][1])
            i+=1

        # a = np.random.rand(10,4)
        # img = axes[0,0].imshow(a,interpolation='nearest')
        # axes[0,0].set_aspect('auto')
        # plt.colorbar(img)
        # plt.title(board)
        # fig.tight_layout()
        # fig.subplots_adjust(top=0.88)
        plt.savefig(fig_file_name)
        plt.clf()


def run_models_from_list(models_file_list, base_heatmap_name, base_matrix_index = None):

    # generate the models you want to include in the heatmaps
    # For example, say that I want to show the layers model (just with path scores, with and without opponent,
    # and compare it to the first moves made by participants) --
    # I create model without the opponent using the layers model
    # data_layers_reg = compute_scores_layers(normalized=False,exp=3,neighborhood_size=2,density='reg',o_weight=0.0, integrate=False)
    # # and the model with the opponent
    # data_layers_reg_withO = compute_scores_layers(normalized=True,exp=3,neighborhood_size=2,density='reg',o_weight=0.5, integrate=False)
    # and then the actual distribution of moves (it's computed from another file but you don't need to edit it)
    data = []

    for file in models_file_list:
        matrices = read_matrices_from_file('data_matrices/cogsci/'+file)
        data_matrices = {}
        for mat in matrices:
            # for k,v in mat:
            if mat.endswith('.json'):
                data_matrices[mat[:-5]] = matrices[mat]
            else:
                data_matrices[mat] = matrices[mat]

        data.append(copy.deepcopy(data_matrices))

    for board in ['6_easy','6_hard','10_easy','10_hard','10_medium']:
        plt.rcParams.update({'font.size': 9})
        fig_file_name = base_heatmap_name + 'markov_' + board + '091518_smalltest.pdf'
        heatmaps = []
        full = board + '_full'
        pruned = board + '_pruned'
        if board.startswith('6'):  # adjust sizes of heatmaps depending on size of boards
            fig, axes = plt.subplots(2, len(data), figsize=(6,6))  # this will create a 2X3 figure with 6 heatmaps, you can modify if you want fewer/more
            # fig, axes = plt.subplots(1, len(data), figsize=(16,8))  # this will create a 2X3 figure with 6 heatmaps, you can modify if you want fewer/more
            # fig, axes = plt.subplots(2, 4, figsize=(10,6))
        else:
            fig, axes = plt.subplots(2, len(data), figsize=(8,8)) # this will create a 2X3 figure with 6 heatmaps, you can modify if you want fewer/more
            # fig, axes = plt.subplots(2, 4, figsize=(18,12))

        fig.suptitle(board)  # add subtitle to the figure based on board name
        i = 0  # this will be used to index into the list of heatmaps
        print(board)  # just printing the board name so I know if they finish, you can remove

        for j in range(len(data)):
            matrix_name_full = models_file_list[j][:-5]
            if matrix_name_full.startswith('model'):
                matrix_name_full = 'Blocking'
            else:
                matrix_name_full = 'Participants First Moves'
            matrix_name_pruned = models_file_list[j][:-5]
            if base_matrix_index != None:
                dist_full = emd(data[j][full],data[base_matrix_index][full]) # earth mover distance for the full board
                dist_pruned = emd(data[j][pruned],data[base_matrix_index][pruned]) # earth mover distance for the full board
                matrix_name_full = matrix_name_full + '\n' + str(round(dist_full, 3))
                matrix_name_pruned = matrix_name_pruned + '\n' + str(round(dist_pruned, 3))
            heatmaps.append((data[j][full], matrix_name_full))

        for j in range(len(data)):
            # matrix_name_full = models_file_list[j][:-5]
            # matrix_name_pruned = models_file_list[j][:-5]
            matrix_name_full = models_file_list[j][:-5]
            matrix_name_pruned = models_file_list[j][:-5]
            if matrix_name_pruned.startswith('model'):
                matrix_name_pruned = 'Blocking'
            else:
                matrix_name_pruned = 'Participants First Moves'
            if base_matrix_index != None:
                dist_full = emd(data[j][full],data[base_matrix_index][full]) # earth mover distance for the full board
                dist_pruned = emd(data[j][pruned],data[base_matrix_index][pruned]) # earth mover distance for the full board
                matrix_name_full = matrix_name_full + '\n' + str(round(dist_full, 3))
                matrix_name_pruned = matrix_name_pruned + '\n' + str(round(dist_pruned, 3))
            heatmaps.append((data[j][pruned], matrix_name_pruned))



        # this creates the actual heatmaps
        for ax in axes.flatten():  # flatten in case you have a second row at some point
            a = np.array(heatmaps[i][0])
            a = np.flip(a,0)
            img = ax.pcolormesh(a)
            for y in range(a.shape[0]):
                for x in range(a.shape[1]):
                    if(a[y,x]==-1) | (a[y,x]==-0.00001):
                        ax.text(x + 0.5, y + 0.5, 'X',
                             horizontalalignment='center',
                             verticalalignment='center',
                             color='white'
                                 )
                    elif((a[y,x]==-2) | (a[y,x]==-0.00002)):
                        ax.text(x + 0.5, y + 0.5, 'O',
                             horizontalalignment='center',
                             verticalalignment='center',
                             color='white'
                        )
                    # elif(a[y,x]!=0):
                    #     ax.text(x + 0.5, y + 0.5, '%.2f' % a[y, x],
                    #              horizontalalignment='center',
                    #              verticalalignment='center',
                    #              color='white'
                    #      )

            cb = fig.colorbar(img, ax=ax)
            # plt.colorbar(img)
            ax.set_aspect('equal')
            # ax.tick_params(labelsize=8)
            ax.set_title(heatmaps[i][1])

            ax.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            i += 1

        plt.savefig(fig_file_name)
        plt.clf()



def compute_distances_for_boards(models_file_list, base_matrix_index):
    data = []
    for file in models_file_list:
        matrices = read_matrices_from_file('data_matrices/cogsci/'+file)
        data_matrices = {}
        for mat in matrices:
            # for k,v in mat:
            if mat.endswith('.json'):
                data_matrices[mat[:-5]] = matrices[mat]
            else:
                data_matrices[mat] = matrices[mat]

        data.append(copy.deepcopy(data_matrices))

    for board in ['6_easy','6_hard','10_easy','10_hard','10_medium']:
        full = board + '_full'
        pruned = board + '_pruned'
        for j in range(len(data)):
            matrix_name_full = models_file_list[j][:-5]
            matrix_name_pruned = models_file_list[j][:-5]
            if base_matrix_index != None:
                dist_full = emd(data[j][full],data[base_matrix_index][full]) # earth mover distance for the full board
                dist_pruned = emd(data[j][pruned],data[base_matrix_index][pruned]) # earth mover distance for the full board
            print (matrix_name_full + ',' + full+ ',' + str(round(dist_full, 3)))
            print (matrix_name_pruned + ',' + pruned+ ',' + str(round(dist_pruned, 3)))

def compute_square_scores_dominance(board, score_matrix, n=1000):
    player = 'X'
    other_player = 'O'

    streak_size = 4  # how many Xs in a row you need to win
    if len(board) == 10:  # if it's a 10X10 board you need 5.
        streak_size = 5
    threshold = 0
    if streak_size == 5:
        threshold = 0
    # find open paths, compute sum score for each open path
    open_paths_data = []  # this list will hold information on all the potential paths, each path will be represented by a pair (length and empty squares, which will be used to check overlap)
    paths = []
    total_paths_score = 0.0
    for row in range(len(board)):
        for col in range(len(board)):
    # check right-down diagonal (there is a more efficient way to look at all the paths, but it was easier for me to debug when separating them :)
            for i in range(streak_size):
                r = row - i
                c = col - i
                if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
                    continue
                blocked = False
                path_length = 0
                path_x_count = 0
                filled_squares = []
                empty_squares = []
                path = []
                path_score = 0.0
                square_row = r
                square_col = c
                while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
                    if board[square_row][square_col] == other_player:
                        blocked = True
                    elif board[square_row][square_col] == player:
                        path_x_count += 1
                        filled_squares.append((square_row,square_col))

                    elif ((square_col != col) | (square_row != row)) | (other_player=='X'):
                        empty_squares.append([square_row,square_col])
                        path_score += score_matrix[square_row][square_col]
                        total_paths_score += path_score
                    path.append([square_row,square_col])
                    square_row += 1
                    square_col += 1
                    path_length += 1

                if (path_length == streak_size) & (not blocked) & (path_x_count>threshold):  # add the path if it's not blocked and if there is already at least one X on it
                    if other_player == 'O':
                        open_paths_data.append((path_x_count+1,empty_squares, path_score))
                    elif (path_x_count>threshold):
                        open_paths_data.append((path_x_count,empty_squares, path_score))

            # check left-down diagonal
            for i in range(streak_size):
                r = row - i
                c = col + i
                if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
                    continue
                blocked = False
                path_length = 0
                path_x_count = 0
                filled_squares = []
                empty_squares = []
                path = []
                path_score = 0.0
                square_row = r
                square_col = c
                while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
                    if board[square_row][square_col] == other_player:
                        blocked = True
                    elif board[square_row][square_col] == player:
                        path_x_count += 1
                        filled_squares.append((square_row,square_col))

                    elif ((square_col != col) | (square_row != row)) | (other_player=='X'):
                        empty_squares.append([square_row,square_col])
                        path_score += score_matrix[square_row][square_col]
                        total_paths_score += path_score
                    path.append([square_row,square_col])
                    square_row += 1
                    square_col -= 1
                    path_length += 1

                if (path_length == streak_size) & (not blocked) & (path_x_count>threshold):  # add the path if it's not blocked and if there is already at least one X on it
                    if other_player == 'O':
                        open_paths_data.append((path_x_count+1,empty_squares, path_score))
                    elif (path_x_count>threshold):
                        open_paths_data.append((path_x_count,empty_squares, path_score))

            # check vertical
            for i in range(streak_size):
                r = row - i
                c = col
                if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
                    continue
                blocked = False
                path_length = 0
                path_x_count = 0
                filled_squares = []
                empty_squares = []
                path = []
                path_score = 0.0
                square_row = r
                square_col = c
                while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
                    if board[square_row][square_col] == other_player:
                        blocked = True
                    elif board[square_row][square_col] == player:
                        path_x_count += 1
                        filled_squares.append((square_row,square_col))

                    elif ((square_col != col) | (square_row != row)) | (other_player=='X'):
                        empty_squares.append([square_row,square_col])
                        path_score += score_matrix[square_row][square_col]
                        total_paths_score += path_score
                    path.append([square_row,square_col])
                    square_row += 1
                    path_length += 1



                if (path_length == streak_size) & (not blocked) & (path_x_count>threshold):  # add the path if it's not blocked and if there is already at least one X on it
                    if other_player == 'O':
                        open_paths_data.append((path_x_count+1,empty_squares, path_score))
                    elif (path_x_count>threshold):
                        open_paths_data.append((path_x_count,empty_squares, path_score))
            # check horizontal
            for i in range(streak_size):
                r = row
                c = col - i
                if (r > len(board)-1) | (r < 0) | (c > len(board)-1) | (c < 0):
                    continue
                blocked = False
                path_length = 0
                path_x_count = 0
                filled_squares = []
                empty_squares = []
                path = []
                path_score = 0.0
                square_row = r
                square_col = c
                while (not blocked) & (path_length < streak_size) & (square_row < len(board)) & (square_row >= 0) & (square_col < len(board)) & (square_col >= 0):
                    if board[square_row][square_col] == other_player:
                        blocked = True
                    elif board[square_row][square_col] == player:
                        path_x_count += 1
                        filled_squares.append((square_row,square_col))

                    elif ((square_col != col) | (square_row != row)) | (other_player=='X'):
                        empty_squares.append([square_row,square_col])
                        path_score += score_matrix[square_row][square_col]
                        total_paths_score += path_score
                    path.append([square_row,square_col])
                    square_col += 1
                    path_length += 1

                if (path_length == streak_size) & (not blocked) & (path_x_count>threshold):  # add the path if it's not blocked and if there is already at least one X on it
                    if other_player == 'O':
                        open_paths_data.append((path_x_count+1,empty_squares, path_score))
                    elif (path_x_count>threshold):
                        open_paths_data.append((path_x_count,empty_squares, path_score))


    move_matrix = copy.deepcopy(board)
    # choose open path with proportion to its sum of scores
    weights = []
    paths_final = []
    for p in open_paths_data:
        weights.append(p[2]/total_paths_score)
        paths_final.append(p[1])


    for run in range(n):
        path = choice(paths_final,weights)
        # choose square from path with proportion to its score
        square_weights = []
        squares = []
        for j in range(len(path)):
            square = path[j]
            if open_paths_data[j][2]!=0:
                square_weights.append(score_matrix[square[0]][square[1]]/open_paths_data[j][2])
            else:
                square_weights.append(0)
            squares.append(square)
        chosen_square = choice(squares, square_weights)
        move_matrix[chosen_square[0]][chosen_square[1]] +=1

    return move_matrix



def cdf(weights):
    total = sum(weights)
    result = []
    cumsum = 0
    for w in weights:
        cumsum += w
        result.append(cumsum / total)
    return result

def choice(population, weights):
    assert len(population) == len(weights)
    cdf_vals = cdf(weights)
    x = random.random()
    idx = bisect.bisect(cdf_vals, x)
    return population[idx]



def compute_square_scores_layers_from_matrix(density_matrix, scoring_matrix, threshold = 0.2):
    density_scores = read_matrices_from_file(density_matrix)
    model_scores = read_matrices_from_file(scoring_matrix)

    data_matrices = {}

    for board in ['6_easy_full','6_easy_pruned','6_hard_full','6_hard_pruned','10_easy_full','10_easy_pruned','10_hard_full','10_hard_pruned','10_medium_full','10_medium_pruned']:
        density_score_matrix = density_scores[board]
        model_score_matrix = model_scores[board]
        score_matrix = copy.deepcopy(model_score_matrix)
       # compute maximal density score for filtering
        max_density_score = -1000000
        for r in range(len(density_score_matrix)):
            for j in range(len(density_score_matrix[r])):
                # score_matrix[r][c] = score_matrix[r][c]/sum_scores
                if (density_score_matrix[r][j]!='X') & (density_score_matrix[r][j]!='O'):
                    if density_score_matrix[r][j] > max_density_score:
                        max_density_score = density_score_matrix[r][j]

        for r in range(len(density_score_matrix)):
            for j in range(len(density_score_matrix[r])):
                if (score_matrix[r][j]!=-0.00001) & (score_matrix[r][j]!=-0.00002):
                    if score_matrix[r][j] <= (0.2*density_score_matrix[r][j]):
                        score_matrix[r][j] = 0

        sum_scores = 0.0
        for row in range(len(score_matrix)):
            for col in range(len(score_matrix[row])):
                if (score_matrix[row][col]!=-0.00001) & (score_matrix[row][col]!=-0.00002):
                    sum_scores += score_matrix[row][col]

        #re-normalize
        for row in range(len(score_matrix)):
            for col in range(len(score_matrix[row])):
                if (score_matrix[row][col]!=-0.00001) & (score_matrix[row][col]!=-0.00002):
                    score_matrix[row][col] = score_matrix[row][col]/sum_scores

        data_matrices[board] = score_matrix

    filename = scoring_matrix[:-5] + '_layers.json'
    write_matrices_to_file(data_matrices, filename)



def get_alpha_beta_scores():
    data_matrices = {}
    game_configs_file = "model_config_blocking50.json"
    configs = cm.get_game_configs(game_configs_file)
    dimension = 6


    for conf in configs:
        data_matrices = {}
        for filename in os.listdir("predefinedBoards/"):
            if filename.startswith("6"):
                file_path = "examples/board_6_4.txt"
                dimension = 6
                # continue
                # if not(filename.startswith("6_easy")):
                #   continue

            else:
                # continue
                # if filename.startswith("10by10_easy"):
                # if (filename.startswith("10_medium")):
                #   continue
                file_path = "examples/board_10_5.txt"
                dimension = 10


            game = cm.start_game(file_path, conf)

            print (filename)

            score_matrix = []

            win_depth = cm.fill_board_from_file("predefinedBoards/"+filename,game)
            board = game.board
            square_scores = board.get_free_spaces_ranked_heuristic_with_scores(player=c.HUMAN,reduced_opponent=game.reduced_opponent, heuristic=game.heuristic, interaction=game.interaction, other_player=game.opponent, prune=game.prune, exp=game.exp, neighborhood=game.neighborhood_size, potential=game.potential)

            for row in range(dimension):
                score_matrix.append([])
                for col in range(dimension):
                    score_matrix[row].append(0)

            for row in range(dimension):
                for col in range(dimension):
                    position = row*dimension + col+1

                    if board.board[position] == 1:
                        score_matrix[row][col] = 'X'
                    elif board.board[position] == 2:
                        score_matrix[row][col] = 'O'

            sum_scores = 0.0
            for i in range(len(square_scores)):
                pos = square_scores[i][0]
                score = square_scores[i][1]
                col = ((pos - 1) % dimension)
                row = (float(pos)/float(dimension))-1
                row = int(math.ceil(row))
                score_matrix[row][col] = score
                sum_scores += score

            for row in range(0,len(score_matrix)):
                for col in range(0,len(score_matrix[row])):
                    if (score_matrix[row][col]=='X'):
                        score_matrix[row][col] = -0.00001
                    elif (score_matrix[row][col]=='O'):
                        score_matrix[row][col] = -0.00002


            for row in range(len(score_matrix)):
                for col in range(len(score_matrix[row])):
                    if (score_matrix[row][col]!=-0.00001) & (score_matrix[row][col]!=-0.00002):
                        score_matrix[row][col] = score_matrix[row][col]/sum_scores

            data_matrices[filename[:-5]] = copy.deepcopy(score_matrix)

        output_name = 'data_matrices/cogsci/' + game_configs_file[:-5] + '_' +conf['name'] + '.json'
        cm.write_matrices_to_file(data_matrices, output_name)

def likelihood(values_file, models_file_list):
    data = []
    for file in models_file_list:
        matrices = read_matrices_from_file('data_matrices/cogsci/'+file)
        data_matrices = {}
        for mat in matrices:
            # for k,v in mat:
            if mat.endswith('.json'):
                data_matrices[mat[:-5]] = matrices[mat]
            else:
                data_matrices[mat] = matrices[mat]

        data.append(copy.deepcopy(data_matrices))

    epsilon = 0.00001

    values = read_matrices_from_file(values_file)



    for board in ['6_easy_full','6_hard_full','10_easy_full','10_hard_full','10_medium_full','6_easy_pruned','6_hard_pruned','10_easy_pruned','10_hard_pruned','10_medium_pruned']:



        for j in range(len(data)):
            model_name = models_file_list[j][:-5]
            probability_dist = data[j]
            first_moves_board = values[board]
            prob_dist = probability_dist[board]
            # probability_dist = read_matrices_from_file(probability_dist_file)
            comm_prob = 1.0

            sum_user_likelihoods = 0.0
            sum_user_log_likelihoods = 0.0
            num_users = 0.0
            num_moves = 0.0
            for user in first_moves_board.keys():
                # print user
                first_moves_user = first_moves_board[user]
                for move in first_moves_user:
                    comm_prob = 1.0
                    # for move in first_moves_user:
                    # move = first_moves_user[0]
                    row = move[0]
                    col = move[1]
                    prob  = prob_dist[row][col]
                    if (prob == 0):
                        prob = epsilon
                    # print prob
                    comm_prob = prob * comm_prob
                    # print comm_prob
                    sum_user_likelihoods += comm_prob
                    sum_user_log_likelihoods += math.log(comm_prob)
                    num_moves += 1
                num_users += 1
                # print board + ',' + model_name + ',' + values_file[:-5] + ',' + str(comm_prob) + ',' + str(math.log(comm_prob))
                print (board + ',' + model_name + ',' + 'all' + ',' + str(comm_prob) + ',' + str(math.log(comm_prob)))


            avg_likelihood_borad = sum_user_likelihoods/num_moves
            avg_log_likelihood_board = sum_user_log_likelihoods/num_moves
            # print model_name +"," + board + ',' + str(avg_log_likelihood_board)
            # print avg_likelihood_borad
            # print avg_log_likelihood_board


def generate_uniform_dist_for_boards(matrix_file):
    data_matrices = {}
    matrices = read_matrices_from_file('data_matrices/cogsci/' + matrix_file)
    for board in ['6_easy_full','6_hard_full','10_easy_full','10_hard_full','10_medium_full','6_easy_pruned','6_hard_pruned','10_easy_pruned','10_hard_pruned','10_medium_pruned']:
        matrix = matrices[board]
        uniform_matrix = copy.deepcopy(matrix)
        free_spaces = 0.0
        for row in range(len(matrix)):
            for col in range(len(matrix)):
                if (matrix[row][col]!=-0.00001) & (matrix[row][col]!=-0.00002):
                    free_spaces += 1

        for row in range(len(matrix)):
            for col in range(len(matrix)):
                if (matrix[row][col]!=-0.00001) & (matrix[row][col]!=-0.00002):
                    uniform_matrix[row][col] = 1.0/free_spaces

        data_matrices[board] = copy.deepcopy(uniform_matrix)
    write_matrices_to_file(data_matrices, 'data_matrices/cogsci/chance.json')


def distancePredictionPeopleLOO():
    base_dir = 'data_matrices/cogsci/'
    people_all = read_matrices_from_file(base_dir+'avg_people_first_moves_all.json')
    sum_distances = 0.0
    boards = ['6_easy_full','6_hard_full','10_easy_full','10_hard_full','10_medium_full','6_easy_pruned','6_hard_pruned','10_easy_pruned','10_hard_pruned','10_medium_pruned']
    sum_distances = 0.0
    for i in range(len(boards)):
        boardsMinus1 = copy.deepcopy(boards)
        del boardsMinus1[i]
        c.BOARDS_MINUS_1 = copy.deepcopy(boardsMinus1)
        # x0 = [0.25,0.25,0.25,0.25]
        x0 = [0.2,0.2,0.2,0.2, 0.2]
        b = (0.0,1.0)
        bnds = (b,b,b,b,b)
        con1 = {'type':'eq','fun':constraint1}
        cons = [con1]
        res = minimize(distancePredictionPeople,x0,method='SLSQP',bounds = bnds, constraints=cons)
        population = res.x
        pop_str = ''
        for p in population:
            pop_str += str(p) + ','
        print (pop_str)

        board = boards[i]
        dist = 0.0
        people = people_all[board]
        models = []
        density = read_matrices_from_file(base_dir+'model_density_nbr=2.json')
        linear = read_matrices_from_file(base_dir+'model_config_opp_linear_layers.json')
        non_linear = read_matrices_from_file(base_dir+'model_config_opp_non-linear_layers.json')
        non_linear_interaction = read_matrices_from_file(base_dir+'model_config_opp_non-linear_interaction_layers.json')
        blocking = read_matrices_from_file(base_dir+'model_config_blocking10_blocking_layers.json')
        models.append(density[board])
        models.append(linear[board])
        models.append(non_linear[board])
        models.append(non_linear_interaction[board])
        models.append(blocking[board])
        joint_matrix = copy.deepcopy(people)

        for row in range(len(joint_matrix)):
            for col in range(len(joint_matrix)):
                if (joint_matrix[row][col]!=-0.00001) & (joint_matrix[row][col]!=-0.00002):
                    agg_score = 0.0
                    for i in range (len(models)):
                        model = models[i]
                        agg_score+=model[row][col]*population[i]
                    joint_matrix[row][col] = agg_score

        sum_distances += emd(joint_matrix,people)
    # print joint_matrix
    # print emd(joint_matrix,people)
    return sum_distances

def distancePredictionPeople(population):
    base_dir = 'data_matrices/cogsci/'
    people_all = read_matrices_from_file(base_dir+'avg_people_first_moves_all.json')
    sum_distances = 0.0
    for board in c.BOARDS_MINUS_1:
        people = people_all[board]
        models = []
        density = read_matrices_from_file(base_dir+'model_density_nbr=2.json')
        # linear = read_matrices_from_file(base_dir+'model_config_opp_linear_layers.json')
        linear = read_matrices_from_file(base_dir+'linear_new.json')
        non_linear = read_matrices_from_file(base_dir+'model_config_opp_non-linear_layers.json')
        non_linear_interaction = read_matrices_from_file(base_dir+'model_config_opp_non-linear_interaction_layers.json')
        blocking = read_matrices_from_file(base_dir+'model_config_blocking10_blocking_layers.json')
        models.append(density[board])
        models.append(linear[board])
        models.append(non_linear[board])
        models.append(non_linear_interaction[board])
        models.append(blocking[board])
        joint_matrix = copy.deepcopy(people)

        for row in range(len(joint_matrix)):
            for col in range(len(joint_matrix)):
                if (joint_matrix[row][col]!=-0.00001) & (joint_matrix[row][col]!=-0.00002):
                    agg_score = 0.0
                    for i in range (len(models)):
                        model = models[i]
                        agg_score+=model[row][col]*population[i]
                    joint_matrix[row][col] = agg_score

        sum_distances+=emd(joint_matrix,people)
    # print joint_matrix
    # print emd(joint_matrix,people)
    return sum_distances



def constraint1(x):
    # return sum(x) == 1
    # if sum(x) == 1:
    #     return 0
    # else:
    #     return 1
    return sum(x) - 1

def constraint2(x):
    for x_val in x:
        if (x<0) | (x>1):
            return False
    return True




if __name__ == "__main__":
    # distancePredictionPeople([0.1,0.1,0.3,0.3,0.2], '6_easy_full')
    # x0 = [0.2,0.2,0.2,0.2,0.2]
    # # x0 = [0.25,0.25,0.25,0.25]
    # b = (0.0,1.0)
    # bnds = (b,b,b,b,b)
    # con1 = {'type':'eq','fun':constraint1}
    # cons = [con1]
    #
    # c.BOARDS_MINUS_1 = ['6_easy_full','6_hard_full','10_easy_full','10_hard_full','10_medium_full','6_easy_pruned','6_hard_pruned','10_easy_pruned','10_hard_pruned','10_medium_pruned']
    # # res = minimize(distancePredictionPeople,x0,method='SLSQP',bounds = bnds, constraints=cons)
    # res = minimize(distancePredictionPeople,x0,bounds = bnds, constraints=cons)
    # print res
    # print '---'
    # print res.x
    #
    # exit()
    #
    # distancePredictionPeopleLOO()
    # exit()
    # generate_uniform_dist_for_boards('model_config_opp_non-linear_interaction.json')
    # models_files = ['model_config_blocking50_blocking.json', 'avg_people_first_moves_all.json']
    # models_files = ['model_config_blocking10_blocking_layers.json', 'model_config_opp_non-linear_layers.json','model_config_opp_linear_layers.json', 'model_config_opp_non-linear_interaction_layers.json','model_density_nbr=2.json','avg_people_first_moves_all.json']
    # models_files = ['model_config_opp_blocking_layers.json', 'model_config_opp_non-linear_layers.json','linear_new.json', 'model_config_opp_non-linear_interaction_layers.json','model_density_nbr=2.json','chance.json']
    models_files = ['model_config_opp_blocking.json', 'model_config_opp_non-linear.json','linear_new.json', 'model_config_opp_non-linear_interaction.json','model_density_nbr=2.json','chance.json']
    # models_files = ['model_config_blocking10_blocking.json', 'model_config_opp_non-linear.json','model_config_opp_linear.json', 'model_config_opp_non-linear_interaction.json','model_density_nbr=2.json','chance.json']
    # models_files = ['first_pruned200818.json','model_config_blocking10_blocking_layers.json']
    # models_files = ['model_config_blocking10_blocking.json','avg_people_first_moves_all.json']
    run_models_from_list(models_files, 'heatmaps/cogsci/')


    likelihood('data_matrices/cogsci/people_first_moves_values_byParticipantall.json',models_files)
    exit()
    # model_files = ['mcts.json','model_config_blocking10_blocking.json','avg_people_first_moves_all.json']
    # model_files = ['model_config_blocking10_blocking_layers.json','avg_people_first_moves_all.json']
    # run_models_from_list(model_files, 'heatmaps/cogsci/cogsci6MCblockingLayers')


    # # # # models_files = ['model_config_opp_linear_layers.json', 'model_config_opp_non-linear_layers.json', 'model_config_opp_non-linear_interaction_layers.json', 'model_config_opp_blocking_layers.json', 'avg_people_first_moves_all.json']
    # compute_distances_for_boards(models_files,1)
    # get_alpha_beta_scores()
    # compute_square_scores_layers_from_matrix('data_matrices/cogsci/model_density_nbr=2.json', 'data_matrices/cogsci/model_config_blocking50_blocking.json', threshold=0.2)
    # compute_square_scores_layers_from_matrix('data_matrices/cogsci/model_density_nbr=2.json', 'data_matrices/cogsci/model_config_opp_non-linear.json', threshold=0.2)
    # compute_square_scores_layers_from_matrix('data_matrices/cogsci/model_density_nbr=2.json', 'data_matrices/cogsci/model_config_opp_non-linear_interaction.json', threshold=0.2)
    # compute_square_scores_layers_from_matrix('data_matrices/cogsci/model_density_nbr=2.json', 'data_matrices/cogsci/model_config_opp_linear.json', threshold=0.2)
    compute_scores_density(normalized=True,neighborhood_size=2)
    # compute_scores_layers(normalized=True, exp=1, neighborhood_size=2, o_weight=0.0, integrate=False, interaction=True, dominance=False, block=False)
    # compute_scores_layers(normalized=True, exp=2, neighborhood_size=2, o_weight=0.5, integrate=False, interaction=True, dominance=False, block=True)
    # compute_scores_layers(normalized=True, exp=1, neighborhood_size=2, o_weight=0.5, integrate=False, interaction=True, dominance=False, block=False)
    # compute_scores_layers(normalized=True, exp=2, neighborhood_size=2, o_weight=0.5, integrate=False, interaction=True, dominance=False, block=False)
    # compute_scores_layers(normalized=True, exp=1, neighborhood_size=2, o_weight=0.5, integrate=False, interaction=True, dominance=False, block=False)
    # compute_scores_layers(normalized=True, exp=2, neighborhood_size=2, o_weight=0.5, integrate=False, interaction=True, dominance=False, block=False)
    # compute_scores_layers(normalized=True, exp=2, neighborhood_size=2, o_weight=0.5, integrate=True, interaction=True, dominance=False)
    # compute_scores_layers(normalized=True, exp=1, neighborhood_size=2, o_weight=0.5, integrate=False, interaction=True, dominance=False)
    # compute_scores_layers(normalized=True, exp=2, neighborhood_size=2, o_weight=0.5, integrate=False, interaction=False, dominance=False)
    # compute_scores_layers(normalized=True, exp=2, neighborhood_size=2, o_weight=0.0, integrate=False, interaction=False, dominance=False)
    # compute_scores_layers(normalized=True, exp=2, neighborhood_size=2, o_weight=0.5, integrate=False, interaction=False)
    # compute_scores_layers(normalized=True, exp=2, neighborhood_size=2, o_weight=0.0, integrate=True)
    # # compute_scores_layers(normalized=True, exp=1, neighborhood_size=2, o_weight=0.0, integrate=False)
    # compute_scores_layers(normalized=True, exp=2, neighborhood_size=2, o_weight=0.5, integrate=False)
    # model_files = ['model_density_nbr=2.json','model_layers_e=1_nbr=2_o=0.5.json','model_layers_e=2_nbr=2_o=0.5.json', 'avg_people_clicks_all.json']
    # model_files = ['avg_people_first_moves_all.json', 'avg_people_clicks_all.json']
    # model_files = ['model_layers_e=2_nbr=2_o=0.5.json','model_layers_e=2_nbr=2_o=0.0.json', 'avg_people_clicks_all.json']
    # # # # run_models()  # calls the function that runs the models
    # # # model_files = ['paths_linear_square_opp.json', 'paths_non-linear_square_opp.json', 'avg_people_clicks_solvedCorrect.json']
    # model_files = ['model_layers_e=2_nbr=2_o=0.5.json','paths_non-linear_square_layers_opp.json', 'avg_people_clicks_all.json']
    # model_files = ['paths_linear_full_layers_opp_potentialBlock50.json','paths_non-linear_full_layers_opp_potentialBlock50.json', 'avg_people_clicks_all.json']
    # model_files = ['model_layers_e=1_nbr=2_o=0.5_interaction.json','model_layers_e=2_nbr=2_o=0.5_interaction.json', 'avg_people_clicks_all.json']
    # # model_files = ['model_layers_e=1_nbr=2_o=0.5.json','model_layers_e=2_nbr=2_o=0.5.json', 'participant_solutions.json']
    # run_models_from_list(model_files, 'heatmaps/cogsci/linearVsNonLinearThreshold0Clicks',2)
    # model_files = ['model_layers_e=2_nbr=2_o=0.5_interaction.json','model_layers_e=4_nbr=2_o=0.5_interaction_potential.json', 'avg_people_clicks_all.json']
    # # # model_files = ['model_layers_e=1_nbr=2_o=0.5.json','model_layers_e=2_nbr=2_o=0.5.json', 'participant_solutions.json']
    # run_models_from_list(model_files, 'heatmaps/cogsci/exp4Vsexp2clicks',2)
    # model_files = ['density.json','paths_linear_square_layers_opp.json','paths_non-linear_square_layers_opp.json', 'avg_people_first_moves_all.json']
    # model_files = ['model_layers_e=2_nbr=2_o=0.5.json','paths_non-linear_square_layers_opp.json', 'participant_solutions.json']
    # model_files = ['model_layers_e=2_nbr=2_o=0.0_interaction.json','model_layers_e=2_nbr=2_o=0.5_interaction.json', 'avg_people_first_moves_all.json']
    # model_files = ['model_layers_e=1_nbr=2_o=0.5_interactionNew.json','model_layers_e=2_nbr=2_o=0.5_interactionNew.json', 'avg_people_first_moves_all.json']
    # model_files = ['model_layers_e=2_nbr=2_o=0.5_interaction_block.json','model_layers_e=2_nbr=2_o=0.5_interaction_block_oInteraction.json', 'avg_people_first_moves_all.json']
    # run_models_from_list(model_files, 'heatmaps/cogsci/BlockFirstMoves_oInteractionVsNot1',2)
    # # model_files = ['model_layers_e=1_nbr=2_o=0.5_interactionNew.json','model_layers_e=2_nbr=2_o=0.5_interactionNew.json', 'avg_people_clicks_all.json']
    # model_files = ['model_layers_e=2_nbr=2_o=0.5_interaction_block.json','model_layers_e=2_nbr=2_o=0.5_interaction_block_oInteraction.json', 'avg_people_clicks_all.json']
    #
    # run_models_from_list(model_files, 'heatmaps/cogsci/BlockClicks_oInteractionVsNot1',2)

