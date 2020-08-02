# -*- coding: utf-8 -*-
"""
@author: Junxiao Song
"""

from __future__ import print_function
import numpy as np
import copy
from FeatureExtractor import FeatureExtractor
import math

WIN_SCORE = 9
FORCING_BONUS = 11

class BoardSlim(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]  # player1 and player2

    def init_board(self, start_player=1, initial_state=None):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player - 1]  # start player
        # keep available moves in a list
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

        if initial_state is not None:
            p1_moves = np.transpose(np.nonzero(initial_state[0])).tolist()
            p2_moves = np.transpose(np.nonzero(initial_state[1])).tolist()
            player_to_moves = {
                1: p1_moves,
                2: p2_moves
            }

            if len(p1_moves) == len(p2_moves):
                self.current_player = start_player
            else:
                self.current_player = (start_player % 2) + 1
            for i in range(len(p1_moves) + len(p2_moves)):
                loc = player_to_moves[self.current_player].pop(0)
                print(loc)
                move = self.location_to_move(loc)
                print(move)
                self.do_move(move)

    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """

        square_state = np.zeros((2, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0

        return square_state[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)

        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )

        self.last_move = move

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row * 2 - 1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))

        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]  # player1 and player2

    def init_board(self, start_player=0, initial_state=None):

        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # start player

        # keep available moves in a list
        self.availables = list(range(self.width * self.height))
        self.states = {}

        self.last_move = -1
        self.last_move_p1 = -1
        self.last_move_p2 = -1

        if initial_state is not None:
            p1_moves = np.transpose(np.nonzero(initial_state[0])).tolist()
            p2_moves = np.transpose(np.nonzero(initial_state[1])).tolist()

            player_to_moves = {
                1: p1_moves,
                2: p2_moves
            }

            self.last_move_p1 = p1_moves[-1] if len(p1_moves) > 0 else -1
            self.last_move_p2 = p2_moves[-1] if len(p2_moves) > 0 else -1

            if abs(len(p1_moves) - len(p2_moves)) > 1:
                raise Exception("Invalid Board ({}'s turn was skipped)".format(
                    "Player1" if len(p2_moves) > len(p1_moves) else "Player2"))

            if len(p1_moves) == len(p2_moves):
                self.current_player = start_player + 1
            else:
                if len(p1_moves) > len(p2_moves):
                    self.last_move = self.last_move_p1
                    self.current_player = self.players[0]

                    if start_player != 0:
                        raise Exception("It cant be that player 2 was first to play (he made less moves)")

                else:
                    self.last_move = self.last_move_p2
                    self.current_player = self.players[1]

                    if start_player != 1:
                        raise Exception("It cant be that player 1 was first to play (he made less moves)")

            for i in range(len(p1_moves) + len(p2_moves)):
                loc = player_to_moves[self.current_player].pop(0)
                # print(loc)
                move = self.location_to_move(loc)
                # print(move)
                self.do_move(move)

    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self, last_move=True):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height or 3*width*height
        """

        if last_move:
            square_state = np.zeros((4, self.width, self.height))
            if self.states:
                moves, players = np.array(list(zip(*self.states.items())))

                move_curr = moves[players == self.current_player]

                move_oppo = moves[players != self.current_player]

                square_state[0][move_curr // self.width,
                                move_curr % self.height] = 1.0

                square_state[1][move_oppo // self.width,
                                move_oppo % self.height] = 1.0

            if len(self.states) % 2 == 0:

                if self.last_move_p1 != -1:
                    # indicate the last move location OF THE CURRENT PLAYER!!!!
                    square_state[2][self.last_move_p1 // self.width,
                                    self.last_move_p1 % self.height] = 1.0

                square_state[3][:, :] = 1.0  # indicate the colour to play

            else:

                if self.last_move_p2 != -1:
                    # indicate the last move location OF THE CURRENT PLAYER!!!!
                    square_state[2][self.last_move_p2 // self.width,
                                    self.last_move_p2 % self.height] = 1.0

            return square_state[:, ::-1, :]

        else:
            square_state = np.zeros((3, self.width, self.height))
            if self.states:
                moves, players = np.array(list(zip(*self.states.items())))

                move_curr = moves[players == self.current_player]
                move_oppo = moves[players != self.current_player]

                square_state[0][move_curr // self.width,
                                move_curr % self.height] = 1.0
                square_state[1][move_oppo // self.width,
                                move_oppo % self.height] = 1.0

            if len(self.states) % 2 == 0:
                square_state[2][:, :] = 1.0  # indicate the colour to play

            return square_state[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)

        if self.current_player == self.players[0]:
            self.last_move_p1 = move
        else:
            self.last_move_p2 = move

        self.last_move = move

        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row * 2 - 1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player

    def calc_all_heuristics(self,
                            exp,
                            o_weight,
                            last_move=True,
                            normalized_density_scores=False,
                            density='reg',
                            sig=3,
                            max_radius_density=-1):

        board_state = self.current_state(last_move=last_move)
        cur_positions = np.flipud(board_state[0])
        opponent_positions = np.flipud(board_state[1])


        width = self.width
        height = self.height

        scores = np.zeros((5, width, height))

        radius = width - 1

        if max_radius_density != -1:
            radius = min(radius, max_radius_density)

        cur_positions_padded = np.zeros((width + 2 * radius, height + 2 * radius))
        cur_positions_padded[radius:radius + width, radius:radius + height] = cur_positions

        gaussian_kernel = []
        if density == 'guassian':
            # create guassians for each X square
            for row in range(width):
                for col in range(height):
                    if cur_positions[row, col] == 'X':
                        gaussian_kernel.append(self.makeGaussian(width, fwhm=sig, center=[row, col]))

        immediate_threats, unavoidable_traps = self.find_all_win_scores_squares()

        for row in range(width):
            for col in range(height):
                row_hat, col_hat = row + radius, col + radius

                if cur_positions[row, col] or opponent_positions[row, col]:  # taken squares get -1
                    scores[:, row, col] = -1
                    continue

                if (row, col) in immediate_threats or (row, col) in unavoidable_traps:
                    scores[:, row, col] = WIN_SCORE
                    continue

                # DENSITY:
                if density == 'guassian':
                    scores[0, row, col] = self.compute_density_guassian(row, col, gaussian_kernel)
                else:
                    for r in range(radius):
                        scores[0, row, col] += (1 / (8 * (r + 1))) \
                                            * (np.sum(cur_positions_padded[row_hat - r - 1, col_hat - r - 1: col_hat + r + 2]) +
                                               np.sum(cur_positions_padded[row_hat + r + 1, col_hat - r - 1: col_hat + r + 2]) +
                                               np.sum(cur_positions_padded[row_hat - r: row_hat + r + 1, col_hat + r + 1]) +
                                               np.sum(cur_positions_padded[row_hat - r: row_hat + r + 1, col_hat - r - 1]))

                #THE REST:
                (all_features_but_blocking, open_paths_data, max_path) = \
                    self.all_features_but_density_and_forcing(exp,
                                                              row,
                                                              col,
                                                              cur_positions,
                                                              opponent_positions)

                scores[1, row, col] = all_features_but_blocking["linear"]
                scores[2, row, col] = all_features_but_blocking["nonlinear"]
                scores[3, row, col] = all_features_but_blocking["interaction"]

                if max_path == self.n_in_row-2:
                    scores[4, row, col] += FORCING_BONUS

                # scores[4, row, col] = self.calc_blocking_bonus(max_path,
                #                                                row,
                #                                                col,
                #                                                cur_positions,
                #                                                opponent_positions,
                #                                                o_weight)


        if normalized_density_scores:
            scores[0,:,:] = self.normalize_matrix(scores[0,:,:], width, height, cur_positions, opponent_positions)

        return scores


    def calc_blocking_bonus(self, max_path, row, col, cur_positions, opponent_positions, o_weight):

        # Calculate blocking scores
        blocking_score_x = 0.0

        # Calculate open paths for Opponent player
        open_paths_data_o, max_path_o = \
            self.find_open_paths(row=row,
                                 col=col,
                                 cur_positions=opponent_positions,
                                 opponent_positions=cur_positions) #!!!!!!!

        blocking_score_o = 0.0

        # Calculate blocking score
        if (max_path == (self.n_in_row - 1)): # give score for forcing O
            blocking_score_x += FORCING_BONUS

        elif (max_path_o == (self.n_in_row - 1)):  # give score for forcing X
            blocking_score_o += FORCING_BONUS

        if o_weight == 0.5:
            blocking_score = blocking_score_x + blocking_score_o

        elif o_weight == 0:
            blocking_score = blocking_score_x  # o blindness for x player disregard O

        elif o_weight == 1.0:
            ### Osher: in this case, shouldn't we ignore the blocking_score_x?
            blocking_score = blocking_score_x  # o blindness - just use for score how good it would be to block x

        if blocking_score > WIN_SCORE:
            blocking_score = WIN_SCORE

        return blocking_score



    def all_features_but_density_and_forcing(self,
                                             exp,
                                             row: int,
                                             col: int,
                                             cur_positions,
                                             opponent_positions,
                                             threshold = 0):

        tmp_score_dict = {
            "linear": 0.0,
            "nonlinear": 0.0,
            "interaction": 0.0,
        }

        exp = exp

        open_paths_data, max_length_path = self.find_open_paths(row, col, cur_positions, opponent_positions, threshold=threshold)

        # compute the linear, nonlinear and interactions scores for the cell based on the potential paths
        for i in range(len(open_paths_data)):

            p1 = open_paths_data[i]

            if self.n_in_row - 1 == p1[0]:
                raise Exception("We missed an immediate threat at {}".format((row, col)))

            tmp_score_dict["linear"] += p1[0]
            tmp_score_dict["nonlinear"] += 1.0 / math.pow((self.n_in_row - p1[0]), exp)  # score for individual path

            # Calculate interaction score:
            for j in range(i+1, len(open_paths_data)):
                p2=open_paths_data[j]

                if self.check_path_overlap(p2[2], p1[2]) == False:
                    raise Exception("It cant be that the paths do not overlap (at {})".format((row, col)))

                if not self.check_path_overlap(p1[1], p2[1], square_to_ignore=(row,col)):

                    numenator = 0.0 + p1[0] * p2[0]
                    denom = ((self.n_in_row - 1) * (self.n_in_row - 1)) - (p1[0] * p2[0])

                    if denom == 0:
                        raise Exception("We missed an immediate threat at {}".format((row, col)))

                    tmp_score_dict["interaction"] += math.pow(numenator / denom, exp)

        return (tmp_score_dict, open_paths_data, max_length_path)


    def find_open_paths(self,
                        row: int,
                        col: int,
                        cur_positions,
                        opponent_positions,
                        threshold=0
                        ):

        open_paths_data = []  # this list will hold information on all the potential paths, each path will be represented by a pair (length and empty squares, which will be used to check overlap)

        max_length_path = 0

        # check right-down diagonal
        for i in range(self.n_in_row):
            r = row - i
            c = col - i
            if (r > self.height - 1) | (r < 0) | (c > self.width - 1) | (c < 0):
                continue
            blocked = False  # indicates whether the current way is blocked
            path_length = 0
            path_cur_pawns_count = 0
            empty_squares = []
            path = []
            square_row = r
            square_col = c
            while (not blocked) & (path_length < self.n_in_row) & (square_row < self.height) & (square_row >= 0) & (
                    square_col < self.width) & (square_col >= 0):
                if opponent_positions[square_row][square_col]:
                    blocked = True
                elif cur_positions[square_row][square_col]:
                    path_cur_pawns_count += 1
                elif ((square_col != col) | (square_row != row)):
                    # Square is empty, add it to empty_squares only if it's NOT the current square in play
                    empty_squares.append([square_row, square_col])
                path.append([square_row, square_col])
                square_row += 1
                square_col += 1
                path_length += 1

            if (path_length == self.n_in_row) & (not blocked) & (path_cur_pawns_count > threshold):
                open_paths_data.append((path_cur_pawns_count, empty_squares, path))

        # check left-down diagonal
        for i in range(self.n_in_row):
            r = row - i
            c = col + i
            if (r > self.height - 1) | (r < 0) | (c > self.width - 1) | (c < 0):
                continue
            blocked = False
            path_length = 0
            path_cur_pawns_count = 0
            empty_squares = []
            path = []
            square_row = r
            square_col = c
            while (not blocked) & (path_length < self.n_in_row) & (square_row < self.height) & (square_row >= 0) & (
                    square_col < self.width) & (square_col >= 0):
                if opponent_positions[square_row][square_col]:
                    blocked = True
                elif cur_positions[square_row][square_col]:
                    path_cur_pawns_count += 1
                elif ((square_col != col) | (square_row != row)):
                    empty_squares.append([square_row, square_col])
                path.append([square_row, square_col])
                square_row += 1
                square_col -= 1
                path_length += 1

            if (path_length == self.n_in_row) & (not blocked) & (path_cur_pawns_count > threshold):
                open_paths_data.append((path_cur_pawns_count, empty_squares, path))

            # check vertical

        # check vertical
        for i in range(self.n_in_row):
            r = row - i
            c = col
            if (r > self.height - 1) | (r < 0) | (c > self.width - 1) | (c < 0):
                continue
            blocked = False
            path_length = 0
            path_cur_pawns_count = 0
            empty_squares = []
            path = []
            square_row = r
            square_col = c
            while (not blocked) & (path_length < self.n_in_row) & (square_row < self.height) & (square_row >= 0) & (
                    square_col < self.width) & (square_col >= 0):
                if opponent_positions[square_row][square_col]:
                    blocked = True
                elif cur_positions[square_row][square_col]:
                    path_cur_pawns_count += 1
                elif ((square_col != col) | (square_row != row)):
                    empty_squares.append([square_row, square_col])

                path.append([square_row, square_col])
                square_row += 1
                path_length += 1

            if (path_length == self.n_in_row) & (not blocked) & (path_cur_pawns_count > threshold):
                open_paths_data.append((path_cur_pawns_count, empty_squares, path))

        # check horizontal
        for i in range(self.n_in_row):
            r = row
            c = col - i
            if (r > self.height - 1) | (r < 0) | (c > self.width - 1) | (c < 0):
                continue
            blocked = False
            path_length = 0
            path_cur_pawns_count = 0
            empty_squares = []
            path = []
            square_row = r
            square_col = c
            while (not blocked) & (path_length < self.n_in_row) & (square_row < self.height) & (square_row >= 0) & (
                    square_col < self.width) & (square_col >= 0):
                if opponent_positions[square_row][square_col]:
                    blocked = True
                elif cur_positions[square_row][square_col]:
                    path_cur_pawns_count += 1
                elif ((square_col != col) | (square_row != row)):
                    empty_squares.append([square_row, square_col])

                path.append([square_row, square_col])
                square_col += 1
                path_length += 1

            if (path_length == self.n_in_row) & (not blocked) & (path_cur_pawns_count > threshold):
                open_paths_data.append((path_cur_pawns_count, empty_squares, path))

        max_length_path = max(open_paths_data)[0] if len(open_paths_data) > 0 else 0

        return (open_paths_data, max_length_path)


    def check_immediate_threats(self):
        # THE IDEA: fill in all available squares one by one, and using already implemented functions check if the
        # game has just ended with your win - and if it did - the square is an immediate threat.
        immediate_threats = []

        width = self.width
        height = self.height

        for move in self.availables:
            row = move // height
            col = move % width

            board_copy = copy.deepcopy(self)
            board_copy.do_move(move)
            has_winner, winner = board_copy.has_a_winner()
            if has_winner and winner == self.current_player:  # you only check the threats that you impose on the opponent
                immediate_threats.append((row, col))

        return immediate_threats


    def find_all_win_scores_squares(self):
        # THE IDEA: fill in all available squares one by one, and using already implemented functions check if the
        # game has just ended with your win - and if it did - the square is an immediate threat. If you do not win by
        # placing a pawn in a given square, then if for every opponent's answer, placing a pawn in the given square
        # means you can win in your next turn, this square is an unavoidable trap.

        """
        EXAMPLE:
            1.
                assuming X has started the game, let the current board status be:
            X _ _
            _ _ _
            O O X
            If X places a pawn at (1,1),
            he is guaranteed to win the game.
            (1,1) is an immediate threat.

            2.
                assuming X has started the game, let the current board status be:
            X _ _
            _ O _
            O _ X
            (1,1) is taken, but if X places a pawn at (0,2) (and in this board he has to),
            he is guaranteed to win the game in his next turn.
            (0,2) is an unavoidable trap.
        """

        unavoidable_traps = []
        immediate_threats = []

        width = self.width
        height = self.height

        for move in self.availables:
            row = move // height
            col = move % width

            board_copy = copy.deepcopy(self)
            board_copy.do_move(move)  # current played

            has_winner, winner = board_copy.has_a_winner()
            if has_winner and winner == self.current_player:  # (row, col) is an immediate threat.
                immediate_threats.append((row, col))
                continue

            if len(board_copy.availables) < 2:  # there is no next turn for you after this turn
                continue

            curr_flag = True
            for opponent_move in board_copy.availables:  # for each opponent's answer

                board_second_copy = copy.deepcopy(board_copy)
                board_second_copy.do_move(opponent_move)
                has_winner_opponent_move, winner_opponent_move = board_second_copy.has_a_winner()

                # if there exists a move the opponent can
                # do which leads to his win or blocks current player's chance of winning in the next turn,
                # current square is not an unavoidable trap

                if (has_winner_opponent_move and winner_opponent_move == board_second_copy.current_player)\
                        or len(board_second_copy.check_immediate_threats()) == 0:
                        curr_flag = False
                        break

            if curr_flag: # (row, col) is an unavoidable trap.
                unavoidable_traps.append((row, col))

        return (immediate_threats, unavoidable_traps)


    @staticmethod
    def compute_density_guassian(row, col, guassian_kernel):
        density_score = 0.0
        for guas in guassian_kernel:
            density_score += guas[row][col]
        return density_score

    @staticmethod
    def makeGaussian(size, fwhm=3, center=None):
        """ Make a square gaussian kernel.
        size is the length of a side of the square
        fwhm is full-width-half-maximum, which
        can be thought of as an effective radius.
        """

        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]

        if center is None:
            x0 = y0 = size // 2 - 1
        else:
            x0 = center[0]
            y0 = center[1]

        # return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
        return np.exp(-1 * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)

    @staticmethod
    def normalize_matrix(scores, width, height, cur_positions, opponent_positions):
        sum_scores = 0.0
        counter = 0.0
        for r in range(width):
            for j in range(height):
                if cur_positions[r][j]:  # Originially: "if score_matrix[r][j] == 'X':"
                    scores[r][j] = -0.00001
                elif opponent_positions[r][j]:  # Originially: "if score_matrix[r][j] == 'O':"
                    scores[r][j] = -0.00002
                else:
                    counter += 1.0
                    if scores[r][j] > 0:
                        sum_scores += scores[r][j]

        for r in range(len(scores)):
            for c in range(len(scores[r])):
                if (scores[r][c] != -0.00001) & (scores[r][c] != -0.00002):
                    if sum_scores == 0:
                        scores[r][c] = 1.0 / counter
                    else:
                        # TODO: change if we don't want to eliminate negative scores
                        if scores[r][c] >= 0:
                            scores[r][c] = scores[r][c] / sum_scores
                        else:
                            scores[r][c] = 0

        return scores

    @staticmethod
    def check_path_overlap(empty1, empty2, square_to_ignore=None):
        for square in empty1:
            if (square_to_ignore is not None and square != square_to_ignore) or (square_to_ignore is None):
                if square in empty2:
                    return True
        return False



class Game(object):
    """game server"""

    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')

        self.board.init_board(start_player)

        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}

        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]


            move = player_in_turn.get_action(self.board)


            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3, is_last_move=True):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # store the data
            states.append(self.board.current_state(is_last_move))
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)

    def start_play_training(self, player, opponent, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, probs, z) for training
        """
        self.board.init_board()
        p1, p2 = self.board.players
        index2player = {
            p1: player,
            p2: opponent
        }
        states, moves = [], []
        while True:
            current_player = index2player[self.board.get_current_player()]
            move = current_player.get_action(self.board,
                                             temp=temp,
                                             return_prob=0)
            if self.board.get_current_player() == p1:
                # store the data
                states.append(self.board.current_state())
                moves.append(move)
                # perform a move
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                reward = 0
                if winner == p1:
                    reward = 1
                elif winner == p2:
                    reward = -1
                return reward, zip(states, moves)
