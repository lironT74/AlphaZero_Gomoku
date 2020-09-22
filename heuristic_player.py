import numpy as np
import copy


class Heuristic_player(object):
    """AI player based on MCTS"""


    def __init__(self, name="", heuristic="forcing", o_weight=0.5, open_path_threshold=-1, exp = 1, last_move=True, normalized_density_scores=False, density='reg', sig=3, max_radius_density=-1):
        self.name = name

        self.heuristic = heuristic

        self.open_path_threshold = open_path_threshold
        self.o_weight = o_weight
        self.exp = exp
        self.last_move = last_move
        self.normalized_density_scores = normalized_density_scores
        self.normalize_all_heuristics = True
        self.density= density
        self.sig = sig
        self.max_radius_density = max_radius_density

    def set_player_ind(self, p):
        self.player = p


    def get_action(self, board):

        board_copy = copy.deepcopy(board)

        width = board_copy.width
        height = board_copy.height
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board_copy.states.get(loc, -1)
                if p == 1:
                    print('X'.center(8), end='')
                elif p == 2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

        board_copy.set_open_paths_threshold(self.open_path_threshold)

        sensible_moves = board.availables
        if len(sensible_moves) > 0:

            heuristic_scores = board_copy.calc_all_heuristics(
                            exp = self.exp,
                            last_move=self.last_move,
                            normalized_density_scores=self.normalized_density_scores,
                            normalize_all_heuristics=self.normalize_all_heuristics,
                            density= self.density,
                            sig = self.sig,
                            max_radius_density=self.max_radius_density,
                            opponent_weight = self.o_weight)

            print(np.array(heuristic_scores[self.heuristic]))


            heuristic_scores = np.array(np.flipud(heuristic_scores[self.heuristic])).flatten()


            move = np.random.choice(range(0, board.width*board.height), p=heuristic_scores)

            # print(move)

            return move


        else:

            print("WARNING: the board is full")


    def __str__(self):
        return "MCTS {}".format(self.player)