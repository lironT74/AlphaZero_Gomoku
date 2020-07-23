import numpy as np


class PolicyPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, policy_network, is_selfplay):
        self._is_selfplay = is_selfplay
        self._policy_network = policy_network

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        pass

    def get_action(self, board, temp=1e-3, return_prob=0):
        sensible_moves = board.availables
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(board.width*board.height)
        if len(sensible_moves) > 0:
            probs_legal = self._policy_network.policy_fn(board)
            # probs_legal = self._policy_network.policy_value_fn(board)[0] #lt

            acts = [p[0] for p in probs_legal]
            probs = [p[1] for p in probs_legal]


            # with the default temp=1e-3, it is almost equivalent
            # to choosing the move with the highest prob
            probs_norm = probs / np.sum(probs)
            move = np.random.choice(acts, p=probs_norm)
            # reset the root node

            if return_prob:
                return move, probs_norm
            else:
                return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "Policy {}".format(self.player)