# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search in AlphaGo Zero style, which uses a policy-value
network to guide the tree search and evaluate the leaf nodes

@author: Junxiao Song
"""

import numpy as np
import copy
import matplotlib.pyplot as plt
import io

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    """A node in the MCTS tree.

    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000, name="MCTS"):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self.name = name

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while (1):
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            state.do_move(action)

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        action_probs, leaf_value = self._policy(state)
        # Check for end of game.
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        else:
            # for end state，return the "true" leaf_value
            if winner == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player() else -1.0
                )

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """

        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]

        acts, visits = zip(*act_visits)

        # print(visits)

        act_probs = softmax((1.0 / temp) * np.log(np.array(visits) + 1e-10))

        # print(act_probs)

        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return str(self.name)


class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, policy_value_function,
                 c_puct=5, n_playout=2000, is_selfplay=0, name="MCTS"):

        self.mcts = MCTS(policy_value_function, c_puct, n_playout, name)
        self._is_selfplay = is_selfplay
        self.name = name

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=0, return_heatmap = False, show=False):
        sensible_moves = board.availables

        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(board.width * board.height)
        if len(sensible_moves) > 0:
            acts, probas = self.mcts.get_move_probs(board, temp)

            acts_policy, probas_policy = zip(*self.mcts._policy(board)[0])

            move_probs[list(acts)] = probas

            if self._is_selfplay:
                # add Dirichlet Noise for exploration (needed for
                # self-play training)
                move = np.random.choice(
                    acts,
                    p=0.75 * probas + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probas)))
                )
                # update the root node and reuse the search tree
                self.mcts.update_with_move(move)
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                move = np.random.choice(acts, p=probas)
                # reset the root node
                self.mcts.update_with_move(-1)
            #                location = board.move_to_location(move)
            #                print("AI move: %d,%d\n" % (location[0], location[1]))

            if return_heatmap:
                buf = self.create_probas_heatmap(acts_policy, probas_policy, acts, probas, board.width, board.height, self.name, board, show)
                if return_prob:
                    return move, move_probs, buf
                else:
                    return move, buf
            else:
                if return_prob:
                    return move, move_probs
                else:
                    return move

        else:
            print("WARNING: the board is full")

    def __str__(self):
        return str(self.name) + " {}".format(self.player)


    def create_probas_heatmap(self, acts_policy, probas_policy, acts_mcts, probas_mcts, width, height, name, board, show=False):

        fontsize = 15

        if hasattr(self, 'player'):

            my_marker = "X" if self.player == 1 else "O"

            if self.player == 1:
                x_positions = board.current_state()[0]
                o_positions = board.current_state()[1]
            else:
                x_positions = board.current_state()[1]
                o_positions = board.current_state()[0]

        else:
            # This is training time. Make sure that in the board you've sent, its X's turn to play (or as you wish)
            my_marker = "X"

            x_positions = board.current_state()[0]
            o_positions = board.current_state()[1]

        y_axis = range(width - 1, -1, -1)
        x_axis = range(0, height, 1)

        fig, axes = plt.subplots(2, figsize=(10,15))
        (ax1, ax2) = axes

        move_probs_mcts = np.zeros(width * height)
        move_probs_mcts[list(acts_mcts)] = probas_mcts
        move_probs_mcts = move_probs_mcts.reshape(width, height)
        move_probs_mcts = np.flipud(move_probs_mcts)
        move_probs_mcts = np.round_(move_probs_mcts, decimals=3)

        im1 = ax1.imshow(move_probs_mcts, cmap='jet')
        fig.colorbar(im1, ax=ax1).ax.tick_params(labelsize=fontsize)

        # We want to show all ticks...
        ax1.set_xticks(np.arange(len(x_axis)))
        ax1.set_yticks(np.arange(len(y_axis)))
        # ... and label them with the respective list entries
        ax1.set_xticklabels(x_axis, fontsize=fontsize)
        ax1.set_yticklabels(y_axis, fontsize=fontsize)
        # Rotate the tick labels and set their alignment.
        plt.setp(ax1.get_xticklabels(), ha="right",
                 rotation_mode="anchor")
        # Loop over data dimensions and create text annotations.
        for i in range(len(y_axis)):
            for j in range(len(x_axis)):
                text = ax1.text(j, i, "X" if x_positions[i, j] == 1 else ("O" if o_positions[i, j] == 1 else move_probs_mcts[i, j]),
                               ha="center", va="center", color="w", fontsize=fontsize)
        ax1.set_title("Heatmap of action probas of \n{} which plays {} ".format(name, my_marker), fontsize=fontsize+4)



        move_probs_policy = np.zeros(width * height)
        move_probs_policy[list(acts_policy)] = probas_policy
        move_probs_policy = move_probs_policy.reshape(width, height)
        move_probs_policy = np.flipud(move_probs_policy)
        move_probs_policy = np.round_(move_probs_policy, decimals=3)

        im2 = ax2.imshow(move_probs_policy, cmap='jet')
        fig.colorbar(im2, ax=ax2).ax.tick_params(labelsize=fontsize)

        ax2.set_xticks(np.arange(len(x_axis)))
        ax2.set_yticks(np.arange(len(y_axis)))
        ax2.set_xticklabels(x_axis, fontsize=fontsize)
        ax2.set_yticklabels(y_axis, fontsize=fontsize)
        plt.setp(ax1.get_xticklabels(), ha="right",
                 rotation_mode="anchor")
        for i in range(len(y_axis)):
            for j in range(len(x_axis)):
                text = ax2.text(j, i, "X" if x_positions[i, j] == 1 else (
                    "O" if o_positions[i, j] == 1 else move_probs_policy[i, j]),
                                ha="center", va="center", color="w", fontsize=fontsize)
        ax2.set_title("Heatmap of action probas of \nthe corresponding policy value fn", fontsize=fontsize+4)

        fig.tight_layout()

        if show:
            plt.show()

        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        return buf
