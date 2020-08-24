# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search in AlphaGo Zero style, which uses a policy-value
network to guide the tree search and evaluate the leaf nodes

@author: Junxiao Song
"""
import string
import numpy as np
import copy
import matplotlib as mpl

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

    def get_move_probs(self, state, temp=1e-3, **kwargs):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """

        return_visits = kwargs.get('return_visits', False)

        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]

        acts, visits = zip(*act_visits)

        act_probs = softmax((1.0 / temp) * np.log(np.array(visits) + 1e-10))

        if return_visits:
            return acts, act_probs, visits

        else:
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

    def get_action(self, board, temp=1e-3, return_prob=False, **kwargs):

        sensible_moves = board.availables

        return_fig = kwargs.get('return_fig', False)
        show_fig = kwargs.get('show_fig', False)

        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(board.width * board.height)
        if len(sensible_moves) > 0:

            acts_mcts, probas_mcts, visits_mcts = self.mcts.get_move_probs(board, temp, return_visits=True)

            acts_policy, probas_policy = zip(*self.mcts._policy(board)[0])

            move_probs[list(acts_mcts)] = probas_mcts

            if self._is_selfplay:
                # add Dirichlet Noise for exploration (needed for
                # self-play training)
                move = np.random.choice(
                    acts_mcts,
                    p=0.75 * probas_mcts + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probas_mcts)))
                )
                # update the root node and reuse the search tree
                self.mcts.update_with_move(move)
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                move = np.random.choice(acts_mcts, p=probas_mcts)
                # reset the root node
                self.mcts.update_with_move(-1)
            #                location = board.move_to_location(move)
            #                print("AI move: %d,%d\n" % (location[0], location[1]))

            if return_fig:
                buf =self.create_probas_heatmap(acts_policy=acts_policy,
                                               probas_policy=probas_policy,
                                               acts_mcts=acts_mcts,
                                               probas_mcts=probas_mcts,
                                               visits_mcts=visits_mcts,
                                               width=board.width,
                                               height=board.height,
                                               board=board,
                                               return_fig=True,
                                               show_fig=False)

                if return_prob:
                    return move, move_probs, buf
                else:
                    return move, buf

            else:
                if show_fig:
                    self.create_probas_heatmap(acts_policy=acts_policy,
                                               probas_policy=probas_policy,
                                               acts_mcts=acts_mcts,
                                               probas_mcts=probas_mcts,
                                               visits_mcts=visits_mcts,
                                               width=board.width,
                                               height=board.height,
                                               board=board,
                                               return_fig=False,
                                               show_fig=True)

                if return_prob:
                    return move, move_probs
                else:
                    return move

        else:
            print("WARNING: the board is full")

    def __str__(self):
        return str(self.name) + " {}".format(self.player)


    def create_probas_heatmap(self, acts_policy, probas_policy, acts_mcts, probas_mcts, visits_mcts, width, height, board, return_fig=False, show_fig=False):

        if return_fig:
            mpl.use('Agg')

        fontsize = 15


        my_marker = "X" if self.player == 1 else "O"

        if self.player == 1:
            x_positions = board.current_state()[0]
            o_positions = board.current_state()[1]
        else:
            x_positions = board.current_state()[1]
            o_positions = board.current_state()[0]



        x_axis = [letter for i, letter in zip(range(width), string.ascii_lowercase)]
        y_axis = range(height, 0, -1)

        # fig, axes = plt.subplots(2, figsize=(10,15))
        fig, axes = plt.subplots(1, 3, figsize=(25,10))


        fig.suptitle(f"Model: {self.name}, MCTS playouts: {self.mcts._n_playout}", fontsize=fontsize+5)

        (ax1, ax2, ax3) = axes


        move_probs_policy = np.zeros(width * height)
        move_probs_policy[list(acts_policy)] = probas_policy
        move_probs_policy = move_probs_policy.reshape(width, height)
        move_probs_policy = np.flipud(move_probs_policy)
        move_probs_policy = np.round_(move_probs_policy, decimals=3)

        im1 = ax1.imshow(move_probs_policy, cmap='jet')
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im1, ax=ax1, cax=cax1).ax.tick_params(labelsize=fontsize)

        ax1.set_xticks(np.arange(len(x_axis)))
        ax1.set_yticks(np.arange(len(y_axis)))
        ax1.set_xticklabels(x_axis, fontsize=fontsize)
        ax1.set_yticklabels(y_axis, fontsize=fontsize)
        plt.setp(ax1.get_xticklabels(), ha="right", rotation_mode="anchor")
        for i in range(len(y_axis)):
            for j in range(len(x_axis)):
                text = ax1.text(j, i, "X" if x_positions[i, j] == 1 else (
                    "O" if o_positions[i, j] == 1 else move_probs_policy[i, j]),
                                ha="center", va="center", color="w", fontsize=fontsize)
        ax1.set_title("Probas of the policy value fn", fontsize=fontsize+4)


        normalized_visits = np.zeros(width * height)
        visits_mcts = visits_mcts / np.sum(visits_mcts)
        normalized_visits[list(acts_mcts)] = visits_mcts
        normalized_visits = normalized_visits.reshape(width, height)
        normalized_visits = np.flipud(normalized_visits)
        normalized_visits = np.round_(normalized_visits, decimals=3)

        im2 = ax2.imshow(normalized_visits, cmap='jet')
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im2, ax=ax2, cax=cax2).ax.tick_params(labelsize=fontsize)

        ax2.set_xticks(np.arange(len(x_axis)))
        ax2.set_yticks(np.arange(len(y_axis)))
        ax2.set_xticklabels(x_axis, fontsize=fontsize)
        ax2.set_yticklabels(y_axis, fontsize=fontsize)
        plt.setp(ax2.get_xticklabels(), ha="right", rotation_mode="anchor")
        for i in range(len(y_axis)):
            for j in range(len(x_axis)):
                text = ax2.text(j, i, "X" if x_positions[i, j] == 1 else (
                    "O" if o_positions[i, j] == 1 else normalized_visits[i, j]),
                                ha="center", va="center", color="w", fontsize=fontsize)
        ax2.set_title("Normalized visit counts of MCTS", fontsize=fontsize + 4)



        move_probs_mcts = np.zeros(width * height)
        move_probs_mcts[list(acts_mcts)] = probas_mcts
        move_probs_mcts = move_probs_mcts.reshape(width, height)
        move_probs_mcts = np.flipud(move_probs_mcts)
        move_probs_mcts = np.round_(move_probs_mcts, decimals=3)

        im3 = ax3.imshow(move_probs_mcts, cmap='jet')
        divider3 = make_axes_locatable(ax3)
        cax3 = divider3.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im3, ax=ax3, cax=cax3).ax.tick_params(labelsize=fontsize)

        ax3.set_xticks(np.arange(len(x_axis)))
        ax3.set_yticks(np.arange(len(y_axis)))
        ax3.set_xticklabels(x_axis, fontsize=fontsize)
        ax3.set_yticklabels(y_axis, fontsize=fontsize)
        plt.setp(ax3.get_xticklabels(), ha="right", rotation_mode="anchor")

        for i in range(len(y_axis)):
            for j in range(len(x_axis)):
                text = ax3.text(j, i, "X" if x_positions[i, j] == 1 else (
                    "O" if o_positions[i, j] == 1 else move_probs_mcts[i, j]),
                                ha="center", va="center", color="w", fontsize=fontsize)
        ax3.set_title(f"Probas of the MCTS which plays {my_marker} ", fontsize=fontsize + 4)

        fig.tight_layout()

        fig.subplots_adjust(top=0.88)

        if return_fig:
            buf = io.BytesIO()
            plt.savefig(buf, format='jpeg')
            buf.seek(0)
            return buf

        elif show_fig:
            plt.show()



