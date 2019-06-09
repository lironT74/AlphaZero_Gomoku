

from __future__ import print_function

from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
import keras.backend as K

from keras.utils import np_utils

import tensorflow as tf

import numpy as np
import pickle


class PolicyNet():
    """policy network """

    def __init__(self, board_width, board_height, model_file=None, pretrained_file=None):
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty
        self.build_net()
        self._loss_train_op(0.001)

        if model_file:
            self.model.load_weights(model_file)
        if pretrained_file:
            self.model.load_weights(pretrained_file, by_name=True)

    def build_net(self):
        """create the policy value network """
        in_x = network = Input((2, self.board_width, self.board_height))

        # conv layers
        network = Conv2D(filters=32, kernel_size=(3, 3), padding="same", data_format="channels_first",
                         activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        network = Conv2D(filters=64, kernel_size=(3, 3), padding="same", data_format="channels_first",
                         activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        network = Conv2D(filters=128, kernel_size=(3, 3), padding="same", data_format="channels_first",
                         activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        # action policy layers
        policy_net = Conv2D(filters=4, kernel_size=(1, 1), data_format="channels_first", activation="relu",
                            kernel_regularizer=l2(self.l2_const))(network)
        policy_net = Flatten()(policy_net)
        self.policy_net = Dense(self.board_width * self.board_height, activation="softmax",
                                kernel_regularizer=l2(self.l2_const))(policy_net)

        self.model = Model(in_x, self.policy_net)

        def policy_value(state_input):
            state_input_union = np.array(state_input)
            results = self.model.predict_on_batch(state_input_union)
            return results

        self.policy_value = policy_value

    def policy_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available action and the score of the board state
        """
        legal_positions = board.availables
        current_state = board.current_state()
        act_probs = self.policy_value(
            current_state.reshape((-1, 2, self.board_width, self.board_height)))
        act_probs = list(zip(legal_positions, act_probs.flatten()[legal_positions]))
        return act_probs

    def _loss_train_op(self, initial_learning_rate):
        """
        Three loss terms：
        loss = (z - v)^2 + pi^T * log(p) + c||theta||^2
        """

        # get the train op
        # opt = Adam()
        self.session = K.get_session()
        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(initial_learning_rate, global_step, 10000, 0.95, True)
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        one_hot_move_ph = tf.placeholder(tf.float32, (None, self.board_width * self.board_height), "moves")
        reward_ph = tf.placeholder(tf.float32, (None,), "rewards")


        def self_entropy(probs):
            return -np.mean(np.sum(probs * np.log(probs + 1e-10), axis=1))

        def loss_op():

            objective = tf.log(tf.nn.softmax(self.model.output[0], axis=-1)) * one_hot_move_ph
            objective = tf.reduce_sum(objective, axis=-1, keepdims=False)
            objective = objective * reward_ph
            return -1 * objective

        self.loss_op = loss_op()
        self.minimize_op = opt.minimize(self.loss_op, global_step=global_step)

        def train_step(states, reward, moves):
            np_state_input = np.array(states)

            np_reward = np.array(reward)
            np_moves = np.eye(self.board_height * self.board_width)[np.array(moves)]


            # K.set_value(self.model.optimizer.lr, learning_rate)

            # loss = self.model.train_on_batch(np_state_input, [np_winner])
            feed_dict = {
                self.model.input: np_state_input,
                one_hot_move_ph: np_moves,
                reward_ph: np_reward
            }
            _, loss, new_probs = self.session.run([self.minimize_op, self.loss_op, self.model.output],
                                                  feed_dict)
            entropy = self_entropy(new_probs)
            return loss, entropy

        self.train_step = train_step

    def get_policy_param(self):
        net_params = self.model.get_weights()
        return net_params

    def save_model(self, model_path):
        """ save model params to file """
        # net_params = self.get_policy_param()
        # pickle.dump(net_params, open(model_file, 'wb'), protocol=2)
        self.model.save_weights(model_path)

    def load_model(self, model_path):
        self.model.load_weights(model_path)