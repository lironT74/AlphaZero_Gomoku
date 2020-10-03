# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
"""

from __future__ import print_function
from mcts_alphaZero import MCTSPlayer
from Game_boards_and_aux import *



def run():
    # n = 5
    # width, height = 8,8
    # model_1_file = './best_policy_8_8_5.model'

    n = 4
    width, height = 6,6

    model_1_file = '/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p3_v7/current_policy_2100.model'

    model_2_file = '/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v10/current_policy_5000.model'

    try:

        initial_board, board_name, last_move_p1, last_move_p2, _, _ = BOARD_1_TRUNCATED

        # i_board, board = initialize_board_without_init_call(height, width, n, input_board=initial_board)
        #

        # game = Game(board)

        # best_policy_1 = PolicyValueNet(width, height, model_file=model_1_file, input_plains_num=3)
        # mcts_player_1 = MCTSPlayer(best_policy_1.policy_value_fn, c_puct=5, n_playout=400, name="pt_6_6_4_p3_v7_2100", input_plains_num=3)
        #
        # best_policy_2 = PolicyValueNet(width, height, model_file=model_2_file, input_plains_num=4)
        # mcts_player_2 = MCTSPlayer(best_policy_2.policy_value_fn, c_puct=5, n_playout=400, name="pt_6_6_4_p4_v10_5000", input_plains_num=4)


        best_policy_1 = PolicyValueNet(width, height, model_file=model_2_file, input_plains_num=4)
        mcts_player_1 = MCTSPlayer(best_policy_1.policy_value_fn, c_puct=5, c=True, name="pt_6_6_4_p4_v10_5000",
                                   input_plains_num=4)

        board = initialize_board_with_init_and_last_moves(last_move_p1=last_move_p1, board_height=height,
                                                          board_width=width, n_in_row=n, input_board=initial_board)

        # mcts_player_1.get_action(board, display=True)


        print(initial_board)
        print(board.keep_only_close_enough_squares(3))
        print(board.keep_only_close_enough_squares(2))
        print(board.keep_only_close_enough_squares(1))
        print(board.keep_only_close_enough_squares(0))


        # print("asdas")
        # print(board.current_state(last_move=True, is_random_last_turn=True))
        # print("asdasas")

        # results = {-1:0, 1:0, 2:0}
        # start_player = 2
        #
        #
        # for i in range(1):
        #     winner = game.start_play(player2=mcts_player_1, player1=mcts_player_2,
        #                              start_player=start_player,
        #                              is_shown=False,
        #                              start_board=i_board,
        #                              last_move_p1=last_move_p1,
        #                              last_move_p2=last_move_p2,
        #                              correct_move_p1=last_move_p1,
        #                              correct_move_p2=last_move_p2,
        #                              savefig=1)
        #
        #     results[winner] += 1
        #     start_player = 3 - start_player
        #
        #     print(f"Game {i+1}: player {start_player} has started, player {winner} has won")
        #
        # print("\n\nWins of player 1: ", results[1])
        # print("Wins of player 1: ", results[2])
        # print("Ties: ", results[-1])
        #

    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
