from __future__ import print_function
from multiprocessing import Pool, set_start_method
from mcts_alphaZero import *
from heuristic_player import Heuristic_player
from scipy.special import comb
import pandas as pd
import os
from mcts_pure import MCTSPlayer as PUREMCTS

def compare_all_players(playerss_list, width=6, height=6, n=4, open_path_threshold=-1):

    jobs = []
    for i in range(len(playerss_list)):
        for j in range(i+1, len(playerss_list)):
            jobs.append((playerss_list[i], playerss_list[j], width, height, n, open_path_threshold))


    with Pool(int(comb(len(playerss_list), 2))) as pool:
        print(f"Using {pool._processes} workers. There are {len(jobs)} jobs: \n")
        pool.starmap(compare_two_models, jobs)
        pool.close()
        pool.join()


def compare_two_models(player1, player2, width, height, n, open_path_threshold):

    main_dir = "matches/matches - new selected models" if not player1.is_random_last_turn and not player2.is_random_last_turn \
        else "matches/no MCTS matches"

    for board_state, board_name, p1, p2, alternative_p1, alternative_p2 in PAPER_6X6_TRUNCATED_BOARDS:

        save_game_res(width=width,
                      height=height,
                      n=n,
                      board_state=board_state,
                      board_name=board_name,
                      player1=player1,
                      player2=player2,
                      last_move_p1=p1,
                      last_move_p2=p2,
                      correct_move_p1=p1,
                      correct_move_p2=p2,
                      start_player=2,
                      open_path_threshold=open_path_threshold,
                      main_dir=main_dir)

        if player1.input_plains_num+player2.input_plains_num >= 7:

            save_game_res(width=width,
                          height=height,
                          n=n,
                          board_state=board_state,
                          board_name=board_name,
                          player1=player1,
                          player2=player2,
                          last_move_p1=alternative_p1,
                          last_move_p2=alternative_p2,
                          correct_move_p1=p1,
                          correct_move_p2=p2,
                          start_player=2,
                          open_path_threshold=open_path_threshold,
                          main_dir=main_dir)


        if player1.input_plains_num+player2.input_plains_num == 8:

            save_game_res(width=width,
                          height=height,
                          n=n,
                          board_state=board_state,
                          board_name=board_name,
                          player1=player1,
                          player2=player2,
                          last_move_p1=p1,
                          last_move_p2=alternative_p2,
                          correct_move_p1=p1,
                          correct_move_p2=p2,
                          start_player=2,
                          open_path_threshold=open_path_threshold,
                          main_dir=main_dir)

            save_game_res(width=width,
                          height=height,
                          n=n,
                          board_state=board_state,
                          board_name=board_name,
                          player1=player1,
                          player2=player2,
                          last_move_p1=alternative_p1,
                          last_move_p2=p2,
                          correct_move_p1=p1,
                          correct_move_p2=p2,
                          start_player=2,
                          open_path_threshold=open_path_threshold,
                          main_dir=main_dir)

    for board_state, board_name, p1, p2, alternative_p1, alternative_p2 in PAPER_FULL_6X6_BOARDS:
        save_game_res(width=width,
                      height=height,
                      n=n,
                      board_state=board_state,
                      board_name=board_name,
                      player1=player1,
                      player2=player2,
                      last_move_p1=p1,
                      last_move_p2=p2,
                      correct_move_p1=p1,
                      correct_move_p2=p2,
                      start_player=2,
                      open_path_threshold=open_path_threshold,
                      main_dir=main_dir)

    board_state, board_name, p1, p2, alternative_p1, alternative_p2 = EMPTY_BOARD_6X6

    save_game_res(width=width,
                  height=height,
                  n=n,
                  board_state=board_state,
                  board_name=board_name,
                  player1=player1,
                  player2=player2,
                  last_move_p1=p1,
                  last_move_p2=p2,
                  correct_move_p1=p1,
                  correct_move_p2=p2,
                  start_player=1,
                  open_path_threshold=open_path_threshold,
                  main_dir=main_dir)


def save_game_res(width, height, n, board_state, board_name, player1, player2, last_move_p1,
                  last_move_p2, correct_move_p1, correct_move_p2, start_player, open_path_threshold, main_dir):


    i_board1, board1 = initialize_board_without_init_call(width, height, n, input_board=board_state, open_path_threshold=open_path_threshold)
    game1 = Game(board1)


    game1.start_play(player1=player1, player2=player2,
                     start_player=start_player,
                     is_shown=0,
                     start_board=i_board1,
                     last_move_p1=last_move_p1,
                     last_move_p2=last_move_p2,
                     correct_move_p1=correct_move_p1,
                     correct_move_p2=correct_move_p2,
                     is_random_last_turn_p1=player1.is_random_last_turn,
                     is_random_last_turn_p2=player2.is_random_last_turn,
                     savefig=1,
                     board_name=board_name,
                     main_dir=main_dir)

    i_board2, board2 = initialize_board_without_init_call(width, height, n, input_board=board_state, open_path_threshold=open_path_threshold)
    game2 = Game(board2)
    game2.start_play(player1=player2, player2=player1,
                     start_player=start_player,
                     is_shown=0,
                     start_board=i_board2,
                     last_move_p1=last_move_p1,
                     last_move_p2=last_move_p2,
                     correct_move_p1=correct_move_p1,
                     correct_move_p2=correct_move_p2,
                     is_random_last_turn_p1=player1.is_random_last_turn,
                     is_random_last_turn_p2=player2.is_random_last_turn,
                     savefig=1,
                     board_name=board_name,
                     main_dir=main_dir)



if __name__ == '__main__':

    # v7 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p3_v7/current_policy_2100.model', 'pt_6_6_4_p3_v7_2100', 3, False)
    # v9 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p3_v9/current_policy_1350.model', 'pt_6_6_4_p3_v9_1350', 3, False)
    # v10 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v10/current_policy_1150.model', 'pt_6_6_4_p4_v10_1150', 4, False)
    #
    # models = [v7, v9, v10]
    # compare_all_players(models)

    pass