from __future__ import print_function
from multiprocessing import Pool
from game import Game
from mcts_alphaZero import MCTSPlayer
from Game_boards_and_aux import *
from scipy.special import comb

def compare_all_models(models_list, width=6, height=6, n=4):

    jobs = []
    for i in range(len(models_list)):
        for j in range(i+1, len(models_list)):
            jobs.append((models_list[i], models_list[j], width, height, n))


    with Pool(int(comb(len(models_list), 2))) as pool:
        print(f"Using {pool._processes} workers. There are {len(jobs)} jobs: \n")
        pool.starmap(compare_two_models, jobs)
        pool.close()
        pool.join()


def compare_two_models(model1, model2, width, height, n):

    path1, name1, plains1 = model1
    path2, name2, plains2 = model2



    best_policy_1 = PolicyValueNet(width, height, model_file=path1, input_plains_num=plains1)
    mcts_player_1 = MCTSPlayer(best_policy_1.policy_value_fn, c_puct=5, n_playout=400, name=name1, input_plains_num=plains1)

    best_policy_2 = PolicyValueNet(width, height, model_file=path2, input_plains_num=plains2)
    mcts_player_2 = MCTSPlayer(best_policy_2.policy_value_fn, c_puct=5, n_playout=400, name=name2, input_plains_num=plains2)


    for board_state, board_name, p1, p2, alternative_p1, alternative_p2 in PAPER_TRUNCATED_BOARDS:

        save_game_res(width=width,
                      height=height,
                      n=n,
                      board_state=board_state,
                      board_name=board_name,
                      mcts_player_1=mcts_player_1,
                      mcts_player_2=mcts_player_2,
                      last_move_p1=p1,
                      last_move_p2=p2,
                      correct_move_p1=p1,
                      correct_move_p2=p2,
                      start_player=2)

        if plains1+plains2 >= 7:

            save_game_res(width=width,
                          height=height,
                          n=n,
                          board_state=board_state,
                          board_name=board_name,
                          mcts_player_1=mcts_player_1,
                          mcts_player_2=mcts_player_2,
                          last_move_p1=alternative_p1,
                          last_move_p2=alternative_p2,
                          correct_move_p1=p1,
                          correct_move_p2=p2,
                          start_player=2)


        if plains1+plains2 == 8:

            save_game_res(width=width,
                          height=height,
                          n=n,
                          board_state=board_state,
                          board_name=board_name,
                          mcts_player_1=mcts_player_1,
                          mcts_player_2=mcts_player_2,
                          last_move_p1=p1,
                          last_move_p2=alternative_p2,
                          correct_move_p1=p1,
                          correct_move_p2=p2,
                          start_player=2)

            save_game_res(width=width,
                          height=height,
                          n=n,
                          board_state=board_state,
                          board_name=board_name,
                          mcts_player_1=mcts_player_1,
                          mcts_player_2=mcts_player_2,
                          last_move_p1=alternative_p1,
                          last_move_p2=p2,
                          correct_move_p1=p1,
                          correct_move_p2=p2,
                          start_player=2)

    for board_state, board_name, p1, p2, alternative_p1, alternative_p2 in PAPER_FULL_BOARDS:
        save_game_res(width=width,
                      height=height,
                      n=n,
                      board_state=board_state,
                      board_name=board_name,
                      mcts_player_1=mcts_player_1,
                      mcts_player_2=mcts_player_2,
                      last_move_p1=p1,
                      last_move_p2=p2,
                      correct_move_p1=p1,
                      correct_move_p2=p2,
                      start_player=2)

    board_state, board_name, p1, p2, alternative_p1, alternative_p2 = EMPTY_BOARD

    save_game_res(width=width,
                  height=height,
                  n=n,
                  board_state=board_state,
                  board_name=board_name,
                  mcts_player_1=mcts_player_1,
                  mcts_player_2=mcts_player_2,
                  last_move_p1=p1,
                  last_move_p2=p2,
                  correct_move_p1=p1,
                  correct_move_p2=p2,
                  start_player=1)


def save_game_res(width, height, n, board_state, board_name, mcts_player_1, mcts_player_2, last_move_p1, last_move_p2, correct_move_p1, correct_move_p2, start_player):
    i_board1, board1 = initialize_board_without_init_call(width, height, n, input_board=board_state)
    game1 = Game(board1)
    game1.start_play(player1=mcts_player_1, player2=mcts_player_2,
                     start_player=start_player,
                     is_shown=0,
                     start_board=i_board1,
                     last_move_p1=last_move_p1,
                     last_move_p2=last_move_p2,
                     correct_move_p1=correct_move_p1,
                     correct_move_p2=correct_move_p2,
                     savefig=1,
                     board_name=board_name)

    i_board2, board2 = initialize_board_without_init_call(width, height, n, input_board=board_state)
    game2 = Game(board2)
    game2.start_play(player1=mcts_player_2, player2=mcts_player_1,
                    start_player=start_player,
                    is_shown=0,
                    start_board=i_board2,
                    last_move_p1=last_move_p1,
                    last_move_p2=last_move_p2,
                    correct_move_p1=correct_move_p1,
                    correct_move_p2=correct_move_p2,
                    savefig=1,
                    board_name=board_name)


if __name__ == '__main__':

    v7 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p3_v7/current_policy_2100.model', 'pt_6_6_4_p3_v7_2100', 3)
    v9 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p3_v9/current_policy_1350.model', 'pt_6_6_4_p3_v9_1350', 3)
    v10 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v10/current_policy_1150.model', 'pt_6_6_4_p4_v10_1150', 4)
    models = [v7, v9, v10]

    compare_all_models(models)
