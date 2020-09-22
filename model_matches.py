from __future__ import print_function
from multiprocessing import Pool, set_start_method
from mcts_alphaZero import MCTSPlayer
from heuristic_player import Heuristic_player
from Game_boards_and_aux import *
from scipy.special import comb
import pandas as pd
import os

def compare_all_models(models_list, width=6, height=6, n=4, open_path_threshold=-1, n_playout=400):

    jobs = []
    for i in range(len(models_list)):
        for j in range(i+1, len(models_list)):
            jobs.append((models_list[i], models_list[j], width, height, n, open_path_threshold, n_playout))


    with Pool(int(comb(len(models_list), 2))) as pool:
        print(f"Using {pool._processes} workers. There are {len(jobs)} jobs: \n")
        pool.starmap(compare_two_models, jobs)
        pool.close()
        pool.join()


def compare_two_models(model1, model2, width, height, n, open_path_threshold, n_playout):

    path1, name1, plains1, no_playouts1, is_random_last_turn1 = model1
    path2, name2, plains2, no_playouts2, is_random_last_turn2 = model2

    best_policy_1 = PolicyValueNet(width, height, model_file=path1, input_plains_num=plains1)
    mcts_player_1 = MCTSPlayer(best_policy_1.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts = no_playouts1, name=name1, input_plains_num=plains1, is_random_last_turn=is_random_last_turn1)

    best_policy_2 = PolicyValueNet(width, height, model_file=path2, input_plains_num=plains2)
    mcts_player_2 = MCTSPlayer(best_policy_2.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts = no_playouts2, name=name2, input_plains_num=plains2, is_random_last_turn=is_random_last_turn2)

    main_dir = "matches/matches - new selected models" if not no_playouts1 and not no_playouts2 \
        else "matches/no MCTS matches"

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
                      start_player=2,
                      open_path_threshold=open_path_threshold,
                      main_dir=main_dir)

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
                          start_player=2,
                          open_path_threshold=open_path_threshold,
                          main_dir=main_dir)


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
                          start_player=2,
                          open_path_threshold=open_path_threshold,
                          main_dir=main_dir)

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
                          start_player=2,
                          open_path_threshold=open_path_threshold,
                          main_dir=main_dir)

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
                      start_player=2,
                      open_path_threshold=open_path_threshold,
                      main_dir=main_dir)

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
                  start_player=1,
                  open_path_threshold=open_path_threshold,
                  main_dir=main_dir)


def save_game_res(width, height, n, board_state, board_name, mcts_player_1, mcts_player_2, last_move_p1,
                  last_move_p2, correct_move_p1, correct_move_p2, start_player, open_path_threshold, main_dir):


    i_board1, board1 = initialize_board_without_init_call(width, height, n, input_board=board_state, open_path_threshold=open_path_threshold)
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
                     board_name=board_name,
                     main_dir=main_dir)

    i_board2, board2 = initialize_board_without_init_call(width, height, n, input_board=board_state, open_path_threshold=open_path_threshold)
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
                    board_name=board_name,
                    main_dir=main_dir)




def compare_all_models_statistics(players_list, opponent_player, width=6, height=6, n=4, num_games=100):

    jobs = []
    for i in range(len(players_list)):
        jobs.append((players_list[i], opponent_player, width, height, n, num_games))


    with Pool(int(comb(len(players_list), 2))) as pool:
        print(f"Using {pool._processes} workers. There are {len(jobs)} jobs: \n")

        pool.starmap(collect_statistics_two_models, jobs)
        pool.close()
        pool.join()



def collect_statistics_two_models(cur_player, opponent_player, width, height, n, num_games):

    # for board_state, board_name, p1, p2, alternative_p1, alternative_p2 in PAPER_TRUNCATED_BOARDS:
    #
    #     save_games_statistics(width=width,
    #                   height=height,
    #                   n=n,
    #                   board_state=board_state,
    #                   board_name=board_name,
    #                   cur_player=cur_player,
    #                   opponent_player=opponent_player,
    #                   last_move_p1=p1,
    #                   last_move_p2=p2,
    #                   correct_move_p1=p1,
    #                   correct_move_p2=p2,
    #                   start_player=2,
    #                   num_games=num_games)
    #
    #
    # for board_state, board_name, p1, p2, alternative_p1, alternative_p2 in PAPER_FULL_BOARDS:
    #     save_games_statistics(width=width,
    #                   height=height,
    #                   n=n,
    #                   board_state=board_state,
    #                   board_name=board_name,
    #                   cur_player=cur_player,
    #                   opponent_player=opponent_player,
    #                   last_move_p1=p1,
    #                   last_move_p2=p2,
    #                   correct_move_p1=p1,
    #                   correct_move_p2=p2,
    #                   start_player=2,
    #                   num_games=num_games)

    board_state, board_name, p1, p2, alternative_p1, alternative_p2 = EMPTY_BOARD

    save_games_statistics(width=width,
                  height=height,
                  n=n,
                  board_state=board_state,
                  board_name=board_name,
                  cur_player=cur_player,
                  opponent_player=opponent_player,
                  last_move_p1=p1,
                  last_move_p2=p2,
                  correct_move_p1=p1,
                  correct_move_p2=p2,
                  start_player=1,
                  num_games=num_games)



def save_games_statistics(width, height, n, board_state, board_name, cur_player,
                          opponent_player, last_move_p1, last_move_p2, correct_move_p1,
                          correct_move_p2, start_player, num_games):

    i_board1, board1 = initialize_board_without_init_call(width, height, n, input_board=board_state)
    game1 = Game(board1)

    wins = []

    shutters = []
    real_last_turn_shutters_cur_player = []
    total_plays = 0


    if start_player == 1:
        player_by_index = {1: cur_player, 2: opponent_player}

    else:
        player_by_index = {2: cur_player, 1: opponent_player}


    for i in range(num_games):

        winner, game_length, shutter_sizes, real_last_move_shutter_sizes = game1.start_play(player1=player_by_index[1],
                                                             player2=player_by_index[2],
                                                             start_player=start_player,
                                                             is_shown=0,
                                                             start_board=i_board1,
                                                             last_move_p1=last_move_p1,
                                                             last_move_p2=last_move_p2,
                                                             correct_move_p1=correct_move_p1,
                                                             correct_move_p2=correct_move_p2,
                                                             savefig=0,
                                                             board_name=board_name,
                                                             return_statistics=1)

        if winner != -1:
            wins.append((player_by_index[winner].name, game_length))



        plays = range(1, game_length + 1, 1)

        start_player_range = [(index, shutter) for index, shutter in zip(plays[0::2], shutter_sizes[start_player]) if
                              shutter != -1]

        shutters.extend([x[1] for x in start_player_range])



        if cur_player.is_random_last_turn:
            start_player_range = [(index, shutter) for index, shutter in zip(plays[0::2], real_last_move_shutter_sizes[start_player])
                                  if
                                  shutter != -3]

            real_last_turn_shutters_cur_player.extend([x[1] for x in start_player_range])


        total_plays += len(plays[0::2])


    columns = [f"average shutter size",
               f"no. of plays which had a shutter",
               "total no. of plays",
               "fraction of plays with shutter" ,
               "no. wins",
               "average game length",
               "average game length for winning games",
               "average game length for loosing games"]


    index = [f"{cur_player.name}"]


    avg_shutter_size = np.average(shutters) if len(shutters) > 0 else -1
    plays_with_shutter = len(shutters)

    total_plays_count = total_plays
    plays_with_shutter_fraction = plays_with_shutter/total_plays_count

    number_of_wins = [x[0] for x in wins].count(cur_player.name)
    average_game_length = np.average([x[1] for x in wins])

    average_game_length_winning_games = np.average([x[1] for x in wins if x[0] == cur_player.name]) if len([x[1] for x in wins if x[0] == cur_player.name]) > 0 else -1
    average_game_length_loosing_games = np.average([x[1] for x in wins if x[0] == opponent_player.name]) if len([x[1] for x in wins if x[0] == opponent_player.name]) > 0 else -1


    np_results = np.array([[avg_shutter_size, plays_with_shutter,
                            total_plays_count, plays_with_shutter_fraction,
                            number_of_wins, average_game_length, average_game_length_winning_games, average_game_length_loosing_games]])

    df = pd.DataFrame(np_results, index=index, columns=columns)

    if cur_player.is_random_last_turn:
        avg_shutter_size_real_last_turn = np.average(real_last_turn_shutters_cur_player) if len(real_last_turn_shutters_cur_player) > 0 else -1
        plays_with_shutter_real_last_turn = len(real_last_turn_shutters_cur_player)

        plays_with_shutter_fraction_last_turn = plays_with_shutter_real_last_turn / total_plays_count

        df[f"average shutter size (real last turn)"] = [avg_shutter_size_real_last_turn]
        df[f"no. of plays which had a shutter (real last turn)"] = [plays_with_shutter_real_last_turn]
        df[f"fraction of plays with shutter (real last turn)"] = [plays_with_shutter_fraction_last_turn]


    print(df.to_string())


    # path = f"/home/lirontyomkin/AlphaZero_Gomoku/matches/statistics/vs {opponent_player.name}/{board_name}/"
    #
    # if not os.path.exists(f"{path}/"):
    #     os.makedirs(f"{path}/")
    #
    # df.to_excel(f"{path}{cur_player.name}/{num_games} games statistics.xlsx", index=True, header=True)
    #
    # f = open(f"{path}{num_games} games results.txt", "a")
    # f.write(df.to_string())
    # f.write('\n\n')
    # f.close()


if __name__ == '__main__':

    # v7 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p3_v7/current_policy_2100.model', 'pt_6_6_4_p3_v7_2100', 3, False)
    # v9 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p3_v9/current_policy_1350.model', 'pt_6_6_4_p3_v9_1350', 3, False)
    # v10 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v10/current_policy_1150.model', 'pt_6_6_4_p4_v10_1150', 4, False)
    #
    # models = [v7, v9, v10]
    # compare_all_models(models)

    n_playout = 400

    v9 = ( '/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p3_v9/current_policy_1500.model',
           'pt_6_6_4_p3_v9_1500', 3,True, False)

    v10 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v10/current_policy_1500.model',
           'pt_6_6_4_p4_v10_1500', 4, True, False)

    v10_random = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v10/current_policy_1500.model',
           'pt_6_6_4_p4_v10_1500_random', 4, True, True)

    v7 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p3_v7/current_policy_1500.model',
          'pt_6_6_4_p3_v7_1500', 3, True, False)



    policy_v7 = PolicyValueNet(6, 6, model_file=v7[0], input_plains_num=v7[2])
    player_v7 = MCTSPlayer(policy_v7.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v7[3],
                           name=v7[1], input_plains_num=v7[2], is_random_last_turn=v7[4])


    policy_v9 = PolicyValueNet(6, 6, model_file=v9[0], input_plains_num=v9[2])
    player_v9 = MCTSPlayer(policy_v9.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v9[3],
                               name=v9[1], input_plains_num=v9[2], is_random_last_turn=v9[4])


    policy_v10 = PolicyValueNet(6, 6, model_file=v10[0], input_plains_num=v10[2])
    player_v10 = MCTSPlayer(policy_v10.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v10[3],
                               name=v10[1], input_plains_num=v10[2], is_random_last_turn=v10[4])


    policy_v10_random = PolicyValueNet(6, 6, model_file=v10_random[0], input_plains_num=v10_random[2])
    player_v10_random = MCTSPlayer(policy_v10_random.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v10_random[3],
                               name=v10_random[1], input_plains_num=v10_random[2], is_random_last_turn=v10_random[4])


    opponent_player = Heuristic_player(name="Forcing heuristic", heuristic="interaction with forcing")


    players = [player_v7, player_v9, player_v10, player_v10_random]

    # compare_all_models(models)

    set_start_method('spawn')
    # compare_all_models_statistics(players, opponent_player, num_games=10)
    # compare_all_models_statistics(players, opponent_player,  num_games=100)
    # compare_all_models_statistics(players, opponent_player,  num_games=1000)
    # compare_all_models_statistics(players, opponent_player,  num_games=10000)

    collect_statistics_two_models(player_v9, opponent_player, 6, 6, 4, 10)