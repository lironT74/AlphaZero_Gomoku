from __future__ import print_function
import pickle
from multiprocessing import Pool, set_start_method
from mcts_alphaZero import *
from heuristic_player import Heuristic_player
from scipy.special import comb
import pandas as pd
import os
from mcts_pure import MCTSPlayer as PUREMCTS


def compare_all_models_statistics(players_list, opponents, width=6, height=6, n=4, num_games=100):

    jobs = []
    for opponent_player in opponents:
        for board in PAPER_TRUNCATED_BOARDS:
            jobs.append((players_list, opponent_player, board, width, height, n, num_games, 2))

        for board in PAPER_FULL_BOARDS:
            jobs.append((players_list, opponent_player, board, width, height, n, num_games, 2))

        jobs.append((players_list, opponent_player, EMPTY_BOARD, width, height, n, num_games, 1))


    with Pool(len(jobs)) as pool:
        print(f"Using {pool._processes} workers. There are {len(jobs)} jobs: \n")
        pool.starmap(compare_all_models_statistics, jobs)
        pool.close()
        pool.join()



def collect_statistics_two_models(players_list, opponent_player, board, width, height, n, num_games, start_player):

    board_state, board_name, p1, p2, alternative_p1, alternative_p2 = board

    columns = [
        f"no. games",
        f"no. wins",
        f"no. losses",
        f"no. ties",
        f"win ratio",

        f"avg game len",
        f"avg game len (wins)",
        f"avg game len (losses)",

        f"avg shutter size",
        f"avg shutter size (wins)",
        f"avg shutter size (losses)",

        f"fraction of plays with shutter",
        f"fraction of plays with shutter (wins)",
        f"fraction of plays with shutter (losses)",

        f"avg shutter size (real last turn)",
        f"avg shutter size (real last turn - wins)",
        f"avg shutter size (real last turn - losses)",

        f"fraction of plays with shutter (real last turn)",
        f"fraction of plays with shutter (real last turn - wins)",
        f"fraction of plays with shutter (real last turn - losses)",

        # f"no. plays with shutter",
        # f"total no. of plays",
        #
        # f"no. plays with shutter (wins)",
        # f"total no. of plays (wins)",
        #
        #
        # f"no. plays with shutter (losses)",
        # f"total no. of plays (losses)",

    ]

    result_df = pd.DataFrame(0, index=[player.name for player in players_list], columns=columns)

    for cur_player in players_list:
        result =              save_games_statistics(width=width,
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
                              start_player=start_player,
                              num_games=num_games)

        print(result)
        result_df.loc[cur_player.name] = result

    print(result_df.to_string())

    path = f"/home/lirontyomkin/AlphaZero_Gomoku/matches/statistics/vs {opponent_player.name}/{board_name}/"

    result_df.to_excel(f"{path}all models {num_games} games results.xlsx")


def save_games_statistics(width, height, n, board_state, board_name, cur_player,
                          opponent_player, last_move_p1, last_move_p2, correct_move_p1,
                          correct_move_p2, start_player, num_games):

    i_board1, board1 = initialize_board_without_init_call(width, height, n, input_board=board_state)
    game1 = Game(board1)

    wins = []

    shutters_wins = []
    real_last_turn_shutters_cur_player_wins = []

    shutters_losses = []
    real_last_turn_shutters_cur_player_losses = []

    shutters_ties = []
    real_last_turn_shutters_cur_player_ties = []

    total_plays = 0
    total_plays_wins = 0
    total_plays_losses = 0

    if start_player == 1:
        player_by_index = {1: cur_player, 2: opponent_player}

    else:
        player_by_index = {2: cur_player, 1: opponent_player}


    games_history = []

    for i in range(num_games):

        winner, game_length, shutter_sizes, real_last_move_shutter_sizes, game_history = game1.start_play(player1=player_by_index[1],
                                                             player2=player_by_index[2],
                                                             start_player=start_player,
                                                             is_shown=0,
                                                             start_board=i_board1,
                                                             last_move_p1=last_move_p1,
                                                             last_move_p2=last_move_p2,
                                                             correct_move_p1=correct_move_p1,
                                                             correct_move_p2=correct_move_p2,
                                                             is_random_last_turn_p1=player_by_index[1].is_random_last_turn,
                                                             is_random_last_turn_p2=player_by_index[2].is_random_last_turn,
                                                             savefig=0,
                                                             board_name=board_name,
                                                             return_statistics=1)

        games_history.append((winner, game_length, shutter_sizes, real_last_move_shutter_sizes, game_history))


        plays = range(1, game_length + 1, 1)
        start_player_range = [(index, shutter) for index, shutter in zip(plays[0::2], shutter_sizes[start_player]) if shutter != -1]


        if winner != -1:
            wins.append((player_by_index[winner].name, game_length))

            if winner == start_player:
                shutters_wins.extend([x[1] for x in start_player_range])
                total_plays_wins += len(plays[0::2])
            else:
                shutters_losses.extend([x[1] for x in start_player_range])
                total_plays_losses += len(plays[0::2])

        else:
            shutters_ties.extend([x[1] for x in start_player_range])



        if cur_player.is_random_last_turn:

            start_player_range = [(index, shutter) for index, shutter in
                                  zip(plays[0::2], real_last_move_shutter_sizes[start_player])
                                  if
                                  shutter != -1]

            if winner == start_player:
                real_last_turn_shutters_cur_player_wins.extend([x[1] for x in start_player_range])
            elif winner == 3 - start_player:
                real_last_turn_shutters_cur_player_losses.extend([x[1] for x in start_player_range])
            else:
                real_last_turn_shutters_cur_player_ties.extend([x[1] for x in start_player_range])


        total_plays += len(plays[0::2])


    no_games = num_games
    no_wins = [x[0] for x in wins].count(cur_player.name)
    no_losses = [x[0] for x in wins].count(opponent_player.name)
    no_ties = num_games - no_wins - no_losses
    win_ratio = 1.0*(no_wins + 0.5*no_ties) / no_games

    average_game_length = np.average([x[1] for x in wins]) if len([x[1] for x in wins]) >0 else -1
    average_game_length_wins = np.average([x[1] for x in wins if x[0] == cur_player.name]) if len([x[1] for x in wins if x[0] == cur_player.name]) > 0 else -1
    average_game_length_losses = np.average([x[1] for x in wins if x[0] == opponent_player.name]) if len([x[1] for x in wins if x[0] == opponent_player.name]) > 0 else -1


    avg_shutter_size = np.average(shutters_wins + shutters_losses + shutters_ties) if len(shutters_wins + shutters_losses + shutters_ties) > 0 else -1
    avg_shutter_size_wins = np.average(shutters_wins) if len(shutters_wins) > 0 else -1
    avg_shutter_size_losses = np.average(shutters_losses) if len(shutters_losses) > 0 else -1

    plays_with_shutter_fraction = len(shutters_wins + shutters_losses + shutters_ties)/total_plays
    plays_with_shutter_fraction_wins = len(shutters_wins)/total_plays_wins
    plays_with_shutter_fraction_losses = len(shutters_losses)/total_plays_losses


    avg_shutter_size_real_last_turn = -1
    avg_shutter_size_wins_real_last_turn = -1
    avg_shutter_size_losses_real_last_turn = -1
    plays_with_shutter_fraction_real_last_turn = -1
    plays_with_shutter_fraction_wins_real_last_turn = -1
    plays_with_shutter_fraction_losses_real_last_turn = -1


    if cur_player.is_random_last_turn:

        avg_shutter_size_real_last_turn = np.average(real_last_turn_shutters_cur_player_wins +
                                                     real_last_turn_shutters_cur_player_losses +
                                                     real_last_turn_shutters_cur_player_ties) if len(
                                                    real_last_turn_shutters_cur_player_wins +
                                                    real_last_turn_shutters_cur_player_losses +
                                                    real_last_turn_shutters_cur_player_ties) > 0 else -1

        avg_shutter_size_wins_real_last_turn = np.average(real_last_turn_shutters_cur_player_wins) if len(real_last_turn_shutters_cur_player_wins) > 0 else -1
        avg_shutter_size_losses_real_last_turn = np.average(real_last_turn_shutters_cur_player_losses) if len(real_last_turn_shutters_cur_player_losses) > 0 else -1

        plays_with_shutter_fraction_real_last_turn = len(real_last_turn_shutters_cur_player_wins +
                                                     real_last_turn_shutters_cur_player_losses +
                                                     real_last_turn_shutters_cur_player_ties) / total_plays

        plays_with_shutter_fraction_wins_real_last_turn = len(real_last_turn_shutters_cur_player_wins) / total_plays_wins
        plays_with_shutter_fraction_losses_real_last_turn = len(real_last_turn_shutters_cur_player_losses) / total_plays_losses


    path = f"/home/lirontyomkin/AlphaZero_Gomoku/matches/statistics/vs {opponent_player.name}/{board_name}/"

    if not os.path.exists(f"{path}{cur_player.name}/"):
        os.makedirs(f"{path}{cur_player.name}/")

    outfile = open(f"{path}{cur_player.name}/full_{num_games}_games_stats", 'wb')
    pickle.dump(games_history, outfile)
    outfile.close()


    result =    [no_games,
                 no_wins,
                 no_losses,
                 no_ties,
                 win_ratio,

                 average_game_length,
                 average_game_length_wins,
                 average_game_length_losses,

                 avg_shutter_size,
                 avg_shutter_size_wins,
                 avg_shutter_size_losses,

                 plays_with_shutter_fraction,
                 plays_with_shutter_fraction_wins,
                 plays_with_shutter_fraction_losses,

                 avg_shutter_size_real_last_turn,
                 avg_shutter_size_wins_real_last_turn,
                 avg_shutter_size_losses_real_last_turn,

                 plays_with_shutter_fraction_real_last_turn,
                 plays_with_shutter_fraction_wins_real_last_turn,
                 plays_with_shutter_fraction_losses_real_last_turn]

    return result


if __name__ == '__main__':
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
    players_list = [player_v7, player_v9, player_v10, player_v10_random]



    opponent_player_1 = Heuristic_player(name="forcing heuristic", heuristic="interaction with forcing")
    opponent_player_2 = PUREMCTS(c_puct=5, n_playout=500, name="pure MCTS 500")


    v9_5000 = ( '/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p3_v9/current_policy_5000.model',
           'v9_5000_no_MCTS', 3, True, False)
    policy_opponent_3 = PolicyValueNet(6, 6, model_file=v9_5000[0], input_plains_num=v9_5000[2])
    opponent_player_3 = MCTSPlayer(policy_opponent_3.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v9_5000[3],
                               name=v9_5000[1], input_plains_num=v9_5000[2], is_random_last_turn=v9_5000[4])


    v10_5000 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v10/current_policy_5000.model',
           f'v10_5000_no_MCTS', 4, True, False)
    policy_opponent_4 = PolicyValueNet(6, 6, model_file=v10_5000[0], input_plains_num=v10_5000[2])
    opponent_player_4 = MCTSPlayer(policy_opponent_4.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v10_5000[3],
                               name=v10_5000[1], input_plains_num=v10_5000[2], is_random_last_turn=v10_5000[4])


    opponents = [opponent_player_1, opponent_player_2, opponent_player_3, opponent_player_4]


    compare_all_models_statistics(players_list, opponents, width=6, height=6, n=4, num_games=1000)