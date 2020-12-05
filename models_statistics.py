from __future__ import print_function
import pickle
from multiprocessing import Pool, set_start_method
from mcts_alphaZero import *
from heuristic_player import Heuristic_player
from scipy.special import comb
import pandas as pd
import os
from mcts_pure import MCTSPlayer as PUREMCTS
import PIL
import warnings

warnings.simplefilter("error", np.VisibleDeprecationWarning)

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


discription_dict = {
    "v7_1500" : "v7_1500",
    "v9_1500" : "v9_1500",
    "v10_1500": "v10_1500",
    "v10_1500_random" : "v10_1500_random",
    "v23_5000": "v23\nsim:50\nshutter:1\nfull:yes\niter:5000",
    "v24_5000": "v24\nsim:50\nshutter:0\nfull:yes\niter:5000",
    "v25_5000": "v25\nsim:50\nshutter:1\nfull:no\niter:5000",
    "v26_5000": "v26\nsim:50\nshutter:0\nfull:no\niter:5000",
    "v27_5000": "v27\nsim:25\nshutter:1\nfull:yes\niter:5000",
    "v28_5000": "v28\nsim:25\nshutter:0\nfull:yes\niter:5000",
    "v29_5000": "v29\nsim:25\nshutter:1\nfull:no\niter:5000",
    "v30_5000": "v30\nsim:25\nshutter:0\nfull:no\niter:5000",
    "v31_5000": "v31\nsim:100\nshutter:1\nfull:yes\niter:5000",
    "v32_5000": "v32\nsim:100\nshutter:0\nfull:yes\niter:5000",
    "v33_5000": "v33\nsim:100\nshutter:1\nfull:no\niter:5000",
    "v34_5000": "v34\nsim:100\nshutter:0\nfull:no\niter:5000"
}




old_models = (["v7_1500", "v9_1500", "v10_1500", "v10_1500_random"], "old_models")
all_new_12_models = (["v27_5000", "v28_5000", "v29_5000", "v30_5000", "v23_5000", "v24_5000", "v25_5000", "v26_5000", "v31_5000", "v32_5000", "v33_5000", "v34_5000"], "all_new_12_models", False)

mcts_25_models = (["v27_5000", "v28_5000", "v29_5000", "v30_5000"], "mcts_25_models", False)
mcts_50_models = (["v23_5000", "v24_5000", "v25_5000", "v26_5000"], "mcts_50_models", False)
mcts_100_models = (["v31_5000", "v32_5000", "v33_5000", "v34_5000"], "mcts_100_models", False)

full_boards_models = (["v27_5000", "v28_5000","v23_5000", "v24_5000","v31_5000", "v32_5000"], "full_boards_models", True)
non_full_boards_models = (["v29_5000", "v30_5000","v25_5000", "v26_5000", "v33_5000", "v34_5000"], "non_full_boards_models", True)

shutter_0_models = (["v28_5000", "v30_5000", "v24_5000", "v26_5000", "v32_5000", "v34_5000"], "shutter_0_models", True)
shutter_1_models = (["v27_5000", "v29_5000", "v23_5000", "v25_5000", "v31_5000", "v33_5000"], "shutter_1_models", True)


all_models_variations = [all_new_12_models, mcts_25_models, mcts_50_models, mcts_100_models, full_boards_models,
                  non_full_boards_models, shutter_0_models, shutter_1_models]



def compare_all_models_statistics(players_list, opponents, width=6, height=6, n=4, num_games=1000, sub_dir = "statistics"):


    print(f"Playing: ")

    jobs = []

    truncated = {6: PAPER_6X6_TRUNCATED_BOARDS, 10: PAPER_10X10_TRUNCATED_BOARDS}
    full = {6: PAPER_FULL_6X6_BOARDS, 10: PAPER_FULL_10X10_BOARDS}
    empty = {6: EMPTY_BOARD_6X6, 10: EMPTY_BOARD_10X10}


    def who_started_dict(board_name):

        X_boards = [
                    "board 3 full",
                    "board 5 full",
                    "board 3 truncated",
                    "board 5 truncated",
                    "empty board"
                   ]

        O_boards = [
                    "board 1 full",
                    "board 2 full",
                    "board 1 truncated",
                    "board 2 truncated",
                    "board 4 full",
                    "board 4 truncated",
                   ]

        if board_name in X_boards:
            return 1
        elif board_name in O_boards:
            return 2
        else:
            raise Exception(f"{board_name} is not a valid board")




    for opponent_player in opponents:

        for board in truncated[width]:

            board_state, board_name, p1, p2, alternative_p1, alternative_p2 = board

            for cur_player in players_list:
                jobs.append((width, height, n, board_state, board_name, cur_player,
                             opponent_player, p1, p2, p1,
                             p2, who_started_dict(board_name), num_games, sub_dir))

        for board in full[width]:

            board_state, board_name, p1, p2, alternative_p1, alternative_p2 = board

            for cur_player in players_list:
                jobs.append((width, height, n, board_state, board_name, cur_player,
                             opponent_player, p1, p2, p1,
                             p2, who_started_dict(board_name), num_games, sub_dir))


        board_state, board_name, p1, p2, alternative_p1, alternative_p2 = empty[width]

        for cur_player in players_list:
            jobs.append((width, height, n, board_state, board_name, cur_player,
                         opponent_player, p1, p2, p1,
                         p2, who_started_dict(board_name), num_games, sub_dir))


    with Pool(20) as pool:
        print(f"Using {pool._processes} workers. There are {len(jobs)} X {num_games} games to play: \n")
        pool.starmap(save_games_statistics, jobs)
        pool.close()
        pool.join()



    print(f"Statistics: ")

    jobs = []
    for opponent_player in opponents:
        for board in PAPER_6X6_TRUNCATED_BOARDS:

            jobs.append((players_list, opponent_player, board, num_games, sub_dir))

        for board in PAPER_FULL_6X6_BOARDS:
            jobs.append((players_list, opponent_player, board, num_games, sub_dir))

        jobs.append((players_list, opponent_player, EMPTY_BOARD_6X6, num_games, sub_dir))


    with Pool() as pool:
        # print(f"Using {pool._processes} workers. There are {len(jobs)} jobs: \n")
        pool.starmap(statistics_of_games_to_df, jobs)
        pool.close()
        pool.join()




def statistics_of_games_to_df(players_list, opponent_player, board, num_games, sub_dir):


    board_state, board_name, p1, p2, alternative_p1, alternative_p2 = board

    path = f"/home/lirontyomkin/AlphaZero_Gomoku/matches/{sub_dir}/vs {opponent_player.name}/{board_name}/"

    if os.path.exists(f"{path}all models {num_games} games results.xlsx"):
        print(f"already saved {opponent_player.name} on {board_name} df")
        return

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

        "CI_wins_losses",

        "CI_game_length",
        "CI_game_length_wins",
        "CI_game_length_losses",

        "CI_shutter_size",
        "CI_shutter_size_wins",
        "CI_shutter_size_losses",

        "CI_plays_with_shutter_all",
        "CI_plays_with_shutter_wins",
        "CI_plays_with_shutter_losses",

        "CI_shutter_size_real_last_turn",
        "CI_shutter_size_wins_real_last_turn",
        "CI_shutter_size_losses_real_last_turn",

        "CI_plays_with_shutter_all_real_last_turn",
        "CI_plays_with_shutter_wins_real_last_turn",
        "CI_plays_with_shutter_losses_real_last_turn"

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

    result_df = pd.DataFrame(index=[player.name for player in players_list], columns=columns)


    for cur_player in players_list:

        result = get_statistics_from_saved_results(board_name, cur_player, opponent_player, num_games, sub_dir = sub_dir)

        for i, col in enumerate(columns):
            result_df.loc[cur_player.name, col] = result[i]


    print(result_df.to_string())

    result_df.to_excel(f"{path}all models {num_games} games results.xlsx")



def save_games_statistics(width, height, n, board_state, board_name, cur_player,
                          opponent_player, last_move_p1, last_move_p2, correct_move_p1,
                          correct_move_p2, start_player, num_games, sub_dir):

    path = f"/home/lirontyomkin/AlphaZero_Gomoku/matches/{sub_dir}/vs {opponent_player.name}/{board_name}/"
    if os.path.exists(f"{path}/{cur_player.name}/full_{num_games}_games_stats"):
        print(f"already saved {opponent_player.name} vs {cur_player.name} on {board_name}")
        return


    i_board1, board1 = initialize_board_without_init_call(width, height, n, input_board=board_state)
    game1 = Game(board1)

    all_games_history = []

    print(f"Starting {num_games} games: {cur_player.name} vs {opponent_player.name} on {board_name} {cur_time()}")

    for i in range(num_games):

        winner, game_length, shutter_sizes, real_last_move_shutter_sizes, game_history = game1.start_play(
                                                             player1=cur_player,
                                                             player2=opponent_player,
                                                             start_player=start_player,
                                                             is_shown=0,
                                                             start_board=i_board1,
                                                             last_move_p1=last_move_p1,
                                                             last_move_p2=last_move_p2,
                                                             correct_move_p1=correct_move_p1,
                                                             correct_move_p2=correct_move_p2,
                                                             is_random_last_turn_p1=cur_player.is_random_last_turn,
                                                             is_random_last_turn_p2=opponent_player.is_random_last_turn,
                                                             savefig=0,
                                                             board_name=board_name,
                                                             return_statistics=1)

        all_games_history.append((start_player, winner, game_length, shutter_sizes, real_last_move_shutter_sizes, game_history))


    print(f"Done {num_games} games: {cur_player.name} vs {opponent_player.name} on {board_name} {cur_time()}")

    path = f"/home/lirontyomkin/AlphaZero_Gomoku/matches/{sub_dir}/vs {opponent_player.name}/{board_name}/"

    if not os.path.exists(f"{path}{cur_player.name}/"):
        os.makedirs(f"{path}{cur_player.name}/")

    outfile = open(f"{path}{cur_player.name}/full_{num_games}_games_stats", 'wb')
    pickle.dump(all_games_history, outfile)
    outfile.close()



def get_statistics_from_saved_results(board_name, cur_player, opponent_player, num_games, sub_dir):

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


    cur_player_player_number = 1


    games_history = pickle.load(
        open(f"/home/lirontyomkin/AlphaZero_Gomoku/matches/{sub_dir}/vs {opponent_player.name}/{board_name}/"
             f"{cur_player.name}/full_{num_games}_games_stats", 'rb'))


    for i in range(num_games):

        _, winner, game_length, shutter_sizes, real_last_move_shutter_sizes, game_history = games_history[i]


        plays = range(1, game_length + 1, 1)
        cur_player_range = [(index, shutter) for index, shutter in zip(plays[0::2], shutter_sizes[cur_player_player_number]) if shutter != -1]


        if winner != -1:
            wins.append((cur_player.name if winner == cur_player_player_number else opponent_player.name, game_length))

            if winner == cur_player_player_number:
                shutters_wins.extend([x[1] for x in cur_player_range])
                total_plays_wins += len(plays[0::2])

            else:
                shutters_losses.extend([x[1] for x in cur_player_range])
                total_plays_losses += len(plays[0::2])

        else:
            shutters_ties.extend([x[1] for x in cur_player_range])


        if cur_player.is_random_last_turn:

            cur_player_range = [(index, shutter) for index, shutter in
                                  zip(plays[0::2], real_last_move_shutter_sizes[cur_player_player_number])
                                  if
                                  shutter != -1]

            if winner == cur_player_player_number:
                real_last_turn_shutters_cur_player_wins.extend([x[1] for x in cur_player_range])
            elif winner == 3 - cur_player_player_number:
                real_last_turn_shutters_cur_player_losses.extend([x[1] for x in cur_player_range])
            else:
                real_last_turn_shutters_cur_player_ties.extend([x[1] for x in cur_player_range])

        total_plays += len(plays[0::2])


    no_games = num_games
    no_wins = [x[0] for x in wins].count(cur_player.name)
    no_losses = [x[0] for x in wins].count(opponent_player.name)
    no_ties = num_games - no_wins - no_losses
    win_ratio = 1.0*(no_wins + 0.5*no_ties) / no_games


    print(f"{cur_player.name} vs {opponent_player.name} on {board_name}: {no_games}, {no_wins}, {no_losses}, {no_ties}")

    average_game_length = np.average([x[1] for x in wins]) if len([x[1] for x in wins]) >0 else -1
    average_game_length_wins = np.average([x[1] for x in wins if x[0] == cur_player.name]) if len([x[1] for x in wins if x[0] == cur_player.name]) > 0 else -1
    average_game_length_losses = np.average([x[1] for x in wins if x[0] == opponent_player.name]) if len([x[1] for x in wins if x[0] == opponent_player.name]) > 0 else -1


    avg_shutter_size = np.average(shutters_wins + shutters_losses + shutters_ties) if len(shutters_wins + shutters_losses + shutters_ties) > 0 else -1
    avg_shutter_size_wins = np.average(shutters_wins) if len(shutters_wins) > 0 else -1
    avg_shutter_size_losses = np.average(shutters_losses) if len(shutters_losses) > 0 else -1

    plays_with_shutter_fraction = len(shutters_wins + shutters_losses + shutters_ties)/total_plays if total_plays > 0 else 0
    plays_with_shutter_fraction_wins = len(shutters_wins)/total_plays_wins if total_plays_wins > 0 else 0
    plays_with_shutter_fraction_losses = len(shutters_losses)/total_plays_losses if total_plays_losses > 0 else 0


    wins_losses_sample = [1 if x[0] == cur_player.name else 0 for x in wins]
    CI_wins_losses = bootstrap_mean(wins_losses_sample)

    CI_game_length = bootstrap_mean([x[1] for x in wins]) if len([x[1] for x in wins]) > 0 else [-1,-1]
    CI_game_length_wins = bootstrap_mean([x[1] for x in wins if x[0] == cur_player.name]) if len([x[1] for x in wins if x[0] == cur_player.name]) > 0 else [-1,-1]
    CI_game_length_losses = bootstrap_mean([x[1] for x in wins if x[0] == opponent_player.name]) if len([x[1] for x in wins if x[0] == opponent_player.name]) > 0 else [-1,-1]

    CI_shutter_size = bootstrap_mean(shutters_wins + shutters_losses + shutters_ties) if len(shutters_wins + shutters_losses + shutters_ties) > 0 else [-1,-1]
    CI_shutter_size_wins = bootstrap_mean(shutters_wins) if len(shutters_wins) > 0 else [-1,-1]
    CI_shutter_size_losses = bootstrap_mean(shutters_losses) if len(shutters_losses) > 0 else [-1,-1]


    plays_with_shutter_all = len(shutters_wins + shutters_losses + shutters_ties)
    plays_without_shutter_all = total_plays - plays_with_shutter_all
    plays_with_shutter_sample_all_plays = np.concatenate((np.ones(plays_with_shutter_all), np.zeros(plays_without_shutter_all)))
    CI_plays_with_shutter_all = bootstrap_mean(plays_with_shutter_sample_all_plays)


    plays_with_shutter_wins = len(shutters_wins)
    plays_without_shutter_wins = total_plays_wins - plays_with_shutter_wins
    plays_with_shutter_sample_wins_plays = np.concatenate((np.ones(plays_with_shutter_wins), np.zeros(plays_without_shutter_wins)))
    CI_plays_with_shutter_wins = bootstrap_mean(plays_with_shutter_sample_wins_plays)


    plays_with_shutter_losses = len(shutters_losses)
    plays_without_shutter_losses = total_plays_losses - plays_with_shutter_losses
    plays_with_shutter_sample_losses_plays = np.concatenate((np.ones(plays_with_shutter_losses), np.zeros(plays_without_shutter_losses)))
    CI_plays_with_shutter_losses = bootstrap_mean(plays_with_shutter_sample_losses_plays)



    avg_shutter_size_real_last_turn = -1
    avg_shutter_size_wins_real_last_turn = -1
    avg_shutter_size_losses_real_last_turn = -1
    plays_with_shutter_fraction_real_last_turn = -1
    plays_with_shutter_fraction_wins_real_last_turn = -1
    plays_with_shutter_fraction_losses_real_last_turn = -1

    CI_shutter_size_real_last_turn = [-1, -1]
    CI_shutter_size_wins_real_last_turn = [-1, -1]
    CI_shutter_size_losses_real_last_turn = [-1, -1]
    CI_plays_with_shutter_all_real_last_turn = [-1, -1]
    CI_plays_with_shutter_wins_real_last_turn = [-1, -1]
    CI_plays_with_shutter_losses_real_last_turn = [-1, -1]


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
                                                     real_last_turn_shutters_cur_player_ties) / total_plays if total_plays > 0 else 0

        plays_with_shutter_fraction_wins_real_last_turn = len(real_last_turn_shutters_cur_player_wins) / total_plays_wins if total_plays_wins > 0 else 0
        plays_with_shutter_fraction_losses_real_last_turn = len(real_last_turn_shutters_cur_player_losses) / total_plays_losses if total_plays_losses > 0 else 0


        CI_shutter_size_real_last_turn = bootstrap_mean(real_last_turn_shutters_cur_player_wins +
                                                     real_last_turn_shutters_cur_player_losses +
                                                     real_last_turn_shutters_cur_player_ties) if len(
                                                    real_last_turn_shutters_cur_player_wins +
                                                    real_last_turn_shutters_cur_player_losses +
                                                    real_last_turn_shutters_cur_player_ties) > 0 else [-1, -1]

        CI_shutter_size_wins_real_last_turn = bootstrap_mean(real_last_turn_shutters_cur_player_wins) if len(real_last_turn_shutters_cur_player_wins) > 0 else [-1, -1]
        CI_shutter_size_losses_real_last_turn = bootstrap_mean(real_last_turn_shutters_cur_player_losses) if len(real_last_turn_shutters_cur_player_losses) > 0 else [-1, -1]


        plays_with_shutter_all_real_last_turn = len(real_last_turn_shutters_cur_player_wins +
                                                     real_last_turn_shutters_cur_player_losses +
                                                     real_last_turn_shutters_cur_player_ties)
        plays_without_shutter_all_real_last_turn = total_plays - plays_with_shutter_all_real_last_turn
        plays_with_shutter_sample_all_plays_real_last_turn =  np.concatenate((np.ones(plays_with_shutter_all_real_last_turn), np.zeros(plays_without_shutter_all_real_last_turn)))
        CI_plays_with_shutter_all_real_last_turn = bootstrap_mean(plays_with_shutter_sample_all_plays_real_last_turn)


        plays_with_shutter_wins_real_last_turn = len(real_last_turn_shutters_cur_player_wins)
        plays_without_shutter_wins_real_last_turn = total_plays_wins - plays_with_shutter_wins_real_last_turn
        plays_with_shutter_sample_wins_plays_real_last_turn = np.concatenate((np.ones(plays_with_shutter_wins_real_last_turn), np.zeros(plays_without_shutter_wins_real_last_turn)))
        CI_plays_with_shutter_wins_real_last_turn = bootstrap_mean(plays_with_shutter_sample_wins_plays_real_last_turn)

        plays_with_shutter_losses_real_last_turn = len(real_last_turn_shutters_cur_player_losses)
        plays_without_shutter_losses_real_last_turn = total_plays_losses - plays_with_shutter_losses_real_last_turn
        plays_with_shutter_sample_losses_plays_real_last_turn = np.concatenate((np.ones(plays_with_shutter_losses_real_last_turn), np.zeros(plays_without_shutter_losses_real_last_turn)))
        CI_plays_with_shutter_losses_real_last_turn = bootstrap_mean(plays_with_shutter_sample_losses_plays_real_last_turn)



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
                 plays_with_shutter_fraction_losses_real_last_turn,


                 np.array2string(np.array(CI_wins_losses)),

                 np.array2string(np.array(CI_game_length)),
                 np.array2string(np.array(CI_game_length_wins)),
                 np.array2string(np.array(CI_game_length_losses)),

                 np.array2string(np.array(CI_shutter_size)),
                 np.array2string(np.array(CI_shutter_size_wins)),
                 np.array2string(np.array(CI_shutter_size_losses)),

                 np.array2string(np.array(CI_plays_with_shutter_all)),
                 np.array2string(np.array(CI_plays_with_shutter_wins)),
                 np.array2string(np.array(CI_plays_with_shutter_losses)),

                 np.array2string(np.array(CI_shutter_size_real_last_turn)),
                 np.array2string(np.array(CI_shutter_size_wins_real_last_turn)),
                 np.array2string(np.array(CI_shutter_size_losses_real_last_turn)),

                 np.array2string(np.array(CI_plays_with_shutter_all_real_last_turn)),
                 np.array2string(np.array(CI_plays_with_shutter_wins_real_last_turn)),
                 np.array2string(np.array(CI_plays_with_shutter_losses_real_last_turn))
    ]


    return result




def plot_all_statistics_results(opponents, num_games, sub_dir):

    jobs = []
    for opponent_player in opponents:
        for board in PAPER_6X6_TRUNCATED_BOARDS:
            board_name = board[1]
            jobs.append((num_games, opponent_player.name, board_name, sub_dir))

        for board in PAPER_FULL_6X6_BOARDS:
            board_name = board[1]
            jobs.append((num_games, opponent_player.name, board_name, sub_dir))

        board_name = EMPTY_BOARD_6X6[1]
        jobs.append((num_games, opponent_player.name, board_name, sub_dir))

    with Pool(len(jobs)) as pool:
    # with Pool(5) as pool:
        pool.starmap(create_statistics_graphics, jobs)
        pool.close()
        pool.join()


    jobs_collage = []
    BOARDS = [BOARD_1_TRUNCATED, BOARD_2_TRUNCATED, BOARD_1_FULL, BOARD_2_FULL, EMPTY_BOARD_6X6]
    for board in BOARDS:
        board_name = board[1]
        jobs_collage.append((board_name, opponents, sub_dir))

    with Pool(len(jobs_collage)) as pool:
        pool.starmap(call_collage_statistics_results, jobs_collage)
        pool.close()
        pool.join()



def create_statistics_graphics(num_games, opponent_name, board_name, sub_dir):

    path = f"/home/lirontyomkin/AlphaZero_Gomoku/matches/{sub_dir}/vs {opponent_name}/{board_name}/"

    width = 40
    height = 10
    font_size = 27


    for variation in all_models_variations:
        models = variation[0]
        group_by = variation[1]
        is_interchangeable_color = variation[2]

        data = pd.read_excel(f"{path}all models {num_games} games results.xlsx", index_col=0)
        data = data.reindex(models + old_models[0])

        save_plays_with_shutter_results(path, num_games, width, height, font_size, opponent_name, data, group_by=group_by)
        save_shutter_size(path, num_games, width, height, font_size, opponent_name, data, group_by=group_by)
        save_save_game_len(path, num_games, width, height, font_size, opponent_name, data, group_by=group_by)
        save_game_results(path, num_games, width, height, font_size, opponent_name, data, group_by=group_by)
        save_win_ratio_no_ties(path, num_games, width, height, font_size, opponent_name, data, group_by=group_by, is_interchangeable_color=is_interchangeable_color)


def save_plays_with_shutter_results(path, num_games, width, height, font_size, opponent_name, data, group_by=None):
    mpl.rcParams.update({'font.size': font_size})

    fig = plt.figure(constrained_layout=True)
    fig.suptitle(f"Fraction of plays with shutter results - {opponent_name} - {group_by}", y=1.05)

    fig.set_size_inches(width, height)
    grid = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[20, 1])

    ind = np.arange(len(data.index) - 1)
    width = 0.25

    ax = fig.add_subplot(grid[0, 0])

    ax.bar(ind-width, data["fraction of plays with shutter"][:-1], width=width, label="fraction of plays with shutter", color="blue")
    ax.bar(ind, data["fraction of plays with shutter (wins)"][:-1], width=width, label="fraction of plays with shutter - wins", color="green")
    ax.bar(ind + width, data["fraction of plays with shutter (losses)"][:-1], width=width, label="fraction of plays with shutter - losses", color="red")


    for index, ci in zip(ind - width, data["CI_plays_with_shutter_all"][:-1]):
        ax.plot((index, index), npstr2tuple(ci) , 'r_-', color='black',  linewidth=4, mew=4, ms=20)

    for index, ci in zip(ind, data["CI_plays_with_shutter_wins"][:-1]):
        try:
            ax.plot((index, index), npstr2tuple(ci) , 'r_-', color='black', linewidth=4, mew=4, ms=20)
        except:
            pass


    for index, ci in zip(ind + width, data["CI_plays_with_shutter_losses"][:-1]):

        try:
            ax.plot((index, index), npstr2tuple(ci), 'r_-', color='black', linewidth=4, mew=4, ms=20)
        except:
            pass




    ax.bar(len(data.index)- 0.5 - 3*width, data["fraction of plays with shutter"][-1], width=width, color="blue")
    ax.bar(len(data.index)- 0.5 - 2*width, data["fraction of plays with shutter (real last turn)"][-1], label="real last turn", width=width, color="cornflowerblue")

    ax.bar(len(data.index)- 0.5 - 0.5*width, data["fraction of plays with shutter (wins)"][-1], width=width, color="green")
    ax.bar(len(data.index)- 0.5 + 0.5*width, data["fraction of plays with shutter (real last turn - wins)"][-1], label="real last turn", width=width, color="greenyellow")

    ax.bar(len(data.index)- 0.5 + 2*width, data["fraction of plays with shutter (losses)"][-1], width=width, color="red")
    ax.bar(len(data.index)- 0.5 + 3*width, data["fraction of plays with shutter (real last turn - losses)"][-1], label="real last turn", width=width, color="lightcoral")


    ax.plot((len(data.index)- 0.5 - 3*width, len(data.index)- 0.5 - 3*width), npstr2tuple(data["CI_plays_with_shutter_all"][-1]) , 'r_-', color='black',  linewidth=4, mew=4, ms=20)
    ax.plot((len(data.index)- 0.5 - 2*width, len(data.index)- 0.5 - 2*width), npstr2tuple(data["CI_plays_with_shutter_all_real_last_turn"][-1]) , 'r_-', color='black',  linewidth=4, mew=4, ms=20)

    ax.plot((len(data.index)- 0.5 - 0.5*width, len(data.index)- 0.5 - 0.5*width), npstr2tuple(data["CI_plays_with_shutter_wins"][-1]) , 'r_-', color='black',  linewidth=4, mew=4, ms=20)
    ax.plot((len(data.index)- 0.5 + 0.5*width, len(data.index)- 0.5 + 0.5*width), npstr2tuple(data["CI_plays_with_shutter_wins_real_last_turn"][-1]) , 'r_-', color='black',  linewidth=4, mew=4, ms=20)

    ax.plot((len(data.index)- 0.5 + 2*width, len(data.index)- 0.5 + 2*width), npstr2tuple(data["CI_plays_with_shutter_losses"][-1]) , 'r_-', color='black',  linewidth=4, mew=4, ms=20)
    ax.plot((len(data.index)- 0.5 + 3*width, len(data.index)- 0.5 + 3*width), npstr2tuple(data["CI_plays_with_shutter_losses_real_last_turn"][-1]) , 'r_-', color='black',  linewidth=4, mew=4, ms=20)



    ax.set_xticks(list(np.arange(len(data.index) - 1)) + [len(data.index) - 0.5])
    # ax.set_xticklabels(data.index)
    ax.set_xticklabels([discription_dict[name] for name in data.index])


    ax.set_ylim([0,1.1])

    lax = fig.add_subplot(grid[1, 0])
    h, l = ax.get_legend_handles_labels()
    lax.legend(h, l, borderaxespad=0, loc="center", fancybox=True, shadow=True, ncol=2)
    lax.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = PIL.Image.open(buf)

    if group_by == None:
        plt.savefig(f"{path}Fraction of plays with shutter results.png", bbox_inches='tight')
    else:

        plt.savefig(f"{path}Fraction of plays with shutter results {group_by}.png", bbox_inches='tight')

    plt.close('all')


def save_shutter_size(path, num_games, width, height, font_size, opponent_name, data, group_by=None):
    mpl.rcParams.update({'font.size': font_size})

    fig = plt.figure(constrained_layout=True)
    fig.suptitle(f"Shutter sizes results  - {opponent_name} - {group_by}", y=1.05)

    fig.set_size_inches(width, height)
    grid = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[20, 1])

    ind = np.arange(len(data.index) - 1)
    width = 0.25

    ax = fig.add_subplot(grid[0, 0])

    ax.bar(ind-width, data["avg shutter size"][:-1], width=width, label="average shutter size", color="blue")
    ax.bar(ind, data["avg shutter size (wins)"][:-1], width=width, label="average shutter size - wins", color="green")
    ax.bar(ind + width, data["avg shutter size (losses)"][:-1], width=width, label="average shutter size - losses", color="red")

    for index, ci in zip(ind - width, data["CI_shutter_size"][:-1]):
        ax.plot((index, index), npstr2tuple(ci), 'r_-', color='black', linewidth=4, mew=4, ms=20)

    for index, ci in zip(ind, data["CI_shutter_size_wins"][:-1]):
        ax.plot((index, index), npstr2tuple(ci), 'r_-', color='black', linewidth=4, mew=4, ms=20)

    for index, ci in zip(ind + width, data["CI_shutter_size_losses"][:-1]):
        ax.plot((index, index), npstr2tuple(ci), 'r_-', color='black', linewidth=4, mew=4, ms=20)


    ax.bar(len(data.index)- 0.5 - 3*width, data["avg shutter size"][-1], width=width, color="blue")
    ax.bar(len(data.index)- 0.5 - 2*width, data["avg shutter size (real last turn)"][-1], label="real last turn", width=width, color="cornflowerblue")

    ax.bar(len(data.index)- 0.5 - 0.5*width, data["avg shutter size (wins)"][-1], width=width, color="green")
    ax.bar(len(data.index)- 0.5 + 0.5*width, data["avg shutter size (real last turn - wins)"][-1], label="real last turn", width=width, color="greenyellow")

    ax.bar(len(data.index)- 0.5 + 2*width, data["avg shutter size (losses)"][-1], width=width, color="red")
    ax.bar(len(data.index)- 0.5 + 3*width, data["avg shutter size (real last turn - losses)"][-1], label="real last turn", width=width, color="lightcoral")


    ax.plot((len(data.index) - 0.5 - 3 * width, len(data.index) - 0.5 - 3 * width),npstr2tuple(data["CI_shutter_size"][-1]), 'r_-', color='black', linewidth=4, mew=4, ms=20)
    ax.plot((len(data.index) - 0.5 - 2 * width, len(data.index) - 0.5 - 2 * width),npstr2tuple(data["CI_shutter_size_real_last_turn"][-1]), 'r_-', color='black', linewidth=4, mew=4,ms=20)

    ax.plot((len(data.index) - 0.5 - 0.5 * width, len(data.index) - 0.5 - 0.5 * width),npstr2tuple(data["CI_shutter_size_wins"][-1]), 'r_-', color='black', linewidth=4, mew=4, ms=20)
    ax.plot((len(data.index) - 0.5 + 0.5 * width, len(data.index) - 0.5 + 0.5 * width),npstr2tuple(data["CI_shutter_size_wins_real_last_turn"][-1]), 'r_-', color='black', linewidth=2,mew=2, ms=20)

    ax.plot((len(data.index) - 0.5 + 2 * width, len(data.index) - 0.5 + 2 * width),npstr2tuple(data["CI_shutter_size_losses"][-1]), 'r_-', color='black', linewidth=4, mew=4, ms=20)
    ax.plot((len(data.index) - 0.5 + 3 * width, len(data.index) - 0.5 + 3 * width),npstr2tuple(data["CI_shutter_size_losses_real_last_turn"][-1]), 'r_-', color='black', linewidth=2,mew=2, ms=20)


    ax.set_xticks(list(np.arange(len(data.index) - 1)) + [len(data.index) - 0.5])
    # ax.set_xticklabels(data.index)
    ax.set_xticklabels([discription_dict[name] for name in data.index])

    lax = fig.add_subplot(grid[1, 0])
    h, l = ax.get_legend_handles_labels()
    lax.legend(h, l, borderaxespad=0, loc="center", fancybox=True, shadow=True, ncol=2)
    lax.axis("off")


    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = PIL.Image.open(buf)

    if group_by == None:
        plt.savefig(f"{path}Shutter sizes results.png", bbox_inches='tight')
    else:

        plt.savefig(f"{path}Shutter sizes results {group_by}.png", bbox_inches='tight')

    plt.close('all')


def save_save_game_len(path, num_games, width, height, font_size, opponent_name, data, group_by=None):
    mpl.rcParams.update({'font.size': font_size})


    fig = plt.figure(constrained_layout=True)
    fig.suptitle(f"Games length results - {opponent_name} - {group_by}",  y=1.05)

    fig.set_size_inches(width, height)
    grid = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[20, 1])

    ind = np.arange(len(data.index))
    width = 0.25

    ax = fig.add_subplot(grid[0, 0])

    ax.bar(ind-width, data["avg game len"], width=width, label="average game len", color="blue")
    ax.bar(ind, data["avg game len (wins)"], width=width, label="average game len - wins", color="green")
    ax.bar(ind + width, data["avg game len (losses)"], width=width, label="average game len - losses", color="red")

    for index, ci in zip(ind - width, data["CI_game_length"]):
        ax.plot((index, index), npstr2tuple(ci), 'r_-', color='black', linewidth=4, mew=4, ms=20)

    for index, ci in zip(ind, data["CI_game_length_wins"]):
        ax.plot((index, index), npstr2tuple(ci), 'r_-', color='black', linewidth=4, mew=4, ms=20)

    for index, ci in zip(ind + width, data["CI_game_length_losses"]):
        ax.plot((index, index), npstr2tuple(ci), 'r_-', color='black', linewidth=4, mew=4, ms=20)



    ax.set_xticks(ind)
    # ax.set_xticklabels(data.index)
    ax.set_xticklabels([discription_dict[name] for name in data.index])

    lax = fig.add_subplot(grid[1, 0])
    h, l = ax.get_legend_handles_labels()
    lax.legend(h, l, borderaxespad=0, loc="center", fancybox=True, shadow=True)
    lax.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = PIL.Image.open(buf)

    if group_by == None:
        plt.savefig(f"{path}Games lengths results.png", bbox_inches='tight')
    else:

        plt.savefig(f"{path}Games lengths results {group_by}.png", bbox_inches='tight')

    plt.close('all')


def save_win_ratio_no_ties(path, num_games, width, height, font_size, opponent_name, data, group_by=None, is_interchangeable_color=False):
    mpl.rcParams.update({'font.size': font_size})


    fig = plt.figure(constrained_layout=True)
    fig.suptitle(f"Win ratio ignoring ties - {opponent_name} - {group_by}", y=1.05)

    fig.set_size_inches(width, height)
    grid = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[20,1])

    ind = np.arange(len(data.index))
    width = 0.5


    ax = fig.add_subplot(grid[0, 0])


    ax.set_ylim([0,1.1])


    if is_interchangeable_color:
        colors = ['blue', 'red'] * (len(data["no. wins"][:-4]) // 2) + ['grey'] * 4
    else:

        if len(data["no. wins"]) < 16:

            if group_by == "mcts_25_models":
                coloring = 'blue'
            elif group_by == "mcts_50_models":
                coloring = 'red'
            else:
                coloring = 'green'

            colors = [coloring] * len(data["no. wins"][:-4]) + ['grey'] * 4

        else:
            colors = ['blue'] * 4 + ['red'] * 4 + ['green'] * 4 + ['grey'] * 4


    ax.bar(ind, data["no. wins"] * (1 / (num_games - data["no. ties"])), width=width, color=colors, alpha=0.5)


    for index, ci in zip(ind, data["CI_wins_losses"]):
        ax.plot((index, index), npstr2tuple(ci), 'r_-', color='black', linewidth=4, mew=4, ms=40)


    ax.set_xticks(ind)
    # ax.set_xticklabels(data.index)
    ax.set_xticklabels([discription_dict[name] for name in data.index])


    lax = fig.add_subplot(grid[1, 0])
    h, l = ax.get_legend_handles_labels()
    lax.legend(h,l, borderaxespad=0, loc="center", fancybox=True, shadow=True)
    lax.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = PIL.Image.open(buf)


    if group_by == None:
        plt.savefig(f"{path}Win ratio ignoring ties.png", bbox_inches='tight')
    else:

        plt.savefig(f"{path}Win ratio ignoring ties {group_by}.png", bbox_inches='tight')


    plt.close('all')


def save_game_results(path, num_games, width, height, font_size, opponent_name, data, group_by=None):
    mpl.rcParams.update({'font.size': font_size})


    fig = plt.figure(constrained_layout=True)
    fig.suptitle(f"Games results - {opponent_name} - {group_by}", y=1.05)

    fig.set_size_inches(width, height)
    grid = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[20,1])

    ind = np.arange(len(data.index))
    width = 0.5

    ax = fig.add_subplot(grid[0, 0])

    ax.set_ylim([0,num_games*1.1])

    ax.bar(ind, data["no. wins"], width=width, label="wins", color="green")
    ax.bar(ind, data["no. losses"], width=width, label="losses", color="red", bottom=data["no. wins"])
    ax.bar(ind, data["no. ties"], width=width, label = "ties", color="yellow", bottom=data["no. wins"]+data["no. losses"])

    ax.set_xticks(ind)
    # ax.set_xticklabels(data.index)
    ax.set_xticklabels([discription_dict[name] for name in data.index])


    lax = fig.add_subplot(grid[1, 0])
    h, l = ax.get_legend_handles_labels()
    lax.legend(h,l, borderaxespad=0, loc="center", fancybox=True, shadow=True)
    lax.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = PIL.Image.open(buf)


    if group_by == None:
        plt.savefig(f"{path}Games results.png", bbox_inches='tight')
    else:

        plt.savefig(f"{path}Games results {group_by}.png", bbox_inches='tight')


    plt.close('all')



def call_collage_statistics_results(board_name, opponents, sub_dir):

    opponents_names = [opp.name for opp in opponents]

    path_collage = f"/home/lirontyomkin/AlphaZero_Gomoku/matches/{sub_dir}/{board_name} summery/"

    if not os.path.exists(path_collage):
        os.makedirs(path_collage)

    for variation in all_models_variations:
        models = variation[0]
        group_by = variation[1]

        listofimages1 = [
            f"/home/lirontyomkin/AlphaZero_Gomoku/matches/{sub_dir}/vs {opponent_name}/{board_name}/Fraction of plays with shutter results {group_by}.png"
            for opponent_name in opponents_names]

        listofimages2 = [
            f"/home/lirontyomkin/AlphaZero_Gomoku/matches/{sub_dir}/vs {opponent_name}/{board_name}/Games lengths results {group_by}.png"
            for opponent_name in opponents_names]

        listofimages3 = [
            f"/home/lirontyomkin/AlphaZero_Gomoku/matches/{sub_dir}/vs {opponent_name}/{board_name}/Games results {group_by}.png"
            for opponent_name in opponents_names]

        listofimages4 = [
            f"/home/lirontyomkin/AlphaZero_Gomoku/matches/{sub_dir}/vs {opponent_name}/{board_name}/Shutter sizes results {group_by}.png"
            for opponent_name in opponents_names]

        listofimages5 = [
            f"/home/lirontyomkin/AlphaZero_Gomoku/matches/{sub_dir}/vs {opponent_name}/{board_name}/Win ratio ignoring ties {group_by}.png"
            for opponent_name in opponents_names]

        create_collages_boards(listofimages1, f"Fraction of plays with shutter results {group_by}", path_collage, group_by)
        create_collages_boards(listofimages2, f"Games lengths results {group_by}", path_collage, group_by)
        create_collages_boards(listofimages3, f"Games results {group_by}", path_collage, group_by)
        create_collages_boards(listofimages4, f"Shutter sizes results {group_by}", path_collage, group_by)
        create_collages_boards(listofimages5, f"Win ratio ignoring ties {group_by}", path_collage, group_by)



def create_collages_boards(listofimages, fig_name, path, group_by=None):

    im_check = PIL.Image.open(listofimages[0])
    width1, height1 = im_check.size

    cols = 1
    rows = len(listofimages)

    width = width1 * cols
    height = height1 * rows

    thumbnail_width = width // cols
    thumbnail_height = height // rows

    size = thumbnail_width, thumbnail_height
    new_im = PIL.Image.new('RGB', (width, height))
    ims = []
    for p in listofimages:
        im = PIL.Image.open(p)
        im.thumbnail(size)
        ims.append(im)
    i = 0
    x = 0
    y = 0
    for col in range(cols):
        for row in range(rows):
            # print(i, x, y)
            new_im.paste(ims[i], (x, y))
            i += 1
            y += thumbnail_height
        x += thumbnail_width
        y = 0

    if group_by == None:
        new_im.save(path + f"{fig_name}.png")
    else:

        if not os.path.exists(path + f"{group_by}/"):
            os.makedirs(path + f"{group_by}/")

        new_im.save(path + f"{group_by}/{fig_name}.png")



def save_states_from_history_empty_board(opponent, players_list, num_games, sub_dir):

    path = f"/home/lirontyomkin/AlphaZero_Gomoku/matches/{sub_dir}/vs {opponent.name}/empty board/"
    start_player = 1

    states = []

    for cur_player in players_list:

        print(cur_player.name, ": \n")

        game_stats_path = f"{path}{cur_player.name}/full_{num_games}_games_stats"
        game_stats = pickle.load(open(game_stats_path, "rb"))
        chosen_games = np.random.choice(num_games, size=3)

        for chosen_game in chosen_games:
            _, _, _, _, _, game_history = game_stats[chosen_game]

            board_state = copy.deepcopy(EMPTY_BOARD_6X6[0])
            print(board_state, "\n")

            cur_player = 1
            last_move_p1 = None
            last_move_p2 = None


            for row, col in game_history:

                board_state[row, col] = cur_player
                print(board_state, "\n")

                if cur_player == 1:
                    last_move_p1 = [board_state.shape[1] - 1 - row, col]
                else:
                    last_move_p2 = [board_state.shape[1] - 1 - row, col]


                if all([(state[0] != board_state).any() for state in states]):

                    states.append((copy.deepcopy(board_state), last_move_p1, last_move_p2, start_player))

                cur_player = 3 - cur_player


    print(f"{len(states)} states")

    names = "_".join([player.name for player in players_list])

    pickle.dump(states, open(f"{path}sampled_states_{names}", "wb"))





def the_twelve_6_X_6():
    n_playout = 400

    v7 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p3_v7/current_policy_1500.model',
          'v7_1500', 3, True, False)
    v9 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p3_v9/current_policy_1500.model',
          'v9_1500', 3, True, False)
    v10 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v10/current_policy_1500.model',
           'v10_1500', 4, True, False)
    v10_random = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v10/current_policy_1500.model',
                  'v10_1500_random', 4, True, True)

    v12 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v12/current_policy_5000.model',
           'v12_5000', 4, True, False)
    v14 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v14/current_policy_5000.model',
           'v14_5000', 4, True, False)
    v16 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v16/current_policy_5000.model',
           'v16_5000', 4, True, False)
    v18 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v18/current_policy_5000.model',
           'v18_5000', 4, True, False)
    v20 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v20/current_policy_5000.model',
           'v20_5000', 4, True, False)
    v22 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v22/current_policy_5000.model',
           'v22_5000', 4, True, False)

    v23 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v23/current_policy_5000.model',
           'v23_5000', 4, True, False)
    v24 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v24/current_policy_5000.model',
           'v24_5000', 4, True, False)
    v25 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v25/current_policy_5000.model',
           'v25_5000', 4, True, False)
    v26 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v26/current_policy_5000.model',
           'v26_5000', 4, True, False)
    v27 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v27/current_policy_5000.model',
           'v27_5000', 4, True, False)
    v28 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v28/current_policy_5000.model',
           'v28_5000', 4, True, False)
    v29 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v29/current_policy_5000.model',
           'v29_5000', 4, True, False)
    v30 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v30/current_policy_5000.model',
           'v30_5000', 4, True, False)
    v31 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v31/current_policy_5000.model',
           'v31_5000', 4, True, False)
    v32 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v32/current_policy_5000.model',
           'v32_5000', 4, True, False)
    v33 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v33/current_policy_5000.model',
           'v33_5000', 4, True, False)
    v34 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v34/current_policy_5000.model',
           'v34_5000', 4, True, False)

    limit_shutter = 0

    policy_v7 = PolicyValueNet(6, 6, model_file=v7[0], input_plains_num=v7[2],
                               shutter_threshold_availables=limit_shutter)
    player_v7 = MCTSPlayer(policy_v7.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v7[3],
                           name=v7[1], input_plains_num=v7[2], is_random_last_turn=v7[4])
    policy_v9 = PolicyValueNet(6, 6, model_file=v9[0], input_plains_num=v9[2],
                               shutter_threshold_availables=limit_shutter)
    player_v9 = MCTSPlayer(policy_v9.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v9[3],
                           name=v9[1], input_plains_num=v9[2], is_random_last_turn=v9[4])
    policy_v10 = PolicyValueNet(6, 6, model_file=v10[0], input_plains_num=v10[2],
                                shutter_threshold_availables=limit_shutter)
    player_v10 = MCTSPlayer(policy_v10.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v10[3],
                            name=v10[1], input_plains_num=v10[2], is_random_last_turn=v10[4])
    policy_v10_random = PolicyValueNet(6, 6, model_file=v10_random[0], input_plains_num=v10_random[2],
                                       shutter_threshold_availables=limit_shutter)
    player_v10_random = MCTSPlayer(policy_v10_random.policy_value_fn, c_puct=5, n_playout=n_playout,
                                   no_playouts=v10_random[3],
                                   name=v10_random[1], input_plains_num=v10_random[2],
                                   is_random_last_turn=v10_random[4])

    # policy_v12 = PolicyValueNet(6, 6, model_file=v12[0], input_plains_num=v12[2])
    # player_v12 = MCTSPlayer(policy_v12.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v12[3],
    #                         name=v12[1], input_plains_num=v12[2], is_random_last_turn=v12[4])
    #
    # policy_v14 = PolicyValueNet(6, 6, model_file=v14[0], input_plains_num=v14[2])
    # player_v14 = MCTSPlayer(policy_v14.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v14[3],
    #                         name=v14[1], input_plains_num=v14[2], is_random_last_turn=v14[4])
    #
    # policy_v16 = PolicyValueNet(6, 6, model_file=v16[0], input_plains_num=v16[2])
    # player_v16 = MCTSPlayer(policy_v16.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v16[3],
    #                         name=v16[1], input_plains_num=v16[2], is_random_last_turn=v16[4])
    #
    # policy_v18 = PolicyValueNet(6, 6, model_file=v18[0], input_plains_num=v18[2])
    # player_v18 = MCTSPlayer(policy_v18.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v18[3],
    #                         name=v18[1], input_plains_num=v18[2], is_random_last_turn=v18[4])
    #
    # policy_v20 = PolicyValueNet(6, 6, model_file=v20[0], input_plains_num=v20[2])
    # player_v20 = MCTSPlayer(policy_v20.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v20[3],
    #                         name=v20[1], input_plains_num=v20[2], is_random_last_turn=v20[4])
    #
    # policy_v22 = PolicyValueNet(6, 6, model_file=v22[0], input_plains_num=v22[2])
    # player_v22 = MCTSPlayer(policy_v22.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v22[3],
    #                         name=v22[1], input_plains_num=v22[2], is_random_last_turn=v22[4])

    policy_v23 = PolicyValueNet(6, 6, model_file=v23[0], input_plains_num=v23[2],
                                shutter_threshold_availables=limit_shutter)
    player_v23 = MCTSPlayer(policy_v23.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v23[3],
                            name=v23[1], input_plains_num=v23[2], is_random_last_turn=v23[4])

    policy_v24 = PolicyValueNet(6, 6, model_file=v24[0], input_plains_num=v24[2],
                                shutter_threshold_availables=limit_shutter)
    player_v24 = MCTSPlayer(policy_v24.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v24[3],
                            name=v24[1], input_plains_num=v24[2], is_random_last_turn=v24[4])

    policy_v25 = PolicyValueNet(6, 6, model_file=v25[0], input_plains_num=v25[2],
                                shutter_threshold_availables=limit_shutter)
    player_v25 = MCTSPlayer(policy_v25.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v25[3],
                            name=v25[1], input_plains_num=v25[2], is_random_last_turn=v25[4])

    policy_v26 = PolicyValueNet(6, 6, model_file=v26[0], input_plains_num=v26[2],
                                shutter_threshold_availables=limit_shutter)
    player_v26 = MCTSPlayer(policy_v26.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v26[3],
                            name=v26[1], input_plains_num=v26[2], is_random_last_turn=v26[4])

    policy_v27 = PolicyValueNet(6, 6, model_file=v27[0], input_plains_num=v27[2],
                                shutter_threshold_availables=limit_shutter)
    player_v27 = MCTSPlayer(policy_v27.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v27[3],
                            name=v27[1], input_plains_num=v27[2], is_random_last_turn=v27[4])

    policy_v28 = PolicyValueNet(6, 6, model_file=v28[0], input_plains_num=v28[2],
                                shutter_threshold_availables=limit_shutter)
    player_v28 = MCTSPlayer(policy_v28.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v28[3],
                            name=v28[1], input_plains_num=v28[2], is_random_last_turn=v28[4])

    policy_v29 = PolicyValueNet(6, 6, model_file=v29[0], input_plains_num=v29[2],
                                shutter_threshold_availables=limit_shutter)
    player_v29 = MCTSPlayer(policy_v29.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v29[3],
                            name=v29[1], input_plains_num=v29[2], is_random_last_turn=v29[4])

    policy_v30 = PolicyValueNet(6, 6, model_file=v30[0], input_plains_num=v30[2],
                                shutter_threshold_availables=limit_shutter)
    player_v30 = MCTSPlayer(policy_v30.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v30[3],
                            name=v30[1], input_plains_num=v30[2], is_random_last_turn=v30[4])

    policy_v31 = PolicyValueNet(6, 6, model_file=v31[0], input_plains_num=v31[2],
                                shutter_threshold_availables=limit_shutter)
    player_v31 = MCTSPlayer(policy_v31.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v31[3],
                            name=v31[1], input_plains_num=v31[2], is_random_last_turn=v31[4])

    policy_v32 = PolicyValueNet(6, 6, model_file=v32[0], input_plains_num=v32[2],
                                shutter_threshold_availables=limit_shutter)
    player_v32 = MCTSPlayer(policy_v32.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v32[3],
                            name=v32[1], input_plains_num=v32[2], is_random_last_turn=v32[4])

    policy_v33 = PolicyValueNet(6, 6, model_file=v33[0], input_plains_num=v33[2],
                                shutter_threshold_availables=limit_shutter)
    player_v33 = MCTSPlayer(policy_v33.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v33[3],
                            name=v33[1], input_plains_num=v33[2], is_random_last_turn=v33[4])

    policy_v34 = PolicyValueNet(6, 6, model_file=v34[0], input_plains_num=v34[2],
                                shutter_threshold_availables=limit_shutter)
    player_v34 = MCTSPlayer(policy_v34.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v34[3],
                            name=v34[1], input_plains_num=v34[2], is_random_last_turn=v34[4])

    players_list = [

        player_v23, player_v24, player_v25, player_v26,
        player_v27, player_v28, player_v29, player_v30,
        player_v31, player_v32, player_v33, player_v34,

        player_v7, player_v9, player_v10, player_v10_random]

    opponent_player_1 = Heuristic_player(name="forcing heuristic", heuristic="interaction with forcing")
    opponent_player_2 = PUREMCTS(c_puct=5, n_playout=500, name="pure MCTS 500")
    opponent_player_3 = PUREMCTS(c_puct=5, n_playout=1000, name="pure MCTS 1000")

    v9_5000 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p3_v9/current_policy_5000.model',
               'v9_5000_no_MCTS', 3, True, False)
    policy_opponent_4 = PolicyValueNet(6, 6, model_file=v9_5000[0], input_plains_num=v9_5000[2])
    opponent_player_4 = MCTSPlayer(policy_opponent_4.policy_value_fn, c_puct=5, n_playout=n_playout,
                                   no_playouts=v9_5000[3],
                                   name=v9_5000[1], input_plains_num=v9_5000[2], is_random_last_turn=v9_5000[4])

    v10_5000 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v10/current_policy_5000.model',
                f'v10_5000_no_MCTS', 4, True, False)
    policy_opponent_5 = PolicyValueNet(6, 6, model_file=v10_5000[0], input_plains_num=v10_5000[2])
    opponent_player_5 = MCTSPlayer(policy_opponent_5.policy_value_fn, c_puct=5, n_playout=n_playout,
                                   no_playouts=v10_5000[3],
                                   name=v10_5000[1], input_plains_num=v10_5000[2], is_random_last_turn=v10_5000[4])

    opponents = [opponent_player_1, opponent_player_2, opponent_player_3, opponent_player_4, opponent_player_5]

    set_start_method("spawn")

    num_games = 1000
    sub_dir = f"statistics_limit_all_to_shutter_{limit_shutter}"

    compare_all_models_statistics(players_list, opponents, width=6, height=6, n=4, num_games=num_games, sub_dir=sub_dir)
    plot_all_statistics_results(opponents, num_games=num_games, sub_dir=sub_dir)
    save_states_from_history_empty_board(opponent_player_1, players_list, 1000, sub_dir=sub_dir)




def the_six_10X10():

    n_playout = 400
    limit_shutter = None


    v_03 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_10_10_5_p3_v0_3/current_policy_5000.model',
           'v_03_5000', 4, True, False)
    v_04 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_10_10_5_p3_v0_4/current_policy_5000.model',
           'v_04_5000', 4, True, False)
    v_05 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_10_10_5_p3_v0_5/current_policy_5000.model',
           'v_05_5000', 4, True, False)
    v_06 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_10_10_5_p3_v0_6/current_policy_5000.model',
           'v_06_5000', 4, True, False)
    v_07 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_10_10_5_p3_v0_1/current_policy_5000.model',
            'v_07_5000', 4, True, False)
    v_08 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_10_10_5_p3_v0_2/current_policy_5000.model',
            'v_08_5000', 4, True, False)



    policy_v_03 = PolicyValueNet(10, 10, model_file=v_03[0], input_plains_num=v_03[2],
                                shutter_threshold_availables=limit_shutter)
    player_v_03 = MCTSPlayer(policy_v_03.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v_03[3],
                            name=v_03[1], input_plains_num=v_03[2], is_random_last_turn=v_03[4])

    policy_v_04 = PolicyValueNet(10, 10, model_file=v_04[0], input_plains_num=v_04[2],
                                shutter_threshold_availables=limit_shutter)
    player_v_04 = MCTSPlayer(policy_v_04.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v_04[3],
                            name=v_04[1], input_plains_num=v_04[2], is_random_last_turn=v_04[4])

    policy_v_05 = PolicyValueNet(10, 10, model_file=v_05[0], input_plains_num=v_05[2],
                                shutter_threshold_availables=limit_shutter)
    player_v_05 = MCTSPlayer(policy_v_05.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v_05[3],
                            name=v_05[1], input_plains_num=v_05[2], is_random_last_turn=v_05[4])

    policy_v_06 = PolicyValueNet(10, 10, model_file=v_06[0], input_plains_num=v_06[2],
                                shutter_threshold_availables=limit_shutter)
    player_v_06 = MCTSPlayer(policy_v_06.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v_06[3],
                            name=v_06[1], input_plains_num=v_06[2], is_random_last_turn=v_06[4])

    policy_v_07 = PolicyValueNet(10, 10, model_file=v_07[0], input_plains_num=v_07[2],
                                 shutter_threshold_availables=limit_shutter)
    player_v_07 = MCTSPlayer(policy_v_07.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v_07[3],
                             name=v_07[1], input_plains_num=v_07[2], is_random_last_turn=v_07[4])

    policy_v_08 = PolicyValueNet(10, 10, model_file=v_08[0], input_plains_num=v_08[2],
                                 shutter_threshold_availables=limit_shutter)
    player_v_08 = MCTSPlayer(policy_v_08.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=v_08[3],
                             name=v_08[1], input_plains_num=v_08[2], is_random_last_turn=v_08[4])



    players_list = [player_v_03, player_v_04, player_v_05, player_v_06, player_v_07, player_v_08]




    opponent_player_1 = Heuristic_player(name="forcing heuristic", heuristic="interaction with forcing")
    opponent_player_2 = PUREMCTS(c_puct=5, n_playout=500, name="pure MCTS 500")
    opponent_player_3 = PUREMCTS(c_puct=5, n_playout=1000, name="pure MCTS 1000")

    v_01 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_10_10_5_p3_v0_1/current_policy_5000.model',
               'v_01_5000_no_MCTS', 3, True, False)
    policy_opponent_4 = PolicyValueNet(10, 10, model_file=v_01[0], input_plains_num=v_01[2])
    opponent_player_4 = MCTSPlayer(policy_opponent_4.policy_value_fn, c_puct=5, n_playout=n_playout,
                                   no_playouts=v_01[3],
                                   name=v_01[1], input_plains_num=v_01[2], is_random_last_turn=v_01[4])

    v_02 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_10_10_5_p3_v0_2/current_policy_5000.model',
                f'v_02_5000_no_MCTS', 4, True, False)
    policy_opponent_5 = PolicyValueNet(10, 10, model_file=v_02[0], input_plains_num=v_02[2])
    opponent_player_5 = MCTSPlayer(policy_opponent_5.policy_value_fn, c_puct=5, n_playout=n_playout,
                                   no_playouts=v_02[3],
                                   name=v_02[1], input_plains_num=v_02[2], is_random_last_turn=v_02[4])

    opponents = [opponent_player_1, opponent_player_2, opponent_player_3, opponent_player_4, opponent_player_5]


    set_start_method("spawn")

    num_games = 1000
    sub_dir = f"10X10_statistics"

    compare_all_models_statistics(players_list, opponents, width=10, height=10, n=5, num_games=num_games, sub_dir=sub_dir)
    plot_all_statistics_results(opponents, num_games=num_games, sub_dir=sub_dir)
    save_states_from_history_empty_board(opponent_player_1, players_list, 1000, sub_dir=sub_dir)



if __name__ == '__main__':
    pass