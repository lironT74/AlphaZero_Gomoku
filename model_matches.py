from __future__ import print_function
from multiprocessing import Pool
from mcts_alphaZero import MCTSPlayer
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

    path1, name1, plains1, no_playouts1 = model1
    path2, name2, plains2, no_playouts2 = model2

    best_policy_1 = PolicyValueNet(width, height, model_file=path1, input_plains_num=plains1)
    mcts_player_1 = MCTSPlayer(best_policy_1.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts = no_playouts1, name=name1, input_plains_num=plains1)

    best_policy_2 = PolicyValueNet(width, height, model_file=path2, input_plains_num=plains2)
    mcts_player_2 = MCTSPlayer(best_policy_2.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts = no_playouts2, name=name2, input_plains_num=plains2)

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



def compare_all_models_statistics(models_list, width=6, height=6, n=4, open_path_threshold=-1, n_playout=400, num_games=100):

    jobs = []
    for i in range(len(models_list)):
        for j in range(i+1, len(models_list)):
            jobs.append((models_list[i], models_list[j], width, height, n, open_path_threshold, n_playout, num_games))


    with Pool(int(comb(len(models_list), 2))) as pool:
        print(f"Using {pool._processes} workers. There are {len(jobs)} jobs: \n")

        pool.starmap(collect_statistics_two_models, jobs)
        pool.close()
        pool.join()



def collect_statistics_two_models(model1, model2, width, height, n, open_path_threshold, n_playout, num_games):


    path1, name1, plains1, no_playouts1 = model1
    path2, name2, plains2, no_playouts2 = model2

    best_policy_1 = PolicyValueNet(width, height, model_file=path1, input_plains_num=plains1)
    mcts_player_1 = MCTSPlayer(best_policy_1.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=no_playouts1,
                               name=name1, input_plains_num=plains1)

    best_policy_2 = PolicyValueNet(width, height, model_file=path2, input_plains_num=plains2)
    mcts_player_2 = MCTSPlayer(best_policy_2.policy_value_fn, c_puct=5, n_playout=n_playout, no_playouts=no_playouts2,
                               name=name2, input_plains_num=plains2)

    for board_state, board_name, p1, p2, alternative_p1, alternative_p2 in PAPER_TRUNCATED_BOARDS:

        save_games_statistics(width=width,
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
                      open_path_threshold=open_path_threshold, num_games=num_games)

        if plains1 + plains2 >= 7:
            save_games_statistics(width=width,
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
                          open_path_threshold=open_path_threshold, num_games=num_games)

        if plains1 + plains2 == 8:
            save_games_statistics(width=width,
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
                          open_path_threshold=open_path_threshold, num_games=num_games)

            save_games_statistics(width=width,
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
                          open_path_threshold=open_path_threshold, num_games=num_games)

    for board_state, board_name, p1, p2, alternative_p1, alternative_p2 in PAPER_FULL_BOARDS:
        save_games_statistics(width=width,
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
                      open_path_threshold=open_path_threshold, num_games=num_games)

    board_state, board_name, p1, p2, alternative_p1, alternative_p2 = EMPTY_BOARD

    save_games_statistics(width=width,
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
                  open_path_threshold=open_path_threshold, num_games=num_games)



def save_games_statistics(width, height, n, board_state, board_name, mcts_player_1,
                          mcts_player_2, last_move_p1, last_move_p2, correct_move_p1,
                          correct_move_p2, start_player, open_path_threshold, num_games):

    i_board1, board1 = initialize_board_without_init_call(width, height, n, input_board=board_state,
                                                          open_path_threshold=open_path_threshold)
    game1 = Game(board1)


    wins = []

    shutters = {mcts_player_1.name: [], mcts_player_2.name: []}

    for i in range(num_games):

        current_players = {2-i%2: mcts_player_1, 1 + i%2: mcts_player_2}

        winner, game_length, shutter_sizes = game1.start_play(player1=current_players[1],
                                                              player2=current_players[2],
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
            wins.append(current_players[winner].name)


        plays = range(1, game_length + 1, 1)
        start_player_range = [(index, shutter) for index, shutter in zip(plays[0::2], shutter_sizes[start_player]) if
                              shutter != -1]
        second_player_range = [(index, shutter) for index, shutter in zip(plays[1::2], shutter_sizes[3 - start_player])
                               if shutter != -1]


        shutters[current_players[start_player].name].extend([x[1] for x in start_player_range])
        shutters[current_players[3 - start_player].name].extend([x[1] for x in second_player_range])



    columns = [f"average shutter size", f"no. of plays which had shutter != -1",  "no. wins"]

    index = [f"{mcts_player_1.name}", f"{mcts_player_2.name}"]

    np_results = np.array([[np.average(shutters[mcts_player_1.name]), len(shutters[mcts_player_1.name]), wins.count(mcts_player_1.name)],
                           [np.average(shutters[mcts_player_2.name]), len(shutters[mcts_player_2.name]), wins.count(mcts_player_2.name)]])

    df = pd.DataFrame(np_results, index=index, columns=columns)

    print(df.to_string())

    if mcts_player_1.no_playouts or mcts_player_2.no_playouts:
        path = f"/home/lirontyomkin/AlphaZero_Gomoku/matches/statistics/no MCTS/{mcts_player_1.name} vs {mcts_player_2.name}/"
    else:
        path = f"/home/lirontyomkin/AlphaZero_Gomoku/matches/statistics/{mcts_player_1.name} vs {mcts_player_2.name}/"

    if not os.path.exists(f"{path}{board_name}/"):
        os.makedirs(f"{path}{board_name}/")

    df.to_excel(f"{path}{board_name}/{num_games} games statistics.xlsx", index=True, header=True)

    f = open(f"{path}{mcts_player_1.name} vs {mcts_player_2.name} {num_games} games results.txt", "a")
    f.write(f"----------> {board_name} <----------\n")
    f.write(df.to_string())
    f.write('\n\n')
    f.close()


if __name__ == '__main__':

    v7 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p3_v7/current_policy_2100.model', 'pt_6_6_4_p3_v7_2100', 3, False)
    v9 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p3_v9/current_policy_1350.model', 'pt_6_6_4_p3_v9_1350', 3, False)
    v10 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v10/current_policy_1150.model', 'pt_6_6_4_p4_v10_1150', 4, False)

    models = [v7, v9, v10]
    compare_all_models(models)


    v9 = ( '/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p3_v9/current_policy_1500.model',
           'pt_6_6_4_p3_v9_1500', 3,True)

    v10 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p4_v10/current_policy_1500.model',
           'pt_6_6_4_p4_v10_1500', 4, True)

    v7 = ('/home/lirontyomkin/AlphaZero_Gomoku/models/pt_6_6_4_p3_v7/current_policy_1500.model',
          'pt_6_6_4_p3_v7_1500', 3, True)

    models = [v7, v9, v10]

    compare_all_models(models)
    # compare_all_models_statistics(models, num_games=10000)

